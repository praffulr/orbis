import os
import glob
import json
import random

from collections import defaultdict, Counter

import numpy as np
import torch


# CONFIG

FEATURE_ROOT = "vjepa_features_tb5"

NORMAL_DIR = os.path.join(FEATURE_ROOT, "normal")
ANOMALY_DIR = os.path.join(FEATURE_ROOT, "anomaly")

ANNOTATION_DIR = "annotations"

OUTPUT_DIR = "cached_features_tb5_new"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_final_mc.pt")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_final_mc.pt")

VAL_RATIO = 0.15
SEED = 42

# Automatically chosen on first feature file
FEATURE_LAYER = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# DoTA CLASS DEFINITIONS

DOTA_CLASS_NAMES = {
    0: "normal",
    1: "start_stop_or_stationary",
    2: "moving_ahead_or_waiting",
    3: "lateral",
    4: "oncoming",
    5: "turning",
    6: "pedestrian",
    7: "obstacle",
    8: "leave_to_right",
    9: "leave_to_left",
}

NUM_CLASSES = len(DOTA_CLASS_NAMES)

# CLASS MAPPING


def build_class_mapping():

    """
    Fixed DoTA class mapping.

    We keep IDs fixed across experiments.
    """

    class_mapping = {name: idx for idx, name in DOTA_CLASS_NAMES.items()}

    print("\nClass Mapping")
    print("-" * 60)

    for name, idx in class_mapping.items():

        print(f"{name:35s} -> {idx}")

    print("-" * 60)

    print("Number of classes:", len(class_mapping))

    return class_mapping


# LOAD ALL ANNOTATIONS


def load_annotations():
    """
    Creates a cache of the form

    annotation_cache[video] = {
        "frames": {
            frame_id -> accident_id
        },
        "video_accident": accident_id
    }

    The video-level accident is exactly how the Orbis
    pipeline defines the class.
    """

    annotation_cache = {}

    annotation_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.json"))

    print(f"\nLoading {len(annotation_files)} annotation files...")

    skipped_unknown = 0

    for file in annotation_files:

        with open(file, "r") as f:
            ann = json.load(f)

        video = ann["video_name"]

        frame_dict = {}

        video_accident = None

        # Parse frame annotations

        for item in ann["labels"]:

            frame_id = item["frame_id"]

            accident_id = int(item["accident_id"])

            frame_dict[frame_id] = accident_id

            if accident_id > 0 and accident_id <= 9:
                video_accident = accident_id

        # Skip unknown accident videos

        if video_accident is None:

            skipped_unknown += 1

            continue

        annotation_cache[video] = {
            "frames": frame_dict,
            "video_accident": video_accident,
        }

    print(f"Loaded annotations : {len(annotation_cache)} videos")
    print(f"Skipped unknown    : {skipped_unknown}")

    return annotation_cache


# DATASET STATISTICS


def print_dataset_statistics(annotation_cache):

    counter = Counter()

    for item in annotation_cache.values():

        counter[item["video_accident"]] += 1

    print("\nVideo-level accident distribution")

    print("-" * 50)

    for cls in range(1, NUM_CLASSES):

        print(f"{cls:2d}  " f"{DOTA_CLASS_NAMES[cls]:28s}" f"{counter[cls]}")

    print("-" * 50)


# FEATURE EXTRACTION


def extract_feature(data):
    """
    Extracts the requested V-JEPA feature tensor.

    Supports both

        feature : Tensor

    and

        feature : Dict[str, Tensor]

    formats.
    """

    global FEATURE_LAYER

    feature = data["feature"]

    # New cache format

    if isinstance(feature, dict):

        # Automatically choose first tensor if layer
        # has not been specified.

        if FEATURE_LAYER is None:

            for key, value in feature.items():

                if torch.is_tensor(value):

                    FEATURE_LAYER = key

                    print(f"\nAutomatically selected feature layer: {FEATURE_LAYER}")

                    break

        if FEATURE_LAYER not in feature:

            raise RuntimeError(
                f"{FEATURE_LAYER} not found.\n"
                f"Available layers: {list(feature.keys())}"
            )

        feature = feature[FEATURE_LAYER]

    if not torch.is_tensor(feature):

        raise RuntimeError("Feature is not a torch tensor.")

    feature = feature.float()

    # Expected V-JEPA shape

    if feature.shape != (576, 768):

        raise RuntimeError(
            f"Expected feature shape (576,768), " f"received {tuple(feature.shape)}"
        )

    return feature


# PROCESS ONE FEATURE FILE


def process_feature_file(
    feature_path,
    annotation_cache,
):
    """
    Converts one extracted V-JEPA feature file into one
    training sample.

    Returns

    {
        feature
        label
        mc_label
        source_mc_label
        video
        indices
    }

    Following the Orbis protocol:

    ------------

    Normal clip

        binary label      = 0
        mc_label          = 0 (normal)
        source_mc_label   = accident type

    ------------

    Anomaly clip

        binary label      = 1
        mc_label          = accident type
        source_mc_label   = accident type

    ------------
    """

    data = torch.load(
        feature_path,
        weights_only=False,
    )

    feature = extract_feature(data)

    video = data["video"]

    indices = data["indices"]

    binary_label = int(data["label"])

    # Basic validation

    if len(indices) != 5:

        raise RuntimeError(f"{video}: Expected 5-frame context.")

    if video not in annotation_cache:

        raise RuntimeError(f"No annotation found for {video}")

    annotation = annotation_cache[video]

    video_accident = annotation["video_accident"]

    # Ignore unknown classes

    if video_accident > 9:

        return None

    # Labels following Orbis

    if binary_label == 0:

        # Normal sample

        mc_label = 0

        source_mc_label = video_accident

    else:

        # Anomalous sample

        mc_label = video_accident

        source_mc_label = video_accident

    return {
        "feature": feature,
        "label": torch.tensor(
            binary_label,
            dtype=torch.long,
        ),
        "mc_label": torch.tensor(
            mc_label,
            dtype=torch.long,
        ),
        "source_mc_label": torch.tensor(
            source_mc_label,
            dtype=torch.long,
        ),
        "video": video,
        "indices": indices,
    }


# BUILD DATASET


def build_dataset(
    feature_files,
    class_mapping,
    annotation_cache,
):
    """
    Builds one cached dataset (train or validation).

    Returns a dictionary containing

        features
        labels
        mc_labels
        source_mc_labels
        videos
        indices

    along with useful metadata.
    """

    features = []

    binary_labels = []

    multiclass_labels = []

    source_multiclass_labels = []

    videos = []

    frame_indices = []

    skipped = 0

    class_counter = Counter()

    binary_counter = Counter()

    for i, feature_path in enumerate(feature_files):

        sample = process_feature_file(
            feature_path,
            annotation_cache,
        )

        # Unknown classes are skipped

        if sample is None:

            skipped += 1

            continue

        features.append(sample["feature"])

        binary_labels.append(sample["label"])

        multiclass_labels.append(sample["mc_label"])

        source_multiclass_labels.append(sample["source_mc_label"])

        videos.append(sample["video"])

        frame_indices.append(sample["indices"])

        binary_counter[int(sample["label"])] += 1

        class_counter[int(sample["mc_label"])] += 1

        if (i + 1) % 500 == 0:

            print(f"Processed " f"{i+1}/{len(feature_files)}")

    if len(features) == 0:

        raise RuntimeError("No valid samples were created.")

    dataset = {
        "features": torch.stack(features),
        "labels": torch.stack(binary_labels),
        "mc_labels": torch.stack(multiclass_labels),
        "source_mc_labels": torch.stack(source_multiclass_labels),
        "videos": videos,
        "indices": frame_indices,
        "feature_layer": FEATURE_LAYER,
        "num_classes": NUM_CLASSES,
        "class_names": DOTA_CLASS_NAMES,
        "class_mapping": class_mapping,
        "binary_distribution": dict(binary_counter),
        "multiclass_distribution": dict(class_counter),
    }

    # Print summary

    print("\nDataset summary")

    print("-" * 60)

    print("Samples :", len(features))

    print("Skipped :", skipped)

    print()

    print("Binary distribution")

    print(dataset["binary_distribution"])

    print()

    print("Multiclass distribution")

    for cls in sorted(class_counter.keys()):

        print(f"{cls:2d} " f"{DOTA_CLASS_NAMES[cls]:28s}" f"{class_counter[cls]}")

    print("-" * 60)

    return dataset


# VIDEO-LEVEL STRATIFIED TRAIN / VAL SPLIT


def stratified_split(
    feature_files,
    annotation_cache,
):
    """
    Splits the dataset by VIDEO rather than by feature file.

    Advantages
    ----------
    ✓ No train/validation leakage
    ✓ Normal and anomaly clips from the same video stay together
    ✓ Roughly preserves multiclass distribution
    """

    # Group files by video

    video_groups = defaultdict(list)

    video_class = {}

    for path in feature_files:

        data = torch.load(
            path,
            weights_only=False,
        )

        video = data["video"]

        video_groups[video].append(path)

        if video not in annotation_cache:
            continue

        cls = annotation_cache[video]["video_accident"]

        if cls > 9:
            continue

        video_class[video] = cls

    # Organize videos by class

    class_to_videos = defaultdict(list)

    for video, cls in video_class.items():

        class_to_videos[cls].append(video)

    print("\nVideo distribution")

    print("-" * 60)

    for cls in sorted(class_to_videos.keys()):

        print(
            f"{cls:2d} "
            f"{DOTA_CLASS_NAMES[cls]:28s}"
            f"{len(class_to_videos[cls])} videos"
        )

    # Split videos

    random.seed(SEED)

    train_files = []

    val_files = []

    for cls in sorted(class_to_videos.keys()):

        videos = class_to_videos[cls]

        random.shuffle(videos)

        n_val = max(
            1,
            int(len(videos) * VAL_RATIO),
        )

        val_videos = videos[:n_val]

        train_videos = videos[n_val:]

        for video in train_videos:

            train_files.extend(video_groups[video])

        for video in val_videos:

            val_files.extend(video_groups[video])

    random.shuffle(train_files)

    random.shuffle(val_files)

    # Statistics

    print("\nSplit summary")

    print("-" * 60)

    print("Training files :", len(train_files))

    print("Validation files :", len(val_files))

    print("-" * 60)

    return train_files, val_files


# DATASET SANITY CHECKS


def check_video_leakage(
    train_files,
    val_files,
):
    """
    Ensures that no video appears in both train and validation.
    """

    train_videos = set()

    val_videos = set()

    for path in train_files:

        data = torch.load(
            path,
            weights_only=False,
        )

        train_videos.add(data["video"])

    for path in val_files:

        data = torch.load(
            path,
            weights_only=False,
        )

        val_videos.add(data["video"])

    overlap = train_videos.intersection(val_videos)

    print("\nVideo leakage check")

    print("-" * 60)

    if len(overlap) == 0:

        print("✓ No video leakage detected")

    else:

        print("❌ Leakage detected")

        print(overlap)

        raise RuntimeError("Train/Val video leakage")

    print("-" * 60)


# FINAL DATASET STATISTICS


def print_cache_statistics(dataset, split):

    print("\n" + "=" * 60)

    print(split)

    print("=" * 60)

    print("Feature shape:", dataset["features"].shape)

    print("Binary labels:", Counter(dataset["labels"].tolist()))

    print("Multi-class labels:")

    counter = Counter(dataset["mc_labels"].tolist())

    inverse_mapping = {v: k for k, v in dataset["class_mapping"].items()}

    for cls, count in sorted(counter.items()):

        print(f"{cls:2d} " f"{inverse_mapping[cls]:30s}" f": {count}")

    print("=" * 60)


# MAIN


if __name__ == "__main__":

    print("\nBuilding V-JEPA Multi-Class Cache")

    # Build mapping

    class_mapping = build_class_mapping()

    # Load annotations

    annotation_cache = load_annotations()

    # Collect feature files

    feature_files = []

    feature_files.extend(glob.glob(os.path.join(NORMAL_DIR, "*.pt")))

    feature_files.extend(glob.glob(os.path.join(ANOMALY_DIR, "*.pt")))

    print("\nTotal feature files:", len(feature_files))

    # Split

    train_files, val_files = stratified_split(
        feature_files,
        annotation_cache,
    )

    # Check leakage

    check_video_leakage(
        train_files,
        val_files,
    )

    # Build datasets

    print("\nBuilding training cache...")

    train_cache = build_dataset(
        train_files,
        class_mapping,
        annotation_cache,
    )

    print("\nBuilding validation cache...")

    val_cache = build_dataset(
        val_files,
        class_mapping,
        annotation_cache,
    )

    # Statistics

    print_cache_statistics(train_cache, "TRAIN")

    print_cache_statistics(val_cache, "VALIDATION")

    # Save

    torch.save(
        train_cache,
        TRAIN_FILE,
    )

    torch.save(
        val_cache,
        VAL_FILE,
    )

    print("\nSaved files")

    print(TRAIN_FILE)

    print(VAL_FILE)

    print("\nDONE")
