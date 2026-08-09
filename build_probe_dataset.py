import os
import glob
import torch
import numpy as np

from sklearn.model_selection import train_test_split


# CONFIG

FEATURE_ROOT = "vjepa_features_tb5"

NORMAL_DIR = os.path.join(FEATURE_ROOT, "normal")

ANOMALY_DIR = os.path.join(FEATURE_ROOT, "anomaly")


LAYERS = ["final"]


OUTPUT_DIR = "cached_features_tb5"


os.makedirs(OUTPUT_DIR, exist_ok=True)


TEST_SIZE = 0.20

SEED = 42

def load_layer_features(folder, layer):

    features = []
    labels = []

    videos = []
    frame_indices = []

    files = sorted(glob.glob(os.path.join(folder, "*.pt")))

    label = 0 if "normal" in folder else 1

    print("\n--------------------------------")
    print("Folder:", folder)
    print("Layer :", layer)
    print("Files :", len(files))
    print("--------------------------------")

    valid = 0

    shapes = set()

    for file in files:

        sample = torch.load(file, map_location="cpu", weights_only=False)

        feature_dict = sample["feature"]

        if layer not in feature_dict:
            continue

        x = feature_dict[layer]

        # remove batch dimension

        x = x.squeeze()

        if x.ndim != 2:

            print("Invalid feature:", x.shape, file)

            continue

        # =====================================================
        # Tubelet=5 expected
        #
        # 5 frames
        # tubelet size = 5
        #
        # temporal tokens = 1
        #
        # spatial = 24*24
        #
        # total tokens = 576
        #
        # =====================================================

        if x.shape[0] != 576:

            print("Unexpected tokens:", x.shape, file)

            continue

        shapes.add(tuple(x.shape))

        features.append(x.float())

        labels.append(label)

        videos.append(sample["video"])

        frame_indices.append(sample["indices"])

        valid += 1

    print("Valid samples:", valid)

    print("Shapes:", shapes)

    if len(shapes) != 1:

        raise RuntimeError(f"Inconsistent shapes {shapes}")

    return (features, labels, videos, frame_indices)

def build_dataset(layer):

    print("\n==============================")
    print("Layer:", layer)
    print("==============================")

    (normal_x, normal_y, normal_videos, normal_indices) = load_layer_features(
        NORMAL_DIR, layer
    )

    (anomaly_x, anomaly_y, anomaly_videos, anomaly_indices) = load_layer_features(
        ANOMALY_DIR, layer
    )

    X = normal_x + anomaly_x

    y = normal_y + anomaly_y

    videos = normal_videos + anomaly_videos

    indices = normal_indices + anomaly_indices

    X = torch.stack(X)

    y = torch.tensor(y, dtype=torch.long)

    print("\nDataset")

    print("Features:", X.shape)

    print("Labels:", y.shape)

    print("Samples:", len(videos))

    print("Normal:", (y == 0).sum().item())

    print("Anomaly:", (y == 1).sum().item())

    idx = np.arange(len(y))

    train_idx, val_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=y.numpy()
    )

    def select(lst, ids):

        return [lst[i] for i in ids]

    train_x = X[train_idx]

    train_y = y[train_idx]

    val_x = X[val_idx]

    val_y = y[val_idx]

    train_meta = {
        "videos": select(videos, train_idx),
        "indices": select(indices, train_idx),
    }

    val_meta = {"videos": select(videos, val_idx), "indices": select(indices, val_idx)}

    print("\nSplit")

    print("Train:", train_x.shape)

    print("Val:", val_x.shape)

    train_path = os.path.join(OUTPUT_DIR, f"train_{layer}.pt")

    val_path = os.path.join(OUTPUT_DIR, f"val_{layer}.pt")

    torch.save(
        {
            "features": train_x,
            "labels": train_y,
            "videos": train_meta["videos"],
            "indices": train_meta["indices"],
            "layer": layer,
        },
        train_path,
    )

    torch.save(
        {
            "features": val_x,
            "labels": val_y,
            "videos": val_meta["videos"],
            "indices": val_meta["indices"],
            "layer": layer,
        },
        val_path,
    )

    print("\nSaved:")
    print(train_path)
    print(val_path)

def main():

    for layer in LAYERS:

        build_dataset(layer)

    print("\nDONE")


if __name__ == "__main__":

    main()
