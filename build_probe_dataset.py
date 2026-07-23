import os
import glob
import torch
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

FEATURE_ROOT = "vjepa_features_tb5"


NORMAL_DIR = os.path.join(FEATURE_ROOT, "normal")


ANOMALY_DIR = os.path.join(FEATURE_ROOT, "anomaly")


# LAYERS = ["final", "block_3", "block_6", "block_9", "block_12"]

LAYERS = ["final"]


OUTPUT_DIR = "cached_features"


os.makedirs(OUTPUT_DIR, exist_ok=True)


TEST_SIZE = 0.20

SEED = 42


# ============================================================
# LOAD SINGLE LAYER
# ============================================================


def load_layer_features(folder, layer):

    features = []
    labels = []

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

        # =================================================
        # FINAL REPRESENTATION
        # =================================================

        if layer=="final":

            x = x.squeeze()

            if x.ndim != 2:

                print(
                    "Invalid final token shape:",
                    x.shape,
                    file
                )

                continue


            # tubelet=5 check
            if x.shape[0] != 576:

                print(
                    "Unexpected token count:",
                    x.shape,
                    file
                )

                continue

        # =================================================
        # BLOCK REPRESENTATION
        # =================================================

        else:

            x = x.squeeze()

            if x.ndim != 2:

                print(
                    "Invalid block shape:",
                    x.shape,
                    file
                )

                continue


            # tubelet=5 check
            if x.shape[0] != 576:

                print(
                    "Unexpected token count:",
                    x.shape,
                    file
                )

                continue

        shapes.add(tuple(x.shape))

        features.append(x.float())

        labels.append(label)

        valid += 1

    print("Valid samples:", valid)

    print("Feature shapes:", shapes)

    if len(shapes) != 1:

        raise RuntimeError(f"Inconsistent shapes detected for {layer}: {shapes}")

    return features, labels


# ============================================================
# BUILD DATASET
# ============================================================


def build_dataset(layer):

    print("\n\n================================")
    print("Processing layer:", layer)
    print("================================")

    normal_features, normal_labels = load_layer_features(NORMAL_DIR, layer)

    anomaly_features, anomaly_labels = load_layer_features(ANOMALY_DIR, layer)

    X = normal_features + anomaly_features

    y = normal_labels + anomaly_labels

    X = torch.stack(X)

    y = torch.tensor(y, dtype=torch.long)

    print("\nComplete dataset")

    print("Features:", X.shape)

    print("Labels:", y.shape)

    print("Normal:", (y == 0).sum().item())

    print("Anomaly:", (y == 1).sum().item())

    # =====================================================
    # TRAIN VALIDATION SPLIT
    # =====================================================

    indices = np.arange(len(y))

    train_idx, val_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=y.numpy()
    )

    train_x = X[train_idx]

    train_y = y[train_idx]

    val_x = X[val_idx]

    val_y = y[val_idx]

    print("\nSplit")

    print("Train:", train_x.shape)

    print("Val:", val_x.shape)

    # =====================================================
    # SAVE
    # =====================================================

    train_path = os.path.join(OUTPUT_DIR, f"train_{layer}.pt")

    val_path = os.path.join(OUTPUT_DIR, f"val_{layer}.pt")

    torch.save(
        {
            "features": train_x,
            "labels": train_y,
            "layer": layer,
            "num_samples": len(train_y),
        },
        train_path,
    )

    torch.save(
        {"features": val_x, "labels": val_y, "layer": layer, "num_samples": len(val_y)},
        val_path,
    )

    print("\nSaved")

    print(train_path)

    print(val_path)


# ============================================================
# MAIN
# ============================================================


def main():

    for layer in LAYERS:

        build_dataset(layer)

    print("\n================================")
    print("ALL PROBE DATASETS CREATED")
    print("================================")


if __name__ == "__main__":

    main()