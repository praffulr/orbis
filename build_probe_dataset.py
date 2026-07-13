import os
import glob
import torch
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

FEATURE_ROOT = "vjepa_features"

NORMAL_DIR = os.path.join(FEATURE_ROOT, "normal")
ANOMALY_DIR = os.path.join(FEATURE_ROOT, "anomaly")

# Build probe datasets for all cached representations
LAYERS = [
    "final",
    "block_3",
    "block_6",
    "block_9",
    "block_12"
]

OUTPUT_FILE = "probe_dataset.pt"

TEST_SIZE = 0.20
SEED = 42


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(folder):

    data = {layer: [] for layer in LAYERS}

    files = sorted(glob.glob(os.path.join(folder, "*.pt")))

    print(f"{folder}: {len(files)} files")

    valid = 0

    for f in files:

        sample = torch.load(
            f,
            map_location="cpu",
            weights_only=False
        )

        feat = sample["feature"]

        ok = True

        current = {}

        for layer in LAYERS:

            if layer not in feat:
                ok = False
                break

            x = feat[layer]

            # ---------------------------------------
            # Final embedding already has shape (768)
            # ---------------------------------------
            if layer == "final":

                if x.ndim != 1:
                    x = x.squeeze()

            # ---------------------------------------
            # Cached activations:
            # (576,768) -> mean over tokens -> (768)
            # ---------------------------------------
            else:

                if x.ndim == 2:
                    x = x.mean(dim=0)

                elif x.ndim == 3:
                    x = x.mean(dim=1).squeeze(0)

            current[layer] = x.numpy()

        if ok:

            valid += 1

            for layer in LAYERS:
                data[layer].append(current[layer])

    print("Valid samples:", valid)

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    normal = load_features(NORMAL_DIR)
    anomaly = load_features(ANOMALY_DIR)

    dataset = {}

    labels_normal = np.zeros(len(normal["final"]), dtype=np.float32)
    labels_anomaly = np.ones(len(anomaly["final"]), dtype=np.float32)

    y = np.concatenate([labels_normal, labels_anomaly])

    for layer in LAYERS:

        X = np.concatenate(
            [
                np.array(normal[layer]),
                np.array(anomaly[layer])
            ],
            axis=0
        )

        print(f"{layer:10s} -> {X.shape}")

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y
        )

        dataset[layer] = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val
        }

    torch.save(dataset, OUTPUT_FILE)

    print("\n==============================")
    print("PROBE DATASET CREATED")
    print("==============================")

    for layer in LAYERS:
        print(layer)


if __name__ == "__main__":
    main()