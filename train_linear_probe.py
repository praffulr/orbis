import torch
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

DATASET = "probe_dataset.pt"

LAYERS = [
    "final",
    "block_3",
    "block_6",
    "block_9",
    "block_12"
]


# ============================================================
# TRAIN SINGLE PROBE
# ============================================================

def train_probe(layer, dataset):

    print("\n" + "=" * 60)
    print(f"Training probe on: {layer}")
    print("=" * 60)

    X_train = dataset[layer]["X_train"]
    X_val = dataset[layer]["X_val"]

    y_train = dataset[layer]["y_train"]
    y_val = dataset[layer]["y_val"]

    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)

    # ---------------------------------------------------------
    # Linear Probe
    # ---------------------------------------------------------

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ))
    ])

    probe.fit(X_train, y_train)

    pred = probe.predict(X_val)

    prob = probe.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, prob)

    print("\nAccuracy :", round(acc, 4))
    print("AUC      :", round(auc, 4))

    print("\nClassification Report\n")
    print(classification_report(y_val, pred, digits=4))

    print("Confusion Matrix\n")
    print(confusion_matrix(y_val, pred))

    save_name = f"probe_{layer}.pkl"

    joblib.dump(probe, save_name)

    print("\nSaved:", save_name)

    return {
        "accuracy": acc,
        "auc": auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = torch.load(
        DATASET,
        map_location="cpu",
        weights_only=False
    )

    results = {}

    for layer in LAYERS:

        if layer not in dataset:
            print(f"Skipping {layer} (not found)")
            continue

        results[layer] = train_probe(layer, dataset)

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    ranking = sorted(
        results.items(),
        key=lambda x: x[1]["auc"],
        reverse=True
    )

    for layer, metric in ranking:

        print(
            f"{layer:10s}"
            f"  Accuracy = {metric['accuracy']:.4f}"
            f"   AUC = {metric['auc']:.4f}"
        )

    best_layer = ranking[0][0]

    print("\nBest representation:", best_layer)


if __name__ == "__main__":
    main()