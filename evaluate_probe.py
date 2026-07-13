import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression


# --------------------
# Load dataset
# --------------------

data = torch.load(
    "probe_dataset.pt",
    weights_only=False,
)

X = data["X"]
y = data["y"]

# --------------------
# Split
# --------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# --------------------
# Load trained probe
# --------------------

probe = torch.load(
    "linear_probe.pt",
    weights_only=False,
)
probe = probe["classifier"]
pred = probe.predict(X_test)
prob = probe.predict_proba(X_test)[:, 1]

print()

print("Accuracy :", accuracy_score(y_test, pred))
print("AUC      :", roc_auc_score(y_test, prob))

print()

print(classification_report(y_test, pred))

print()

print(confusion_matrix(y_test, pred))