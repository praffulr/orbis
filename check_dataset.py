import torch
import numpy as np


DATASET = "probe_dataset.pt"


data = torch.load(
    DATASET,
    weights_only=False
)


X = data["X"]
y = data["y"]


if isinstance(X, np.ndarray):
    X = torch.from_numpy(X)

if isinstance(y, np.ndarray):
    y = torch.from_numpy(y)


print("="*60)
print("DATASET SUMMARY")
print("="*60)

print("Feature shape :", X.shape)
print("Label shape   :", y.shape)

print()

print("Feature dtype :", X.dtype)
print("Label dtype   :", y.dtype)

print()

print("Classes")
print("----------------")
print("Normal  :", (y == 0).sum().item())
print("Anomaly :", (y == 1).sum().item())

print()

print("Feature statistics")
print("----------------")
print("Mean :", X.mean().item())
print("Std  :", X.std().item())
print("Min  :", X.min().item())
print("Max  :", X.max().item())


print("="*60)