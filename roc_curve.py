import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)



data=torch.load(
    "probe_dataset.pt",
    weights_only=False
)


X_val=data["X_val"].numpy()
y_val=data["y_val"].numpy()



probe=torch.load(
    "linear_probe.pt",
    weights_only=False
)



scores=probe.predict_proba(
    X_val
)[:,1]



fpr,tpr,_=roc_curve(
    y_val,
    scores
)


roc_auc=auc(
    fpr,
    tpr
)



plt.figure(figsize=(6,6))


plt.plot(
    fpr,
    tpr,
    label=f"AUC={roc_auc:.3f}"
)


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "V-JEPA2 Linear Probe ROC"
)


plt.legend()

plt.grid()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()