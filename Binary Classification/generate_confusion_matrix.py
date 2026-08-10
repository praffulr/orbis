import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

CHECKPOINT = "checkpoints/best_vjepa_attention.pt"
VAL_FILE = "cached_features/val_vjepa_final_mc.pt"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = {
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

data = torch.load(
    VAL_FILE,
    weights_only=False
)

features = data["features"].float()

# Use 'labels' for the actual 0 (Normal) or 1 (Anomaly) ground truth
binary_labels = data["labels"].numpy()

# Use 'source_mc_labels' strictly to group the sequences by scenario (1-9)
source_mc_labels = data["source_mc_labels"].numpy()

print("Features:", features.shape)
print("Binary Labels:", binary_labels.shape)
print("Source MC Labels:", source_mc_labels.shape)

print("\nValidation class distribution (by Source MC Group)\n")

unique, counts = np.unique(source_mc_labels, return_counts=True)

for u, c in zip(unique, counts):
    name = CLASS_NAMES.get(u, f"Unknown_{u}")
    print(f"{u}: {name} -> {c}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

cfg = checkpoint["config"]


class AttentionProbe(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout
    ):

        super().__init__()

        self.query = torch.nn.Parameter(
            torch.randn(1, 1, input_dim)
        )

        self.norm1 = torch.nn.LayerNorm(input_dim)

        self.attention = torch.nn.MultiheadAttention(
            input_dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = torch.nn.LayerNorm(input_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.classifier = torch.nn.Linear(
            input_dim,
            2
        )

    def forward(self, x):

        B = x.shape[0]

        q = self.query.expand(B, -1, -1)

        x = self.norm1(x)

        out, _ = self.attention(
            q,
            x,
            x
        )

        out = out.squeeze(1)

        out = self.norm2(out)

        out = self.dropout(out)

        return self.classifier(out)


model = AttentionProbe(
    input_dim=768,
    num_heads=cfg["num_heads"],
    dropout=cfg["dropout"]
)

model.load_state_dict(checkpoint["model"])

model.to(DEVICE)

model.eval()

predictions = []
probabilities = []

with torch.no_grad():

    for i in range(len(features)):

        x = features[i].unsqueeze(0).to(DEVICE)

        logits = model(x)

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1)

        predictions.append(pred.item())

        probabilities.append(probs[:, 1].item())

predictions = np.array(predictions)
probabilities = np.array(probabilities)

print("\nFinished inference on", len(predictions), "samples.")

def safe_percent(x, total):
    if total == 0:
        return 0.0
    return 100 * x / total


def calculate_metrics(gts, preds, probs):
    """
    Computes Acc (%), AUROC, Precision (%), Recall (%), and F1 (%).
    """
    total = len(preds)
    correct = np.sum(gts == preds)
    acc = safe_percent(correct, total) if total > 0 else 0.0

    prec = precision_score(gts, preds, zero_division=0) * 100.0
    rec = recall_score(gts, preds, zero_division=0) * 100.0
    f1 = f1_score(gts, preds, zero_division=0) * 100.0

    try:
        if len(np.unique(gts)) > 1:
            auc = roc_auc_score(gts, probs)
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")

    return acc, auc, prec, rec, f1


def make_class_confusion_matrix(class_id):
    """
    Groups samples by the scenario (source_mc_labels), but 
    evaluates using the actual binary normal/anomalous ground truth.
    """
    mask = source_mc_labels == class_id
    
    cls_preds = predictions[mask]
    cls_gts = binary_labels[mask]
    cls_probs = probabilities[mask]
    
    N = len(cls_preds)
    
    TN = np.sum((cls_gts == 0) & (cls_preds == 0))
    FP = np.sum((cls_gts == 0) & (cls_preds == 1))
    FN = np.sum((cls_gts == 1) & (cls_preds == 0))
    TP = np.sum((cls_gts == 1) & (cls_preds == 1))

    cm = np.array([
        [TN, FP],
        [FN, TP]
    ])
    
    acc, auc, prec, rec, f1 = calculate_metrics(cls_gts, cls_preds, cls_probs)

    return cm, N, TN, FP, FN, TP, acc, auc, prec, rec, f1


def make_overall_confusion_matrix():
    
    TN = np.sum((binary_labels == 0) & (predictions == 0))
    FP = np.sum((binary_labels == 0) & (predictions == 1))
    FN = np.sum((binary_labels == 1) & (predictions == 0))
    TP = np.sum((binary_labels == 1) & (predictions == 1))

    cm = np.array([
        [TN, FP],
        [FN, TP]
    ])
    
    acc, auc, prec, rec, f1 = calculate_metrics(
        binary_labels,
        predictions,
        probabilities
    )

    return cm, TN, FP, FN, TP, acc, auc, prec, rec, f1


def draw_styled_matrix(ax, cm, TN, FP, FN, TP, title):
    """
    Draws a color-coded confusion matrix mimicking the target reference image layout.
    """
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        ax=ax,
        linewidths=2,
        linecolor="white"
    )

    n_norm = TN + FP
    n_anom = FN + TP

    tn_pct = f"{safe_percent(TN, n_norm):.1f}%"
    fp_pct = f"{safe_percent(FP, n_norm):.1f}%"
    fn_pct = f"{safe_percent(FN, n_anom):.1f}%"
    tp_pct = f"{safe_percent(TP, n_anom):.1f}%"

    text_annotations = [
        [f"TN/Corr: {TN}\n({tn_pct})", f"FP/Err: {FP}\n({fp_pct})"],
        [f"FN: {FN}\n({fn_pct})", f"TP: {TP}\n({tp_pct})"]
    ]

    for r in range(2):
        for c in range(2):
            cell_color = (
                "red"
                if (r != c and cm[r, c] > 0)
                else "darkgreen"
                if (r == c and cm[r, c] > 0)
                else "black"
            )
            ax.text(
                c + 0.5,
                r + 0.5,
                text_annotations[r][c],
                ha="center",
                va="center",
                color=cell_color,
                fontsize=12,
                fontweight="bold"
            )

    ax.set_xticklabels(
        ["Pred Normal (0)", "Pred Anom (1)"],
        fontsize=10,
        fontweight="bold"
    )
    ax.set_yticklabels(
        ["True Normal (0)", "True Anom (1)"],
        fontsize=10,
        fontweight="bold",
        rotation=0
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

# Use a 2x5 grid mapping to the image layout
fig, axes = plt.subplots(
    2,
    5,
    figsize=(26, 11),
    dpi=300
)

axes = axes.flatten()
plot_idx = 0

# Lists to store the metrics
per_class_acc = []
per_class_auc = []
per_class_prec = []
per_class_rec = []
per_class_f1 = []

for cls_id in range(1, 10):
    cm, N, TN, FP, FN, TP, acc, auc, prec, rec, f1 = make_class_confusion_matrix(cls_id)
    
    per_class_acc.append(acc)
    per_class_auc.append(auc)
    per_class_prec.append(prec)
    per_class_rec.append(rec)
    per_class_f1.append(f1)

    print(
        f"{CLASS_NAMES[cls_id]:26s}"
        f"N={N:3d} | "
        f"Acc={acc:5.2f}% | AUC={auc:6.4f} | Prec={prec:5.2f}% | Rec={rec:5.2f}% | F1={f1:5.2f}%"
    )
    
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    title = f"Class {cls_id}: {CLASS_NAMES[cls_id]} (N={N})\nAcc: {acc:.1f}% | AUC: {auc_str}"
    
    draw_styled_matrix(
        axes[plot_idx],
        cm,
        TN,
        FP,
        FN,
        TP,
        title
    )
    plot_idx += 1

overall_cm, TN, FP, FN, TP, overall_acc, overall_auc, overall_prec, overall_rec, overall_f1 = make_overall_confusion_matrix()
total = overall_cm.sum()

overall_auc_str = f"{overall_auc:.4f}" if not np.isnan(overall_auc) else "N/A"
overall_title = f"OVERALL VALIDATION (N={total})\nAcc: {overall_acc:.1f}% | AUC: {overall_auc_str}"

draw_styled_matrix(
    axes[plot_idx],
    overall_cm,
    TN,
    FP,
    FN,
    TP,
    overall_title
)
plot_idx += 1

for i in range(plot_idx, len(axes)):
    axes[i].axis("off")

plt.suptitle(
    "Color-Coded Confusion Matrices Per Source Class & Overall (Binary)",
    fontsize=20,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "per_class_confusion_matrices_binary_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n" + "="*70)
print("METRICS SUMMARY LISTS")
print("="*70)

formatted_acc = [round(val, 2) for val in per_class_acc]
formatted_auc = [round(val, 4) if not np.isnan(val) else None for val in per_class_auc]
formatted_prec = [round(val, 2) for val in per_class_prec]
formatted_rec = [round(val, 2) for val in per_class_rec]
formatted_f1 = [round(val, 2) for val in per_class_f1]

print("per_class_acc_list  =", formatted_acc)
print("per_class_auc_list  =", formatted_auc)
print("per_class_prec_list =", formatted_prec)
print("per_class_rec_list  =", formatted_rec)
print("per_class_f1_list   =", formatted_f1)

print("\n--- Overall Summary ---")
print(f"Overall Accuracy:  {overall_acc:.2f}%")
print(f"Overall AUROC:     {overall_auc_str}")
print(f"Overall Precision: {overall_prec:.2f}%")
print(f"Overall Recall:    {overall_rec:.2f}%")
print(f"Overall F1:        {overall_f1:.2f}%")

print("\nSaved:")
print("per_class_confusion_matrices_binary_grid.png")