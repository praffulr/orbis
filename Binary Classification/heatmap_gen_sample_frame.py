import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "checkpoints/best_vjepa_attention.pt"

TRAIN_FILE = "cached_features/train_vjepa_final_mc.pt"
VAL_FILE   = "cached_features/val_vjepa_final_mc.pt"

DOTA_FILE = "DOTA_training/DoTA_training.pt"

OUTPUT_DIR = "heatmaps_train_live"

TARGET_VIDEO = "d2SCftR5sWc_002095"

GRID_H = 24
GRID_W = 24

TOP_PERCENT = 5

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# ATTENTION PROBE
# ============================================================

class AttentionProbe(nn.Module):

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout
    ):

        super().__init__()

        self.query = nn.Parameter(
            torch.randn(
                1,
                1,
                input_dim
            )
        )

        self.norm1 = nn.LayerNorm(
            input_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            input_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            input_dim,
            2
        )

    def forward(
        self,
        x,
        return_attention=False
    ):

        B = x.size(0)

        q = self.query.expand(
            B,
            -1,
            -1
        )

        x = self.norm1(x)

        out, weights = self.attention(
            q,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False
        )

        out = out.squeeze(1)

        out = self.norm2(out)

        out = self.dropout(out)

        logits = self.classifier(out)

        if return_attention:
            return logits, weights

        return logits


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

cfg = checkpoint["config"]

model = AttentionProbe(
    input_dim=768,
    num_heads=cfg["num_heads"],
    dropout=cfg["dropout"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD DOTA
# ============================================================

print("\nLoading DOTA metadata...")

dota = torch.load(
    DOTA_FILE,
    map_location="cpu",
    weights_only=False
)

print(
    "DOTA samples:",
    len(dota["labels"])
)


# ============================================================
# FIND TARGET FEATURE
# ============================================================

print("\nSearching cached feature files...")

target_feature = None
found_split = None

for file_path in [
    TRAIN_FILE,
    VAL_FILE
]:

    if not os.path.exists(file_path):
        continue

    print("Checking", file_path)

    data = torch.load(
        file_path,
        weights_only=False
    )

    features = data["features"].float()
    video_ids = data["video_ids"]

    for i, vid in enumerate(video_ids):

        if TARGET_VIDEO in str(vid):

            target_feature = features[i]

            found_split = file_path

            print(
                f"Found target in {file_path}"
            )

            break

    if target_feature is not None:
        break

if target_feature is None:

    raise RuntimeError(
        "Target sequence not found."
    )


# ============================================================
# FIND TARGET INDEX INSIDE DOTA
# ============================================================

target_dota_idx = None

for i, vid in enumerate(dota["video_ids"]):

    if TARGET_VIDEO in str(vid):

        target_dota_idx = i
        break

if target_dota_idx is None:

    raise RuntimeError(
        "Target video not found inside DOTA metadata."
    )


# ============================================================
# LIVE FORWARD PASS
# ============================================================

print("\nRunning live forward pass...")

sample = target_feature.unsqueeze(0).to(DEVICE)

with torch.no_grad():

    logits, attn_weights = model(
        sample,
        return_attention=True
    )

    probs = torch.softmax(
        logits,
        dim=1
    )

pred_label = logits.argmax(dim=1).item()

confidence = probs.max().item()

attn_weights = (
    attn_weights
    .squeeze(0)
    .squeeze(1)
    .cpu()
)

print(
    "Attention shape:",
    attn_weights.shape
)


# ============================================================
# GLOBAL ATTENTION BASELINE
# ============================================================

print("\nComputing global attention baseline...")

baseline = torch.load(
    found_split,
    weights_only=False
)

base_features = baseline["features"].float()

all_attn = []

subset = min(
    100,
    len(base_features)
)

with torch.no_grad():

    for i in range(subset):

        feat = (
            base_features[i]
            .unsqueeze(0)
            .to(DEVICE)
        )

        _, weights = model(
            feat,
            return_attention=True
        )

        weights = (
            weights
            .squeeze(0)
            .squeeze(1)
        )

        mean_head = weights.mean(
            dim=0
        )

        all_attn.append(
            mean_head.cpu().numpy()
        )

GLOBAL_MEAN = np.stack(
    all_attn
).mean(
    axis=0
)

print(
    "Global attention:",
    GLOBAL_MEAN.shape
)

# ============================================================
# LOAD IMAGE
# ============================================================

def load_frame(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# RELATIVE ATTENTION
# ============================================================

def compute_relative_attention(attention):

    if torch.is_tensor(attention):

        attention = (
            attention
            .detach()
            .cpu()
            .numpy()
        )

    # 576 tokens -> 24x24

    att_map = attention.reshape(
        GRID_H,
        GRID_W
    )

    global_map = GLOBAL_MEAN.reshape(
        GRID_H,
        GRID_W
    )

    # remove common attention

    relative = (
        att_map -
        global_map
    )

    # normalize

    max_val = np.max(
        np.abs(relative)
    )

    if max_val > 1e-8:
        relative /= max_val

    # keep only positive attention

    positive = np.clip(
        relative,
        0,
        None
    )

    # keep only top-k%

    flat = positive.flatten()

    if flat.max() > 0:

        k = max(
            1,
            int(
                len(flat)
                *
                TOP_PERCENT
                /
                100
            )
        )

        threshold = np.partition(
            flat,
            -k
        )[-k]

        positive[
            positive < threshold
        ] = 0

    return positive


# ============================================================
# CREATE HEATMAP
# ============================================================

def process_attention_map(
        attention,
        image
):

    H, W = image.shape[:2]

    attn_2d = attention.reshape(
        GRID_H,
        GRID_W
    )

    # normalize

    attn_2d = (
        attn_2d -
        attn_2d.min()
    ) / (
        attn_2d.max()
        -
        attn_2d.min()
        +
        1e-8
    )

    # resize to image

    attn_resized = cv2.resize(
        attn_2d,
        (W, H),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap_bgr = cv2.applyColorMap(
        np.uint8(
            255 *
            attn_resized
        ),
        cv2.COLORMAP_JET
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    overlay_bgr = cv2.addWeighted(
        image_bgr,
        0.5,
        heatmap_bgr,
        0.5,
        0
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB
    )

    return overlay_rgb


# ============================================================
# VISUALIZE SAMPLE
# ============================================================

def visualize_sample():

    label = int(
        dota["labels"][target_dota_idx]
    )

    video = dota["video_ids"][target_dota_idx]

    clip_paths = dota["clip_paths"][target_dota_idx]

    if isinstance(
        clip_paths,
        str
    ):
        clip_paths = [
            clip_paths
        ]

    if len(clip_paths) == 0:
        raise RuntimeError(
            "No clip paths found."
        )

    # ----------------------------------------------------
    # Load last frame
    # ----------------------------------------------------

    image = load_frame(
        clip_paths[-1]
    )

    # ----------------------------------------------------
    # EXACT SAME PIPELINE AS VALIDATION SCRIPT
    # ----------------------------------------------------

    attention = attn_weights.mean(
        dim=0
    )

    attention_map = compute_relative_attention(
        attention
    )

    overlay_img = process_attention_map(
        attention_map,
        image
    )

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        dpi=200
    )

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Original Last Frame",
        fontsize=16
    )

    axes[0].axis(
        "off"
    )

    axes[1].imshow(
        overlay_img
    )

    axes[1].set_title(
        "Attention Overlay",
        fontsize=16
    )

    axes[1].axis(
        "off"
    )

    gt = (
        "ANOMALY"
        if label == 1
        else
        "NORMAL"
    )

    pred = (
        "ANOMALY"
        if pred_label == 1
        else
        "NORMAL"
    )

    fig.suptitle(
        f"{video}\n"
        f"GT: {gt} | "
        f"Pred: {pred} | "
        f"Confidence: {confidence:.3f}",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    folder = (
        "anomaly"
        if label == 1
        else
        "normal"
    )

    save_dir = (
        Path(OUTPUT_DIR)
        / folder
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = (
        save_dir
        /
        f"{video}_live_heatmap.png"
    )

    plt.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(
        "Saved:",
        save_path
    )

# ============================================================
# MAIN EXECUTION
# ============================================================

print("\nGenerating training sample heatmap...")

try:

    visualize_sample()

    print(
        "\nSuccessfully generated heatmap!"
    )

except Exception as e:

    print(
        "\nFAILED:",
        e
    )


print(
    "\n=============================="
)

print(
    "DONE"
)

print(
    "=============================="
)