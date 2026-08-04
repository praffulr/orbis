import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# CONFIG
# ============================================================


ATTENTION_FILE = "best_val_attention_weights_tb5.pt"

DOTA_FILE = "DOTA_training/DoTA_training.pt"


OUTPUT_DIR = "heatmaps_binary"


IMG_SIZE = 384


GRID_H = 24
GRID_W = 24


NUM_CONTEXT_FRAMES = 5


TOP_PERCENT = 5


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DOTA METADATA
# ============================================================


print("\nLoading DOTA metadata...")


dota = torch.load(DOTA_FILE, map_location="cpu", weights_only=False)


print(dota.keys())


print("\nDOTA samples:")
print(len(dota["labels"]))


# ============================================================
# LOAD ATTENTION
# ============================================================


print("\nLoading attention...")


attention_data = torch.load(ATTENTION_FILE, map_location="cpu", weights_only=False)


print("Attention samples:", len(attention_data))


example = next(iter(attention_data.values()))


print("\nAttention keys")

for k in example:
    print(k, type(example[k]))


print("Attention shape:", example["attention"].shape)


# ============================================================
# GLOBAL ATTENTION BASELINE
# ============================================================


print("\nComputing global attention...")


all_attn = []


for sample in attention_data.values():

    attn = sample["attention"]

    # heads x tokens

    mean_head = attn.mean(dim=0)

    all_attn.append(mean_head.numpy())


all_attn = np.stack(all_attn)


GLOBAL_MEAN = all_attn.mean(axis=0)


print("Global shape:", GLOBAL_MEAN.shape)


# ============================================================
# PATH FROM DOTA CLIP
# ============================================================


def get_frame_path(clip_path):

    """
    clip_path example:

    DOTA_training/data/train/
    0RJPQ_97dcs_000199_anomalous/frame_0000.jpg

    """

    return Path(clip_path)


# ============================================================
# LOAD IMAGE
# ============================================================


def load_frame(path):

    image = cv2.imread(str(path))

    if image is None:

        raise FileNotFoundError(path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# ============================================================
# ATTENTION MAP
# ============================================================


def compute_relative_attention(attention):

    if torch.is_tensor(attention):

        attention = attention.detach().cpu().numpy()

    att_map = attention.reshape(GRID_H, GRID_W)

    global_map = GLOBAL_MEAN.reshape(GRID_H, GRID_W)

    relative = att_map - global_map

    mx = np.max(np.abs(relative))

    if mx > 1e-8:

        relative /= mx

    positive = np.clip(relative, 0, None)

    # keep top attention patches

    flat = positive.flatten()

    if flat.max() > 0:

        k = max(1, int(len(flat) * TOP_PERCENT / 100))

        threshold = np.partition(flat, -k)[-k]

        positive = np.where(positive >= threshold, positive, 0)

    return positive


# ============================================================
# HEATMAP
# ============================================================


def create_heatmap(attention, image):

    H, W = image.shape[:2]

    heat = cv2.resize(attention, (W, H), interpolation=cv2.INTER_CUBIC)

    heat = cv2.GaussianBlur(heat, (5, 5), 0)

    if heat.max() > 0:

        heat /= heat.max()

    heat_color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)

    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    return heat_color


# ============================================================
# OVERLAY
# ============================================================


def overlay(image, heat, alpha=0.35):

    return np.clip(image * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


# ============================================================
# VISUALIZE ONE SAMPLE
# ============================================================


def visualize_sample(att_idx, sample):

    # -----------------------------
    # DOTA metadata lookup
    # -----------------------------

    label = int(dota["labels"][att_idx])

    video = dota["video_ids"][att_idx]

    clip_paths = dota["clip_paths"][att_idx]

    if isinstance(clip_paths, str):

        clip_paths = [clip_paths]

    # -----------------------------
    # attention
    # -----------------------------

    attention = sample["attention"]

    attention_map = compute_relative_attention(attention.mean(dim=0))

    images = []

    for p in clip_paths[:NUM_CONTEXT_FRAMES]:

        img = load_frame(p)

        images.append(img)

    if len(images) < NUM_CONTEXT_FRAMES:

        return

    # -----------------------------
    # plot
    # -----------------------------

    fig, axes = plt.subplots(3, NUM_CONTEXT_FRAMES, figsize=(20, 10))

    for i, img in enumerate(images):

        axes[0, i].imshow(img)

        axes[0, i].axis("off")

        axes[0, i].set_title(f"Frame {i}")

        heat = create_heatmap(attention_map, img)

        axes[1, i].imshow(heat)

        axes[1, i].axis("off")

        axes[1, i].set_title("Attention")

        axes[2, i].imshow(overlay(img, heat))

        axes[2, i].axis("off")

        axes[2, i].set_title("Overlay")

    gt = "ANOMALY" if label == 1 else "NORMAL"

    prediction = sample.get("prediction", -1)

    confidence = sample.get("confidence", 0)

    fig.suptitle(
        f"{video}\n" f"GT:{gt} " f"Pred:{prediction} " f"Conf:{confidence:.3f}",
        fontsize=16,
    )

    plt.tight_layout()

    folder = "anomaly" if label == 1 else "normal"

    save_dir = Path(OUTPUT_DIR) / folder

    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / (f"{video}_{att_idx}.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()

    print("Saved:", save_path)


# ============================================================
# MAIN LOOP
# ============================================================


success = 0
failed = 0


print("\nGenerating heatmaps...")


for idx, sample in attention_data.items():

    try:

        idx = int(idx)

        visualize_sample(idx, sample)

        success += 1

    except Exception as e:

        failed += 1

        print("\nFAILED:", idx)

        print(e)


print("\n==============================")
print("DONE")
print("Success:", success)
print("Failed:", failed)
print("==============================")
