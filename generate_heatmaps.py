import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

ATTENTION_FILE = "best_val_attention_weights_tb5.pt"

FRAME_ROOT = "frames"

OUTPUT_DIR = "attention_heatmaps_tb5_5th"

GRID_H = 24
GRID_W = 24

os.makedirs(OUTPUT_DIR, exist_ok=True)

attention = torch.load(ATTENTION_FILE, map_location="cpu")

# =====================================================
# HELPER
# =====================================================

def process_attention(attn_1d, image):

    H, W, _ = image.shape

    weights = attn_1d.reshape(GRID_H, GRID_W).numpy()

    weights = (
        weights - weights.min()
    ) / (
        weights.max() - weights.min() + 1e-8
    )

    weights = cv2.resize(
        weights,
        (W, H),
        interpolation=cv2.INTER_CUBIC
    )

    heat = cv2.applyColorMap(
        np.uint8(weights * 255),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.7,
        heat,
        0.3,
        0
    )

    return (
        cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    )


# =====================================================
# VIDEOS
# =====================================================

videos = sorted([
    d for d in os.listdir(FRAME_ROOT)
    if os.path.isdir(os.path.join(FRAME_ROOT, d))
])

print("Videos:", len(videos))

# =====================================================
# MAIN
# =====================================================

for idx, video in enumerate(videos):

    if idx not in attention:
        continue

    frame_dir = os.path.join(FRAME_ROOT, video, "images")

    if not os.path.exists(frame_dir):
        continue

    frames = sorted(os.listdir(frame_dir))

    if len(frames) < 5:
        continue

    # show last context frame
    img_path = os.path.join(frame_dir, frames[4])

    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.resize(image, (384,384))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    attn_heads = attention[idx]["attention"]

    num_heads = attn_heads.shape[0]

    mean_attention = attn_heads.mean(0)

    mean_heat, mean_overlay = process_attention(
        mean_attention,
        image
    )

    cols = max(3, num_heads)

    fig = plt.figure(
        figsize=(4*cols,12)
    )

    # =====================================================
    # Top row
    # =====================================================

    plt.subplot(3, cols, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3, cols, 2)
    plt.imshow(mean_heat)
    plt.title("Mean Attention")
    plt.axis("off")

    plt.subplot(3, cols, 3)
    plt.imshow(mean_overlay)
    plt.title("Mean Overlay")
    plt.axis("off")

    # =====================================================
    # Row 2
    # =====================================================

    for h in range(num_heads):

        heat,_ = process_attention(
            attn_heads[h],
            image
        )

        plt.subplot(
            3,
            cols,
            cols+h+1
        )

        plt.imshow(heat)

        plt.title(f"Head {h+1}")

        plt.axis("off")

    # =====================================================
    # Row 3
    # =====================================================

    for h in range(num_heads):

        _,overlay = process_attention(
            attn_heads[h],
            image
        )

        plt.subplot(
            3,
            cols,
            2*cols+h+1
        )

        plt.imshow(overlay)

        plt.title(f"Head {h+1} Overlay")

        plt.axis("off")

    plt.tight_layout()

    label = attention[idx]["label"]

    category = "anomaly" if label == 1 else "normal"

    save_dir = os.path.join(
        OUTPUT_DIR,
        category
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    save_path = os.path.join(
        save_dir,
        video + ".png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)

print("Finished.")