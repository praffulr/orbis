import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================


ATTENTION_FILE = "checkpoints_vjepa_dota_mc/" "best_val_attention_weights_mc.pt"


VAL_FILE = "cached_features_tb5_new/" "val_final_mc.pt"


FRAME_ROOT = "frames"


OUTPUT_DIR = "attention_viz_clean_mc"


GRID_H = 24
GRID_W = 24

IMG_SIZE = 384

NUM_CONTEXT_FRAMES = 5

TOP_PERCENT = 5


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASSES
# ============================================================


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


# ============================================================
# LOAD
# ============================================================


print("\nLoading attention weights...")


attention_data = torch.load(ATTENTION_FILE, map_location="cpu", weights_only=False)


print("Attention videos:", len(attention_data))


print("\nLoading validation metadata...")


val_data = torch.load(VAL_FILE, weights_only=False)


print("Validation samples:", len(val_data["videos"]))


# ============================================================
# CLASS ATTENTION STATISTICS
# ============================================================


print("\nComputing attention statistics...")


class_attention = {}


for video, sample in attention_data.items():

    label = int(sample["label"])

    att = sample["attention"]

    # [heads,1,576]

    att = att.mean(dim=0)

    att = att.squeeze(0)

    if label not in class_attention:

        class_attention[label] = []

    class_attention[label].append(att.numpy())


CLASS_MEANS = {}


for cls, values in class_attention.items():

    CLASS_MEANS[cls] = np.mean(np.stack(values), axis=0)


print("\nAvailable class means")

for cls in CLASS_MEANS:

    print(cls, CLASS_NAMES[cls], CLASS_MEANS[cls].shape)


# ============================================================
# FRAME LOADER
# ============================================================


def load_frame(video, frame_idx):

    path = os.path.join(FRAME_ROOT, video, "images", f"{int(frame_idx):06d}.jpg")

    image = cv2.imread(path)

    if image is None:

        raise FileNotFoundError(path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# ============================================================
# ATTENTION PROCESSING
# ============================================================


def process_attention(attention, label):

    """
    attention:
        [576]

    """

    attention = attention.detach().cpu().numpy()

    att_map = attention.reshape(GRID_H, GRID_W)

    # compare against class behaviour

    class_mean = CLASS_MEANS[label].reshape(GRID_H, GRID_W)

    relative = att_map - class_mean

    max_abs = np.max(np.abs(relative))

    if max_abs > 1e-8:

        relative /= max_abs

    positive = np.clip(relative, 0, None)

    # keep top-k regions

    flat = positive.flatten()

    if flat.max() > 0:

        k = max(1, int(len(flat) * TOP_PERCENT / 100))

        threshold = np.partition(flat, -k)[-k]

        positive = np.where(positive >= threshold, positive, 0)

    return positive


# ============================================================
# HEATMAP
# ============================================================


def create_heatmap(attention):

    attention = cv2.GaussianBlur(attention.astype(np.float32), (5, 5), 0)

    attention = cv2.resize(
        attention, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC
    )

    if attention.max() > 0:

        attention /= attention.max()

    heat = cv2.applyColorMap(np.uint8(attention * 255), cv2.COLORMAP_TURBO)

    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    return heat


def overlay(image, heat, alpha=0.30):

    out = image * (1 - alpha) + heat * alpha

    return np.uint8(np.clip(out, 0, 255))


# ============================================================
# VISUALIZATION
# ============================================================


def visualize(video, sample_id, sample):

    label = int(sample["label"])

    prediction = int(sample["prediction"])

    confidence = sample["confidence"]

    indices = sample["indices"]

    attention = sample["attention"]

    # average heads

    attention = attention.mean(dim=0).squeeze(0)

    positive = process_attention(attention, label)

    frames = []

    for idx in indices:

        frames.append(load_frame(video, idx))

    fig, axes = plt.subplots(3, NUM_CONTEXT_FRAMES, figsize=(20, 12))

    for i, img in enumerate(frames):

        heat = create_heatmap(positive)

        over = overlay(img, heat)

        axes[0, i].imshow(img)

        axes[0, i].set_title(f"Frame {indices[i]}")

        axes[1, i].imshow(heat)

        axes[1, i].set_title("Attention")

        axes[2, i].imshow(over)

        axes[2, i].set_title("Overlay")

        for r in range(3):

            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=14)

    axes[1, 0].set_ylabel("Heatmap", fontsize=14)

    axes[2, 0].set_ylabel("Overlay", fontsize=14)

    gt = CLASS_NAMES[label]

    pred = CLASS_NAMES[prediction]

    status = "CORRECT" if label == prediction else "WRONG"

    fig.suptitle(
        f"""
{video}

GT: {gt}

Prediction: {pred}

Confidence: {confidence:.3f}

{status}
""",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.90))

    folder = os.path.join(OUTPUT_DIR, gt, status)

    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{sample_id}_{video}.png")

    plt.savefig(path, dpi=200, bbox_inches="tight")

    plt.close()

    print("Saved:", path)


# ============================================================
# GENERATE
# ============================================================


print("\nGenerating visualizations...")


success = 0
failed = 0


for video, sample in tqdm(attention_data.items()):

    try:

        visualize(video, video, sample)

        success += 1

    except Exception as e:

        failed += 1

        print("\nFAILED:", video)

        print(e)


print("\n" + "=" * 70)

print("DONE")

print("Successful:", success)

print("Failed:", failed)

print("Saved:", OUTPUT_DIR)

print("=" * 70)
