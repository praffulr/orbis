import os
import torch
import numpy as np
from PIL import Image

from sklearn.decomposition import PCA

import torchvision.transforms.functional as TF

from utils import load_clip_frames, sample_frames


# ----------------------------
# CONFIG
# ----------------------------
FEATURE_DIR = "vjepa_feats"
FRAME_DIR = "frames"
OUTPUT_DIR = "vjepa_vis"

IMG_SIZE = 384
PATCH_SIZE = 16

GRID = IMG_SIZE // PATCH_SIZE  # 24


# ----------------------------
# LOAD FEATURES
# ----------------------------
def load_features(path):
    feat = torch.load(path, map_location="cpu")
    return feat.squeeze(0)  # (N, D)


# ----------------------------
# RESHAPE TOKENS → (T, H, W, D)
# ----------------------------
def reshape_tokens(tokens):
    """
    tokens: (N, D)

    N = T * H * W
    """

    N, D = tokens.shape

    num_spatial = GRID * GRID

    T = N // num_spatial

    tokens = tokens[: T * num_spatial]

    tokens = tokens.reshape(T, GRID, GRID, D)

    return tokens


# ----------------------------
# PCA → RGB
# ----------------------------
def apply_pca(tokens):
    """
    tokens: (T, H, W, D)
    """

    T, H, W, D = tokens.shape

    flat = tokens.reshape(-1, D)

    pca = PCA(n_components=3)

    reduced = pca.fit_transform(flat)

    reduced = reduced.reshape(T, H, W, 3)

    # normalize 0-1
    reduced = reduced - reduced.min()
    reduced = reduced / (reduced.max() + 1e-8)

    return reduced


# ----------------------------
# OVERLAY
# ----------------------------
def overlay(image, heatmap, alpha=0.5):
    """
    image: PIL
    heatmap: (H, W, 3)
    """

    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = Image.fromarray(heatmap).resize(image.size)

    return Image.blend(image.convert("RGB"), heatmap, alpha)


# ----------------------------
# VISUALIZE ONE CLIP
# ----------------------------
def visualize_clip(clip_name):
    feat_path = os.path.join(FEATURE_DIR, clip_name + ".pt")

    frame_path = os.path.join(FRAME_DIR, clip_name, "images")

    frames = sorted(os.listdir(frame_path))

    frames = [os.path.join(frame_path, f) for f in frames]

    frames = sample_frames(frames, num_frames=10)

    tokens = load_features(feat_path)

    tokens = reshape_tokens(tokens)

    heatmaps = apply_pca(tokens)

    os.makedirs(os.path.join(OUTPUT_DIR, clip_name), exist_ok=True)

    for i in range(min(len(frames), heatmaps.shape[0])):

        img = Image.open(frames[i]).convert("RGB")

        vis = overlay(img, heatmaps[i])

        save_path = os.path.join(
            OUTPUT_DIR,
            clip_name,
            f"{i:04d}.png",
        )

        vis.save(save_path)

    print(f"[SAVED VISUALIZATION] {clip_name}")


# ----------------------------
# RUN ALL
# ----------------------------
def run_all():
    clips = os.listdir(FEATURE_DIR)

    for clip in clips:
        if clip.endswith(".pt"):
            clip_name = clip.replace(".pt", "")
            print(f"[INFO] Visualizing {clip_name}")
            visualize_clip(clip_name)


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    run_all()