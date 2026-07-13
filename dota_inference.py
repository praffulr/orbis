import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.models.vision_transformer import vit_base
from src.models.utils.pos_embs import get_3d_sincos_pos_embed


# -----------------------------
# CONFIG
# -----------------------------
CKPT_PATH = "vjepa_ckpt/vjepa2_1_vitb_dist_vitG_384.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
NUM_FRAMES = 16


# -----------------------------
# LOAD CLIP
# -----------------------------
def load_clip(clip_path):
    frames = sorted(glob.glob(os.path.join(clip_path, "images", "*.jpg")))
    assert len(frames) > 0, "No frames found"
    return frames


# -----------------------------
# SAMPLE FRAMES
# -----------------------------
def sample_frames(frames, num_frames=NUM_FRAMES):
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [frames[i] for i in idx]


# -----------------------------
# TRANSFORM
# -----------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])


def load_video_tensor(frame_paths):
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        img = transform(img)
        frames.append(img)

    video = torch.stack(frames, dim=0)  # (T, C, H, W)
    return video.unsqueeze(0)           # (B, T, C, H, W)


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_vjepa():
    model = vit_base(
        img_size=IMG_SIZE,
        num_frames=NUM_FRAMES,
        patch_size=16
    )

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["encoder"], strict=False)

    model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# FORWARD PASS
# -----------------------------
@torch.no_grad()
def run_model(model, video):
    # VJEPA expects: B, C, T, H, W
    video = video.permute(0, 2, 1, 3, 4).to(DEVICE)

    features = model(video)
    return features


# -----------------------------
# RUN ON ONE CLIP
# -----------------------------
def run_clip(clip_path):
    print(f"[INFO] Processing: {clip_path}")

    frames = load_clip(clip_path)
    frames = sample_frames(frames)

    video = load_video_tensor(frames)

    model = load_vjepa()

    features = run_model(model, video)

    print("[DONE] Feature shape:", features.shape)

    # Save features
    save_path = clip_path.replace("frames", "vjepa_feats") + ".pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(features.cpu(), save_path)
    print("[SAVED]", save_path)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    clip_path = "frames/0RJPQ_97dcs_000199"
    run_clip(clip_path)