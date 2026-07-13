import os
import glob
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


IMG_SIZE = 384


transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def load_clip_frames(clip_path):
    """
    Returns sorted jpg files inside

    clip/images/
    """
    frame_paths = sorted(
        glob.glob(
            os.path.join(
                clip_path,
                "images",
                "*.jpg",
            )
        )
    )

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {clip_path}")

    return frame_paths


def sample_frames(frame_paths, num_frames=8):
    """
    Uniform sampling.
    """

    idx = np.linspace(
        0,
        len(frame_paths) - 1,
        num_frames,
    ).astype(int)

    return [frame_paths[i] for i in idx]


def load_video_tensor(frame_paths):
    """
    Returns

    (1,C,T,H,W)
    """

    imgs = []

    for path in frame_paths:

        img = Image.open(path).convert("RGB")
        img = transform(img)

        imgs.append(img)

    video = torch.stack(imgs)

    # (T,C,H,W)

    video = video.permute(1, 0, 2, 3)

    # (C,T,H,W)

    return video.unsqueeze(0)