import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = (
    "vjepa_features/normal/0RJPQ_97dcs_003475.pt"
)


FRAME_ROOT = "frames"


LAYER = "block_6"


FRAME_NUMBER = 8
# choose frame inside 16-frame clip


IMG_SIZE = 384



# ============================================================
# LOAD FEATURE
# ============================================================


data = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)


video_id = data["video"]


indices = data["indices"]


activation = data["feature"][LAYER]


print("Video:",video_id)
print("Frame indices:",indices)
print(
    "Activation:",
    activation.shape
)



# ============================================================
# LOAD FRAME
# ============================================================


frames = sorted(
    [
        os.path.join(
            FRAME_ROOT,
            video_id,
            "images",
            f
        )

        for f in os.listdir(
            os.path.join(
                FRAME_ROOT,
                video_id,
                "images"
            )
        )

        if f.endswith(".jpg")
    ]
)



frame_path = frames[
    indices[FRAME_NUMBER]
]


img=cv2.imread(
    frame_path
)


img=cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


img=cv2.resize(
    img,
    (IMG_SIZE,IMG_SIZE)
)



# ============================================================
# TEMPORAL-SPATIAL TOKEN PROCESSING
# ============================================================

tokens = activation.numpy()


print(
    "Tokens:",
    tokens.shape
)


# V-JEPA2:
#
# tokens:
# [Tubes, Spatial_Patches, Feature]
#
# Here:
#
# 4608 = 8 * 576
#
# 576 = 24*24


TEMPORAL_TOKENS = 8
PATCHES_PER_FRAME = 24*24


tokens = tokens.reshape(
    TEMPORAL_TOKENS,
    PATCHES_PER_FRAME,
    768
)



# choose temporal location
#
# 0-7 possible

TEMPORAL_INDEX = 4



spatial_tokens = tokens[
    TEMPORAL_INDEX
]


print(
    "Spatial tokens:",
    spatial_tokens.shape
)



# feature magnitude

heatmap = np.mean(
    np.abs(spatial_tokens),
    axis=1
)



heatmap = heatmap.reshape(
    24,
    24
)



# normalize

heatmap = (
    heatmap -
    heatmap.min()
)

heatmap /= (
    heatmap.max()+1e-8
)



# resize to image

heatmap=cv2.resize(
    heatmap,
    (IMG_SIZE,IMG_SIZE)
)



# apply colormap

heatmap_color=cv2.applyColorMap(
    np.uint8(
        heatmap*255
    ),
    cv2.COLORMAP_JET
)


heatmap_color=cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)



# ============================================================
# OVERLAY
# ============================================================


overlay = (
    0.6*img +
    0.4*heatmap_color
)


overlay=np.uint8(
    overlay
)



# ============================================================
# SAVE
# ============================================================


plt.figure(
    figsize=(12,5)
)


plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")


plt.subplot(1,3,2)
plt.imshow(
    heatmap,
    cmap="jet"
)
plt.title(
    LAYER+" activation"
)
plt.axis("off")


plt.subplot(1,3,3)
plt.imshow(
    overlay
)
plt.title(
    "Overlay"
)
plt.axis("off")



plt.tight_layout()


plt.savefig(
    "activation_heatmap.png",
    dpi=300
)


print(
    "Saved activation_heatmap.png"
)