import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = (
    # "vjepa_features/anomaly/0RJPQ_97dcs_001437.pt"
    # "vjepa_features/anomaly/2TmFM9p1KF8_005238.pt"
    # "vjepa_features/anomaly/3Sqeb-l1RPA_001183.pt"
    # "vjepa_features/normal/0RJPQ_97dcs_002296.pt"
    # "vjepa_features/normal/0RJPQ_97dcs_002409.pt"
    "vjepa_features/normal/0RJPQ_97dcs_003475.pt"
)


FRAME_ROOT="frames"


LAYER="block_12"


FRAME_NUMBER=2


IMG_SIZE=384



# ============================================================
# LOAD FEATURE
# ============================================================


data=torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)


video_id=data["video"]


indices=data["indices"]


activation=data["feature"][LAYER]


print("Video:",video_id)

print(
    "Sampled frame indices:",
    indices
)


print(
    "Activation shape:",
    activation.shape
)



# ============================================================
# LOAD ORIGINAL FRAME
# ============================================================


image_dir=os.path.join(
    FRAME_ROOT,
    video_id,
    "images"
)



frames=sorted(
    [
        os.path.join(
            image_dir,
            f
        )

        for f in os.listdir(image_dir)

        if f.endswith(".jpg")
    ]
)



frame_path=frames[
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
# TOKEN PROCESSING
# ============================================================


tokens=activation.numpy()


print(
    "Total tokens:",
    tokens.shape[0]
)



PATCHES_PER_FRAME=24*24



assert tokens.shape[0] % PATCHES_PER_FRAME == 0, \
    "Token count cannot be divided into spatial patches"



TEMPORAL_TOKENS = (
    tokens.shape[0] //
    PATCHES_PER_FRAME
)


print(
    "Temporal tokens:",
    TEMPORAL_TOKENS
)


tokens=tokens.reshape(
    TEMPORAL_TOKENS,
    PATCHES_PER_FRAME,
    768
)



# ============================================================
# SELECT TEMPORAL TOKEN
# ============================================================


# Map original frame number
# to temporal token


temporal_index = min(
    FRAME_NUMBER // 2,
    TEMPORAL_TOKENS-1
)



print(
    "Using temporal token:",
    temporal_index
)



spatial_tokens=tokens[
    temporal_index
]



print(
    "Spatial tokens:",
    spatial_tokens.shape
)



# ============================================================
# CREATE HEATMAP
# ============================================================


heatmap=np.mean(
    np.abs(spatial_tokens),
    axis=1
)



heatmap=heatmap.reshape(
    24,
    24
)



heatmap-=heatmap.min()

heatmap/=(
    heatmap.max()+1e-8
)



heatmap=cv2.resize(
    heatmap,
    (IMG_SIZE,IMG_SIZE)
)



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


overlay=(
    0.6*img+
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

plt.title(
    "Original"
)

plt.axis("off")



plt.subplot(1,3,2)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    f"{LAYER} activation"
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

