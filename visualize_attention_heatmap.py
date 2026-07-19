import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os



# ======================================================
# CONFIG
# ======================================================

CHECKPOINT = "best_attention_probe_block_12.pt"

FEATURE_FILE = "./cached_features/val_block_12.pt"


SAVE_DIR = "attention_maps"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



# ======================================================
# MODEL
# ======================================================


class AttentionProbe(nn.Module):

    def __init__(
        self,
        input_dim=768,
        num_classes=2,
        num_heads=8
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


        self.classifier = nn.Linear(
            input_dim,
            num_classes
        )



    def forward(self,x):

        B=x.size(0)


        q=self.query.expand(
            B,
            -1,
            -1
        )


        x=self.norm1(x)


        attended, weights = self.attention(
            q,
            x,
            x,
            need_weights=True,
            average_attn_weights=False
        )


        pooled=attended.squeeze(1)

        pooled=self.norm2(
            pooled
        )


        logits=self.classifier(
            pooled
        )


        return logits, weights




# ======================================================
# LOAD MODEL
# ======================================================


model=AttentionProbe()

checkpoint=torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint
)


model.to(DEVICE)

model.eval()


print("Model loaded")



# ======================================================
# LOAD FEATURES
# ======================================================


data=torch.load(
    FEATURE_FILE,
    weights_only=False
)


features=data["features"]

labels=data["labels"]



print(
    features.shape
)


# Example:
#
# [159,1152,768]
#



# ======================================================
# PICK SAMPLE
# ======================================================


sample_id=0


x=features[
    sample_id
]


label=labels[
    sample_id
]


x=x.unsqueeze(0).to(
    DEVICE
)


print(
    "GT label:",
    label.item()
)



# ======================================================
# INFERENCE
# ======================================================


with torch.no_grad():

    logits, weights = model(x)



prediction=torch.argmax(
    logits,
    dim=1
)


print(
    "Prediction:",
    prediction.item()
)



print(
    "Attention shape:",
    weights.shape
)



# ======================================================
# PROCESS ATTENTION
# ======================================================


#
# weights:
#
# [1,8,1,1152]
#


attention = weights.squeeze(
    0
)


# [8,1,1152]


attention = attention.squeeze(
    1
)


# [8,1152]


attention = attention.mean(
    dim=0
)


# [1152]



attention = attention.cpu().numpy()



# normalize

attention = (
    attention -
    attention.min()
)


attention = (
    attention /
    attention.max()
)



# ======================================================
# RESHAPE TOKENS
# ======================================================


#
# V-JEPA block features:
#
# 1152 tokens
#
# Assuming:
#
# temporal tokens x spatial tokens
#
#


# Find factorization

print(
    "Attention tokens:",
    attention.shape
)



# For your V-JEPA setup:
#
# 1152 = 8 temporal * 144 spatial
#
# 144 = 12x12
#


T=8

H=12

W=12



attention_map = attention.reshape(
    T,
    H,
    W
)



# average over time

spatial_attention = attention_map.mean(
    axis=0
)



# ======================================================
# PLOT HEATMAP
# ======================================================


plt.figure(
    figsize=(6,6)
)


plt.imshow(
    spatial_attention,
    cmap="jet"
)


plt.colorbar()


plt.title(
    f"Attention Heatmap | Label {label.item()}"
)


plt.axis(
    "off"
)



plt.savefig(
    f"{SAVE_DIR}/sample_{sample_id}.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "Saved heatmap"
)