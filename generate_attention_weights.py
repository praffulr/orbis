import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

# =====================================================
# CONFIG
# =====================================================

CHECKPOINT = "checkpoints/best_final_attention.pt"

VAL_FEATURES = "cached_features/val_final.pt"

OUT_FILE = "best_val_attention_weights_tb5.pt"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# =====================================================
# DATASET
# =====================================================


class CachedDataset(Dataset):
    def __init__(self, path):

        data = torch.load(path, weights_only=False)

        self.features = data["features"].float()
        self.labels = data["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        return self.features[idx], self.labels[idx], idx


# =====================================================
# PROBE
# =====================================================


class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):

        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        self.norm1 = nn.LayerNorm(input_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x, return_attention=False):

        B = x.size(0)

        q = self.query.expand(B, -1, -1)

        x = self.norm1(x)

        out, weights = self.attention(
            q, x, x, need_weights=return_attention, average_attn_weights=False
        )

        out = out.squeeze(1)

        out = self.norm2(out)

        out = self.dropout(out)

        logits = self.classifier(out)

        if return_attention:
            return logits, weights

        return logits


# =====================================================
# LOAD MODEL
# =====================================================

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

cfg = checkpoint["config"]

dataset = CachedDataset(VAL_FEATURES)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

input_dim = dataset.features.shape[-1]

model = AttentionProbe(
    input_dim=input_dim, num_heads=cfg["num_heads"], dropout=cfg["dropout"]
).to(DEVICE)

model.load_state_dict(checkpoint["model"])

model.eval()

print("Loaded model.")

# =====================================================
# GENERATE ATTENTION
# =====================================================

attention_dict = {}

with torch.no_grad():

    for x, y, idx in loader:

        x = x.to(DEVICE)
        print(x.shape)
        logits, attn = model(x, return_attention=True)

        # Remove batch dimension
        # [1, heads, 1, 576]
        # ->
        # [heads, 1, 576]

        attn = attn.squeeze(0)

        # remove query dimension
        # ->
        # [heads,576]

        attn = attn.squeeze(1)

        attention_dict[int(idx)] = {
            "attention": attn.cpu(),
            "label": int(y.item())
        }

torch.save(attention_dict, OUT_FILE)

print("Saved:", OUT_FILE)
