import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path


# CONFIG

CHECKPOINT = "checkpoints/best_vjepa_attention.pt"

VAL_FEATURES = "cached_features/val_vjepa_final_mc.pt"

ATTENTION_OUTPUT = "best_val_attention_weights_tb5.pt"

FINAL_MODEL_OUTPUT = "best_vjepa_attention_final.pt"


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


print("Using device:", DEVICE)

# DATASET


class CachedDataset(Dataset):
    def __init__(self, path):

        data = torch.load(path, map_location="cpu", weights_only=False)

        self.features = data["features"].float()

        self.labels = data["labels"].long()

        self.video_ids = data["video_ids"]

        self.frame_ids = data["target_frame_ids"]

        print("\nDataset Loaded")
        print("---------------------")
        print("Features:", self.features.shape)
        print("Labels:", self.labels.shape)
        print("Videos:", len(self.video_ids))

        assert self.features.ndim == 3

        assert self.features.shape[1] == 576

        assert self.features.shape[2] == 768

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.features[idx],
            self.labels[idx],
            idx,
            self.video_ids[idx],
            self.frame_ids[idx],
        )


# ATTENTION PROBE


class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):

        super().__init__()

        # learnable query token

        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        self.norm1 = nn.LayerNorm(input_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x, return_attention=False):

        B = x.shape[0]

        # expand query for batch

        q = self.query.expand(B, -1, -1)

        x = self.norm1(x)

        attn_out, weights = self.attention(
            q, x, x, need_weights=return_attention, average_attn_weights=False
        )

        # remove query dimension

        x = attn_out.squeeze(1)

        x = self.norm2(x)

        x = self.dropout(x)

        logits = self.classifier(x)

        if return_attention:

            return logits, weights

        return logits


# LOAD CHECKPOINT


print("\nLoading checkpoint...")


checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)


cfg = checkpoint["config"]


print("\nCheckpoint config:")
for k, v in cfg.items():
    print(k, ":", v)


# CREATE MODEL


model = AttentionProbe(
    input_dim=768, num_heads=cfg["num_heads"], dropout=cfg["dropout"]
)


model.load_state_dict(checkpoint["model"])


model.to(DEVICE)

model.eval()


print("\nModel loaded successfully")


# SAVE FINAL INFERENCE MODEL


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "input_dim": 768,
            "num_heads": cfg["num_heads"],
            "dropout": cfg["dropout"],
        },
        "training_config": cfg,
    },
    FINAL_MODEL_OUTPUT,
)


print("Saved final model:", FINAL_MODEL_OUTPUT)


# LOAD VALIDATION DATA


dataset = CachedDataset(VAL_FEATURES)


loader = DataLoader(dataset, batch_size=1, shuffle=False)

# FORWARD PASS + ATTENTION EXTRACTION


attention_dict = {}


correct = 0

total = 0


print("\nRunning validation forward pass...")


with torch.no_grad():

    for (features, labels, idx, videos, frame_ids) in loader:

        features = features.to(DEVICE)

        labels = labels.to(DEVICE)

        # FORWARD PASS

        logits, attn = model(features, return_attention=True)

        # prediction

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1)

        confidence = probs.max().item()

        correct += (pred == labels).sum().item()

        total += 1

        #
        # ATTENTION PROCESSING
        #

        # Original:
        # [1, heads, 1, 576]

        attn = attn.squeeze(0)

        # [heads,576]

        attn = attn.squeeze(1)

        attention_dict[int(idx.item())] = {
            "attention": attn.cpu(),
            "label": int(labels.item()),
            "prediction": int(pred.item()),
            "confidence": float(confidence),
            "video": videos[0],
            "indices": frame_ids[0],
        }

        print(
            f"{idx.item():4d} | "
            f"{videos[0]} | "
            f"GT={labels.item()} "
            f"PRED={pred.item()} "
            f"CONF={confidence:.3f}"
        )

# SAVE ATTENTION FILE
torch.save(attention_dict, ATTENTION_OUTPUT)


print("\n")
print("DONE")
print("")

print("Attention saved:", ATTENTION_OUTPUT)


print("Validation Accuracy:", 100 * correct / total)

print("Samples:", total)
