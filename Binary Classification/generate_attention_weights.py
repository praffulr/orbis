import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# CONFIG


CHECKPOINT = "checkpoints/best_vjepa_attention.pt"

VAL_FEATURES = "cached_features/val_vjepa_final_mc.pt"

OUT_FILE = "best_val_attention_weights_tb5.pt"


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# CUSTOM COLLATE


def custom_collate(batch):

    features = torch.stack([item[0] for item in batch])

    labels = torch.stack([item[1] for item in batch])

    indices = [item[2] for item in batch]

    videos = [item[3] for item in batch]

    frame_indices = [item[4] for item in batch]

    return (features, labels, indices, videos, frame_indices)


# DATASET


class CachedDataset(Dataset):
    def __init__(self, path):

        data = torch.load(path, weights_only=False)

        self.features = data["features"].float()

        self.labels = data["labels"].long()

        self.videos = data["video_ids"]

        self.indices = data["target_frame_ids"]

        print("\nLoaded validation dataset")
        print("----------------------------")

        print("Features :", self.features.shape)

        print("Labels   :", self.labels.shape)

        print("Videos   :", len(self.videos))

        print("Indices  :", len(self.indices))

        print("\nExample metadata")

        print("Video  :", self.videos[0])

        print("Frames :", self.indices[0])

        print("Length :", len(self.indices[0]))

        assert self.features.ndim == 3

        assert self.features.shape[1] == 576

        assert len(self.videos) == len(self.features)

        assert len(self.indices) == len(self.features)

    def __len__(self):

        return len(self.features)

    def __getitem__(self, idx):

        return (
            self.features[idx],
            self.labels[idx],
            idx,
            self.videos[idx],
            self.indices[idx],
        )


# ATTENTION PROBE


class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):

        super().__init__()

        # Learnable query token
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

        # Expand query for batch
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


# LOAD CHECKPOINT


checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)


cfg = checkpoint["config"]


print("\nCheckpoint loaded")

print(cfg)


dataset = CachedDataset(VAL_FEATURES)


loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)


input_dim = dataset.features.shape[-1]


model = AttentionProbe(
    input_dim=input_dim, num_heads=cfg["num_heads"], dropout=cfg["dropout"]
).to(DEVICE)


model.load_state_dict(checkpoint["model"])


model.eval()


print("\nProbe loaded successfully")


# GENERATE ATTENTION


attention_dict = {}


correct = 0

total = 0


with torch.no_grad():

    for (x, y, idx, videos, frame_indices) in loader:

        x = x.to(DEVICE)

        logits, attn = model(x, return_attention=True)

        pred = logits.argmax(dim=1)

        probs = torch.softmax(logits, dim=1)

        confidence = probs.max().item()

        correct += (pred.cpu() == y).sum().item()

        total += 1

        # =========================================
        # Attention extraction
        #
        # Original:
        # [1, num_heads, 1, 576]
        #
        # After squeeze:
        # [num_heads,576]
        #
        # =========================================

        attn = attn.squeeze(0)

        attn = attn.squeeze(1)

        assert attn.shape == (cfg["num_heads"], 576)

        # =========================================
        # Metadata
        # =========================================

        video_name = videos[0]

        # Keep filenames as strings
        frames = frame_indices[0]

        attention_dict[int(idx[0])] = {
            "attention": attn.cpu(),
            "label": int(y.item()),
            "prediction": int(pred.item()),
            "confidence": float(confidence),
            "video": video_name,
            "indices": frames,
        }

        print(
            f"{idx[0]:4d} | "
            f"{video_name} | "
            f"frames={frames} | "
            f"GT={y.item()} "
            f"PRED={pred.item()} "
            f"CONF={confidence:.3f}"
        )


# SAVE


torch.save(attention_dict, OUT_FILE)


print("\nSaved:")

print(OUT_FILE)


print(f"\nValidation accuracy: " f"{100*correct/total:.2f}%")
