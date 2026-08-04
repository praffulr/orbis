import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score

import wandb


# =====================================================
# CONFIG
# =====================================================


TRAIN_FILE = "./cached_features/train_vjepa_final_mc.pt"
VAL_FILE   = "./cached_features/val_vjepa_final_mc.pt"


CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


EPOCHS = 30


# =====================================================
# SWEEP CONFIG
# =====================================================

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        "weight_decay": {"distribution": "uniform", "min": 0, "max": 0.1},
        "batch_size": {"values": [16, 32, 64]},
        "beta1": {"values": [0.9, 0.95]},
        "beta2": {"values": [0.99, 0.999]},
        "num_heads": {"values": [4, 8, 16]},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "early_stopping_patience": {"value": 10},
    },
}


# =====================================================
# SEED
# =====================================================


def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# =====================================================
# DATASET
# =====================================================


class CachedFeatureDataset(Dataset):
    def __init__(self, path):

        data = torch.load(path, weights_only=False)

        self.features = data["features"].float()

        self.labels = data["labels"].long()

        print("\nLoaded:", path)

        print("Features:", self.features.shape)

        print("Labels:", self.labels.shape)

        assert len(self.features.shape) == 3, "Expected [N,T,D] token features"

        print(
            "Tokens per sample:",
            self.features.shape[1]
        )

        assert self.features.shape[1] == 576, \
            f"Expected 576 tokens for tubelet=5, got {self.features.shape[1]}"

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (self.features[index], self.labels[index])


# =====================================================
# ATTENTION POOLING PROBE
# =====================================================


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

        B = x.size(0)

        q = self.query.expand(B, -1, -1)

        x = self.norm1(x)

        attn_out, attn_weights = self.attention(
            q, x, x, need_weights=return_attention, average_attn_weights=False
        )

        x = attn_out.squeeze(1)

        x = self.norm2(x)

        x = self.dropout(x)

        logits = self.classifier(x)

        if return_attention:

            return logits, attn_weights

        return logits


# =====================================================
# DEVICE
# =====================================================


def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    elif torch.backends.mps.is_available():

        torch.set_float32_matmul_precision("high")

        return torch.device("mps")

    return torch.device("cpu")


# =====================================================
# TRAIN
# =====================================================


def train():

    wandb.init(project="vjepa-final-attention-probe-binary")

    config = wandb.config

    seed_everything()

    device = get_device()

    print("Device:", device)

    train_dataset = CachedFeatureDataset(TRAIN_FILE)

    val_dataset = CachedFeatureDataset(VAL_FILE)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    input_dim = train_dataset[0][0].shape[-1]

    print("Embedding dimension:", input_dim)

    model = AttentionProbe(input_dim, config.num_heads, config.dropout).to(device)

    print(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    patience_counter = 0

    for epoch in range(EPOCHS):

        # =====================
        # TRAIN
        # =====================

        model.train()

        train_loss = 0

        correct = 0

        total = 0

        for x, y in train_loader:

            x = x.to(device)

            y = y.to(device)

            optimizer.zero_grad()

            # only logits

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            pred = logits.argmax(1)

            correct += (pred == y).sum().item()

            total += len(y)

        train_loss /= len(train_loader)

        train_acc = 100 * correct / total

        # =====================
        # VALIDATION
        # =====================

        model.eval()

        val_loss = 0

        preds = []

        probs = []

        labels = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)

                y = y.to(device)

                logits = model(x)

                loss = criterion(logits, y)

                val_loss += loss.item()

                preds.extend(logits.argmax(1).cpu().numpy())

                probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())

                labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)

        val_acc = accuracy_score(labels, preds) * 100

        try:

            auc = roc_auc_score(labels, probs)

        except:

            auc = 0.0

        print(
            f"""
Epoch {epoch+1}

Train Loss : {train_loss:.4f}
Train Acc  : {train_acc:.2f}

Val Loss   : {val_loss:.4f}
Val Acc    : {val_acc:.2f}

AUC        : {auc:.4f}

"""
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": auc,
            }
        )

        if val_loss < best_loss:

            best_loss = val_loss

            patience_counter = 0

            checkpoint = {

                "model": model.state_dict(),

                "config": {

                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "batch_size": config.batch_size,
                    "beta1": config.beta1,
                    "beta2": config.beta2,
                    "num_heads": config.num_heads,
                    "dropout": config.dropout,

                },

                "vjepa_config": {

                    "tubelet_size": 5,
                    "num_frames": 5,
                    "grid_size": (24,24)

                }

            }

            torch.save(checkpoint, f"{CHECKPOINT_DIR}/best_vjepa_attention.pt")

            print("Saved best checkpoint")

        else:

            patience_counter += 1

            if patience_counter >= config.early_stopping_patience:

                print("Early stopping")

                break

    wandb.finish()


# =====================================================
# RUN SWEEP
# =====================================================


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project="vjepa-final-attention-probe-binary")

    wandb.agent(sweep_id, function=train, count=10)
