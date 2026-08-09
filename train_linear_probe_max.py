import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import wandb


# CONFIG


LAYER = "block_12"
TRAIN_FILE = f"./cached_features/train_{LAYER}.pt"
VAL_FILE = f"./cached_features/val_{LAYER}.pt"

CHECKPOINT_DIR = "checkpoints_maxpool"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 50


# W&B SWEEP CONFIG


sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.1
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "beta1": {
            "values": [0.9, 0.95]
        },
        "beta2": {
            "values": [0.99, 0.999]
        },
        "dropout": {
            "values": [0.0, 0.1, 0.2]
        },
        "early_stopping_patience": {
            "value": 10
        }
    }
}


# RANDOM SEED


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# DATASET


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, weights_only=False)
        self.features = data["features"].float()
        self.labels = data["labels"].long()

        print("\nLoaded:", cache_path)
        print("Feature shape:", self.features.shape)
        print("Labels:", self.labels.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# MAX POOL PROBE


class MaxPoolProbe(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Input: [Batch, Tokens, Embedding] (e.g., [64, 1152, 768])
        # Token-wise Max Pooling -> [Batch, Embedding]
        x = torch.max(x, dim=1)[0]
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# DEVICE


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("mps")
    return torch.device("cpu")


# TRAINING FUNCTION


def train_maxpool_probe():
    wandb.init()
    config = wandb.config
    seed_everything()
    device = get_device()
    print("\nDevice:", device)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = CachedFeatureDataset(TRAIN_FILE)
    val_dataset = CachedFeatureDataset(VAL_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    input_dim = train_dataset[0][0].shape[-1]  # Safely infer input dimension

    model = MaxPoolProbe(
        input_dim=input_dim,
        num_classes=2,
        dropout=config.dropout,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # --------------------------------------------------------
    # Epoch Loop
    # --------------------------------------------------------
    for epoch in range(EPOCHS):
        
        # TRAIN
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # The model handles max pooling internally now
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        epoch_time = time.time() - start

        
        # VALIDATION
        
        model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = F.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds) * 100

        precision = precision_score(all_labels, all_preds, zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, zero_division=0) * 100
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(all_labels, all_preds)

        
        # PRINT
        
        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("=" * 70)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Acc    : {val_acc:.2f}%")
        print(f"Precision  : {precision:.2f}%")
        print(f"Recall     : {recall:.2f}%")
        print(f"AUC        : {auc:.4f}")
        print(f"Epoch Time : {epoch_time:.2f} sec")
        print("\nConfusion Matrix")
        print(cm)

        
        # W&B LOGGING
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_auc": auc,
            "epoch_time": epoch_time,
        })

        
        # SAVE BEST & EARLY STOPPING
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                f"{CHECKPOINT_DIR}/best_maxpool_probe_{LAYER}.pt",
            )
            print("✓ Saved Best Model")
        else:
            patience_counter += 1
            print(f"Early Stopping Patience: {patience_counter}/{config.early_stopping_patience}")
            if patience_counter >= config.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

    wandb.finish()


# MAIN

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        project="vjepa-dota-maxpool-probe",
    )

    wandb.agent(
        sweep_id,
        function=train_maxpool_probe,
        count=20,
    )