import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

import numpy as np

import wandb

from collections import Counter



# CONFIGURATION



TRAIN_FILE = "cached_features_tb5_new/train_final_mc.pt"

VAL_FILE = "cached_features_tb5_new/val_final_mc.pt"


CHECKPOINT_DIR = "checkpoints_vjepa_dota_mc"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print("\nUsing device:", DEVICE)



# DATASET



class VJEPAFeatureDataset(Dataset):

    """
    Dataset wrapper for cached V-JEPA embeddings.

    Input:
        [576,768]

    Output:
        feature tokens
        multiclass target
        metadata
    """

    def __init__(self, cache_path):

        print("\nLoading cache:")
        print(cache_path)

        data = torch.load(cache_path, weights_only=False)

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        self.features = data["features"].float()

        # ----------------------------------------------------
        # Target
        # IMPORTANT:
        #
        # 0 = normal
        # 1-9 = accident categories
        #
        # ----------------------------------------------------

        self.labels = data["mc_labels"].long()

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.videos = data["videos"]

        self.indices = data["indices"]

        # ----------------------------------------------------
        # Class information
        # ----------------------------------------------------

        self.class_names = data["class_names"]

        self.num_classes = data["num_classes"]

        self.feature_layer = data["feature_layer"]

        self.multiclass_distribution = data["multiclass_distribution"]

        # ----------------------------------------------------
        # Printing information
        # ----------------------------------------------------

        print("\n------------------------------")
        print("Dataset Information")
        print("------------------------------")

        print("Samples:", len(self.features))

        print("Feature shape:", self.features.shape)

        print("Label shape:", self.labels.shape)

        print("Feature layer:", self.feature_layer)

        print("Number of classes:", self.num_classes)

        print("\nClass distribution:")

        for cls, count in sorted(self.multiclass_distribution.items()):

            print(f"{cls:2d} " f"{self.class_names[cls]:35s}" f"{count}")

        print("------------------------------\n")

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        assert self.features.ndim == 3, "Expected feature tensor " "[N,T,D]"

        assert self.features.shape[1] == 576, "Expected 576 spatial tokens"

        assert self.features.shape[2] == 768, "Expected embedding dimension 768"

        assert len(self.features) == len(self.labels)

    def __len__(self):

        return len(self.features)

    def __getitem__(self, idx):

        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "video": self.videos[idx],
            "indices": torch.tensor(self.indices[idx], dtype=torch.long),
        }



# ATTENTION PROBE MODEL



class AttentionProbe(nn.Module):

    """
    Learnable attention pooling over V-JEPA tokens.

    Input:
        [B,576,768]

    Output:
        logits:
            [B,num_classes]

        attention:
            [B,num_heads,1,576]
    """

    def __init__(self, input_dim, num_classes, num_heads=8, dropout=0.1):

        super().__init__()

        # ----------------------------------------------------
        # Learnable query token
        # ----------------------------------------------------

        self.query = nn.Parameter(torch.empty(1, 1, input_dim))

        # Better initialization
        nn.init.xavier_uniform_(self.query)

        # ----------------------------------------------------
        # Pre attention normalization
        # ----------------------------------------------------

        self.norm_tokens = nn.LayerNorm(input_dim)

        # ----------------------------------------------------
        # Cross Attention
        # ----------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # ----------------------------------------------------
        # Output processing
        # ----------------------------------------------------

        self.norm_output = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x, return_attention=False):

        """
        x:
            [B,576,768]
        """

        B = x.shape[0]

        # ----------------------------------------------------
        # Expand query
        # ----------------------------------------------------

        query = self.query.expand(B, -1, -1)

        # ----------------------------------------------------
        # Normalize V-JEPA tokens
        # ----------------------------------------------------

        x = self.norm_tokens(x)

        # ----------------------------------------------------
        # Cross attention
        # ----------------------------------------------------

        pooled, weights = self.attention(
            query=query, key=x, value=x, need_weights=True, average_attn_weights=False
        )

        # pooled:
        # [B,1,768]

        pooled = pooled.squeeze(1)

        pooled = self.norm_output(pooled)

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        if return_attention:

            return logits, weights

        return logits


# INITIALIZATION


def compute_class_weights(labels, num_classes):

    counts = torch.bincount(
        labels,
        minlength=num_classes
    )

    print("\nClass counts:")

    for i,c in enumerate(counts):
        print(i, c.item())


    counts = counts.float()

    weights = torch.zeros_like(counts)

    valid = counts > 0

    weights[valid] = len(labels) / (
        num_classes * counts[valid]
    )

    return weights



def initialize_training(config):

    # -------------------------------
    # Load datasets
    # -------------------------------

    train_dataset = VJEPAFeatureDataset(TRAIN_FILE)

    val_dataset = VJEPAFeatureDataset(VAL_FILE)

    num_classes = train_dataset.num_classes

    class_counts = torch.bincount(
        train_dataset.labels,
        minlength=num_classes
    )


    class_weights_sampler = 1.0 / class_counts.float()


    sample_weights = class_weights_sampler[
        train_dataset.labels
    ]


    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )

    # -------------------------------
    # Class weights
    # -------------------------------

    class_weights = compute_class_weights(
        train_dataset.labels,
        num_classes
    )

    class_weights = class_weights.to(DEVICE)


    # -------------------------------
    # Model
    # -------------------------------

    model = AttentionProbe(
        input_dim=768,
        num_classes=num_classes,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model = model.to(DEVICE)


    # -------------------------------
    # Loss
    # -------------------------------

    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    val_criterion = nn.CrossEntropyLoss()


    # -------------------------------
    # Optimizer
    # -------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(
            config.beta1,
            config.beta2
        )
    )


    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        model,
        train_criterion,
        val_criterion,
        optimizer,
        num_classes,
    )



# TRAINING FUNCTION



def train():

    wandb.init()

    config = wandb.config

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        model,
        train_criterion,
        val_criterion,
        optimizer,
        num_classes,
    ) = initialize_training(config)

    epochs = 50

    best_val_loss = float("inf")

    patience_counter = 0

    patience = config.early_stopping_patience

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(epochs):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_loss = 0.0

        train_preds = []

        train_targets = []

        for batch in train_loader:

            features = batch["features"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(features)

            loss = train_criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            train_preds.extend(preds.cpu().numpy())

            train_targets.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)

        train_acc = accuracy_score(train_targets, train_preds) * 100

        train_f1 = f1_score(
            train_targets, train_preds, average="macro", zero_division=0
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_running_loss = 0.0

        val_targets = []

        val_preds = []

        val_probs = []

        epoch_attention = {}

        with torch.no_grad():

            for batch in val_loader:

                features = batch["features"].to(DEVICE)

                labels = batch["label"].to(DEVICE)

                logits, attention = model(features, return_attention=True)

                loss = val_criterion(logits, labels)

                val_running_loss += loss.item()

                probs = F.softmax(logits, dim=1)

                preds = torch.argmax(logits, dim=1)

                val_targets.extend(labels.cpu().numpy())

                val_preds.extend(preds.cpu().numpy())

                val_probs.extend(probs.cpu().numpy())

                # --------------------------------------------
                # Save attention
                # --------------------------------------------

                videos = batch["video"]

                indices = batch["indices"]

                for i, video in enumerate(videos):

                    epoch_attention[video] = {
                        "attention": attention[i].detach().cpu(),
                        "label": int(labels[i].cpu()),
                        "prediction": int(preds[i].cpu()),
                        "confidence": float(probs[i].max()),
                        "indices": indices[i].tolist(),
                    }

        avg_val_loss = val_running_loss / len(val_loader)

        val_acc = accuracy_score(val_targets, val_preds) * 100

        val_precision = precision_score(
            val_targets, val_preds, average="macro", zero_division=0
        )

        val_recall = recall_score(
            val_targets, val_preds, average="macro", zero_division=0
        )

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        # ====================================================
        # AUC
        # ====================================================

        try:

            val_auc = roc_auc_score(
                np.array(val_targets), np.array(val_probs), multi_class="ovr"
            )

        except Exception:

            val_auc = float("nan")

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(val_targets, val_preds)

        print("\n" + "=" * 70)

        print(f"Epoch {epoch+1}/{epochs}")

        print(f"Train Loss : {avg_train_loss:.4f}")

        print(f"Val Loss   : {avg_val_loss:.4f}")

        print(f"Train Acc  : {train_acc:.2f}%")

        print(f"Val Acc    : {val_acc:.2f}%")

        print(f"Val F1     : {val_f1:.4f}")

        print(f"Val AUC    : {val_auc:.4f}")

        print("=" * 70)

        # ====================================================
        # W&B LOGGING
        # ====================================================

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_precision_macro": val_precision,
                "val_recall_macro": val_recall,
                "val_f1_macro": val_f1,
                "val_auc": val_auc,
            }
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if avg_val_loss < best_val_loss:

            print(">>> Saving best model")

            best_val_loss = avg_val_loss

            patience_counter = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": train_dataset.class_names,
                    "feature_layer": train_dataset.feature_layer,
                    "config": dict(config),
                },
                os.path.join(CHECKPOINT_DIR, "best_vjepa_mc_attention.pt"),
            )

            torch.save(
                epoch_attention,
                os.path.join(CHECKPOINT_DIR, "best_val_attention_weights_mc.pt"),
            )

            torch.save(cm, os.path.join(CHECKPOINT_DIR, "best_confusion_matrix.pt"))

        else:

            patience_counter += 1

            print(f"Early stopping: " f"{patience_counter}/{patience}")

            if patience_counter >= patience:

                print("Early stopping triggered")

                break

    wandb.finish()



# WANDB SWEEP CONFIGURATION



sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_f1_macro", "goal": "maximize"},
    "parameters": {
        # --------------------------------------------
        # Learning rate
        # --------------------------------------------
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 5e-4,
        },
        # --------------------------------------------
        # Weight decay
        # --------------------------------------------
        "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.05},
        # --------------------------------------------
        # Batch size
        # --------------------------------------------
        "batch_size": {"values": [8, 16, 32]},
        # --------------------------------------------
        # Attention heads
        # --------------------------------------------
        "num_heads": {"values": [4, 8]},
        # --------------------------------------------
        # Dropout
        # --------------------------------------------
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        # --------------------------------------------
        # Label smoothing
        # --------------------------------------------
        "label_smoothing": {"values": [0.0, 0.05, 0.1]},
        # --------------------------------------------
        # Adam betas
        # --------------------------------------------
        "beta1": {"values": [0.9, 0.95]},
        "beta2": {"values": [0.99, 0.999]},
        # --------------------------------------------
        # Early stopping
        # --------------------------------------------
        "early_stopping_patience": {"value": 8},
    },
}



# MAIN



if __name__ == "__main__":

    print("\nStarting V-JEPA DoTA Multi-Class Attention Probe")

    print("=" * 70)

    sweep_id = wandb.sweep(sweep_config, project="vjepa-dota-multiclass-attention-new")

    print("\nSweep ID:", sweep_id)

    wandb.agent(sweep_id, function=train, count=20)

