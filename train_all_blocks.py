import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import wandb
import json
import os



# ============================================================
# CONFIG
# ============================================================


LAYERS=[
    "final",
    "block_3",
    "block_6",
    "block_9",
    "block_12"
]


CACHE_DIR="./cached_features"


BATCH_SIZE=64

EPOCHS=50


RESULT_FILE="probe_results.json"



# ============================================================
# DATASET
# ============================================================


class CachedFeatureDataset(Dataset):

    def __init__(self,path):

        data=torch.load(
            path,
            weights_only=False
        )

        self.features=data["features"]

        self.labels=data["labels"].long()


        print(
            "Loaded:",
            path
        )

        print(
            "Shape:",
            self.features.shape
        )


    def __len__(self):

        return len(self.labels)



    def __getitem__(self,i):

        return (
            self.features[i],
            self.labels[i]
        )



# ============================================================
# MODEL
# ============================================================


class LinearProbe(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.classifier=nn.Linear(
            input_dim,
            2
        )


    def forward(self,x):

        return self.classifier(x)



# ============================================================
# SINGLE TRAIN RUN
# ============================================================


def train_probe(layer):


    wandb.init(
        name=f"{layer}",
        reinit=True
    )


    config=wandb.config



    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "mps"
        if torch.backends.mps.is_available()
        else
        "cpu"
    )


    train_dataset=CachedFeatureDataset(
        f"{CACHE_DIR}/train_{layer}.pt"
    )


    val_dataset=CachedFeatureDataset(
        f"{CACHE_DIR}/val_{layer}.pt"
    )


    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



    sample,_=train_dataset[0]


    input_dim=sample.numel()



    model=LinearProbe(
        input_dim
    ).to(device)



    criterion=nn.CrossEntropyLoss()



    optimizer=optim.AdamW(

        model.parameters(),

        lr=config.learning_rate,

        weight_decay=config.weight_decay,

        betas=(
            config.beta1,
            config.beta2
        )
    )



    best_loss=float("inf")

    best_auc=0

    best_metrics={}


    patience=config.early_stopping_patience

    counter=0



    for epoch in range(EPOCHS):


        # =====================
        # TRAIN
        # =====================


        model.train()


        total_loss=0

        correct=0

        total=0



        for x,y in train_loader:


            x=x.to(device)

            y=y.to(device)



            x=x.reshape(
                x.size(0),
                -1
            )


            optimizer.zero_grad()


            out=model(x)


            loss=criterion(
                out,
                y
            )


            loss.backward()

            optimizer.step()



            total_loss+=loss.item()


            pred=out.argmax(1)


            correct+=(pred==y).sum().item()

            total+=y.size(0)



        train_loss=total_loss/len(train_loader)

        train_acc=100*correct/total



        # =====================
        # VALIDATION
        # =====================


        model.eval()


        val_loss=0


        labels=[]

        preds=[]

        probs=[]



        with torch.no_grad():


            for x,y in val_loader:


                x=x.to(device)

                y=y.to(device)



                x=x.reshape(
                    x.size(0),
                    -1
                )



                out=model(x)


                loss=criterion(
                    out,
                    y
                )


                val_loss+=loss.item()



                pred=out.argmax(1)



                p=F.softmax(
                    out,
                    dim=1
                )[:,1]



                labels.extend(
                    y.cpu().numpy()
                )


                preds.extend(
                    pred.cpu().numpy()
                )


                probs.extend(
                    p.cpu().numpy()
                )



        val_loss/=len(val_loader)



        acc=accuracy_score(
            labels,
            preds
        )


        precision=precision_score(
            labels,
            preds,
            zero_division=0
        )


        recall=recall_score(
            labels,
            preds,
            zero_division=0
        )


        auc=roc_auc_score(
            labels,
            probs
        )



        print(
            f"{layer} | Epoch {epoch+1}"
        )

        print(
            f"Loss {val_loss:.4f}"
            f" AUC {auc:.4f}"
        )



        wandb.log({

            "epoch":epoch,

            "train_loss":train_loss,

            "train_accuracy":train_acc,

            "val_loss":val_loss,

            "val_accuracy":acc*100,

            "val_precision":precision*100,

            "val_recall":recall*100,

            "val_auc":auc

        })



        if val_loss < best_loss:


            best_loss=val_loss

            counter=0


            best_auc=auc


            best_metrics={

                "accuracy":acc,

                "precision":precision,

                "recall":recall,

                "auc":auc

            }



            torch.save(
                model.state_dict(),
                f"best_probe_{layer}.pt"
            )



        else:

            counter+=1


            if counter>=patience:

                break



    wandb.finish()


    return best_metrics



# ============================================================
# SWEEP
# ============================================================


def run_layer(layer):


    sweep_config={


        "method":"bayes",

        "metric":{

            "name":"val_loss",

            "goal":"minimize"

        },


        "parameters":{


            "learning_rate":{

                "distribution":
                "log_uniform_values",

                "min":1e-6,

                "max":1e-3

            },


            "weight_decay":{

                "distribution":"uniform",

                "min":0.0,

                "max":0.1

            },


            "beta1":{

                "values":[0.9,0.95]

            },


            "beta2":{

                "values":[0.99,0.999]

            },


            "early_stopping_patience":{

                "value":10

            }

        }

    }



    sweep_id=wandb.sweep(
        sweep_config,
        project=f"vjepa-{layer}"
    )



    results=[]



    def wrapper():

        metric=train_probe(layer)

        results.append(metric)



    wandb.agent(
        sweep_id,
        function=wrapper,
        count=20
    )



    best=max(
        results,
        key=lambda x:x["auc"]
    )


    return best



# ============================================================
# MAIN
# ============================================================


def main():


    final_results={}


    for layer in LAYERS:


        print(
            "\n\n=============================="
        )

        print(
            "TRAINING:",
            layer
        )

        print(
            "=============================="
        )


        final_results[layer]=run_layer(layer)



    with open(
        RESULT_FILE,
        "w"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=4
        )



    print("\n\n")
    print("="*70)
    print("FINAL REPRESENTATION COMPARISON")
    print("="*70)



    ranking=sorted(
        final_results.items(),
        key=lambda x:x[1]["auc"],
        reverse=True
    )


    print(
        f"{'Layer':12s}"
        f"{'Accuracy':12s}"
        f"{'Precision':12s}"
        f"{'Recall':12s}"
        f"{'AUC':10s}"
    )


    print("-"*70)



    for layer,m in ranking:


        print(
            f"{layer:12s}"
            f"{m['accuracy']:.4f}      "
            f"{m['precision']:.4f}      "
            f"{m['recall']:.4f}      "
            f"{m['auc']:.4f}"
        )



    print("\nBest Layer:")
    print(
        ranking[0][0]
    )



if __name__=="__main__":

    main()