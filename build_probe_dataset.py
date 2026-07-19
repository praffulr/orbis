import os
import glob
import torch
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

FEATURE_ROOT = "vjepa_features"

NORMAL_DIR = os.path.join(
    FEATURE_ROOT,
    "normal"
)

ANOMALY_DIR = os.path.join(
    FEATURE_ROOT,
    "anomaly"
)


LAYERS = [
    "final",
    "block_3",
    "block_6",
    "block_9",
    "block_12"
]


OUTPUT_DIR = "cached_features"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TEST_SIZE = 0.20
SEED = 42



# ============================================================
# LOAD FEATURES
# ============================================================


def load_features(folder, layer):

    features=[]
    labels=[]


    files=sorted(
        glob.glob(
            os.path.join(folder,"*.pt")
        )
    )


    print("\nLoading:",folder)
    print("Files:",len(files))


    label=0 if "normal" in folder else 1


    valid=0
    shapes=set()


    for f in files:


        sample=torch.load(
            f,
            map_location="cpu",
            weights_only=False
        )


        feat=sample["feature"]


        if layer not in feat:
            continue


        x=feat[layer]


        # -------------------------------
        # FINAL EMBEDDING
        # -------------------------------

        if layer=="final":

            x=x.squeeze()


        # -------------------------------
        # BLOCK ACTIVATIONS
        #
        # keep unpooled tokens
        #
        # (1152,768)
        # -------------------------------

        else:

            if x.ndim==3:
                x=x.squeeze(0)


            if x.ndim!=2:
                print(
                    "Invalid block shape:",
                    layer,
                    x.shape,
                    f
                )
                continue


        shapes.add(
            tuple(x.shape)
        )


        features.append(
            x.float()
        )

        labels.append(
            label
        )


        valid+=1



    print("Valid:",valid)

    print("Shapes:",shapes)


    if len(shapes)>1:

        raise RuntimeError(
            f"Inconsistent shapes for {layer}: {shapes}"
        )


    return features,labels




# ============================================================
# BUILD DATASET
# ============================================================


def build_layer_dataset(layer):


    print("\n================================")
    print("Building:",layer)
    print("================================")


    normal_x,normal_y=load_features(
        NORMAL_DIR,
        layer
    )


    anomaly_x,anomaly_y=load_features(
        ANOMALY_DIR,
        layer
    )


    X=normal_x+anomaly_x
    y=normal_y+anomaly_y



    X=torch.stack(
        X
    )


    y=torch.tensor(
        y,
        dtype=torch.long
    )


    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Labels:",
        y.shape
    )



    indices=np.arange(
        len(y)
    )


    train_idx,val_idx=train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y.numpy()
    )


    train_features=X[train_idx]
    val_features=X[val_idx]

    train_labels=y[train_idx]
    val_labels=y[val_idx]



    print(
        "Train:",
        train_features.shape
    )

    print(
        "Val:",
        val_features.shape
    )



    train_file=os.path.join(
        OUTPUT_DIR,
        f"train_{layer}.pt"
    )


    val_file=os.path.join(
        OUTPUT_DIR,
        f"val_{layer}.pt"
    )



    torch.save(
        {
            "features":train_features,
            "labels":train_labels
        },
        train_file
    )


    torch.save(
        {
            "features":val_features,
            "labels":val_labels
        },
        val_file
    )



    print("Saved:")
    print(train_file)
    print(val_file)





# ============================================================
# MAIN
# ============================================================


def main():

    for layer in LAYERS:

        build_layer_dataset(layer)


    print("\n==============================")
    print("PROBE DATASET COMPLETE")
    print("==============================")



if __name__=="__main__":

    main()
