import os
import cv2
import glob
import json

import torch
import numpy as np

import matplotlib.pyplot as plt

from src.models.vision_transformer import vit_base


############################################################

FRAME_ROOT="frames"

ANN_ROOT="annotations"

VIDEO="bFGmOp9H3MA_000355"

BLOCK="block_6"

IMG_SIZE=384

NUM_FRAMES=16

FRAME_STRIDE=2

############################################################

device="mps" if torch.backends.mps.is_available() else "cpu"


############################################################

model=vit_base(
    img_size=(384,384),
    num_frames=16,
    use_rope=True
)

ckpt=torch.load(
    "vjepa_ckpt/vjepa2_1_vitb_dist_vitG_384.pt",
    map_location="cpu",
    weights_only=True
)

state=ckpt["encoder"]

state={
    k.replace("module.","").replace("backbone.",""):v
    for k,v in state.items()
}

model.load_state_dict(state,strict=False)

model.eval()

model.to(device)

############################################################
# activation hook
############################################################

activations={}

layer_id=int(BLOCK.split("_")[1])

def hook(module,input,output):

    activations["x"]=output.detach()

handle=model.blocks[layer_id-1].register_forward_hook(hook)

############################################################

ann=json.load(
    open(
        os.path.join(
            ANN_ROOT,
            VIDEO+".json"
        )
    )
)

start=ann["anomaly_start"]

############################################################

frames=sorted(
    glob.glob(
        os.path.join(
            FRAME_ROOT,
            VIDEO,
            "images",
            "*.jpg"
        )
    )
)

idx=[
    start+i*FRAME_STRIDE
    for i in range(NUM_FRAMES)
]

clip=[]

for i in idx:

    img=cv2.imread(frames[i])

    img=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img=cv2.resize(
        img,
        (384,384)
    )

    clip.append(img)

x=np.stack(clip)

tensor=torch.from_numpy(x).float()/255.

tensor=tensor.permute(3,0,1,2)

tensor=tensor.unsqueeze(0).to(device)

############################################################

with torch.no_grad():

    model(tensor)

act=activations["x"][0]

############################################################
# Token -> activation score
############################################################

score=torch.norm(
    act,
    dim=1
)

score=score.cpu().numpy()

############################################################
# reshape
############################################################

score=score.reshape(
    16,
    24,
    24
)

############################################################

os.makedirs("activation_maps",exist_ok=True)

############################################################

for t in range(16):

    heat=score[t]

    heat=(heat-heat.min())/(heat.max()-heat.min()+1e-8)

    heat=cv2.resize(
        heat,
        (384,384),
        interpolation=cv2.INTER_CUBIC
    )

    frame=clip[t].astype(np.float32)/255.

    plt.figure(figsize=(6,6))

    plt.imshow(frame)

    plt.imshow(
        heat,
        alpha=0.55,
        cmap="jet"
    )

    plt.axis("off")

    plt.title(
        f"{BLOCK}  Frame {t}"
    )

    plt.savefig(
        f"activation_maps/frame_{t:02d}.png",
        bbox_inches="tight"
    )

    plt.close()

handle.remove()

print("Done.")