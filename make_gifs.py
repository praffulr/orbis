import os
import json
import glob
import random
from PIL import Image


FRAME_ROOT = "frames"
ANN_ROOT = "annotations"

OUT = "visualization"

os.makedirs(
    os.path.join(OUT,"normal"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUT,"anomaly"),
    exist_ok=True
)


def sample_clip(frames, start, step=2, length=16):

    idx = [
        start + i*step
        for i in range(length)
    ]

    if max(idx) >= len(frames):
        return None

    return [
        frames[i]
        for i in idx
    ]



videos = [
    os.path.basename(x).replace(".json","")
    for x in glob.glob(
        "annotations/*.json"
    )
]


normal = []
anomaly = []


for vid in videos:

    ann_file = os.path.join(
        ANN_ROOT,
        vid+".json"
    )

    frames = sorted(
        glob.glob(
            os.path.join(
                FRAME_ROOT,
                vid,
                "images",
                "*.jpg"
            )
        )
    )

    if len(frames)<32:
        continue


    with open(ann_file) as f:
        ann=json.load(f)


    a_start = ann["anomaly_start"]


    # anomaly
    clip = sample_clip(
        frames,
        a_start
    )

    if clip:
        anomaly.append(
            (vid,clip)
        )


    # normal
    if a_start>=32:

        clip = sample_clip(
            frames,
            0
        )

        if clip:
            normal.append(
                (vid,clip)
            )


print(
    "Normal:",
    len(normal)
)

print(
    "Anomaly:",
    len(anomaly)
)



def save_gif(sample, folder):

    vid,frames = sample

    imgs=[]

    for f in frames:
        img=Image.open(f)
        img=img.resize(
            (512,288)
        )
        imgs.append(img)


    path=os.path.join(
        OUT,
        folder,
        vid+".gif"
    )


    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=200,
        loop=0
    )



for s in random.sample(
    normal,
    min(20,len(normal))
):
    save_gif(
        s,
        "normal"
    )


for s in random.sample(
    anomaly,
    min(20,len(anomaly))
):
    save_gif(
        s,
        "anomaly"
    )


print("GIF generation completed")