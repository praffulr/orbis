import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T


from src.models.vision_transformer import vit_base


# CONFIG

FRAME_ROOT = "frames"
ANN_ROOT = "annotations"

OUT_ROOT = "vjepa_features_tb5"

NORMAL_DIR = os.path.join(OUT_ROOT, "normal")

ANOMALY_DIR = os.path.join(OUT_ROOT, "anomaly")



os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ANOMALY_DIR, exist_ok=True)


# V-JEPA SETTINGS

NUM_FRAMES = 5

FRAME_STRIDE = 2

IMG_SIZE = 384

# Proper preprocessing: Resize shortest edge, CenterCrop, and ImageNet Normalize
preprocess = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


RAW_WINDOW = (NUM_FRAMES * FRAME_STRIDE) - 1


print("=" * 60)
print("V-JEPA 2.1 FEATURE EXTRACTION")
print("=" * 60)

print("Context frames:", NUM_FRAMES)

print("Frame stride:", FRAME_STRIDE)

print("Effective window:", RAW_WINDOW)

print("=" * 60)


# DEVICE


if torch.cuda.is_available():

    DEVICE = "cuda"


elif torch.backends.mps.is_available():

    DEVICE = "mps"

    torch.set_float32_matmul_precision("high")


else:

    DEVICE = "cpu"


print("Device:", DEVICE)


# LOAD MODEL


def load_model():

    print("\nLoading V-JEPA...\n")

    model = vit_base(
        img_size=(IMG_SIZE, IMG_SIZE), num_frames=NUM_FRAMES, use_rope=True
    )

    ckpt = torch.load(
        "vjepa_ckpt/vjepa2_1_vitb_dist_vitG_384.pt",
        map_location="cpu",
        weights_only=True,
    )

    state = ckpt["encoder"]

    state = {
        k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()
    }


    # =====================================================
    # ADAPT TUBELET SIZE
    # =====================================================

    old_weight = state["patch_embed.proj.weight"]

    print(
        "Original patch embedding:",
        old_weight.shape
    )


    TARGET_TUBELET = 5


    if old_weight.shape[2] != TARGET_TUBELET:

        print("Interpolating temporal kernel")

        out_c, in_c, old_t, h, w = old_weight.shape


        # Merge spatial dimensions
        # [768,3,2,16,16]
        # ->
        # [768*3*16*16, 2]

        weight = old_weight.permute(
            0,1,3,4,2
        ).reshape(
            -1,
            old_t
        )


        # interpolate temporal dimension

        weight = torch.nn.functional.interpolate(
            weight.unsqueeze(1),
            size=TARGET_TUBELET,
            mode="linear",
            align_corners=False
        )


        # [N,1,5]
        # ->
        # [768,3,16,16,5]

        weight = weight.squeeze(1).reshape(
            out_c,
            in_c,
            h,
            w,
            TARGET_TUBELET
        )


        # ->
        # [768,3,5,16,16]

        new_weight = weight.permute(
            0,
            1,
            4,
            2,
            3
        )


        state["patch_embed.proj.weight"] = new_weight


    print(
        "New patch embedding:",
        state["patch_embed.proj.weight"].shape
    )

    msg = model.load_state_dict(state, strict=False)

    print(msg)

    model.eval()

    model.to(DEVICE)

    print("Model loaded\n")

    return model


# ACTIVATION CACHE


class ActivationCache:
    def __init__(self, model):

        self.outputs = {}

        self.handles = []

        # last block + intermediate
        for idx in [3, 6, 9, 12]:

            handle = model.blocks[idx - 1].register_forward_hook(self.save_output(idx))

            self.handles.append(handle)

    def save_output(self, idx):
        def hook(module, input, output):

            self.outputs[f"block_{idx}"] = output.detach()

        return hook

    def clear(self):

        self.outputs = {}

    def remove(self):

        for h in self.handles:

            h.remove()


# FRAME UTILITIES


def get_frames(video_id):

    path = os.path.join(FRAME_ROOT, video_id, "images", "*.jpg")

    return sorted(glob.glob(path))


def make_clip(frames, indices):

    return [frames[i] for i in indices]


# SAMPLING


def sample_indices(start, total_frames):

    indices = [start + i * FRAME_STRIDE for i in range(NUM_FRAMES)]

    if indices[-1] >= total_frames:

        return None

    return indices


# IMAGE PREPROCESS


def frames_to_tensor(paths):


    imgs = []

    for p in paths:

        img = Image.open(p).convert("RGB")
        img = preprocess(img)

        imgs.append(img)

    # Stack into [T, C, H, W] tensor
    x = torch.stack(imgs)

    # T,C,H,W
    # ->
    # C,T,H,W

    x = x.permute(1, 0, 2, 3)

    return x


# FEATURE EXTRACTION


@torch.no_grad()
def extract_feature(model, cache, clip):

    cache.clear()

    x = frames_to_tensor(clip)

    print("Input:", x.shape)

    # [C,T,H,W]
    x = x.unsqueeze(0)

    # [B,C,T,H,W]

    x = x.to(DEVICE)

    output = model(x)

    print("Model output:", output.shape)

    result = {}

    # =====================================================
    # FINAL LAYER TOKENS
    # =====================================================

    result["final"] = output.squeeze(0).cpu()

    print("Final:", result["final"].shape)

    # =====================================================
    # BLOCK ACTIVATIONS
    # =====================================================

    for name, val in cache.outputs.items():

        result[name] = val.squeeze(0).cpu()

        print(name, result[name].shape)

    return result


# PROCESS VIDEO


def process_video(model, cache, vid):

    ann_file = os.path.join(ANN_ROOT, vid + ".json")

    if not os.path.exists(ann_file):

        return

    with open(ann_file) as f:

        ann = json.load(f)

    if str(ann.get("ignore", "false")).lower() != "false":

        return

    anomaly_start = ann.get("anomaly_start", -1)

    anomaly_end = ann.get("anomaly_end", -1)

    if anomaly_start < 0 or anomaly_end < 0:

        return

    frames = get_frames(vid)

    total = len(frames)

    if total < ((NUM_FRAMES - 1) * FRAME_STRIDE + 1):

        return

    print("\nProcessing:", vid)

    print("Frames:", total)

    print("Anomaly:", anomaly_start, anomaly_end)

    # =====================================================
    # OOD SAMPLE
    # =====================================================

    idx = sample_indices(anomaly_start, total)

    if idx is not None:

        save = os.path.join(ANOMALY_DIR, vid + ".pt")

        if not os.path.exists(save):

            feat = extract_feature(model, cache, make_clip(frames, idx))

            torch.save(
                {"feature": feat, "label": 1, "indices": idx, "video": vid}, save
            )

    # =====================================================
    # ID SAMPLE
    # =====================================================

    normal_idx = None

    # before anomaly

    idx = sample_indices(0, total)

    if idx is not None and idx[-1] < anomaly_start:

        normal_idx = idx

    # after anomaly

    if normal_idx is None:

        idx = sample_indices(anomaly_end + 1, total)

        if idx is not None:

            normal_idx = idx

    if normal_idx is None:

        return

    save = os.path.join(NORMAL_DIR, vid + ".pt")

    if not os.path.exists(save):

        feat = extract_feature(model, cache, make_clip(frames, normal_idx))

        torch.save(
            {"feature": feat, "label": 0, "indices": normal_idx, "video": vid}, save
        )


# MAIN


def main():

    model = load_model()

    cache = ActivationCache(model)

    videos = [
        os.path.basename(x).replace(".json", "")
        for x in glob.glob(ANN_ROOT + "/*.json")
    ]

    print("Videos:", len(videos))

    for vid in tqdm(videos):

        try:

            process_video(model, cache, vid)

        except Exception as e:

            print("FAILED:", vid)

            print(e)

    cache.remove()

    print("\nDONE")

    print("Normal:", len(glob.glob(NORMAL_DIR + "/*.pt")))

    print("Anomaly:", len(glob.glob(ANOMALY_DIR + "/*.pt")))


if __name__ == "__main__":

    main()