import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm

from src.models.vision_transformer import vit_base


# ============================================================
# CONFIG
# ============================================================

FRAME_ROOT = "frames"
ANN_ROOT = "annotations"

OUT_ROOT = "vjepa_features"

NORMAL_DIR = os.path.join(OUT_ROOT, "normal")

ANOMALY_DIR = os.path.join(OUT_ROOT, "anomaly")


os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ANOMALY_DIR, exist_ok=True)


# ============================================================
# V-JEPA SETTINGS
# ============================================================

NUM_FRAMES = 5

FRAME_STRIDE = 2

IMG_SIZE = 384


RAW_WINDOW = (NUM_FRAMES * FRAME_STRIDE) - 1


print("=" * 60)
print("V-JEPA 2.1 FEATURE EXTRACTION")
print("=" * 60)

print("Context frames:", NUM_FRAMES)

print("Frame stride:", FRAME_STRIDE)

print("Effective window:", RAW_WINDOW)

print("=" * 60)


# ============================================================
# DEVICE
# ============================================================


if torch.cuda.is_available():

    DEVICE = "cuda"


elif torch.backends.mps.is_available():

    DEVICE = "mps"

    torch.set_float32_matmul_precision("high")


else:

    DEVICE = "cpu"


print("Device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================


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

    msg = model.load_state_dict(state, strict=False)

    print(msg)

    model.eval()

    model.to(DEVICE)

    print("Model loaded\n")

    return model


# ============================================================
# ACTIVATION CACHE
# ============================================================


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


# ============================================================
# FRAME UTILITIES
# ============================================================


def get_frames(video_id):

    path = os.path.join(FRAME_ROOT, video_id, "images", "*.jpg")

    return sorted(glob.glob(path))


def make_clip(frames, indices):

    return [frames[i] for i in indices]


# ============================================================
# SAMPLING
# ============================================================


def sample_indices(start, total_frames):

    indices = [start + i * FRAME_STRIDE for i in range(NUM_FRAMES)]

    if indices[-1] >= total_frames:

        return None

    return indices


# ============================================================
# IMAGE PREPROCESS
# ============================================================


def frames_to_tensor(paths):

    import cv2

    imgs = []

    for p in paths:

        img = cv2.imread(p)

        if img is None:

            raise RuntimeError(f"Cannot read {p}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        imgs.append(img)

    x = np.stack(imgs)

    x = torch.from_numpy(x).float()

    x /= 255.0

    # T,H,W,C
    # ->
    # C,T,H,W

    x = x.permute(3, 0, 1, 2)

    return x


# ============================================================
# FEATURE EXTRACTION
# ============================================================


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


# ============================================================
# PROCESS VIDEO
# ============================================================


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


# ============================================================
# MAIN
# ============================================================


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