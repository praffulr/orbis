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


NUM_FRAMES = 16

FRAME_STRIDE = 2
# 10 FPS -> 5 FPS


IMG_SIZE = 384


# ------------------------------
# DEVICE
# ------------------------------


if torch.cuda.is_available():

    DEVICE = "cuda"


elif torch.backends.mps.is_available():

    DEVICE = "mps"

else:

    DEVICE = "cpu"


if DEVICE == "mps":

    torch.set_float32_matmul_precision("high")


# ============================================================
# LOAD MODEL
# ============================================================


def load_model():

    print("Loading V-JEPA2")

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

    return model


# ============================================================
# ACTIVATION CACHE
# ============================================================


class ActivationCache:
    def __init__(self, model):

        self.outputs = {}

        self.handles = []

        blocks = [3, 6, 9, 12]

        for idx in blocks:

            handle = model.blocks[idx - 1].register_forward_hook(self.save_output(idx))

            self.handles.append(handle)

    def save_output(self, idx):
        def hook(module, inp, out):

            self.outputs[f"block_{idx}"] = out.detach()

        return hook

    def clear(self):

        self.outputs = {}

    def remove(self):

        for h in self.handles:

            h.remove()


# ============================================================
# FRAME LOADING
# ============================================================


def get_frames(video_id):

    path = os.path.join(FRAME_ROOT, video_id, "images", "*.jpg")

    return sorted(glob.glob(path))

# ============================================================
# V-JEPA TEMPORAL SAMPLING
# ============================================================


def sample_indices(start, total_frames):

    """
    DoTA:
        10 FPS

    V-JEPA:
        5 FPS input

    Therefore:
        stride = 2

    Example:

    raw frames:
    40 41 42 43 ... 70

    sampled:
    40 42 44 ... 70

    Total:
    16 frames
    """

    indices = [start + i * FRAME_STRIDE for i in range(NUM_FRAMES)]

    # validation:
    # last sampled frame must exist

    if indices[-1] >= total_frames:
        return None

    return indices


def make_clip(frames, idx):

    return [frames[i] for i in idx]


# ============================================================
# ORBIS SAMPLING
# ============================================================


def anomaly_clip(frames, start):

    idx = sample_indices(start, len(frames))

    if idx is None:

        return None, None

    return make_clip(frames, idx), idx


def normal_clip(frames, anomaly_start, anomaly_end):

    total = len(frames)

    # =====================================================
    # CASE 1:
    # take beginning of video
    # =====================================================

    idx = sample_indices(0, total)

    if idx is not None:

        if idx[-1] < anomaly_start:

            return (make_clip(frames, idx), idx)

    # =====================================================
    # CASE 2:
    # take frames after anomaly
    # =====================================================

    start = anomaly_end + 1

    idx = sample_indices(start, total)

    if idx is not None:

        return (make_clip(frames, idx), idx)

    # =====================================================
    # CASE 3:
    # take frames before anomaly
    # =====================================================

    start = anomaly_start - (NUM_FRAMES - 1) * FRAME_STRIDE

    if start >= 0:

        idx = sample_indices(start, total)

        if idx is not None:

            if idx[-1] < anomaly_start:

                return (make_clip(frames, idx), idx)

    return None, None


# ============================================================
# IMAGE PREPROCESS
# ============================================================


def frames_to_tensor(paths):

    import cv2

    imgs = []

    for p in paths:

        img = cv2.imread(p)

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

    # B,C,T,H,W

    x = x.unsqueeze(0)

    x = x.to(DEVICE, non_blocking=True)

    output = model(x)

    result = {}

    # =================================================
    # FINAL EMBEDDING
    # Used for linear probing
    # =================================================

    if output.ndim == 3:

        output = output.mean(dim=1)

    result["final"] = output.squeeze(0).cpu()

    # =================================================
    # INTERMEDIATE ACTIVATIONS
    # Used for heatmaps
    # =================================================

    for name, val in cache.outputs.items():

        # KEEP TOKENS
        #
        # before:
        # [1,576,768]
        #
        # after:
        # [576,768]

        result[name] = val.squeeze(0).cpu()

    return result


# ============================================================
# PROCESS VIDEO
# ============================================================


def process_video(model, cache, vid):

    ann_path = os.path.join(ANN_ROOT, vid + ".json")

    if not os.path.exists(ann_path):

        return

    with open(ann_path) as f:

        ann = json.load(f)

    if ann.get("ignore", False):

        return

    # ============================================================
    # ANNOTATION VALIDATION
    # ============================================================

    if str(ann.get("ignore", "false")).lower() != "false":

        return

    start = ann.get("anomaly_start", -1)

    end = ann.get("anomaly_end", -1)

    num_frames = ann.get("num_frames", -1)

    if start < 0:

        return

    if end < 0:

        return

    if num_frames < 0:

        return

    frames = get_frames(vid)

    # verify annotation matches actual frames

    if len(frames) != num_frames:

        print("Frame mismatch:", vid, len(frames), num_frames)

    total_frames = len(frames)

    # minimum frames required:
    #
    # 16 frames
    # stride 2
    #
    # span:
    # 0,2,4,...30
    #
    # requires 31 raw frames

    minimum_required = (NUM_FRAMES - 1) * FRAME_STRIDE + 1

    if total_frames < minimum_required:

        return

    # anomaly

    clip, idx = anomaly_clip(frames, start)

    if clip:

        save = os.path.join(ANOMALY_DIR, vid + ".pt")

        if not os.path.exists(save):

            feat = extract_feature(model, cache, clip)

            torch.save(
                {"feature": feat, "label": 1, "indices": idx, "video": vid}, save
            )

    # normal

    clip, idx = normal_clip(frames, start, end)

    if clip:

        save = os.path.join(NORMAL_DIR, vid + ".pt")

        if not os.path.exists(save):

            feat = extract_feature(model, cache, clip)

            torch.save(
                {"feature": feat, "label": 0, "indices": idx, "video": vid}, save
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

            print("FAILED", vid, e)

    cache.remove()

    print("\n==============================")

    print("FEATURE EXTRACTION COMPLETE")

    print("Normal:", len(glob.glob(NORMAL_DIR + "/*.pt")))

    print("Anomaly:", len(glob.glob(ANOMALY_DIR + "/*.pt")))


if __name__ == "__main__":

    main()