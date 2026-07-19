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

# ------------------------------------------------------------
# V-JEPA SETTINGS
# ------------------------------------------------------------

NUM_FRAMES = 5  # context frames
FRAME_STRIDE = 2  # DoTA 10 FPS -> sample every 2 frames
IMG_SIZE = 384

RAW_WINDOW = (NUM_FRAMES * FRAME_STRIDE) - 1

print("=" * 60)
print("V-JEPA FEATURE EXTRACTION")
print("=" * 60)
print(f"Context Frames : {NUM_FRAMES}")
print(f"Frame Stride   : {FRAME_STRIDE}")
print(f"Raw Window     : {RAW_WINDOW} frames")
print("=" * 60)

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Device:", DEVICE)

if DEVICE == "mps":
    torch.set_float32_matmul_precision("high")

# ============================================================
# MODEL
# ============================================================


def load_model():

    print("\nLoading V-JEPA 2.1...\n")

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

    print("Model Loaded.\n")

    return model


# ============================================================
# ACTIVATION CACHE
# ============================================================


class ActivationCache:
    def __init__(self, model):

        self.outputs = {}
        self.handles = []

        for idx in [3, 6, 9, 12]:

            h = model.blocks[idx - 1].register_forward_hook(self.save_output(idx))

            self.handles.append(h)

        
    
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
# ANOMALY CLIP
# ============================================================


def anomaly_clip(frames, anomaly_start):

    idx = sample_indices(anomaly_start, len(frames))

    if idx is None:
        return None, None

    return make_clip(frames, idx), idx


# ============================================================
# NORMAL CLIP
# Same logic as DoTA loader
# ============================================================


def normal_clip(frames, anomaly_start, anomaly_end):

    total = len(frames)

    # --------------------------------------------------------
    # Case 1
    # Beginning of video
    # --------------------------------------------------------

    if anomaly_start >= RAW_WINDOW:

        idx = sample_indices(0, total)

        if idx is not None:

            print("Normal clip : Beginning")

            return make_clip(frames, idx), idx

    # --------------------------------------------------------
    # Case 2
    # End of video
    # --------------------------------------------------------

    elif (total - RAW_WINDOW) > anomaly_end:

        start = total - RAW_WINDOW

        idx = sample_indices(start, total)

        if idx is not None:

            print("Normal clip : End")

            return make_clip(frames, idx), idx

    print("No valid normal clip.")

    return None, None


# ============================================================
# IMAGE PREPROCESS
# ============================================================


def frames_to_tensor(paths):

    import cv2

    imgs = []

    for p in paths:

        img = cv2.imread(p)

        if img is None:
            raise ValueError(f"Could not read image: {p}")

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

    print("Input tensor:", x.shape)

    # B,C,T,H,W

    x = x.unsqueeze(0)

    x = x.to(DEVICE, non_blocking=True)

    print("Model input:", x.shape)

    output = model(x)

    print("Model output:", output.shape)

    result = {}

    # =====================================================
    # FINAL REPRESENTATION
    # =====================================================

    if output.ndim == 3:

        output = output.mean(dim=1)

    result["final"] = output.squeeze(0).cpu()

    print("Final embedding:", result["final"].shape)

    # =====================================================
    # INTERMEDIATE ACTIVATIONS
    # =====================================================

    for name, val in cache.outputs.items():

        result[name] = val.squeeze(0).cpu()

        print(name, result[name].shape)

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

    # =====================================================
    # VALIDATION
    # =====================================================

    if str(ann.get("ignore", "false")).lower() != "false":
        return

    anomaly_start = ann.get("anomaly_start", -1)
    anomaly_end = ann.get("anomaly_end", -1)
    num_frames = ann.get("num_frames", -1)

    if anomaly_start < 0 or anomaly_end < 0 or num_frames < 0:
        return

    # =====================================================
    # LOAD FRAMES
    # =====================================================

    frames = get_frames(vid)

    # -----------------------------------------------------
    # Skip videos whose frames were never extracted
    # (expected because only a subset of DoTA is used)
    # -----------------------------------------------------

    if len(frames) == 0:
        print(f"Skipping {vid}: frame folder not available.")
        return

    total_frames = len(frames)

    # -----------------------------------------------------
    # Warn only if frame count differs
    # -----------------------------------------------------

    if total_frames != num_frames:
        print(
            f"WARNING: Frame count mismatch for {vid} "
            f"(disk={total_frames}, annotation={num_frames})"
        )

    # =====================================================
    # Minimum frames required
    # =====================================================

    minimum_required = (NUM_FRAMES - 1) * FRAME_STRIDE + 1

    if total_frames < minimum_required:
        print(
            f"Skipping {vid}: "
            f"only {total_frames} frames "
            f"(minimum {minimum_required})"
        )
        return

    print("\n================================================")
    print("Processing :", vid)
    print("Frames     :", total_frames)
    print("Anomaly    :", anomaly_start, "-", anomaly_end)
    print("================================================")

    # =====================================================
    # ANOMALY CLIP
    # =====================================================

    anomaly_idx = sample_indices(anomaly_start, total_frames)

    if anomaly_idx is None:
        print("Cannot sample anomaly clip.")
    else:

        print("Anomaly indices:", anomaly_idx)

        anomaly_clip = make_clip(frames, anomaly_idx)

        save = os.path.join(ANOMALY_DIR, vid + ".pt")

        if not os.path.exists(save):

            feat = extract_feature(model, cache, anomaly_clip)

            torch.save(
                {
                    "feature": feat,
                    "label": 1,
                    "indices": anomaly_idx,
                    "video": vid,
                },
                save,
            )

            print("Saved anomaly feature.")

    # =====================================================
    # NORMAL CLIP
    # =====================================================

    normal_idx = None

    # -----------------------------------------------------
    # Case 1
    # Beginning of video
    # -----------------------------------------------------

    idx = sample_indices(0, total_frames)

    if idx is not None and idx[-1] < anomaly_start:
        normal_idx = idx
        print("Normal clip from beginning.")

    # -----------------------------------------------------
    # Case 2
    # After anomaly
    # -----------------------------------------------------

    if normal_idx is None:

        idx = sample_indices(anomaly_end + 1, total_frames)

        if idx is not None:
            normal_idx = idx
            print("Normal clip after anomaly.")

    # -----------------------------------------------------
    # Case 3
    # Before anomaly
    # -----------------------------------------------------

    if normal_idx is None:

        start = anomaly_start - (NUM_FRAMES - 1) * FRAME_STRIDE

        if start >= 0:

            idx = sample_indices(start, total_frames)

            if idx is not None and idx[-1] < anomaly_start:
                normal_idx = idx
                print("Normal clip immediately before anomaly.")

    if normal_idx is None:
        print("No valid normal clip found.")
        return

    print("Normal indices:", normal_idx)

    normal_clip = make_clip(frames, normal_idx)

    save = os.path.join(NORMAL_DIR, vid + ".pt")

    if not os.path.exists(save):

        feat = extract_feature(model, cache, normal_clip)

        torch.save(
            {
                "feature": feat,
                "label": 0,
                "indices": normal_idx,
                "video": vid,
            },
            save,
        )

        print("Saved normal feature.")


# ============================================================
# IMAGE PREPROCESS
# ============================================================


def frames_to_tensor(paths):

    import cv2

    imgs = []

    for p in paths:

        img = cv2.imread(p)

        if img is None:
            raise RuntimeError(f"Cannot read image: {p}")

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

    print("Input clip tensor:", x.shape)
    # expected:
    # [3,5,384,384]

    # B,C,T,H,W

    x = x.unsqueeze(0)

    print("Model input:", x.shape)
    # expected:
    # [1,3,5,384,384]

    x = x.to(DEVICE, non_blocking=True)

    output = model(x)

    result = {}

    # =====================================================
    # FINAL REPRESENTATION
    # =====================================================

    print("Model output:", output.shape)

    if output.ndim == 3:

        output = output.mean(dim=1)

    result["final"] = output.squeeze(0).cpu()

    print("Final embedding:", result["final"].shape)

    # =====================================================
    # INTERMEDIATE BLOCK FEATURES
    # =====================================================

    for name, val in cache.outputs.items():

        print(name, "activation:", val.shape)

        result[name] = val.squeeze(0).cpu()

    return result


# ============================================================
# PROCESS SINGLE VIDEO
# ============================================================


def process_video(model, cache, vid):

    ann_path = os.path.join(ANN_ROOT, vid + ".json")

    if not os.path.exists(ann_path):

        return

    with open(ann_path) as f:

        ann = json.load(f)

    print("\n==============================")
    print("Processing:", vid)
    print("==============================")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if str(ann.get("ignore", "false")).lower() != "false":

        print("Ignored video")

        return

    anomaly_start = ann.get("anomaly_start", -1)

    anomaly_end = ann.get("anomaly_end", -1)

    num_frames = ann.get("num_frames", -1)

    if anomaly_start < 0 or anomaly_end < 0:

        print("Invalid anomaly range")

        return

    frames = get_frames(vid)

    if len(frames) != num_frames:

        print("WARNING frame mismatch:", len(frames), num_frames)

    total = len(frames)

    minimum_required = (NUM_FRAMES - 1) * FRAME_STRIDE + 1

    if total < minimum_required:

        print("Not enough frames")

        return

    print("Total frames:", total)

    print("Anomaly:", anomaly_start, "-", anomaly_end)

    # =====================================================
    # ANOMALY SAMPLE
    # =====================================================

    anomaly_clip_paths, anomaly_indices = anomaly_clip(frames, anomaly_start)

    if anomaly_clip_paths:

        print("Anomaly indices:", anomaly_indices)

        save = os.path.join(ANOMALY_DIR, vid + ".pt")

        if not os.path.exists(save):

            feat = extract_feature(model, cache, anomaly_clip_paths)

            torch.save(
                {"feature": feat, "label": 1, "indices": anomaly_indices, "video": vid},
                save,
            )

            print("Saved anomaly feature")

    else:

        print("No valid anomaly clip")

    # =====================================================
    # NORMAL SAMPLE
    # =====================================================

    normal_clip_paths, normal_indices = normal_clip(frames, anomaly_start, anomaly_end)

    if normal_clip_paths:

        print("Normal indices:", normal_indices)

        save = os.path.join(NORMAL_DIR, vid + ".pt")

        if not os.path.exists(save):

            feat = extract_feature(model, cache, normal_clip_paths)

            torch.save(
                {"feature": feat, "label": 0, "indices": normal_indices, "video": vid},
                save,
            )

            print("Saved normal feature")

    else:

        print("No valid normal clip")


# ============================================================
# MAIN
# ============================================================


def main():

    model = load_model()

    # for name, module in model.named_modules(): 
    #         if name: # Skip the empty string which represents the root model
    #             print(f"Component: {name:<30} | Type: {type(module).__name__}") 
    #             print("\n" + "="*60 + "\n")

    cache = ActivationCache(model)

    videos = [
        os.path.basename(x).replace(".json", "")
        for x in glob.glob(ANN_ROOT + "/*.json")
    ]

    print("Total videos:", len(videos))

    success = 0

    for vid in tqdm(videos):

        try:

            process_video(model, cache, vid)

            success += 1

        except Exception as e:

            print("FAILED:", vid)

            print(e)

    cache.remove()

    print("\n==============================")
    print("FEATURE EXTRACTION COMPLETE")
    print("==============================")

    print("Processed:", success)

    print("Normal samples:", len(glob.glob(NORMAL_DIR + "/*.pt")))

    print("Anomaly samples:", len(glob.glob(ANOMALY_DIR + "/*.pt")))

    

if __name__ == "__main__":

    main()
