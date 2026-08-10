import argparse
from pathlib import Path

import torch
from tqdm import tqdm


def normalize_sample_scores(
    raw_scores_path="sample_scores.pt",
    calib_stats_path="calib_stats_combined_4d.pt",
    output_path="normalized.pt",
    eps=1e-8,
):
    raw_path = Path(raw_scores_path)
    stats_path = Path(calib_stats_path)
    out_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw scores file not found at: {raw_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Calibration stats file not found at: {stats_path}")

    print(f"Loading raw sample scores from: {raw_path}")
    raw_samples = torch.load(raw_path, map_location="cpu")
    print(f"Loaded {len(raw_samples)} clip entries.")

    print(f"Loading 4D calibration stats from: {stats_path}")
    calib_stats = torch.load(stats_path, map_location="cpu")

    # Load 'combined' stats directly
    if "combined" in calib_stats:
        mean = calib_stats["combined"]["mean"].float()  # [T, 32, H, W]
        std = calib_stats["combined"]["std"].float()    # [T, 32, H, W]
    else:
        mean = calib_stats["mean"].float()
        std = calib_stats["std"].float()

    normalized_samples = []

    print("\nNormalizing 'combined' head maps across noise levels [T, C, H, W]...")
    for item in tqdm(raw_samples, desc="Normalizing samples", total=len(raw_samples), unit="sample"):
        norm_item = {k: v for k, v in item.items() if k != "head_maps"}

        # Extract only the combined head map
        raw_hm = item["head_maps"]["combined"]
        if not isinstance(raw_hm, torch.Tensor):
            raw_hm = torch.tensor(raw_hm)

        raw_hm_f32 = raw_hm.float()  # Shape: [T, 32, H, W]

        assert raw_hm_f32.shape == mean.shape, (
            f"Shape mismatch: raw shape {raw_hm_f32.shape} vs calib stat shape {mean.shape}"
        )

        # Z-Score Normalization
        norm_hm = (raw_hm_f32 - mean) / (std + eps)

        # Store only the normalized combined head map in float16
        norm_item["head_maps"] = {"combined": norm_hm.half()}
        normalized_samples.append(norm_item)

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalized_samples, out_path)
    print(f"\nSuccessfully saved {len(normalized_samples)} normalized samples to: '{out_path}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize retro-surprise 'combined' sample scores across 4D calibration stats."
    )
    parser.add_argument("--raw_scores_path", type=str, default="sample_scores.pt")
    parser.add_argument("--calib_stats_path", type=str, default="calib_stats_combined_4d.pt")
    parser.add_argument("--output_path", type=str, default="normalized.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize_sample_scores(
        raw_scores_path=args.raw_scores_path,
        calib_stats_path=args.calib_stats_path,
        output_path=args.output_path,
    )