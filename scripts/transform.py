import torch

input_path = "calib_stats_combined.pt"
output_path = "calib_stats_combined_4d.pt"

stats = torch.load(input_path, map_location="cpu")
out = {}

for head, t_data in stats.items():
    t_grid = sorted(t_data.keys())
    means = [t_data[t]["mean"].float().squeeze() for t in t_grid]
    stds = [t_data[t]["std"].float().squeeze() for t in t_grid]
    
    out[head] = {
        "mean": torch.stack(means, dim=0),  # Shape: [T, C, H, W]
        "std": torch.stack(stds, dim=0),    # Shape: [T, C, H, W]
        "t_grid": t_grid,
    }

torch.save(out, output_path)
print(f"Saved: {output_path}")