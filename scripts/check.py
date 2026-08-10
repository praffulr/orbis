from pathlib import Path
import torch


def inspect_normalized_pt(file_path="normalized.pt"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading '{path}'...")
    data = torch.load(path, map_location="cpu")

    # Check data type and total count
    print(f"\n--- General Info ---")
    print(f"Data type: {type(data)}")
    
    if isinstance(data, list):
        print(f"Total samples: {len(data)}")
        if len(data) == 0:
            print("Dataset is empty.")
            return
        
        sample = data[0]
        print(f"\n--- First Sample Keys ---")
        for k, v in sample.items():
            if k != "head_maps":
                print(f"  {k}: {v} (Type: {type(v).__name__})")

        print(f"\n--- Head Maps (Tensors) ---")
        head_maps = sample.get("head_maps", {})
        if isinstance(head_maps, dict):
            for head_name, tensor in head_maps.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  Head '{head_name}':")
                    print(f"    - Shape: {list(tensor.shape)} [T, C, H, W]")
                    print(f"    - Dtype: {tensor.dtype}")
                    print(f"    - Device: {tensor.device}")
                    print(f"    - Min value: {tensor.min().item():.4f}")
                    print(f"    - Max value: {tensor.max().item():.4f}")
                    print(f"    - Mean value: {tensor.mean().item():.4f}")
                    print(f"    - Std value: {tensor.std().item():.4f}")
                else:
                    print(f"  Head '{head_name}': Not a Tensor (Type: {type(tensor).__name__})")
        else:
            print(f"  'head_maps' is not a dict (Type: {type(head_maps).__name__})")

    elif isinstance(data, dict):
        print(f"Root keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor shape = {list(v.shape)}, dtype = {v.dtype}")
            elif isinstance(v, dict):
                print(f"  {k}: Dictionary with keys = {list(v.keys())}")


if __name__ == "__main__":
    inspect_normalized_pt()