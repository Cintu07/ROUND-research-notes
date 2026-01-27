
import torch
import sys
import os

def inspect_pt(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        # weights_only=False is required to load dictionary containers with numpy arrays
        data = torch.load(file_path, weights_only=False)
        print(f"--- Data Inspection: {os.path.basename(file_path)} ---")
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"Key: '{key}' | Type: Tensor/Array | Shape: {value.shape}")
                elif isinstance(value, (int, float, str)):
                    print(f"Key: '{key}' | Value: {value}")
                else:
                    print(f"Key: '{key}' | Type: {type(value)}")
        else:
            print(f"Loaded data type: {type(data)}")
            print(data)
            
    except Exception as e:
        print(f"Failed to load file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pt.py <path_to_file.pt>")
    else:
        inspect_pt(sys.argv[1])
