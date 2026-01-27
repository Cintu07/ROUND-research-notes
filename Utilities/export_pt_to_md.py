
import torch
import sys
import os
import numpy as np

def export_to_md(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        data = torch.load(file_path, weights_only=False)
        base_name = os.path.basename(file_path)
        out_path = file_path.replace(".pt", "_export.md")
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Data Export: {base_name}\n\n")
            
            # 1. Metadata Section
            f.write("## 1. Experiment Metadata\n\n")
            f.write("| Key | Value |\n")
            f.write("| :--- | :--- |\n")
            
            simple_keys = ["neurons", "lr", "r_success", "g_success"]
            for k in simple_keys:
                if k in data:
                    val = data[k]
                    # Handle numpy scalars
                    if hasattr(val, 'item'):
                        val = val.item()
                        
                    if isinstance(val, (float, np.floating)):
                        val_str = f"{val:.4f}" if val > 0.0001 else f"{val:.8f}"
                        if "success" in k: val_str += f" ({val*100:.1f}%)"
                    else:
                        val_str = str(val)
                    f.write(f"| `{k}` | **{val_str}** |\n")
            f.write("\n---\n\n")

            # 2. Bit Persistence Grid (The Heatmap)
            if "r_grid" in data:
                grid = data["r_grid"]
                rows, cols = grid.shape
                f.write(f"## 2. Bit Persistence Grid ({rows}x{cols})\n")
                f.write("> **Legend**: `1` = Retrieved (Green), `0` = Lost (Red)\n\n")
                
                f.write("```text\n")
                f.write("      | MSB 7 6 5 4 3 2 1 0 LSB\n")
                f.write("------+-------------------------\n")
                for i in range(rows):
                    row_data = grid[i]
                    # Convert floats to int for clean 0/1 display
                    bits = "".join(["1 " if x > 0.5 else ". " for x in row_data])
                    f.write(f"ID {i:03d}|     {bits}\n")
                f.write("```\n\n")

            # 3. Phasic Manifold Coordinates (Scatter Data)
            # Combine X and Y into (x,y) points
            if "r_scatter_x" in data and "r_scatter_y" in data:
                rx = np.array(data["r_scatter_x"]).flatten()
                ry = np.array(data["r_scatter_y"]).flatten()
                count = len(rx)
                
                f.write(f"## 3. Phasic Manifold Coordinates ({count} points)\n")
                f.write("Coordinates from the 512D -> 2D projection. Grouped by TimeStep (8 steps per Sequence ID).\n\n")
                
                f.write("| Index | Seq ID | Step | X Coordinate | Y Coordinate |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                
                # Assuming 8 steps per sequence (from code logic)
                steps_per_seq = 8
                
                for i in range(count):
                    seq_id = i // steps_per_seq
                    step_id = i % steps_per_seq
                    f.write(f"| {i} | {seq_id} | {step_id} | `{float(rx[i]):.6f}` | `{float(ry[i]):.6f}` |\n")

    
        print(f"Export successful! Created: {out_path}")

    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_pt_to_md.py <path_to_file.pt>")
    else:
        export_to_md(sys.argv[1])
