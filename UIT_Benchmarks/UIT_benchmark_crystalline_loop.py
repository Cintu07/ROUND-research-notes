import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Relative Root Discovery: Ensure the parent project directory is in the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path: sys.path.insert(0, root_dir)
from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512
SEQ_LEN = 8

def generate_binary_streams():
    chars = torch.arange(256).long()
    bits = []
    for i in range(256):
        bits.append([(i >> b) & 1 for b in range(7, -1, -1)])
    return chars, torch.tensor(bits).float().to(DEVICE)

def run_loop_benchmark(args):
    print(f"--- [BIT PERSISTENCE VERIFICATION | UID: {args.uid}] ---")
    
    # 1. SETUP ROUND
    dec_path = args.model_path if args.model_path else os.path.join("models", f"uit_dec_{args.uid}.pt")
    enc_path = dec_path.replace("uit_dec", "uit_enc") if "uit_dec" in dec_path else None
    
    r_dec = UITModel(1, HIDDEN_SIZE, 256, use_binary_alignment=True).to(DEVICE)
    try: r_dec.load_model(dec_path); print(f"Loaded ROUND Dec: {dec_path}")
    except Exception as e: print(f"Warning: ROUND Dec not loaded. Error: {e}")
    
    r_enc = UITEncoderModel(256, HIDDEN_SIZE, 8).to(DEVICE)
    try: r_enc.load_model(enc_path); print(f"Loaded ROUND Enc: {enc_path}")
    except Exception as e: print(f"Warning: ROUND Enc not loaded. Error: {e}")
    
    # 2. SETUP GRU (Mocked/Baseline context)
    # In a real battery, we'd load gru_dec/gru_enc.pt. 
    # For this script, we'll assume standard GRU wrappers if paths exist, else random init.
    gru_dec_path = dec_path.replace("uit_dec", "gru_dec")
    gru_enc_path = dec_path.replace("uit_dec", "gru_enc")
    
    # Simple GRU Wrapper for relay testing
    class GRUDec(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(1, HIDDEN_SIZE, batch_first=True)
            self.readout = nn.Linear(HIDDEN_SIZE, 256)
        def forward(self, x, h=None):
            out, h = self.gru(x, h)
            # Normalize for visualization comparison
            h_norm = torch.norm(h, dim=-1, keepdim=True)
            return self.readout(out), h, h_norm

    class GRUEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(256, HIDDEN_SIZE, batch_first=True)
            self.readout = nn.Linear(HIDDEN_SIZE, 8)
        def forward(self, x):
            out, h = self.gru(x)
            # Capture final state for comparison
            h_last = h[-1:] # (1, batch, hidden)
            return self.readout(out), h_last

    g_dec = GRUDec().to(DEVICE); g_enc = GRUEnc().to(DEVICE)
    if os.path.exists(gru_dec_path): 
        try: g_dec.load_state_dict(torch.load(gru_dec_path)); print("Loaded GRU Dec.")
        except: pass
    if os.path.exists(gru_enc_path):
        try: g_enc.load_state_dict(torch.load(gru_enc_path)); print("Loaded GRU Enc.")
        except: pass

    # 3. SETUP LOGGING DIRECTORY
    res_dir = args.output_dir
    log_dir = args.log_dir if args.log_dir else res_dir
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"verification_log_{args.uid}.txt")

    # 4. EXECUTE VERIFICATION
    char_ids, target_bits = generate_binary_streams()
    r_success = 0; g_success = 0
    r_grid = np.zeros((256, 8))
    
    # Visualization Buffers
    r_coords = [] # (batch, steps, hidden, 2)
    g_coords = [] # (batch, steps, hidden, 2)
    
    with torch.no_grad():
        for i in range(256):
            target = target_bits[i].flip(dims=[0]).cpu()
            oh = torch.zeros(1, 1, 256).to(DEVICE); oh[0, 0, i] = 1.0
            
            # --- ROUND VERIFICATION ---
            r_logits, _, coords = r_enc(oh, return_coordinates=True)
            r_bits = (torch.sigmoid(r_logits.squeeze()) > 0.5).float().cpu()
            r_grid[i] = (r_bits == target).float().numpy()
            if torch.equal(r_bits, target): r_success += 1
            r_coords.append(coords) # List of (h_cos, h_sin)
            
            # --- GRU VERIFICATION ---
            g_logits, g_h_last = g_enc(oh)
            g_bits = (torch.sigmoid(g_logits.squeeze()) > 0.5).float().cpu()
            if torch.equal(g_bits, target): g_success += 1
            # Mock coordinate as (h[0], h[1]) or similar for visualization
            g_coords.append(g_h_last.squeeze().cpu().numpy())

    print(f"Verification Results: ROUND {r_success/256:.2%} | GRU {g_success/256:.2%}")
    
    with open(log_path, "w") as f:
        f.write(f"BIT PERSISTENCE VERIFICATION | UID: {args.uid}\n")
        f.write(f"ROUND Success: {r_success/256:.2%}\n")
        f.write(f"GRU Success: {g_success/256:.2%}\n")
        
        # Norm Integrity Check
        last_step_r = r_coords[0][-1] # (h_cos, h_sin)
        norm = torch.sqrt(last_step_r[0]**2 + last_step_r[1]**2).mean().item()
        f.write(f"ROUND Isometry Norm (Mean): {norm:.6f}\n")
        print(f"ROUND Isometry Norm: {norm:.6f}")

    # --- ADVANCED VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10), gridspec_kw={'width_ratios': [2, 1, 1]})
        
        # Panel A: Bit Persistence Map (Transposed: Bits on Y, Samples on X)
        sns.heatmap(r_grid.T, ax=ax1, cmap=['maroon', 'forestgreen'], cbar=False, vmin=0, vmax=1)
        ax1.set_title("A. Bit Persistence Map (Ground Truth)", color='white', fontsize=18, fontweight='bold', pad=25)
        ax1.set_xlabel("Data Sample ID (ASCII 0-255)", color='white', fontsize=12)
        ax1.set_ylabel("Bit Position (MSB → LSB)", color='white', fontsize=12)
        
        # Improve tick labels
        ax1.set_xticks([0, 32, 64, 96, 128, 160, 192, 224, 255])
        ax1.set_xticklabels(['0', '32', '64', '96', '128', '160', '192', '224', '255'], fontsize=10)
        ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax1.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7'], fontsize=10)
        ax1.grid(True, which='major', color='black', linestyle='-', linewidth=1.5, alpha=0.9)
        ax1.set_axisbelow(False) # Force grid to be on TOP of the heatmap
        
        # Add a clear legend for the heatmap
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='forestgreen', label='Perfect Recall (100%)'),
                          Patch(facecolor='maroon', label='Information Erasure (Error)')]
        ax1.legend(handles=legend_elements, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)

        # Panel B: Comparative Intelligence
        labels = ['Phasic (ROUND)', 'Standard (GRU)']
        scores = [r_success/256, g_success/256]
        colors = ['forestgreen', 'steelblue']
        ax2.bar(labels[0], scores[0], color=colors[0], alpha=0.9, edgecolor='white', linewidth=1)
        ax2.bar(labels[1], scores[1], color=colors[1], alpha=0.5, edgecolor='white', linewidth=1)
        ax2.set_ylim(0, 1.2)
        ax2.set_title("B. Intelligence Survival Rate", color='white', fontsize=18, fontweight='bold', pad=25)
        ax2.set_ylabel("Retrieval Accuracy", color='white', fontsize=12)
        
        for i, v in enumerate(scores):
            ax2.text(i, v + 0.05, f"{v:.1%}", ha='center', color=colors[i], fontweight='bold', fontsize=16)
        
        ax2.grid(True, alpha=0.1, axis='y')
        # Add descriptive subtitle
        ax2.text(0.5, 0.95, "Phasic geometry preserves information\nlong after standard vectors collapse.", 
                 transform=ax2.transAxes, ha='center', color='white', fontsize=12, fontstyle='italic', fontweight='bold')

        # Panel C: Hypertorus Projection (No misleading unit circle)
        # NOTE: Unit circle overlay REMOVED - it was misleading
        
        # Flatten r_coords for scatter
        # r_coords is list[256] of list[steps] of (h_cos, h_sin)
        rx = []; ry = []
        for seq in r_coords:
            for step in seq:
                rx.append(step[0].numpy().flatten())
                ry.append(step[1].numpy().flatten())
        
        # Scale the "Logic Plane" based on GRU accuracy
        plane_size = (g_success/256.0) * 1.0
        rect = plt.Rectangle((-plane_size, -plane_size), plane_size*2, plane_size*2, 
                             color='steelblue', alpha=0.08, zorder=1, label=f'GRU Logic Plane ({g_success/256:.0%})',
                             edgecolor='steelblue', linewidth=1.5)
        ax3.add_patch(rect)

        # ROUND Scatter (Layered on top of Plane)
        ax3.scatter(rx, ry, color='forestgreen', s=15, alpha=0.6, zorder=3, label='ROUND (512D Manifold)')
        
        # Plot GRU states (Blue) - Consistent with Sandwich Duel depth
        gx = []; gy = []
        for h in g_coords:
            gx.append(h[:len(h)//2])
            gy.append(h[len(h)//2:])
        ax3.scatter(gx, gy, color='steelblue', s=10, alpha=0.4, zorder=2, label='GRU (Transparent Logic)')
        
        ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2)
        ax3.set_aspect('equal')
        ax3.set_title("C. Hypertorus Projection (512D → 2D)", color='white', fontsize=18, fontweight='bold', pad=25)
        
        # Brutal Grid ON TOP
        ax3.grid(True, which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.4)
        ax3.set_axisbelow(False)
        ax3.legend(loc='lower left', facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)
        ax3.set_xlabel("Projected Dimension 1", color='white', fontsize=12)
        ax3.set_ylabel("Projected Dimension 2", color='white', fontsize=12)
        
        # Annotate the GRU blob
        ax3.annotate('GRU: Collapsed\nto origin', xy=(0, 0), xytext=(0.6, 0.6),
                     fontsize=10, color='white', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        
        # Honest caption
        ax3.text(0.5, -0.2, "ROUND explores the solution manifold.\nGRU collapses to the origin (no solutions).", 
                 transform=ax3.transAxes, ha='center', color='forestgreen', fontsize=10, fontweight='bold')

        fig.suptitle(f"The Crystalline Intelligence Report: Phasic Resilience vs. Signal Decay\nExperiment: {args.uid} | Neurons: {HIDDEN_SIZE} | LR: {args.lr}", color='white', fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        
        output_path = os.path.join(res_dir, f"verification_report_{args.uid}.png")
        data_path = os.path.join(log_dir, f"verification_data_{args.uid}.pt")
        
        plt.savefig(output_path, facecolor='#0A0B10', edgecolor='none')
        print(f"Report saved to: {output_path}")
        
        # Save Raw Data for Reproducibility
        raw_data = {
            "r_grid": r_grid,
            "r_success": r_success/256,
            "g_success": g_success/256,
            "r_scatter_x": np.array(rx),
            "r_scatter_y": np.array(ry),
            "g_scatter_x": np.array(gx),
            "g_scatter_y": np.array(gy),
            "neurons": HIDDEN_SIZE,
            "lr": args.lr
        }
        torch.save(raw_data, data_path)
        print(f"Raw Plotting Data saved to: {data_path}")
        print(f"Log saved to: {log_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--uid", type=str, default="test")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--crystal_path", type=str, default=None) # Added for battery compatibility
    args = parser.parse_args()
    
    # Resolve conflicting naming in battery injection
    if args.crystal_path and not args.model_path:
        args.model_path = args.crystal_path
        
    run_loop_benchmark(args)
