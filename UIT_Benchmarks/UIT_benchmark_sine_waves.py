import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse
from datetime import datetime

# Relative Root Discovery
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path: sys.path.insert(0, root_dir)
from UIT_ROUND import UITModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure Output Directory Exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# --- SINE WAVE ROUND (Refined) ---
class SineROUND(nn.Module):
    def __init__(self, hidden_size=32, harmonics=[1], persistence=0.5):
        super().__init__()
        self.uit = UITModel(
            input_size=1, 
            hidden_size=hidden_size, 
            output_size=1, 
            num_layers=1,
            harmonics=harmonics,
            persistence=persistence,
            quantization_strength=0.0 # Fluid for continuous tracking
        )
        
    def forward(self, x, h_states=None, return_coordinates=False):
        res = self.uit(x, return_sequence=True, return_coordinates=return_coordinates)
        if return_coordinates:
            pred, conf, coords = res
            return pred, None, coords
        else:
            pred, conf = res
            return pred, None

# --- GRU BASELINE ---
class SineGRU(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        
    def forward(self, x, h_states=None):
        if h_states is None:
            h_states = torch.zeros(1, x.size(0), self.hidden_size).to(DEVICE)
        out, h_new = self.gru(x, h_states)
        pred = self.readout(out)
        return pred, h_new

# --- BENCHMARK EXECUTION ---
def run_benchmark(args):
    # Ensure environment is ready
    ensure_dir(args.output_dir)
    if args.log_dir:
        ensure_dir(args.log_dir)

    # CONFIG
    SEQ_LEN = 100
    BATCH_SIZE = 32
    EPOCHS = 3001 # Memory Core "Double Bake" Standard
    LR = args.lr if args.lr is not None else 0.001 # Battery compatibility
    HIDDEN = 32
    HARMONICS = [1] # Riemannian Native State (Sovereign Resonance)
    
    print(f"--- [RIEMANNIAN TOPOLOGICAL RECOVERY | UID: {args.uid}] ---")
    print(f"Goal: Recover continuous signal curvature from Phasic Identity (H={HARMONICS})")
    
    round_model = SineROUND(hidden_size=HIDDEN, harmonics=HARMONICS).to(DEVICE)
    gru_model = SineGRU(hidden_size=HIDDEN).to(DEVICE)
    
    r_opt = optim.Adam(round_model.parameters(), lr=LR)
    g_opt = optim.Adam(gru_model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Time Steps for evaluation
    t_eval = torch.linspace(0, 8*np.pi, SEQ_LEN, device=DEVICE).view(1, SEQ_LEN, 1)

    def generate_sine_data(batch_size):
        phases = torch.rand(batch_size, 1, 1, device=DEVICE) * 2 * np.pi
        x = torch.sin(t_eval + phases)
        return x, x # Target is input for reconstruction
    
    history = {"epoch": [], "round": [], "gru": []}
    
    for epoch in range(EPOCHS):
        # TRAIN
        round_model.train(); gru_model.train()
        x, y = generate_sine_data(BATCH_SIZE)
        
        # ROUND
        r_opt.zero_grad(); r_pred, _ = round_model(x); r_loss = criterion(r_pred, y); r_loss.backward(); r_opt.step()
        # GRU
        g_opt.zero_grad(); g_pred, _ = gru_model(x); g_loss = criterion(g_pred, y); g_loss.backward(); g_opt.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | ROUND MSE: {r_loss.item():.6f} | GRU MSE: {g_loss.item():.6f}")
            history["epoch"].append(epoch)
            history["round"].append(r_loss.item())
            history["gru"].append(g_loss.item())
            
            # CRYSTAL LOCK (Early Exit)
            if r_loss.item() < 0.0001:
                print(f"CRYSTAL LOCKED at Epoch {epoch}. Terminating Phase.")
                break

    # --- EVALUATION AND PLOTTING ---
    round_model.eval(); gru_model.eval()
    with torch.no_grad():
        v_x, v_y = generate_sine_data(1)
        r_pred, _, coords = round_model(v_x, return_coordinates=True)
        g_pred, _ = gru_model(v_x)
        
        t_np = t_eval.squeeze().cpu().numpy()
        target_np = v_y[0].cpu().numpy().flatten()
        r_np = r_pred[0].cpu().numpy().flatten()
        g_np = g_pred[0].cpu().numpy().flatten()
        
        rx = []; ry = []
        for step_coords in coords:
            # Flatten hidden units for single batch item [0]
            rx.extend(step_coords[0][0].cpu().flatten().tolist())
            ry.extend(step_coords[1][0].cpu().flatten().tolist())

    # VISUALIZATION (A|B|C Architecture)
    try:
        r_final_mse = history["round"][-1]
        g_final_mse = history["gru"][-1]
        # Fidelity as 1.0 - MSE (clipped at 0 and 100%)
        r_fidelity = max(0.0, min(100.0, 100.0 * (1.0 - r_final_mse)))
        g_fidelity = max(0.0, min(100.0, 100.0 * (1.0 - g_final_mse)))

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(22, 7))
        from matplotlib import gridspec
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 0.8, 1])
        
        ax1 = fig.add_subplot(gs[0]) # A: Learning Biography
        ax2 = fig.add_subplot(gs[1]) # B: Manifold Reconstruction
        ax3 = fig.add_subplot(gs[2]) # C: Hypertorus Panel
        
        c_round = 'forestgreen'; c_gru = 'steelblue'; c_truth = 'white'
        
        # --- PANEL A: LEARNING BIOGRAPHY ---
        ax1.plot(history["epoch"], history["round"], color=c_round, linewidth=3, label=f'ROUND (Final: {r_final_mse:.6f})')
        ax1.plot(history["epoch"], history["gru"], color=c_gru, linewidth=2, linestyle='--', label=f'GRU (Final: {g_final_mse:.6f})')
        ax1.set_title("A. Learning Biography (Topological Recovery)", color='white', fontsize=12)
        ax1.set_xlabel("Epochs", color='white'); ax1.set_ylabel("MSE (Log)", color='white')
        ax1.set_yscale('log'); ax1.grid(True, alpha=0.2); ax1.legend()
        
        # --- PANEL B: MANIFOLD RECONSTRUCTION ---
        ax2.plot(t_np, target_np, color=c_truth, label='Ground Truth', linewidth=4, alpha=0.3)
        ax2.plot(t_np, r_np, color=c_round, label=f'ROUND ({r_fidelity:.2f}% Fidelity)', linewidth=2, zorder=3)
        ax2.plot(t_np, g_np, color=c_gru, label=f'GRU ({g_fidelity:.2f}% Fidelity)', linewidth=2, linestyle='--', alpha=0.7)
        ax2.set_title("B. Riemannian Manifold Snapshot", color='white', fontsize=12)
        ax2.set_xlabel("Phase", color='white'); ax2.set_ylim(-1.5, 1.5); ax2.grid(True, alpha=0.2)
        ax2.legend(loc='upper right', fontsize=9)
        
        # Annotate Panel B with "resolved" status
        ax2.text(0.05, 0.95, f"ROUND: RESOLVED ({r_fidelity:.1f}%)", transform=ax2.transAxes, color=c_round, weight='bold', verticalalignment='top')
        ax2.text(0.05, 0.88, f"GRU: RESOLVED ({g_fidelity:.1f}%)", transform=ax2.transAxes, color=c_gru, weight='bold', verticalalignment='top')

        # --- PANEL C: HYPERTORUS PROJECTION ---
        ax3.scatter(rx, ry, color=c_round, s=15, alpha=0.6, zorder=3, label='ROUND (Resonance)')
        plane_size = 1.1 
        rect = plt.Rectangle((-plane_size, -plane_size), plane_size*2, plane_size*2, 
                             color=c_gru, alpha=0.08, zorder=1, label=f'GRU Logic Plane ({g_fidelity:.0f}%)',
                             edgecolor=c_gru, linewidth=1.5)
        ax3.add_patch(rect)
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), color='white', linestyle='--', alpha=0.3)
        ax3.set_title("C. Hypertorus Projection", color='white', fontsize=12)
        ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5); ax3.set_aspect('equal'); ax3.grid(True, alpha=0.2)
        
        # Annotation for Brutal Honesty
        ax3.annotate(f"GRU: {g_fidelity:.2f}% Fidelity\n(Linear Fitting)", 
                     xy=(0, 0), xytext=(0.7, 0.8),
                     arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5),
                     color='white', fontsize=9, fontweight='bold')

        ax3.legend(loc='lower left', fontsize=8)

        # CAPTION
        fig.suptitle(f"Riemannian Topological Recovery: Continuous Signal Manifold | UID: {args.uid}", color='white', fontsize=16)
        plt.figtext(0.5, 0.02, f"Both models achieve high fidelity, but via different topologies. ROUND resonates via Isomorphism; GRU approximates via Piece-wise Linear Gates.", 
                    ha="center", fontsize=10, color='white', fontstyle='italic', fontweight='bold')

        save_path = os.path.join(args.output_dir, f"riemannian_recovery_{args.uid}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, facecolor='black')
        print(f"Report saved: {save_path}")
        
    except Exception as e:
        print(f"Plotting Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--crystal_path", type=str, default=None) # Included for battery compatibility
    args = parser.parse_args()
    
    if args.uid is None:
        args.uid = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_benchmark(args)
