# UIT-ROUND v1.3.14
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Relative Root Discovery
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path: sys.path.insert(0, root_dir)
from UIT_ROUND import UITModel

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--uid", type=str, default="prism_restored")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 2000 # budget for logic crystallization
from config import TOPOLOGY_CONFIG
LEARNING_RATE = args.lr if args.lr is not None else TOPOLOGY_CONFIG['LR']
HIDDEN_SIZE = 18  # Match modular space (like Color Algebra's 64/64)

class PrismROUND(nn.Module):
    def __init__(self):
        super().__init__()
        # Restored Phasic Sieve (quantization_strength=0.125) for gradient stability
        # Standard Modular LR: 0.001
        self.uit = UITModel(input_size=18, hidden_size=HIDDEN_SIZE, output_size=18, num_layers=1, quantization_strength=0.125, persistence=0.5)
        self.classifier = nn.Linear(HIDDEN_SIZE * 3, 18)
        
    def forward(self, xl, xp, return_coords=False):
        # Explicit Phase Passing (ColorROUND Pattern)
        h = torch.zeros(xl.size(0), HIDDEN_SIZE).to(xl.device)
        # Step 1: Process xl (the Lens), accumulate phase into h
        _, h, _, _, _ = self.uit.layers[0](xl[:, 0, :], h)
        # Step 2: Process xp (the Light) through the prism state h
        feat, h, _, h_cos, h_sin = self.uit.layers[0](xp[:, 0, :], h)
        # Readout: Combine standard output with harmonic features
        combined = torch.cat([feat, h_cos, h_sin], dim=-1)
        logits = self.classifier(combined)
        if return_coords:
            return logits, h_cos, h_sin
        return logits

class PrismGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(18, HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.readout = nn.Linear(HIDDEN_SIZE, 18)
    def forward(self, xl, xp, return_coords=False):
        x_seq = torch.cat([xl, xp], dim=1)
        _, h = self.gru(x_seq)
        h_flat = h[-1].squeeze(0)
        logits = self.readout(h_flat)
        if return_coords:
            return logits, h_flat
        return logits

def run_benchmark():
    print(f"--- [v1.3.12 PRISM STACK RESTORATION | UID: {UID}] ---")
    r_model = PrismROUND().to(DEVICE)
    g_model = PrismGRU().to(DEVICE)
    
    # Using 0.02 LR to force movement in weights
    r_opt = optim.Adam(r_model.parameters(), lr=LEARNING_RATE)
    g_opt = optim.Adam(g_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {"round": [], "gru": [], "round_acc": [], "gru_acc": [], "epochs": []}
    
    for epoch in range(EPOCHS):
        xl_idx = torch.randint(0, 18, (BATCH_SIZE,)).to(DEVICE)
        xp_idx = torch.randint(0, 18, (BATCH_SIZE,)).to(DEVICE)
        
        xl = torch.zeros(BATCH_SIZE, 1, 18).to(DEVICE).scatter_(2, xl_idx.view(-1, 1, 1), 1.0)
        xp = torch.zeros(BATCH_SIZE, 1, 18).to(DEVICE).scatter_(2, xp_idx.view(-1, 1, 1), 1.0)
        y = (xl_idx + xp_idx) % 18
        
        # Training
        r_model.train(); g_model.train()
        r_opt.zero_grad(); r_loss = criterion(r_model(xl, xp), y); r_loss.backward(); r_opt.step()
        g_opt.zero_grad(); g_loss = criterion(g_model(xl, xp), y); g_loss.backward(); g_opt.step()
        
        if epoch % 100 == 0:
            r_model.eval(); g_model.eval()
            with torch.no_grad():
                # Validation Pass for Accuracy Tracking
                vl_idx = torch.randint(0, 18, (BATCH_SIZE,)).to(DEVICE)
                vp_idx = torch.randint(0, 18, (BATCH_SIZE,)).to(DEVICE)
                vl = torch.zeros(BATCH_SIZE, 1, 18).to(DEVICE).scatter_(2, vl_idx.view(-1, 1, 1), 1.0)
                vp = torch.zeros(BATCH_SIZE, 1, 18).to(DEVICE).scatter_(2, vp_idx.view(-1, 1, 1), 1.0)
                vy = (vl_idx + vp_idx) % 18
                
                r_acc = (torch.argmax(r_model(vl, vp), dim=1) == vy).float().mean().item()
                g_acc = (torch.argmax(g_model(vl, vp), dim=1) == vy).float().mean().item()
                
                history["round"].append(r_loss.item())
                history["gru"].append(g_loss.item())
                history["round_acc"].append(r_acc)
                history["gru_acc"].append(g_acc)
                history["epochs"].append(epoch)
                
                print(f"Epoch {epoch:4d} | ROUND: {r_loss.item():.4f} ({r_acc:.1%}) | GRU: {g_loss.item():.4f} ({g_acc:.1%})")
                
                # CRYSTAL LOCK: Immediate exit when ROUND achieves perfect resonance
                if r_acc > 0.999: 
                    print(f"CRYSTAL LOCKED at Epoch {epoch}. Phasic Identity achieved perfect resonance.")
                    break

    # --- FINAL EVALUATION FOR MANIFOLD SNAPSHOT ---
    r_model.eval(); g_model.eval()
    with torch.no_grad():
        v_size = 200 # Higher density for visualization
        vl_idx = torch.randint(0, 18, (v_size,)).to(DEVICE)
        vp_idx = torch.randint(0, 18, (v_size,)).to(DEVICE)
        vl = torch.zeros(v_size, 1, 18).to(DEVICE).scatter_(2, vl_idx.view(-1, 1, 1), 1.0)
        vp = torch.zeros(v_size, 1, 18).to(DEVICE).scatter_(2, vp_idx.view(-1, 1, 1), 1.0)
        vy = (vl_idx + vp_idx) % 18
        
        r_logits, r_cos, r_sin = r_model(vl, vp, return_coords=True)
        g_logits, g_h = g_model(vl, vp, return_coords=True)
        
        r_final_acc = (torch.argmax(r_logits, dim=1) == vy).float().mean().item()
        g_final_acc = (torch.argmax(g_logits, dim=1) == vy).float().mean().item()
        
        r_coords = (r_cos.cpu().numpy(), r_sin.cpu().numpy())
        half = HIDDEN_SIZE // 2
        g_h_cpu = g_h.cpu().numpy()
        gx = g_h_cpu[:, :half].flatten()
        gy = g_h_cpu[:, half:].flatten()
        g_coords = (gx, gy)

    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Panel A: Learning Biography
        ax1.plot(history["epochs"], history["round_acc"], color='forestgreen', linewidth=3, label='ROUND (Resonant)')
        ax1.plot(history["epochs"], history["gru_acc"], color='steelblue', linewidth=3, label='GRU (Stochastic)')
        ax1.set_title("A. Learning Biography (Prism Accuracy)", color='white', fontsize=12)
        ax1.set_xlabel("Epochs", color='white')
        ax1.set_ylabel("Accuracy", color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        # Panel B: Final Performance
        labels = ['ROUND', 'GRU']
        ax2.bar(labels[0], r_final_acc, color='forestgreen', alpha=0.9, label='ROUND (Resonant)')
        ax2.bar(labels[1], g_final_acc, color='steelblue', alpha=0.5, label='GRU (Transparent Logic)')
        ax2.set_ylim(0, 1.1)
        ax2.set_title("B. Final Accuracy Duel", color='white', fontsize=12)
        ax2.set_ylabel("Accuracy", color='white')
        ax2.text(0, r_final_acc + 0.02, f"{r_final_acc:.2%}", ha='center', color='white', fontweight='bold')
        ax2.text(1, g_final_acc + 0.02, f"{g_final_acc:.1%}", ha='center', color='white', fontweight='bold')

        # Panel C: Hypertorus Projection
        # Scale the "Logic Plane" (Rectangle) based on GRU accuracy
        plane_size = g_final_acc * 1.0 # Scale from 0 to 1.0
        rect = plt.Rectangle((-plane_size, -plane_size), plane_size*2, plane_size*2, 
                             color='steelblue', alpha=0.08, zorder=1, label=f'GRU Logic Plane ({g_final_acc:.0%})',
                             edgecolor='steelblue', linewidth=1.5)
        ax3.add_patch(rect)

        # Plot ROUND (Layered on top of Plane)
        ax3.scatter(r_coords[0], r_coords[1], color='forestgreen', s=15, alpha=0.6, zorder=3, label='ROUND (Manifold)')
        
        # Plot GRU second with 40% transparency to show "Window Panes"
        ax3.scatter(g_coords[0], g_coords[1], color='steelblue', s=10, alpha=0.4, zorder=2, label='GRU (Transparent Panes)')
        
        # Trajectory hint (Ground Truth Circle)
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), color='white', linestyle='--', alpha=0.3)
        
        # Brutal Grid ON TOP
        ax3.grid(True, which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.4)
        ax3.set_axisbelow(False)

        ax3.set_title("C. Hypertorus Projection (Internal State)", color='white', fontsize=12)
        ax3.set_xlabel("Projected Dimension 1", color='white')
        ax3.set_ylabel("Projected Dimension 2", color='white')
        ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)
        ax3.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='lower left', fontsize=8)
        
        # Dynamic Annotations for Brutal Honesty
        ax3.annotate(f"GRU: {g_final_acc:.0%} Accuracy\n(Rigid Vector)", 
                     xy=(0, 0), xytext=(0.8, 0.8),
                     arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5),
                     color='white', fontsize=9, fontweight='bold')
        
        ax3.text(-1.4, -1.4, "Phasic geometry preserves flow.\nVector logic creates grids.", 
                 color='white', fontsize=9, fontstyle='italic', alpha=0.9, weight='bold')

        fig.suptitle(f"Prism Stacking Head-to-Head Duel | UID: {UID}", color='white', fontsize=14)
        
        plot_path = os.path.join(OUTPUT_DIR, f"prism_stack_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    run_benchmark()
