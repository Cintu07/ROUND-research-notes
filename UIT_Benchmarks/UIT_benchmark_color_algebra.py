# UIT-ROUND v1.3.14
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Relative Root Discovery
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path: sys.path.insert(0, root_dir)
from UIT_ROUND import UITModel

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--uid", type=str, default="color_duel")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None) # Included for battery compatibility
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
LOG_DIR = args.log_dir if args.log_dir else args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 1000 
from config import COLORS_CONFIG
LEARNING_RATE = args.lr if args.lr is not None else COLORS_CONFIG['LR']
HIDDEN_SIZE = 64 
NUM_COLORS = 64 

# --- DATA GENERATION ---
def get_color_phase(idx):
    return (idx / NUM_COLORS) * 2 * np.pi

def get_ground_truth_mixture(idx_a, idx_b):
    phi_a = get_color_phase(idx_a)
    phi_b = get_color_phase(idx_b)
    vec_a = np.array([np.cos(phi_a), np.sin(phi_a)])
    vec_b = np.array([np.cos(phi_b), np.sin(phi_b)])
    vec_mid = (vec_a + vec_b) / 2.0
    norm = np.linalg.norm(vec_mid)
    if norm < 1e-6: return (idx_a + idx_b) // 2 
    vec_mid = vec_mid / norm
    phi_mid = np.arctan2(vec_mid[1], vec_mid[0])
    if phi_mid < 0: phi_mid += 2 * np.pi
    return np.argmin([np.abs(get_color_phase(i) - phi_mid) for i in range(NUM_COLORS)])

def generate_color_data(batch_size):
    idx_a = torch.randint(0, NUM_COLORS, (batch_size,))
    idx_b = torch.randint(0, NUM_COLORS, (batch_size,))
    x = torch.zeros(batch_size, 2, NUM_COLORS)
    x.scatter_(2, idx_a.unsqueeze(1).unsqueeze(2), 1.0)
    x.scatter_(2, idx_b.unsqueeze(1).unsqueeze(2), 1.0)
    targets = [get_ground_truth_mixture(idx_a[i].item(), idx_b[i].item()) for i in range(batch_size)]
    return x.to(DEVICE), torch.tensor(targets).long().to(DEVICE)

# --- MODELS ---
class ColorROUND(nn.Module):
    def __init__(self):
        super().__init__()
        self.uit = UITModel(input_size=NUM_COLORS, hidden_size=HIDDEN_SIZE, output_size=NUM_COLORS, num_layers=1, persistence=0.5)
        self.classifier = nn.Linear(HIDDEN_SIZE * 3, NUM_COLORS)
    def forward(self, x, return_coords=False):
        h = torch.zeros(x.size(0), HIDDEN_SIZE).to(DEVICE)
        _, h, _, _, _ = self.uit.layers[0](x[:, 0, :], h)
        feat_2, h, _, h_cos_2, h_sin_2 = self.uit.layers[0](x[:, 1, :], h)
        combined = torch.cat([feat_2, h_cos_2, h_sin_2], dim=-1)
        logits = self.classifier(combined)
        if return_coords:
            return logits, h_cos_2, h_sin_2
        return logits

class ColorGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(NUM_COLORS, HIDDEN_SIZE, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_COLORS)
    def forward(self, x, return_coords=False):
        _, h = self.gru(x) # h: [1, B, H]
        h_flat = h.squeeze(0)
        logits = self.classifier(h_flat)
        if return_coords:
            return logits, h_flat
        return logits

def run_benchmark():
    print(f"--- [UIT-ROUND vs GRU: COLOR ALGEBRA DUEL | UID: {UID}] ---")
    round_model = ColorROUND().to(DEVICE)
    gru_model = ColorGRU().to(DEVICE)
    
    r_opt = optim.Adam(round_model.parameters(), lr=LEARNING_RATE)
    g_opt = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {"round": [], "gru": []}
    
    for epoch in range(EPOCHS):
        # Training
        round_model.train(); gru_model.train()
        x, y = generate_color_data(BATCH_SIZE)
        
        # ROUND
        r_opt.zero_grad(); r_loss = criterion(round_model(x), y); r_loss.backward(); r_opt.step()
        # GRU
        g_opt.zero_grad(); g_loss = criterion(gru_model(x), y); g_loss.backward(); g_opt.step()
        
        if epoch % 50 == 0:
            round_model.eval(); gru_model.eval()
            with torch.no_grad():
                vx, vy = generate_color_data(100)
                r_acc = (torch.argmax(round_model(vx), dim=1) == vy).float().mean().item()
                g_acc = (torch.argmax(gru_model(vx), dim=1) == vy).float().mean().item()
                history["round"].append(r_acc)
                history["gru"].append(g_acc)
                print(f"Epoch {epoch:4d} | ROUND: {r_acc:7.2%} | GRU: {g_acc:7.2%}")
            if r_acc > 0.999: 
                print(f"CRYSTAL LOCKED at Epoch {epoch}. Terminating Duel.")
                break
        
    # Final Evaluation for Manifold Data
    round_model.eval(); gru_model.eval()
    with torch.no_grad():
        vx, vy = generate_color_data(200) # Larger sample for better visual density
        r_logits, r_cos, r_sin = round_model(vx, return_coords=True)
        g_logits, g_h = gru_model(vx, return_coords=True) # g_h: [200, 64]
        
        # Capture Final Accuracy for Annotations
        r_final_acc = (torch.argmax(r_logits, dim=1) == vy).float().mean().item()
        g_final_acc = (torch.argmax(g_logits, dim=1) == vy).float().mean().item()

        # Coordinates
        r_coords = (r_cos.cpu().numpy(), r_sin.cpu().numpy())
        
        # GRU "Window Pane" logic: Split the vector into two 32D groups
        g_h_cpu = g_h.cpu().numpy()
        half = HIDDEN_SIZE // 2
        gx = g_h_cpu[:, :half].flatten()
        gy = g_h_cpu[:, half:].flatten()
        g_coords = (gx, gy)

    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Panel A: Learning Curves
        ax1.plot(history["round"], color='forestgreen', linewidth=3, label='ROUND (Resonant)')
        ax1.plot(history["gru"], color='steelblue', linewidth=3, label='GRU (Stochastic)')
        ax1.set_title("A. Learning Curves (Crayola-64 Mixture)", color='white', fontsize=12)
        ax1.set_xlabel("Epochs (x50)", color='white')
        ax1.set_ylabel("Accuracy", color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        # Panel B: Final Performance
        labels = ['ROUND', 'GRU']
        accs = [history["round"][-1], history["gru"][-1]]
        ax2.bar(labels[0], accs[0], color='forestgreen', alpha=0.9, label='ROUND (Resonant)')
        ax2.bar(labels[1], accs[1], color='steelblue', alpha=0.5, label='GRU (Transparent Logic)')
        ax2.set_ylim(0, 1.1)
        ax2.set_title("B. Final Accuracy Duel", color='white', fontsize=12)
        ax2.set_ylabel("Accuracy", color='white')
        for i, v in enumerate(accs):
            ax2.text(i, v + 0.02, f"{v:.2%}", ha='center', color='white', fontweight='bold')

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
        
        # Brutal Grid ON TOP (zorder high or axisbelow)
        ax3.grid(True, which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.4)
        ax3.set_axisbelow(False)

        ax3.set_title("C. Hypertorus Projection (Internal State)", color='white', fontsize=12)
        ax3.set_xlabel("Projected Dimension 1", color='white')
        ax3.set_ylabel("Projected Dimension 2", color='white')
        ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)
        ax3.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        # Dynamic Annotations for Brutal Honesty
        ax3.annotate(f"GRU: {g_final_acc:.0%} Accuracy\n(Rigid Vector)", 
                     xy=(0, 0), xytext=(0.8, 0.8),
                     arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5),
                     color='white', fontsize=9, fontweight='bold')
        
        ax3.text(-1.4, -1.4, "Phasic geometry preserves flow.\nVector logic creates grids.", 
                 color='white', fontsize=9, fontstyle='italic', alpha=0.9, weight='bold')

        fig.suptitle(f"Color Algebra Head-to-Head Duel | UID: {UID}", color='white', fontsize=14)
        
        plot_path = os.path.join(OUTPUT_DIR, f"color_algebra_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

    with open(os.path.join(LOG_DIR, f"color_log_{UID}.txt"), "w") as f:
        f.write(f"Final ROUND Acc: {history['round'][-1]:.4f}\n")
        f.write(f"Final GRU Acc: {history['gru'][-1]:.4f}\n")

if __name__ == "__main__":
    run_benchmark()
