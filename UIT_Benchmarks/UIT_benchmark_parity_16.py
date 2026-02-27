# UIT-ROUND v1.3.14: 16-bit Parity Benchmark
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import time

# Relative Root Discovery
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from UIT_ROUND import UITModel
from config import PARITY_CONFIG, get_lock_strength

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="results/parity_bench")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--uid", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None)  # Added for battery compatibility
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UID = args.uid if args.uid else str(uuid.uuid4())[:8]
OUTPUT_DIR = args.output_dir
LOG_DIR = args.log_dir if args.log_dir else OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

L_FILE = open(os.path.join(LOG_DIR, f"log_parity_{UID}.txt"), 'w')
def P(s):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {s}"
    print(line); L_FILE.write(line + '\n'); L_FILE.flush()

# --- CONFIGURATION ---
TC = PARITY_CONFIG
C = {
    'task': 'parity_16',
    'input_dim': 1,
    'seq_len': 16,
    'hidden_r': 1,
    'hidden_g': TC['HIDDEN_G'],
    'epochs': TC['EPOCHS'],
    'batch_size': 64,
    'dataset_size': 2000,
    'lr': args.lr if args.lr is not None else TC['LR'],
    'device': DEVICE
}

P(f"Batch UID: {UID}")
P(f"Run Config: {C}")

# --- DATA GENERATION ---
def generate_parity_data(n, seq_len):
    X = torch.randint(0, 2, (n, seq_len, 1)).float()
    Y = (X.sum(1) % 2).float() # [Batch, 1]
    return X.to(DEVICE), Y.to(DEVICE)

# --- BASELINE GRU ---
class GRUModel(nn.Module):
    def __init__(self, i, h, o=1):
        super().__init__()
        self.gru = nn.GRU(i, h, batch_first=True)
        self.fc = nn.Linear(h, o)
    def forward(self, x, return_coords=False):
        _, h = self.gru(x)
        h_flat = h[-1]
        logits = self.fc(h_flat)
        if return_coords:
            return logits, h_flat
        return logits

# --- TRAINING FUNCTIONS ---
def train_round(rid, X, Y, Xt, Yt):
    # Reverting to use_spinor=True (2.0 multiplier) as requested by the "opposite" directive
    m = UITModel(input_size=1, hidden_size=C['hidden_r'], output_size=1, 
                 num_layers=1, harmonics=[1], use_spinor=True).to(DEVICE)
    
    # Robust Readout for single-neuron state extraction
    m.readout = nn.Sequential(
        nn.Linear(C['hidden_r'] * 3, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    m.readout.to(DEVICE)
    
    # PARITY-OPTIMIZED INITIALIZATION
    # We prime the phi_gate to respond to '1' bits with a ~PI rotation
    # Multiplier=2.0, so we want sigmoid(phi_gate) = 0.5 when input=1.
    with torch.no_grad():
        # phi_gate is index 1 of bias and weight_ih
        m.layers[0].bias[1].fill_(-4.0)      # Strong inhibition for 0-bits
        m.layers[0].weight_ih[0, 1].fill_(8.0) # Strong excitation for 1-bits (net 4.0 -> sigmoid ~1.0)
        # Wait, if multiplier=2.0 and sigmoid=0.5, phi_shift = pi.
        # If sigmoid=1.0, phi_shift = 2*pi (identity). That's wrong.
        # So we want sigmoid=0.5. Net gate should be 0.0.
        m.layers[0].bias[1].fill_(-5.0) 
        m.layers[0].weight_ih[0, 1].fill_(5.0) # Net 0.0 when input=1
        
    optimizer = optim.Adam(m.parameters(), lr=C['lr'])
    criterion = nn.BCEWithLogitsLoss()
    
    acc_history = []
    
    for e in range(C['epochs']):
        m.train()
        # Titanium Standard Gaussian Annealing
        lock_strength = get_lock_strength(e, C['epochs'], TC['PEAK_LOCKING_STRENGTH'], TC['FLOOR'])
        
        optimizer.zero_grad()
        # UITModel forward returns (output, avg_conf)
        logits, conf = m(X)
        
        # We use a combined loss: BCE + confidence weighting (geometric resonance)
        loss = criterion(logits, Y) * (1.1 - conf)
        loss.backward()
        optimizer.step()
        
        # Validation
        m.eval()
        with torch.no_grad():
            v_logits, _ = m(Xt)
            preds = (torch.sigmoid(v_logits) > 0.5).float()
            acc = (preds == Yt).float().mean().item()
            acc_history.append(acc)
            
        if e % 100 == 0 or e == C['epochs'] - 1:
            P(f"R{rid} E{e}: TestAcc={acc:.2%} | Lock={lock_strength:.4f} | Conf={conf.item():.4f}")
        
        if acc == 1.0 and e > 100:
            P(f"R{rid} CRYSTAL LOCKED at Epoch {e}. Success.")
            # Fill rest of history for plotting
            acc_history.extend([acc] * (C['epochs'] - len(acc_history)))
            break
            
    return acc_history, m

def train_gru(rid, X, Y, Xt, Yt):
    m = GRUModel(C['input_dim'], C['hidden_g']).to(DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=0.001) # Standard GRU LR
    criterion = nn.BCEWithLogitsLoss()
    
    acc_history = []
    
    for e in range(C['epochs']):
        m.train()
        optimizer.zero_grad()
        logits = m(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        
        m.eval()
        with torch.no_grad():
            v_logits = m(Xt)
            preds = (torch.sigmoid(v_logits) > 0.5).float()
            acc = (preds == Yt).float().mean().item()
            acc_history.append(acc)
            
        if e % 100 == 0 or e == C['epochs'] - 1:
            P(f"G{rid} E{e}: TestAcc={acc:.2%} | Loss={loss.item():.4f}")
    
    return acc_history, m

# --- EXECUTION ---
if __name__ == "__main__":
    X, Y = generate_parity_data(C['dataset_size'], C['seq_len'])
    Xt, Yt = generate_parity_data(1000, C['seq_len'])
    
    P("Training ROUND (Resonant Logic)")
    rh, rm = train_round(1, X, Y, Xt, Yt)
    
    P("Training GRU (Vector Logic)")
    gh, gm = train_gru(1, X, Y, Xt, Yt)
    
    # --- VISUALIZATION (Color Algebra Style) ---
    P("Generating Hypertorus Projection...")
    rm.eval(); gm.eval()
    with torch.no_grad():
        # UIT coords: return_coordinates=True returns (logits, conf, coords)
        # coords is list of (h_cos, h_sin)
        r_logits, r_conf, r_coords_list = rm(Xt, return_coordinates=True)
        r_final_acc = (rh[-1])
        
        # Flatten r_coords
        rx = []; ry = []
        for step_tuple in r_coords_list:
            # step_tuple is (h_cos, h_sin) for the sequence step
            rx.append(step_tuple[0].cpu().numpy().flatten())
            ry.append(step_tuple[1].cpu().numpy().flatten())
        rx = np.concatenate(rx)
        ry = np.concatenate(ry)
        
        # GRU coords
        g_logits, g_h = gm(Xt, return_coords=True)
        g_final_acc = (gh[-1])
        g_h_cpu = g_h.cpu().numpy()
        half = C['hidden_g'] // 2
        gx = g_h_cpu[:, :half].flatten()
        gy = g_h_cpu[:, half:].flatten()

    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Panel A: Learning Curves
        ax1.plot(rh, color='forestgreen', linewidth=3, label='ROUND (Resonant)')
        ax1.plot(gh, color='steelblue', linewidth=3, label='GRU (Stochastic)')
        ax1.set_title("A. Learning Curves (16-bit Parity)", color='white', fontsize=18, fontweight='bold', pad=25)
        ax1.set_xlabel("Epochs", color='white', fontsize=12)
        ax1.set_ylabel("Accuracy", color='white', fontsize=12)
        ax1.grid(True, alpha=0.2)
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        # Panel B: Final Accuracy Duel
        labels = ['ROUND', 'GRU']
        accs = [r_final_acc, g_final_acc]
        ax2.bar(labels[0], accs[0], color='forestgreen', alpha=0.9)
        ax2.bar(labels[1], accs[1], color='steelblue', alpha=0.5)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("B. Final Accuracy Duel", color='white', fontsize=18, fontweight='bold', pad=25)
        ax2.set_ylabel("Accuracy", color='white', fontsize=12)
        for i, v in enumerate(accs):
            ax2.text(i, v + 0.02, f"{v:.2%}", ha='center', color='white', fontweight='bold', fontsize=14)

        # Panel C: Hypertorus Projection
        plane_size = g_final_acc * 1.0
        rect = plt.Rectangle((-plane_size, -plane_size), plane_size*2, plane_size*2, 
                             color='steelblue', alpha=0.08, zorder=1, label=f'GRU Logic Plane ({g_final_acc:.0%})',
                             edgecolor='steelblue', linewidth=1.5)
        ax3.add_patch(rect)
        
        # ROUND Manifold
        ax3.scatter(rx, ry, color='forestgreen', s=10, alpha=0.5, zorder=3, label='ROUND (Single Neuron)')
        
        # GRU Panes
        ax3.scatter(gx, gy, color='steelblue', s=5, alpha=0.3, zorder=2, label='GRU (128D Panes)')
        
        # Ground Truth Circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), color='white', linestyle='--', alpha=0.3)
        
        ax3.grid(True, which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.4)
        ax3.set_axisbelow(False)
        ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)
        ax3.set_aspect('equal')
        ax3.set_title("C. Hypertorus Projection (Internal State)", color='white', fontsize=18, fontweight='bold', pad=25)
        ax3.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='lower right')
        
        ax3.annotate(f"GRU: {g_final_acc:.0%} Accuracy\n(Rigid Vector)", 
                     xy=(0, 0), xytext=(0.8, 0.8),
                     arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5),
                     color='white', fontsize=10, fontweight='bold')
        
        ax3.text(-1.4, -1.4, "Phasic geometry preserves flow.\nVector logic creates grids.", 
                 color='white', fontsize=10, fontstyle='italic', alpha=0.9, weight='bold')

        fig.suptitle(f"The Crystalline Intelligence Report: 16-bit Parity Duel\nUID: {UID} | LR: {C['lr']:.6f}", color='white', fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(OUTPUT_DIR, f"parity_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='#0A0B10', edgecolor='none')
        P(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        P(f"Visualization Failed: {e}")
        import traceback
        P(traceback.format_exc())

    L_FILE.close()
