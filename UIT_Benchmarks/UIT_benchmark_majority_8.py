# UIT-ROUND v1.3.14: 8-bit Majority Vote Benchmark
# Task: Given 8 sequential bits, output 1 if count(1s) > 4, else 0.
# HTML reference: round_bloch.html, task="majority"
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
import uuid
import time

# Relative Root Discovery
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from UIT_ROUND import UITModel
from config import MAJORITY_CONFIG, get_lock_strength

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="results/majority_bench")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--uid", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None)  # Battery compat (unused)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UID = args.uid if args.uid else str(uuid.uuid4())[:8]
OUTPUT_DIR = args.output_dir
LOG_DIR = args.log_dir if args.log_dir else OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

L_FILE = open(os.path.join(LOG_DIR, f"log_majority_{UID}.txt"), 'w')
def P(s):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {s}"
    print(line); L_FILE.write(line + '\n'); L_FILE.flush()

# --- CONFIGURATION ---
TC = MAJORITY_CONFIG
C = {
    'task': 'majority_8',
    'input_dim': 1,
    'seq_len': 8,
    'hidden_r': TC['HIDDEN_R'],
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
# Matches HTML genMajority(): count > 4 → 1, else 0.
def generate_majority_data(n, seq_len=8):
    X = torch.randint(0, 2, (n, seq_len, 1)).float()
    Y = (X.sum(1) > 4).float()  # [Batch, 1] — strictly greater than 4
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
    m = UITModel(input_size=1, hidden_size=C['hidden_r'], output_size=1,
                 num_layers=1, harmonics=TC['HARMONICS'], use_spinor=True).to(DEVICE)
    
    optimizer = optim.Adam(m.parameters(), lr=C['lr'])
    criterion = nn.BCEWithLogitsLoss()
    
    acc_history = []
    
    for e in range(C['epochs']):
        m.train()
        lock_strength = get_lock_strength(e, C['epochs'], TC['PEAK_LOCKING_STRENGTH'], TC['FLOOR'])
        
        optimizer.zero_grad()
        logits, conf = m(X)
        
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
        
        if acc == 1.0 and e > 50:
            P(f"R{rid} CRYSTAL LOCKED at Epoch {e}. Success.")
            acc_history.extend([acc] * (C['epochs'] - len(acc_history)))
            break
            
    return acc_history, m

def train_gru(rid, X, Y, Xt, Yt):
    m = GRUModel(C['input_dim'], C['hidden_g']).to(DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=0.001)
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
    X, Y = generate_majority_data(C['dataset_size'])
    Xt, Yt = generate_majority_data(1000)
    
    P("Training ROUND (Resonant Logic)")
    rh, rm = train_round(1, X, Y, Xt, Yt)
    
    P("Training GRU (Vector Logic)")
    gh, gm = train_gru(1, X, Y, Xt, Yt)
    
    # --- VISUALIZATION ---
    P("Generating Majority Duel Report...")
    rm.eval(); gm.eval()
    with torch.no_grad():
        r_logits, r_conf, r_coords_list = rm(Xt, return_coordinates=True)
        r_final_acc = rh[-1]
        
        rx = []; ry = []
        for step_tuple in r_coords_list:
            rx.append(step_tuple[0].cpu().numpy().flatten())
            ry.append(step_tuple[1].cpu().numpy().flatten())
        rx = np.concatenate(rx)
        ry = np.concatenate(ry)
        
        g_logits, g_h = gm(Xt, return_coords=True)
        g_final_acc = gh[-1]
        g_h_cpu = g_h.cpu().numpy()
        half = C['hidden_g'] // 2
        gx = g_h_cpu[:, :half].flatten()
        gy = g_h_cpu[:, half:].flatten()

    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Panel A: Learning Curves
        ax1.plot(rh, color='forestgreen', linewidth=3, label=f'ROUND (H={C["hidden_r"]})')
        ax1.plot(gh, color='steelblue', linewidth=3, label=f'GRU (H={C["hidden_g"]})')
        ax1.set_title("A. Learning Curves (8-bit Majority)", color='white', fontsize=18, fontweight='bold', pad=25)
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
        ax3.scatter(rx, ry, color='forestgreen', s=10, alpha=0.5, zorder=3, label='ROUND Phase Manifold')
        ax3.scatter(gx, gy, color='steelblue', s=5, alpha=0.3, zorder=2, label=f'GRU ({C["hidden_g"]}D Panes)')
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), color='white', linestyle='--', alpha=0.3)
        ax3.grid(True, which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.4)
        ax3.set_axisbelow(False)
        ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)
        ax3.set_aspect('equal')
        ax3.set_title("C. Hypertorus Projection (Internal State)", color='white', fontsize=18, fontweight='bold', pad=25)
        ax3.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='lower right')

        fig.suptitle(f"The Crystalline Intelligence Report: 8-bit Majority Vote Duel\nUID: {UID} | LR: {C['lr']:.6f}", color='white', fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(OUTPUT_DIR, f"majority_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='#0A0B10', edgecolor='none')
        P(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        P(f"Visualization Failed: {e}")
        import traceback
        P(traceback.format_exc())

    L_FILE.close()
