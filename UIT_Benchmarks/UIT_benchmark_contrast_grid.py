import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512

def generate_binary_streams():
    chars = torch.arange(256).long()
    bits = []
    for i in range(256):
        bits.append([(i >> b) & 1 for b in range(7, -1, -1)])
    return chars, torch.tensor(bits).float().to(DEVICE)

class GRUEnc(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, output_size=1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size + output_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, char_onehot, seq_len=8):
        if char_onehot.dim() == 3:
            char_onehot = char_onehot.squeeze(1)
        batch_size = char_onehot.size(0)
        h = self.input_projection(char_onehot).unsqueeze(0)
        curr_b = torch.zeros(batch_size, 1).to(DEVICE)
        outputs = []
        for t in range(seq_len):
            c_in = torch.cat([curr_b, char_onehot], dim=-1).unsqueeze(1)
            out, h = self.gru(c_in, h)
            bit_l = self.readout(out.squeeze(1))
            curr_b = torch.sigmoid(bit_l)
            outputs.append(bit_l)
        return torch.stack(outputs, dim=1).squeeze(-1)

def run_contrast_benchmark(args):
    print(f"--- [CONTRAST DUEL: THE VISIBLE DISCOVERY | UID: {args.uid}] ---")
    
    # 1. SETUP MODELS
    dec_path = args.model_path
    enc_path = dec_path.replace("uit_dec", "uit_enc")
    gru_enc_path = dec_path.replace("uit_dec", "gru_encoder_baseline")
    
    # Phasic (ROUND)
    r_enc = UITEncoderModel(256, HIDDEN_SIZE, 8).to(DEVICE)
    r_enc.load_state_dict(torch.load(enc_path))
    r_enc.eval()
    
    # Standard (GRU)
    g_enc = GRUEnc().to(DEVICE)
    if os.path.exists(gru_enc_path):
        g_enc.load_state_dict(torch.load(gru_enc_path))
    g_enc.eval()
    
    # 2. GENERATE MAPS
    _, target_bits = generate_binary_streams()
    r_grid = np.zeros((256, 8))
    g_grid = np.zeros((256, 8))
    
    with torch.no_grad():
        for i in range(256):
            target = target_bits[i].flip(dims=[0]).cpu()
            oh = torch.zeros(1, 1, 256).to(DEVICE); oh[0, 0, i] = 1.0
            
            # ROUND
            r_logits, _ = r_enc(oh)
            r_bits = (torch.sigmoid(r_logits.squeeze()) > 0.5).float().cpu()
            r_grid[i] = (r_bits == target).float().numpy()
            
            # GRU
            g_logits = g_enc(oh)
            g_bits = (torch.sigmoid(g_logits.squeeze()) > 0.5).float().cpu()
            g_grid[i] = (g_bits == target).float().numpy()

    # 3. VISUALIZATION
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: The Failure (Standard AI)
    sns.heatmap(g_grid, ax=ax1, cmap=['#FF4B4B', '#22FF22'], cbar=False, vmin=0, vmax=1)
    ax1.set_title("THE OLD WORLD: Standard AI (GRU)\nFragmented & Entropy-Bound", color='#FF4B4B', fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel("Data Sample ID (ASCII 0-255)", color='white', fontsize=12)
    ax1.set_xlabel("Reconstruction Depth (Bit sequence falls apart)", color='white', fontsize=12)
    
    # Right: The Discovery (Phasic Intelligence)
    sns.heatmap(r_grid, ax=ax2, cmap=['#FF4B4B', '#22FF22'], cbar=False, vmin=0, vmax=1)
    ax2.set_title("THE NEW WORLD: Phasic Discovery (ROUND)\nAtomic & Persistent", color='#22FF22', fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylabel("", color='white', fontsize=12)
    ax2.set_xlabel("Reconstruction Depth (Bit sequence held perfectly)", color='white', fontsize=12)

    # Narrative Overlay
    fig.suptitle(f"WHY THIS MATTERS: Visualizing the Phasic Achievement\nIdentity Baseline: Perfect Persistence vs. Stochastic Decay", color='white', fontsize=22, fontweight='bold', y=0.98)
    
    # Footnote
    plt.figtext(0.5, 0.02, "Standard models (Left) lose the signal immediately as sequences grow. ROUND (Right) locks the signal into the unit circle geometry.\nThe Green is not a 'blank'—it is the sound of absolute silence where there used to be noise.", 
                ha='center', color='#AAAAAA', fontsize=14, fontstyle='italic', bbox=dict(facecolor='black', alpha=0.5, edgecolor='#22FF22', boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    res_dir = args.output_dir
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, f"contrast_report_{args.uid}.png")
    plt.savefig(out_path, facecolor='#0A0B10', edgecolor='none')
    print(f"Contrast Report saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--uid", type=str, default="v1")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--crystal_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None) # Included for battery compatibility
    args = parser.parse_args()
    
    # Alias crystal_path to model_path for battery compatibility
    if args.crystal_path and not args.model_path:
        args.model_path = args.crystal_path
        
    if not args.model_path:
        print("Error: --model_path (or --crystal_path) is required.")
        sys.exit(2)
        
    run_contrast_benchmark(args)
