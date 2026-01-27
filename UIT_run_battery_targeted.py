import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
import time
import uuid
import matplotlib.pyplot as plt
import seaborn as sns

# Relative Root Discovery: Ensure the parent project directory is in the path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIGURATION (Targeted) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = time.strftime("%Y-%m-%d_%H%M")
UID = f"{TIMESTAMP}_SCIENTIFIC_PROTOCOL_TARGETED"
BASE_DIR = f"results/{UID}"
LOG_DIR = f"{BASE_DIR}/logs"
CRYSTAL_DIR = f"{BASE_DIR}/crystals"
PLOT_DIR = f"{BASE_DIR}/plots"

for d in [LOG_DIR, CRYSTAL_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

class WorkflowLogger:
    def __init__(self, filename):
        self.filename = filename
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")

W_LOG = WorkflowLogger(f"{LOG_DIR}/scientific_duel_{UID}.txt")

# --- DATA GENERATORS ---
def get_full_charter_set():
    char_ids = torch.arange(256).long().to(DEVICE)
    bits_msb = []
    bits_lsb = []
    for cid in char_ids:
        bits_msb.append([(cid.item() >> i) & 1 for i in range(7, -1, -1)])
        bits_lsb.append([(cid.item() >> i) & 1 for i in range(8)])
    
    x_msb = torch.tensor(bits_msb).float().unsqueeze(-1).to(DEVICE)
    y_id = char_ids.to(DEVICE)
    x_oh = nn.functional.one_hot(char_ids, 256).float().unsqueeze(1).to(DEVICE)
    y_lsb = torch.tensor(bits_lsb).float().to(DEVICE)
    return x_msb, y_id, x_oh, y_lsb

def generate_ascii_data(batch_size):
    char_ids = torch.randint(0, 256, (batch_size,)).long()
    bits_msb = []
    bits_lsb = []
    for cid in char_ids:
        bits_msb.append([(cid.item() >> i) & 1 for i in range(7, -1, -1)])
        bits_lsb.append([(cid.item() >> i) & 1 for i in range(8)])
    
    x_msb = torch.tensor(bits_msb).float().unsqueeze(-1).to(DEVICE)
    y_id = char_ids.to(DEVICE)
    x_onehot = nn.functional.one_hot(char_ids, 256).float().unsqueeze(1).to(DEVICE) # [Batch, 1, 256]
    y_lsb = torch.tensor(bits_lsb).float().to(DEVICE)
    
    return x_msb, y_id, x_onehot, y_lsb

# --- GRU BASELINES ---
class GRUDecoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, output_size=256):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, h = self.gru(x)
        logits = self.readout(h.squeeze(0))
        return logits, h.squeeze(0)

class GRUEncoder(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, output_size=1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size + output_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, char_onehot, seq_len=8):
        # char_onehot is [Batch, 1, 256]. Squeeze to [Batch, 256] for concatenation
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

# --- PHASE 0: GRU BASELINE TRAINING ---
def train_gru_baseline():
    W_LOG.log("--- PHASE 0: GRU BASELINE TRAINING ---")
    dec = GRUDecoder().to(DEVICE)
    enc = GRUEncoder().to(DEVICE)
    
    dec_opt = optim.Adam(dec.parameters(), lr=1e-3)
    enc_opt = optim.Adam(enc.parameters(), lr=1e-3)
    
    # Static Global Set for absolute honesty
    gx_m, gy_id, gx_oh, gy_l = get_full_charter_set()
    
    for epoch in range(5001): 
        x_m, y_id, x_oh, y_l = generate_ascii_data(64)
        
        dec.train(); dec_opt.zero_grad()
        loss_d = nn.CrossEntropyLoss()(dec(x_m)[0], y_id)
        loss_d.backward()
        dec_opt.step()
        
        enc.train(); enc_opt.zero_grad()
        loss_e = nn.BCEWithLogitsLoss()(enc(x_oh), y_l)
        loss_e.backward()
        enc_opt.step()
        
        # Real-time Global Check
        dec.eval(); enc.eval()
        with torch.no_grad():
            d_acc = (dec(gx_m)[0].argmax(dim=1) == gy_id).float().mean().item()
            e_acc = ((enc(gx_oh) > 0) == gy_l).all(dim=1).float().mean().item()
            
            if epoch % 100 == 0:
                W_LOG.log(f"GRU Step {epoch} | Global Dec: {d_acc:.2%} | Global Enc: {e_acc:.2%}")
                
            if d_acc == 1.0 and e_acc == 1.0 and epoch > 300:
                W_LOG.log(f"--- [GRU BASELINE LOCKED GLOBALLY AT EPOCH {epoch}] ---")
                break
    
    torch.save(dec.state_dict(), f"{CRYSTAL_DIR}/gru_dec_{UID}.pt")
    torch.save(enc.state_dict(), f"{CRYSTAL_DIR}/gru_enc_{UID}.pt")
    return dec, enc, None

# --- PHASE 1: UIT-ROUND CRYSTALLIZATION ---
def crystallize_uit():
    W_LOG.log("--- PHASE 1: UIT-ROUND CRYSTALLIZATION ---")
    r_dec = UITModel(input_size=1, hidden_size=512, output_size=256, use_binary_alignment=True, persistence=1.0).to(DEVICE)
    r_dec_opt = optim.Adam(r_dec.parameters(), lr=2**-7)
    
    # Static Global Set
    gx_m, gy_id, gx_oh, gy_l = get_full_charter_set()
    
    for epoch in range(5001): 
        r_dec.train()
        x_m, y_id, _, _ = generate_ascii_data(64)
        r_dec_opt.zero_grad()
        logits, conf = r_dec(x_m)
        loss = nn.CrossEntropyLoss()(logits, y_id) * (1.1 - conf.item())
        loss.backward()
        r_dec_opt.step()
        
        # Real-Time Global Check
        r_dec.eval()
        with torch.no_grad():
            full_logits, full_conf = r_dec(gx_m)
            acc = (full_logits.argmax(dim=1) == gy_id).float().mean().item()
            
            if epoch % 50 == 0:
                W_LOG.log(f"UIT Decoder {epoch} | Global Acc: {acc:.2%} | Conf: {full_conf.item():.4f}")
            
            if acc == 1.0 and epoch > 100:
                W_LOG.log(f"--- [UIT DECODER LOCKED GLOBALLY AT EPOCH {epoch}] ---")
                break
    
    # Save Map for Encoder
    map_path = f"{CRYSTAL_DIR}/phasic_map_{UID}.pt"
    # Logic for identity seeding usually uses a fixed or specific map retrieval
    
    # Train Encoder
    W_LOG.log("--- PHASE 2: UIT ENCODER CRYSTALLIZATION ---")
    r_enc = UITEncoderModel(input_size=256, hidden_size=512, output_size=8, use_binary_alignment=False, persistence=0.0).to(DEVICE)
    r_enc.renormalize_identity(map_path)
    r_enc_opt = optim.Adam(r_enc.parameters(), lr=1e-3)
    
    for epoch in range(5001): 
        r_enc.train()
        _, _, x_oh, y_l = generate_ascii_data(64)
        
        r_enc_opt.zero_grad()
        outs_raw, conf = r_enc(x_oh) 
        outs = outs_raw.squeeze(1) 
        loss = nn.BCEWithLogitsLoss()(outs, y_l)
        loss.backward()
        r_enc_opt.step()
        
        # Real-Time Global Check
        r_enc.eval()
        with torch.no_grad():
            full_outs_raw, full_conf = r_enc(gx_oh)
            full_outs = full_outs_raw.squeeze(1)
            acc = ((full_outs > 0) == gy_l).all(dim=1).float().mean().item()
            
            if epoch % 50 == 0:
                W_LOG.log(f"UIT Encoder {epoch} | Global Acc: {acc:.2%} | Conf: {full_conf.item():.4f}")
            
            if acc == 1.0 and epoch > 100:
                W_LOG.log(f"--- [UIT ENCODER LOCKED GLOBALLY AT EPOCH {epoch}] ---")
                break
                    
    torch.save(r_dec.state_dict(), f"{CRYSTAL_DIR}/uit_dec_{UID}.pt")
    torch.save(r_enc.state_dict(), f"{CRYSTAL_DIR}/uit_enc_{UID}.pt")
    return r_dec, r_enc, None

def get_lr_for_task(script_name):
    if "prism_stack" in script_name: return 0.001 
    if "sine_waves" in script_name: return 0.001
    if "color_algebra" in script_name: return 0.0078125 
    return 0.0078125 

# --- PHASE 6: EXTERNAL BENCHMARK INTEGRATION ---
def run_external_benchmarks():
    W_LOG.log("--- PHASE 6: EXTERNAL BENCHMARK SUITE (TARGETED) ---")
    
    # TARGETED SUITE: Only the loop benchmark
    suite = [
        "UIT_benchmark_crystalline_loop.py",
        "UIT_benchmark_sandwich_duel.py",
        "UIT_benchmark_color_algebra.py",
        "UIT_benchmark_prism_stack.py",
        "UIT_benchmark_sine_waves.py"
    ]

    for script_name in suite:
        script_path = os.path.join("UIT_Benchmarks", script_name)
        
        if not os.path.exists(script_path):
             W_LOG.log(f"SKIP: {script_name} not found in UIT_Benchmarks")
             continue
             
        lr = get_lr_for_task(script_name)
        
        # Point output_dir to 'plots' and log_dir to 'logs'
        plots_dir = os.path.join(BASE_DIR, "plots")
        logs_dir = os.path.join(BASE_DIR, "logs")
        cmd = [sys.executable, script_path, "--output_dir", plots_dir, "--log_dir", logs_dir, "--uid", UID, "--lr", str(lr)]
        
        # Crystal Path Injection
        crystal_file = os.path.join(BASE_DIR, "crystals", f"uit_dec_{UID}.pt") 
        if os.path.exists(crystal_file):
            cmd.extend(["--crystal_path", crystal_file])

        W_LOG.log(f"RUNNING: {script_name}")
        try:
            subprocess.run(cmd, check=True)
            W_LOG.log(f"SUCCESS: {script_name}")
        except subprocess.CalledProcessError as e:
            W_LOG.log(f"FAILURE: {script_name} (Exit Code: {e.returncode})")

if __name__ == "__main__":
    W_LOG.log(f"Starting Industrial Crystalline Duel (TARGETED) | UID: {UID}")
    
    # Phase 0: Train GRU (Requested for Fairness)
    g_dec, g_enc, g_hist = train_gru_baseline()
    
    # Phase 1: Crystallize UIT
    r_dec, r_enc, r_hist = crystallize_uit()
    
    # Note: Skipping Relay Duel & Story Visualization for speed
    
    # Phase 6: Run Targeted Benchmarks
    run_external_benchmarks()
    
    W_LOG.log(f"Targeted Workshop complete. Results in: {BASE_DIR}")
