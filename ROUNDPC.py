# ROUNDPC v1.0 (Riemannian-Optimized Unified Neural Dynamic Predictive Coder)
import torch
import torch.nn as nn
import numpy as np

"""
Principles:
1. Phasic Identity: Information is stored as a residue in the graded ring (Phase Angle).
2. Phase-Differential Encoding: Bit-streams are encoded via recursive half-shifts (phi = 0.5*phi + bit*pi).
3. Recursive State Displacement: Bits are generated via recursive doubling (Bernoulli Unwinding).
4. Dynamic Stability: Stability is maintained by damping updates as resonance (confidence) grows.
5. Predictive Coding (NEW): BPTT is severed. Weight updates are driven by instantaneous local Kuramoto tension and Phi Tail trace.
"""

HARMONICS_STANDARD = [1, 2, 4, 8]

class ROUNDPCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, harmonics=[1, 2, 4, 8], use_spinor=True, quantization_strength=0.125, use_binary_alignment=False, unwinding_mode=False, persistence=1.0, spin_multiplier=None): 
        super(ROUNDPCNeuron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.use_spinor = use_spinor
        self.quantization_strength = quantization_strength
        self.use_binary_alignment = use_binary_alignment
        self.unwinding_mode = unwinding_mode
        self.persistence = persistence
        
        if spin_multiplier is not None:
            self.spin_multiplier = spin_multiplier
        else:
            self.spin_multiplier = 2.0 if use_spinor else 1.0
        
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.epsilon = nn.Parameter(torch.Tensor(hidden_size))
        self.diagnostic_harmonics = nn.Parameter(torch.Tensor(hidden_size, len(harmonics)))
        
        # New: Kuramoto Hebbian Coupling Weight K
        self.K = nn.Parameter(torch.ones(hidden_size) * 0.1)
        
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.zeros_(p.data)
            elif 'bias' in name:
                nn.init.zeros_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.kaiming_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
        with torch.no_grad():
            for j in range(self.hidden_size):
                self.epsilon[j] = 0.125 * (0.5 ** (j % 5)) 
            nn.init.uniform_(self.diagnostic_harmonics, 0.0, 1.0)
            spread = (2.0 * np.pi) / self.hidden_size
            for j in range(self.hidden_size):
                self.bias[self.hidden_size + j] = j * spread
                
    def forward(self, x, h_prev, target_phase_raw=None, coupling_scalar=1.0):
        # --- STATE NEURON (Stimulus & Memory) ---
        gates = x @ self.weight_ih + h_prev @ self.weight_hh + self.bias
        x_gate, phi_gate = gates.chunk(2, dim=-1)
        standard_part = x_gate # Removed tanh to prevent amplitude squashing
        multiplier = self.spin_multiplier
        
        phi_shift = (torch.sigmoid(phi_gate) * np.pi * multiplier)
        
        # The Phi Tail (Eligibility Trace Memory)
        phi_next_nominal = (h_prev * self.persistence) + phi_shift
        
        # Binary Alignment (Legacy Mode Compatibility)
        if self.use_binary_alignment:
            if self.unwinding_mode:
                bit_out = (h_prev >= (np.pi - 1e-7)).float()
                phi_next_nominal = (h_prev - bit_out * np.pi) * 2.0
                q_grid = np.pi / 128.0 
                phi_next_nominal = torch.round(phi_next_nominal / q_grid) * q_grid 
            else:
                incoming_bit = x[:, 0:1] 
                phi_next_nominal = (h_prev * 0.5) + (incoming_bit * np.pi)
                bit_out = incoming_bit
        
        # --- ERROR NEURON (Kuramoto Tension) ---
        error_signal = 0
        local_loss = None
        
        if target_phase_raw is not None and not self.use_binary_alignment:
            # Topological Gradient Pull (Replaces explicit asin sawtooth trap)
            # The tension drives the phase down the gradient of the scalar difference: (sin(phi) - target)^2
            current_estimate = torch.sin(phi_next_nominal)
            error_signal = - (current_estimate - target_phase_raw) * torch.cos(phi_next_nominal)
            
            # Local Kuramoto Phase Pull (Fractional Damping for Stability)
            phi_next = phi_next_nominal + (self.K * coupling_scalar * error_signal * (np.pi / 4))
            
            # Local Objective (Instantaneous geometric alignment)
            # Generating loss allows the wrapping model to execute local_loss.backward()
            local_loss = torch.mean((torch.sin(phi_next) - target_phase_raw)**2)
            
        else:
            # Native Quantization Fallback (If no explicit target provided)
            q_sieve = torch.round(phi_next_nominal / (np.pi / 4)) * (np.pi / 4)
            phi_next = phi_next_nominal + self.quantization_strength * (q_sieve - phi_next_nominal)

        phi_next = torch.remainder(phi_next, 2.0 * np.pi * multiplier)
        h_cos = torch.zeros_like(phi_next)
        h_sin = torch.zeros_like(phi_next)
        for idx, h in enumerate(self.harmonics):
            h_cos += self.diagnostic_harmonics[:, idx] * torch.cos(h * phi_next)
            h_sin += self.diagnostic_harmonics[:, idx] * torch.sin(h * phi_next)
            
        confidence = (h_cos.abs() / len(self.harmonics)).detach()
        output = standard_part * (1.0 + self.epsilon * h_cos)
        
        if self.use_binary_alignment:
            output = output + 0.1 * h_cos
            if self.unwinding_mode:
                output = bit_out
                
        return output, phi_next, confidence, h_cos, h_sin, local_loss

class ROUNDPCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, harmonics=[1, 2, 4, 8], use_spinor=True, use_binary_alignment=False, unwinding_mode=False, persistence=1.0, quantization_strength=0.125, spin_multiplier=None):
        super(ROUNDPCModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_binary_alignment = use_binary_alignment
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(ROUNDPCNeuron(layer_input, hidden_size, harmonics, use_spinor=use_spinor, quantization_strength=quantization_strength, use_binary_alignment=use_binary_alignment, unwinding_mode=unwinding_mode, persistence=persistence, spin_multiplier=spin_multiplier))
            
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            # Removed Tanh to prevent amplitude clipping of the internal representation
            nn.Linear(hidden_size, output_size)
        )
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, input_seq, target_seq=None, coupling_scalar=1.0, return_sequence=False, return_coordinates=False, discrete_bce_mode=False):
        batch_size, seq_len, _ = input_seq.size()
        outputs = []
        h_states = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
        confidences = []
        coords = []
        total_local_loss = 0
        
        for t in range(seq_len):
            current_input = input_seq[:, t, :]
            current_target = target_seq[:, t, :] if target_seq is not None else None
            
            step_local_loss = 0
            neuron_target = None if discrete_bce_mode else current_target
            
            for i, layer in enumerate(self.layers):
                current_input, h_next, conf, h_cos, h_sin, local_loss = layer(current_input, h_states[i], neuron_target, coupling_scalar)
                confidences.append(conf)
                
                if local_loss is not None:
                    step_local_loss += local_loss
                
                # SURGICAL REMOVAL OF BPTT
                # The memory propagates forward via the detached phi_tail, but gradients are severed.
                if target_seq is not None:
                    h_states[i] = h_next.detach()
                else:
                    h_states[i] = h_next # Legacy compatibility (BPTT active)

            # Instantaneous Local Backpropagation (Continuous Metric)
            if step_local_loss != 0 and target_seq is not None and self.training:
                step_local_loss.backward(retain_graph=True)
                total_local_loss += step_local_loss.item()
                
            if return_coordinates:
                coords.append((h_cos.detach().cpu(), h_sin.detach().cpu()))

            if return_sequence:
                feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
                step_logits = self.readout(feats)
                outputs.append(step_logits)
                
                # Instantaneous Local Backpropagation (Discrete Metric)
                if discrete_bce_mode and current_target is not None and self.training:
                    # Unstrangled BCE Local Tension
                    step_bce_loss = nn.BCEWithLogitsLoss()(step_logits, current_target)
                    # Time graph is detached via h_next.detach(), so retain_graph=True just preserves layer weights
                    step_bce_loss.backward(retain_graph=True)
                    total_local_loss += step_bce_loss.item()
        
        avg_confidence = torch.stack(confidences).mean()
        
        # Structure Return
        if return_sequence:
            res = (torch.stack(outputs, dim=1), avg_confidence)
            if return_coordinates: res = res + (coords,)
            return res
            
        final_feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
        res = (self.readout(final_feats), avg_confidence)
        if return_coordinates: res = res + (coords,)
        return res

    def save_model(self, path): torch.save(self.state_dict(), path)
    def load_model(self, path, freeze=True):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        if freeze:
            for p in self.parameters(): p.requires_grad = False
