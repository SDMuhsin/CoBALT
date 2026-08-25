"""
CPR-Sparse Evaluation: Compare SINQ, CPR-SINQ, SINQ-Sparse, and CPR-Sparse

This script compares:
1. FP16 baseline
2. SINQ 4-bit (uniform quantization)
3. CPR-SINQ (multi-precision: 25% @ 6-bit, 75% @ 5-bit -> avg 5.25 bits)
4. SINQ-Sparse 4-bit + 35% sparsity
5. CPR-SINQ + 35% Sparsity (new combination)

Evaluation on Qwen2.5-1.5B with WikiText-2 perplexity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_dual_scale_shift, quantize_rtn


# =============================================================================
# Data Loading
# =============================================================================

def get_wikitext2(tokenizer, seq_len=2048, n_samples=16):
    """Load WikiText-2 test set."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]
    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])
    return torch.stack(samples[:n_samples])


def get_calibration_data(tokenizer, n_samples=16, seq_len=512):
    """Get calibration data from WikiText-2 train."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt', max_length=seq_len * n_samples, truncation=True)
    input_ids = encodings.input_ids[0]
    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])
    return torch.stack(samples[:n_samples])


# =============================================================================
# Perplexity Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_perplexity(model, test_data, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(len(test_data)):
        batch = test_data[i:i + 1].to(device)
        outputs = model(batch, labels=batch)
        total_loss += outputs.loss.item() * batch.numel()
        total_tokens += batch.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


# =============================================================================
# Sparsity Functions
# =============================================================================

def batched_row_obs_prune(W, X, sparsity):
    """
    Batched row-wise OBS pruning with per-row sparsity.
    Uses Wanda-style importance and OBS compensation.
    """
    K, N = W.shape
    device = W.device
    dtype = W.dtype

    if sparsity == 0:
        return W.clone(), torch.ones(K, N, device=device)

    W = W.float()
    X = X.float().to(device)

    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])
    n_samples = min(X.shape[0], 512)
    X = X[:n_samples]

    # Compute activation norms (Wanda-style)
    act_col_norms = torch.norm(X, dim=0)

    # Build Hessian
    H = X.T @ X
    damping = max(0.01 * H.diag().mean().item(), 1e-3)
    H = H + damping * torch.eye(N, device=device, dtype=H.dtype)

    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except:
        H = H + 0.1 * torch.eye(N, device=device, dtype=H.dtype)
        H_inv = torch.linalg.inv(H)

    H_inv_diag = H_inv.diag()
    n_prune_per_row = int(N * sparsity)

    W_out = W.clone()
    mask = torch.ones(K, N, device=device)

    # Compute importance and select weights to prune per row
    importance = W.abs() * act_col_norms.unsqueeze(0)
    _, sorted_indices = importance.sort(dim=1)
    prune_indices = sorted_indices[:, :n_prune_per_row]
    mask.scatter_(1, prune_indices, 0)

    # Apply OBS compensation
    batch_size = 64
    for batch_start in range(0, K, batch_size):
        batch_end = min(batch_start + batch_size, K)
        batch_indices = prune_indices[batch_start:batch_end]
        batch_W = W_out[batch_start:batch_end]
        batch_pruned_weights = batch_W.gather(1, batch_indices)

        for i in range(batch_end - batch_start):
            J = batch_indices[i]
            w_J = batch_pruned_weights[i]
            H_inv_J = H_inv[:, J]
            H_inv_JJ = H_inv_diag[J]

            valid = H_inv_JJ.abs() > 1e-10
            scale = torch.zeros_like(w_J)
            scale[valid] = w_J[valid] / H_inv_JJ[valid]
            delta = -H_inv_J @ scale
            delta[J] = 0
            W_out[batch_start + i] = W_out[batch_start + i] + delta

    W_out = W_out * mask
    return W_out.to(dtype), mask


# =============================================================================
# CPR-SINQ Functions
# =============================================================================

def compute_column_errors(W, nbits=5, tile_size=128):
    """Compute per-column quantization error."""
    W_float = W.float()
    n_rows, n_cols = W.shape
    device = W.device

    col_errors = torch.zeros(n_cols, device=device)
    n_col_tiles = (n_cols + tile_size - 1) // tile_size

    for t in range(n_col_tiles):
        c_start = t * tile_size
        c_end = min(c_start + tile_size, n_cols)
        tile = W_float[:, c_start:c_end]

        min_max = (0, 2**nbits - 1)
        q, s1, s2, z = quantize_dual_scale_shift(tile, min_max, method='sinq')
        tile_deq = (q.float() - z) * s1 * s2

        error_sq = (tile - tile_deq) ** 2
        col_errors[c_start:c_end] = error_sq.sum(dim=0)

    return col_errors


def quantize_cpr_sparse(W, activations=None, sparsity=0.0, high_frac=0.25,
                        high_bits=6, low_bits=5, tile_size=128):
    """
    CPR-SINQ with optional sparsity.

    1. Apply sparsity (if > 0) with OBS compensation
    2. Identify high-error columns
    3. Quantize high-error columns at high_bits
    4. Quantize low-error columns at low_bits

    Returns quantized data that can be dequantized.
    """
    device = W.device
    dtype = W.dtype
    K, N = W.shape

    # Step 1: Apply sparsity if requested
    if sparsity > 0 and activations is not None:
        W_pruned, mask = batched_row_obs_prune(W, activations, sparsity)
    else:
        W_pruned = W.clone()
        mask = torch.ones(K, N, device=device)

    W_float = W_pruned.float()

    # Step 2: Compute column errors at low precision
    col_errors = compute_column_errors(W_pruned, nbits=low_bits, tile_size=tile_size)

    # Step 3: Identify high-error columns
    n_high = int(high_frac * N)
    _, high_indices = torch.topk(col_errors, n_high)
    high_mask_col = torch.zeros(N, dtype=torch.bool, device=device)
    high_mask_col[high_indices] = True
    low_indices = (~high_mask_col).nonzero(as_tuple=True)[0]

    # Step 4: Create permutation
    col_indices = torch.cat([high_indices, low_indices])

    # Step 5: Reorder columns
    W_perm = W_float[:, col_indices]
    W_high_region = W_perm[:, :n_high]
    W_low_region = W_perm[:, n_high:]

    # Step 6: Quantize each region using SINQ
    def quantize_region_sinq(W_region, nbits):
        """Quantize region using SINQ's dual-scale approach."""
        min_max = (0, 2**nbits - 1)
        n_cols = W_region.shape[1]
        n_tiles = (n_cols + tile_size - 1) // tile_size

        q_list, s1_list, s2_list, z_list = [], [], [], []

        for t in range(n_tiles):
            c_start = t * tile_size
            c_end = min(c_start + tile_size, n_cols)
            tile = W_region[:, c_start:c_end]

            q, s1, s2, z = quantize_dual_scale_shift(tile, min_max, method='sinq')
            q_list.append(q)
            s1_list.append(s1)
            s2_list.append(s2)
            z_list.append(z)

        return q_list, s1_list, s2_list, z_list

    high_q, high_s1, high_s2, high_z = quantize_region_sinq(W_high_region, high_bits)
    low_q, low_s1, low_s2, low_z = quantize_region_sinq(W_low_region, low_bits)

    return {
        'high_q': high_q, 'high_s1': high_s1, 'high_s2': high_s2, 'high_z': high_z,
        'low_q': low_q, 'low_s1': low_s1, 'low_s2': low_s2, 'low_z': low_z,
        'col_indices': col_indices,
        'n_high': n_high,
        'high_bits': high_bits,
        'low_bits': low_bits,
        'shape': (K, N),
        'mask': mask,
        'sparsity': sparsity,
    }


def dequantize_cpr_sparse(quant_data):
    """Dequantize CPR-Sparse quantized weights."""
    def dequant_region(q_list, s1_list, s2_list, z_list):
        tiles = []
        for q, s1, s2, z in zip(q_list, s1_list, s2_list, z_list):
            tile = (q.float() - z) * s1 * s2
            tiles.append(tile)
        return torch.cat(tiles, dim=1)

    W_high = dequant_region(
        quant_data['high_q'], quant_data['high_s1'],
        quant_data['high_s2'], quant_data['high_z']
    )
    W_low = dequant_region(
        quant_data['low_q'], quant_data['low_s1'],
        quant_data['low_s2'], quant_data['low_z']
    )

    W_perm = torch.cat([W_high, W_low], dim=1)

    # Inverse permute
    col_indices = quant_data['col_indices']
    W = torch.zeros_like(W_perm)
    W[:, col_indices] = W_perm

    # Apply sparsity mask
    mask = quant_data['mask']
    W = W * mask.float()

    return W


# =============================================================================
# Linear Layer Classes
# =============================================================================

class CPRSparseLinear(nn.Module):
    """Linear layer with CPR-Sparse quantization."""

    def __init__(self, quant_data, bias, device):
        super().__init__()
        self.quant_data = quant_data
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._W_cached = None
        self._device = device

    def forward(self, x):
        if self._W_cached is None:
            self._W_cached = dequantize_cpr_sparse(self.quant_data).to(x.dtype).to(self._device)
        out = torch.matmul(x, self._W_cached.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class SINQSparseLinear(nn.Module):
    """Linear layer with SINQ + Sparse quantization (4-bit uniform)."""

    def __init__(self, W_q, scales, zeros, mask, scale2, bias, meta):
        super().__init__()
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('scale2', scale2)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.meta = meta
        self._W_cached = None

    def forward(self, x):
        if self._W_cached is None:
            # Standard SINQ dequantization with mask
            W_deq = (self.W_q.float() - self.zeros.float()) * self.scales.float()
            W_deq = W_deq * self.scale2.float()
            W_deq = W_deq * self.mask.float()
            self._W_cached = W_deq.to(x.dtype)
        out = torch.matmul(x, self._W_cached.t())
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# Model Quantization Functions
# =============================================================================

def collect_activations(model, calibration_data, device):
    """Collect activations for all linear layers."""
    layer_activations = {}
    hooks = []
    activation_cache = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_cache:
                activation_cache[name] = []
            activation_cache[name].append(input[0].detach().cpu())
        return hook_fn

    for layer_idx, layer in enumerate(model.model.layers):
        layer = layer.to(device)
        for attr_path in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                         'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            parts = attr_path.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            if isinstance(module, nn.Linear):
                name = f'layer_{layer_idx}.{attr_path}'
                hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i in range(min(4, len(calibration_data))):
            batch = calibration_data[i:i+1].to(device)
            try:
                model(batch)
            except:
                pass

    for h in hooks:
        h.remove()

    for name, acts in activation_cache.items():
        layer_activations[name] = torch.cat(acts, dim=0)

    return layer_activations


def quantize_model_cpr_sparse(model, calibration_data, sparsity=0.0,
                               high_frac=0.25, high_bits=6, low_bits=5, device='cuda'):
    """Apply CPR-Sparse quantization to model."""
    print(f"CPR-Sparse: sparsity={sparsity*100:.0f}%, high={high_frac*100:.0f}%@{high_bits}bit, low={100-high_frac*100:.0f}%@{low_bits}bit")

    layer_activations = collect_activations(model, calibration_data, device)

    for layer_idx, layer in enumerate(model.model.layers):
        layer = layer.to(device)

        for attr_path in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                         'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            parts = attr_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            linear = getattr(parent, parts[-1])

            if not isinstance(linear, nn.Linear):
                continue

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            quant_data = quantize_cpr_sparse(
                W, activations, sparsity=sparsity,
                high_frac=high_frac, high_bits=high_bits, low_bits=low_bits
            )

            new_layer = CPRSparseLinear(quant_data, bias, device)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        model.model.layers[layer_idx] = layer
        gc.collect()
        torch.cuda.empty_cache()

    return model


def quantize_model_sinq_sparse(model, calibration_data, sparsity=0.0, nbits=4, device='cuda'):
    """Apply SINQ-Sparse (uniform bit-width) quantization to model."""
    print(f"SINQ-Sparse: sparsity={sparsity*100:.0f}%, bits={nbits}")

    layer_activations = collect_activations(model, calibration_data, device)

    for layer_idx, layer in enumerate(model.model.layers):
        layer = layer.to(device)

        for attr_path in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                         'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            parts = attr_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            linear = getattr(parent, parts[-1])

            if not isinstance(linear, nn.Linear):
                continue

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            # Apply sparsity with OBS compensation
            if sparsity > 0 and activations is not None:
                W_pruned, mask = batched_row_obs_prune(W, activations, sparsity)
            else:
                W_pruned = W.clone()
                K, N = W.shape
                mask = torch.ones(K, N, device=device)

            # Quantize using SINQ (use quantize_dual_scale_shift for correct handling)
            W_float = W_pruned.float()
            min_max = [0, 2**nbits - 1]
            q, scales, scale2, zeros = quantize_dual_scale_shift(W_float, min_max, method='sinq')

            # quantize_dual_scale_shift returns:
            # - q: quantized weights [K, N]
            # - scales: combined row scale (s1 * mu2) [K, 1]
            # - scale2: column scale (mu1) [N]
            # - zeros: zero points [K, 1]
            scales = scales.to(q.dtype)
            scale2 = scale2.to(q.dtype)
            zeros = zeros.to(q.dtype)

            meta = {'nbits': nbits, 'sparsity': sparsity, 'shape': W.shape}

            new_layer = SINQSparseLinear(q, scales, zeros, mask.to(q.dtype), scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        model.model.layers[layer_idx] = layer
        gc.collect()
        torch.cuda.empty_cache()

    return model


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 70)
    print("CPR-Sparse vs SINQ-Sparse Comparison")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-1.5B"

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=16)
    calibration_data = get_calibration_data(tokenizer, n_samples=16, seq_len=512)
    print(f"Test samples: {len(test_data)}, Calibration samples: {len(calibration_data)}")

    results = {}

    # =========================================================================
    # 1. FP16 Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. FP16 BASELINE")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True
    )
    ppl_fp16 = evaluate_perplexity(model, test_data, device)
    print(f"FP16 PPL: {ppl_fp16:.2f}")
    results['FP16'] = ppl_fp16

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 2. SINQ 4-bit (Dense)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SINQ 4-bit (Dense)")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu',
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model = quantize_model_sinq_sparse(model, calibration_data, sparsity=0.0, nbits=4, device=device)
    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    ppl_sinq = evaluate_perplexity(model, test_data, device)
    print(f"SINQ 4-bit PPL: {ppl_sinq:.2f} ({ppl_sinq/ppl_fp16*100:.1f}% of FP16)")
    results['SINQ_4bit'] = ppl_sinq

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 3. CPR-SINQ (25% @ 6-bit, 75% @ 5-bit, avg=5.25 bits) - Dense
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. CPR-SINQ (25%@6bit, 75%@5bit) - Dense")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu',
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model = quantize_model_cpr_sparse(
        model, calibration_data, sparsity=0.0,
        high_frac=0.25, high_bits=6, low_bits=5, device=device
    )
    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    ppl_cpr = evaluate_perplexity(model, test_data, device)
    print(f"CPR-SINQ (5.25 avg bits) PPL: {ppl_cpr:.2f} ({ppl_cpr/ppl_fp16*100:.1f}% of FP16)")
    results['CPR_5.25bit'] = ppl_cpr

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. SINQ-Sparse 4-bit + 35% Sparsity
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. SINQ-Sparse 4-bit + 35% Sparsity")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu',
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model = quantize_model_sinq_sparse(model, calibration_data, sparsity=0.35, nbits=4, device=device)
    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    ppl_sinq_sparse = evaluate_perplexity(model, test_data, device)
    print(f"SINQ-Sparse 4-bit+35% PPL: {ppl_sinq_sparse:.2f} ({ppl_sinq_sparse/ppl_fp16*100:.1f}% of FP16, {ppl_sinq_sparse/ppl_sinq*100:.1f}% of SINQ)")
    results['SINQ_4bit_35sparse'] = ppl_sinq_sparse

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 5. CPR-Sparse (5.25 avg bits + 35% Sparsity)
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. CPR-Sparse (5.25 avg bits + 35% Sparsity)")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu',
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model = quantize_model_cpr_sparse(
        model, calibration_data, sparsity=0.35,
        high_frac=0.25, high_bits=6, low_bits=5, device=device
    )
    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    ppl_cpr_sparse = evaluate_perplexity(model, test_data, device)
    print(f"CPR-Sparse (5.25bit+35%) PPL: {ppl_cpr_sparse:.2f} ({ppl_cpr_sparse/ppl_fp16*100:.1f}% of FP16, {ppl_cpr_sparse/ppl_cpr*100:.1f}% of CPR)")
    results['CPR_5.25bit_35sparse'] = ppl_cpr_sparse

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<35} | {'PPL':>8} | {'vs FP16':>10} | {'Compression':>12}")
    print("-" * 70)

    # Compression calculation:
    # FP16 = 16 bits/weight
    # 4-bit = 4 bits/weight -> 4x compression
    # 5.25-bit = 5.25 bits/weight -> 3.05x compression
    # 35% sparsity = 1.54x additional compression (on non-zero weights)

    compression_info = {
        'FP16': ('16 bits', 1.0),
        'SINQ_4bit': ('4 bits', 16/4),
        'CPR_5.25bit': ('5.25 bits', 16/5.25),
        'SINQ_4bit_35sparse': ('4bit+35%', (16/4) * (1/0.65)),  # effective compression
        'CPR_5.25bit_35sparse': ('5.25bit+35%', (16/5.25) * (1/0.65)),
    }

    for name, ppl in results.items():
        vs_fp16 = ppl / ppl_fp16 * 100
        bits, compress = compression_info[name]
        print(f"{name:<35} | {ppl:>8.2f} | {vs_fp16:>9.1f}% | {compress:>10.2f}x ({bits})")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\n1. SINQ 4-bit baseline: PPL={results['SINQ_4bit']:.2f}")
    print(f"2. CPR multi-precision improves by: {(1 - results['CPR_5.25bit']/results['SINQ_4bit'])*100:.1f}% (but uses more bits)")
    print(f"3. SINQ-Sparse 35% degrades by: {(results['SINQ_4bit_35sparse']/results['SINQ_4bit'] - 1)*100:.1f}%")
    print(f"4. CPR-Sparse 35% degrades by: {(results['CPR_5.25bit_35sparse']/results['CPR_5.25bit'] - 1)*100:.1f}%")

    # Effective bits comparison
    eff_bits_sinq_sparse = 4 * 0.65  # 4-bit * 65% non-zero = 2.6 effective bits
    eff_bits_cpr_sparse = 5.25 * 0.65  # 5.25-bit * 65% non-zero = 3.41 effective bits

    print(f"\nEffective bits (accounting for sparsity):")
    print(f"  SINQ-Sparse: {eff_bits_sinq_sparse:.2f} effective bits/weight")
    print(f"  CPR-Sparse:  {eff_bits_cpr_sparse:.2f} effective bits/weight")

    # Is CPR-Sparse worth the extra bits?
    ppl_improvement = (results['SINQ_4bit_35sparse'] - results['CPR_5.25bit_35sparse']) / results['SINQ_4bit_35sparse'] * 100
    bit_overhead = (eff_bits_cpr_sparse - eff_bits_sinq_sparse) / eff_bits_sinq_sparse * 100

    print(f"\nCPR-Sparse vs SINQ-Sparse (both 35% sparse):")
    print(f"  PPL improvement: {ppl_improvement:.1f}%")
    print(f"  Bit overhead: {bit_overhead:.1f}%")


if __name__ == '__main__':
    main()
