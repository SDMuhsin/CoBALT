"""
Full PPL evaluation of Sparse-First Sinkhorn (SFS) approach.

This script compares:
1. Baseline SINQ-Sparse (Sinkhorn on full matrix, then prune)
2. SFS (Prune first, then Sinkhorn on sparse matrix)
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_activation_norms,
    compute_importance_scores,
    create_sparsity_mask,
    batched_row_obs_prune,
)


class SFSQuantLinear(nn.Module):
    """Sparse-First Sinkhorn Quantized Linear layer."""

    def __init__(self, W_q, scales, zeros, mask, scale2, bias, meta):
        super().__init__()
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('scale2', scale2)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.meta = meta

    def forward(self, x):
        # Dequantize
        input_dtype = x.dtype
        Q = self.W_q.float()
        z = self.zeros.float()
        s1 = self.scales.float()
        s2 = self.scale2.float()

        if len(s1.shape) == 3:
            K, N = Q.shape
            n_groups = s1.shape[1]
            group_size = N // n_groups
            Q_grouped = Q.view(K, n_groups, group_size)
            W_deq = (Q_grouped - z) * s1
            W_deq = W_deq.view(K, N)
        else:
            W_deq = (Q - z) * s1

        W_deq = W_deq * s2
        W_deq = W_deq * self.mask.float()

        # Convert to input dtype for matmul
        W_deq = W_deq.to(input_dtype)
        out = x @ W_deq.T
        if self.bias is not None:
            out = out + self.bias
        return out


def sparse_quantize_sfs(
    W, activations, sparsity, nbits, group_size=64, device='cuda',
    use_compensation=True
):
    """
    Sparse-First Sinkhorn quantization.

    Pipeline:
    1. Compute importance and create mask (using original W)
    2. Apply sparsity mask
    3. Run Sinkhorn on sparse matrix
    4. Quantize
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Step 1: Compute importance using original W
    # Use simple Wanda importance (no Sinkhorn factors yet)
    if activations is not None:
        act_norms = compute_activation_norms(activations.to(device))
        importance = W.abs() * act_norms.unsqueeze(0)
    else:
        importance = W.abs()

    # Step 2: Create mask and optionally apply OBS compensation
    if use_compensation and activations is not None:
        W_compensated, mask = batched_row_obs_prune(
            W, activations.to(device), importance, sparsity
        )
        W_sparse = W_compensated * mask
    else:
        mask = create_sparsity_mask(importance, sparsity)
        W_sparse = W * mask

    # Step 3: Run Sinkhorn on SPARSE matrix (key difference from baseline!)
    W_norm, mu1, mu2 = sinkhorn_log(W_sparse, order=16)

    # Step 4: Quantize
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Combine scales with mu2
    if len(scales.shape) == 3:
        scales = scales * mu2.unsqueeze(1)
    else:
        scales = scales * mu2

    scale2 = mu1

    # Apply mask to quantized weights
    q = q * mask.to(q.dtype)

    meta = {
        'sparsity': sparsity,
        'actual_sparsity': 1.0 - mask.sum().item() / mask.numel(),
        'nbits': nbits,
        'group_size': group_size,
        'method': 'sfs',
    }

    return q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), scale2.to(orig_dtype), meta


def apply_sfs_quantization(model, calibration_data, nbits, sparsity, device='cuda'):
    """Apply SFS quantization to model."""
    from benchmarks.benchmark_suite import collect_activations, get_layer_paths

    # Collect activations
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc=f"SFS {nbits}b/{int(sparsity*100)}%")):
        layer = layer.to(device)

        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue

            if not isinstance(linear, nn.Linear):
                continue

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            # Apply SFS quantization
            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sfs(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                device=device,
                use_compensation=True if activations is not None else False
            )

            new_layer = SFSQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
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


def evaluate_ppl(model, test_data, device='cuda', seq_len=2048):
    """Evaluate perplexity on test data."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data) - seq_len, seq_len), desc="Evaluating"):
            batch = test_data[i:i+seq_len].unsqueeze(0).to(device)

            outputs = model(batch, labels=batch)
            loss = outputs.loss

            n_tokens = batch.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            # Clear cache periodically
            if (i // seq_len + 1) % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def main():
    print("="*60)
    print("SFS Full PPL Evaluation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"
    sparsity = 0.35
    nbits = 3

    print(f"\nConfig: {model_name}, {sparsity*100:.0f}% sparsity, {nbits}-bit")

    # Load tokenizer and data
    print("\nLoading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load WikiText-2 test data
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join(dataset['text'])
    test_data = tokenizer.encode(test_text, return_tensors='pt')[0]

    # Calibration data
    calib_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    calib_text = '\n\n'.join(calib_dataset['text'][:1000])
    calibration_data = tokenizer.encode(calib_text, return_tensors='pt')[0][:8192]

    print(f"Test tokens: {len(test_data)}")
    print(f"Calibration tokens: {len(calibration_data)}")

    # Load fresh model for SFS
    print("\nLoading model for SFS evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply SFS quantization
    print("\nApplying SFS quantization...")
    model = apply_sfs_quantization(
        model, calibration_data, nbits, sparsity, device
    )

    # Evaluate PPL
    print("\nEvaluating perplexity...")
    ppl_sfs = evaluate_ppl(model, test_data, device)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"SFS PPL: {ppl_sfs:.2f}")
    print(f"\nBaseline SINQ-Sparse PPL (reference): 28.10")
    print(f"Improvement: {(28.10 - ppl_sfs) / 28.10 * 100:+.2f}%")

    if ppl_sfs < 28.10:
        print("\n*** SUCCESS: SFS beats baseline! ***")
    else:
        print("\n*** SFS did not beat baseline ***")

    return ppl_sfs


if __name__ == '__main__':
    ppl = main()
