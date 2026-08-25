"""
Sparse-Aware Sinkhorn Normalization Test

Compare two approaches:
1. Standard Sinkhorn: Compute μ factors on all weights (including zeros after pruning)
2. Sparse-Aware Sinkhorn: Compute μ factors only on non-zero weights

Key insight: After pruning, many weights are zero. Standard Sinkhorn includes these
zeros in variance calculation, which distorts the μ factors.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_activation_norms,
    compute_importance_scores,
)


def sinkhorn_log_sparse_aware(W, mask, order=16, eps=1e-4):
    """
    Sparse-Aware Sinkhorn: Computes normalization factors only on non-zero weights.

    Args:
        W: Weight matrix [K, N]
        mask: Binary mask [K, N] where 1 = kept, 0 = pruned
        order: Number of Sinkhorn iterations
        eps: Small constant for numerical stability

    Returns:
        W_norm: Normalized weights
        mu1: Column scaling factors
        mu2: Row scaling factors
    """
    W = W.float()
    mask = mask.float()
    K, N = W.shape
    device = W.device

    # Initialize log-scale factors
    log_mu1 = torch.zeros(N, device=device)
    log_mu2 = torch.zeros(K, device=device)

    # Identify empty rows/columns (all zeros after masking)
    row_count = mask.sum(dim=1)
    col_count = mask.sum(dim=0)
    empty_rows = (row_count == 0)
    empty_cols = (col_count == 0)

    # Use tighter clamp to prevent overflow (exp(10) ≈ 22000, exp(-10) ≈ 4.5e-5)
    CLAMP_MIN, CLAMP_MAX = -10, 10

    for _ in range(order):
        # Apply current scaling (clamp to prevent overflow)
        log_scale = (-log_mu2.view(-1, 1) - log_mu1.view(1, -1)).clamp(CLAMP_MIN, CLAMP_MAX)
        W_scaled = W * torch.exp(log_scale)

        # Compute MASKED variance for each row (only count non-zero entries)
        row_sum_sq = (mask * W_scaled ** 2).sum(dim=1)
        row_count_safe = row_count.clamp(min=1)
        row_sum = (mask * W_scaled).sum(dim=1)
        row_mean = row_sum / row_count_safe
        row_var = row_sum_sq / row_count_safe - row_mean ** 2
        row_std = (row_var.clamp(min=eps ** 2)).sqrt()

        # Update row factors (don't update empty rows)
        log_update = torch.log(row_std.clamp(min=eps))
        log_update[empty_rows] = 0  # Keep factor at 1 for empty rows
        log_mu2 = (log_mu2 + log_update).clamp(CLAMP_MIN, CLAMP_MAX)

        # Re-apply scaling
        log_scale = (-log_mu2.view(-1, 1) - log_mu1.view(1, -1)).clamp(CLAMP_MIN, CLAMP_MAX)
        W_scaled = W * torch.exp(log_scale)

        # Compute MASKED variance for each column
        col_sum_sq = (mask * W_scaled ** 2).sum(dim=0)
        col_count_safe = col_count.clamp(min=1)
        col_sum = (mask * W_scaled).sum(dim=0)
        col_mean = col_sum / col_count_safe
        col_var = col_sum_sq / col_count_safe - col_mean ** 2
        col_std = (col_var.clamp(min=eps ** 2)).sqrt()

        # Update column factors (don't update empty columns)
        log_update = torch.log(col_std.clamp(min=eps))
        log_update[empty_cols] = 0  # Keep factor at 1 for empty columns
        log_mu1 = (log_mu1 + log_update).clamp(CLAMP_MIN, CLAMP_MAX)

    # Final scaling factors
    mu1 = torch.exp(log_mu1)
    mu2 = torch.exp(log_mu2)

    # Normalize weights
    W_norm = W / (mu2.view(-1, 1) * mu1.view(1, -1))

    return W_norm, mu1, mu2


def quantize_with_sinkhorn(W, X, sparsity, nbits=4, group_size=64, device='cuda', sparse_aware=False):
    """
    Quantize with either standard or sparse-aware Sinkhorn.

    Uses simple Wanda importance for pruning (no OBS compensation to avoid numerical issues).
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Compute Wanda importance
    X_flat = X.float().view(-1, X.shape[-1])[:256].to(device)
    act_norms = torch.norm(X_flat, dim=0)
    importance = W.abs() * act_norms.view(1, -1)

    # Create mask
    n_prune = int(K * N * sparsity)
    if n_prune > 0:
        flat_imp = importance.view(-1)
        threshold = torch.kthvalue(flat_imp, n_prune).values
        mask = (flat_imp > threshold).view(K, N).float()
    else:
        mask = torch.ones(K, N, device=device)

    # Apply mask
    W_sparse = W * mask

    # Compute Sinkhorn factors
    if sparse_aware:
        # Sparse-Aware: Only consider non-zero weights
        W_norm, mu1, mu2 = sinkhorn_log_sparse_aware(W_sparse, mask, order=16)
    else:
        # Standard: Compute on all weights (including zeros)
        # Add tiny noise to avoid zero-row/col issues
        W_for_sinkhorn = W_sparse.clone()
        W_for_sinkhorn[W_for_sinkhorn.abs() < 1e-10] = 1e-8
        W_norm, mu1, mu2 = sinkhorn_log(W_for_sinkhorn, order=16)
        # Re-normalize original sparse weights with these factors
        W_norm = W_sparse / (mu2.view(-1, 1) * mu1.view(1, -1))

    # Quantize
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    if len(scales.shape) == 3:
        scales = scales * mu2.view(-1, 1, 1)
    else:
        scales = scales * mu2.view(-1, 1)
    scale2 = mu1

    q = q * mask.to(q.dtype)

    method = 'sparse_aware_sinkhorn' if sparse_aware else 'standard_sinkhorn'
    meta = {'sparsity': sparsity, 'method': method}

    return q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), scale2.to(orig_dtype), meta


def dequantize(W_q, scales, zeros, mask, scale2, meta):
    Q = W_q.float()
    s1 = scales.float()
    s2 = scale2.float()
    z = zeros.float()

    if len(s1.shape) == 3:
        K, N = Q.shape
        n_groups = s1.shape[1]
        group_size = N // n_groups
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = (Q_grouped - z) * s1
        W_deq = W_deq.view(K, N)
    else:
        W_deq = (Q - z) * s1
    W_deq = W_deq * s2.view(1, -1) * mask.float()
    return W_deq


def evaluate_ppl(model, tokenizer, dataset_text, seq_len=2048, n_samples=16):
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer("\n\n".join(dataset_text), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    nlls = []
    for i in range(min(n_samples, len(input_ids) // seq_len)):
        begin, end = i * seq_len, (i + 1) * seq_len
        batch = input_ids[begin:end].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            nlls.append(outputs.loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


def run_comparison():
    print("="*70)
    print("SPARSE-AWARE SINKHORN vs STANDARD SINKHORN")
    print("="*70)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]
    calib_tokens = tokenizer("\n\n".join(dataset_text[:16]), return_tensors="pt",
                             truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []

    for sparsity in [0.35, 0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        for method_name, sparse_aware in [
            ('Standard Sinkhorn', False),
            ('Sparse-Aware Sinkhorn', True),
        ]:
            print(f"\nTesting: {method_name}...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map=device, trust_remote_code=True
            )

            for layer_idx in range(len(model.model.layers)):
                layer = model.model.layers[layer_idx]

                activations_cache = {}
                def make_hook(name):
                    def hook(m, inp, out):
                        activations_cache[name] = inp[0].detach()
                    return hook

                handles = []
                for name, module in [
                    ('q_proj', layer.self_attn.q_proj),
                    ('k_proj', layer.self_attn.k_proj),
                    ('v_proj', layer.self_attn.v_proj),
                    ('o_proj', layer.self_attn.o_proj),
                    ('up_proj', layer.mlp.up_proj),
                    ('gate_proj', layer.mlp.gate_proj),
                    ('down_proj', layer.mlp.down_proj),
                ]:
                    handles.append(module.register_forward_hook(make_hook(name)))

                with torch.no_grad():
                    _ = model(calib_ids)

                for h in handles:
                    h.remove()

                for name, module in [
                    ('q_proj', layer.self_attn.q_proj),
                    ('k_proj', layer.self_attn.k_proj),
                    ('v_proj', layer.self_attn.v_proj),
                    ('o_proj', layer.self_attn.o_proj),
                    ('up_proj', layer.mlp.up_proj),
                    ('gate_proj', layer.mlp.gate_proj),
                    ('down_proj', layer.mlp.down_proj),
                ]:
                    W = module.weight.data.clone()
                    X = activations_cache[name]

                    W_q, scales, zeros, mask, scale2, meta = quantize_with_sinkhorn(
                        W, X, sparsity, nbits=4, group_size=64,
                        device=device, sparse_aware=sparse_aware
                    )
                    W_deq = dequantize(W_q, scales, zeros, mask, scale2, meta)

                    module.weight.data = W_deq.to(module.weight.dtype)

            ppl = evaluate_ppl(model, tokenizer, dataset_text)
            print(f"  {method_name}: PPL = {ppl:.2f}")

            results.append({
                'sparsity': sparsity,
                'method': method_name,
                'ppl': ppl
            })

            del model
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Sparsity':<12} {'Standard':<15} {'Sparse-Aware':<15} {'Improvement'}")
    print("-"*55)

    for sp in [0.35, 0.50, 0.60]:
        std = next(r['ppl'] for r in results if r['sparsity'] == sp and 'Standard' in r['method'])
        spa = next(r['ppl'] for r in results if r['sparsity'] == sp and 'Sparse-Aware' in r['method'])
        improvement = (std - spa) / std * 100
        print(f"{sp*100:>5.0f}%       {std:<15.2f} {spa:<15.2f} {improvement:>+.1f}%")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
Sparse-Aware Sinkhorn computes μ factors only on non-zero weights.

Theoretical motivation:
- Standard Sinkhorn computes variance over ALL entries (including zeros)
- After pruning, zeros distort the variance estimation
- Sparse-Aware Sinkhorn gives variance estimates for KEPT weights only

If Sparse-Aware shows improvement, it validates the hypothesis that
Sinkhorn factors should be adapted to the sparse structure.
""")

    return results


if __name__ == "__main__":
    run_comparison()
