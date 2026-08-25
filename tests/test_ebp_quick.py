"""
Quick test: Error-Based Pruning (EBP)

THEORETICAL INSIGHT:
For each weight, we have two choices:
1. KEEP: Error = Q(w) - w (quantization error)
2. PRUNE: Error = -w (pruning error = full weight magnitude)

The optimal decision:
- If |Q(w) - w| < |w|: KEEP (quantization has smaller error)
- If |Q(w) - w| > |w|: PRUNE (pruning has smaller error)

This gives a fundamentally different criterion from Wanda!

EBP importance: |w| / |Q(w) - w|
High = keep (weight large relative to quant error)
Low = prune (quant error large relative to weight)

Combined with activation norms: EBP × ||X||
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_ebp():
    print("="*70)
    print("ERROR-BASED PRUNING (EBP) QUICK TEST")
    print("="*70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real weights
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}]")

    # Create activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

    # Compute Sinkhorn on weights
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Compute activation norms
    act_norms = torch.norm(X, dim=0)

    # First, quantize to get Q(w)
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]
    n_groups = N // group_size

    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Dequantize to get quantized weights
    Q_grouped = Q.view(K, n_groups, group_size)
    W_q_norm = (Q_grouped - zeros) * scales
    W_q_norm = W_q_norm.view(K, N)

    # Quantization error (in normalized space)
    quant_error = (W_norm - W_q_norm).abs()

    # Standard importance (inverse μ)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # EBP importance: |W_norm| / |Q(W_norm) - W_norm|
    # High = keep, Low = prune
    ebp_ratio = W_norm.abs() / (quant_error + 1e-8)

    # EBP with activation norms
    importance_ebp = ebp_ratio * act_norms.unsqueeze(0)

    # Combined: EBP × inverse μ (use both signals)
    importance_combined = importance_ebp / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Hybrid: Average of standard and EBP (normalized)
    importance_std_norm = (importance_std - importance_std.mean()) / importance_std.std()
    importance_ebp_norm = (importance_ebp - importance_ebp.mean()) / importance_ebp.std()
    importance_hybrid = importance_std_norm + importance_ebp_norm

    # Create masks (35% sparsity)
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_ebp = create_mask(importance_ebp)
    mask_combined = create_mask(importance_combined)
    mask_hybrid = create_mask(importance_hybrid)

    # Evaluate
    Y_ref = X @ W.T

    def evaluate_mask(mask, name):
        # Apply mask and re-quantize
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        # Output MSE
        Y_approx = X @ W_deq.T
        out_mse = ((Y_ref - Y_approx) ** 2).mean().item()

        return out_mse

    out_std = evaluate_mask(mask_std, "Standard")
    out_ebp = evaluate_mask(mask_ebp, "EBP")
    out_combined = evaluate_mask(mask_combined, "Combined")
    out_hybrid = evaluate_mask(mask_hybrid, "Hybrid")

    print(f"\n--- Output MSE Results ---")
    print(f"Standard (inverse μ):    {out_std:.6f}")
    print(f"EBP only:                {out_ebp:.6f} ({(out_std-out_ebp)/out_std*100:+.2f}%)")
    print(f"EBP × inverse μ:         {out_combined:.6f} ({(out_std-out_combined)/out_std*100:+.2f}%)")
    print(f"Hybrid (avg):            {out_hybrid:.6f} ({(out_std-out_hybrid)/out_std*100:+.2f}%)")

    # Analyze the EBP ratio distribution
    print(f"\n--- EBP Ratio Analysis ---")
    print(f"Mean: {ebp_ratio.mean():.4f}")
    print(f"Median: {ebp_ratio.median():.4f}")
    print(f"Fraction where |w| < |quant_error| (prune better): {(ebp_ratio < 1).float().mean():.2%}")

    # Mask overlap
    overlap_ebp = (mask_std * mask_ebp).sum() / mask_std.sum()
    overlap_combined = (mask_std * mask_combined).sum() / mask_std.sum()
    overlap_hybrid = (mask_std * mask_hybrid).sum() / mask_std.sum()

    print(f"\n--- Mask Overlap with Standard ---")
    print(f"EBP: {overlap_ebp:.1%}")
    print(f"Combined: {overlap_combined:.1%}")
    print(f"Hybrid: {overlap_hybrid:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_ebp()
