"""
Hypothesis 40: Quantization-Error-Minimizing Sparsity (QEMS)

Core insight: Standard importance minimizes output error: ||X @ W' - X @ W||
What if we minimize quantization error instead: ||Q(W') - W'||?

These are DIFFERENT objectives!
- High output error: wrong activations flow to next layer
- High quant error: weights are poorly represented

QEMS: Prune weights that cause the most quantization error.

Mathematical formulation:
- For each weight, compute: quant_error[i,j] = |Q(W[i,j]) - W[i,j]|
- Also consider: how much this error gets amplified by μ during dequant
- Total contribution to error: |quant_error| × μ₁ × μ₂

Prune weights with HIGH quant_error × μ (they're poorly quantized AND amplified).
Keep weights with LOW quant_error × μ (they're well quantized OR damped).

This is OPPOSITE to importance-based! We're pruning BAD weights, not unimportant ones.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_qems():
    print("="*70)
    print("QUANTIZATION-ERROR-MINIMIZING SPARSITY (QEMS) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
    ]

    sparsity = 0.35
    nbits = 4
    group_size = 64

    for name, W_orig in test_layers:
        print(f"\n{'='*70}")
        print(f"Layer: {name}")
        print("="*70)

        W = W_orig.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]
        n_prune = int(K * N * sparsity)

        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Standard importance
        act_norms = torch.norm(X, dim=0)
        importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Quantize full matrix to get per-weight quant error
        Q_full, s_full, z_full, _ = quantize_rtn(W_norm, min_max, group_size=group_size)
        Q_grouped = Q_full.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z_full) * s_full
        W_deq_norm_flat = W_deq_norm.view(K, N)

        # Per-weight quantization error (in normalized space)
        quant_error = (W_norm - W_deq_norm_flat).abs()

        # Error amplification factor (μ scaling during dequant)
        mu_scale = mu1.unsqueeze(0) * mu2.unsqueeze(1)

        # Total contribution to reconstruction error
        # Weights with high (quant_error × μ) contribute most error
        total_error_contrib = quant_error * mu_scale

        print(f"Quant error stats: mean={quant_error.mean():.6f}, std={quant_error.std():.6f}")
        print(f"Error contrib stats: mean={total_error_contrib.mean():.6f}, std={total_error_contrib.std():.6f}")

        # Correlation between importance and error contribution
        corr = torch.corrcoef(torch.stack([
            importance_std.flatten(),
            total_error_contrib.flatten()
        ]))[0, 1].item()
        print(f"Correlation(importance, error_contrib): {corr:.4f}")

        def evaluate_mask(mask, label):
            W_sparse = W_norm * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            mse = ((Y_ref - Y_approx) ** 2).mean().item()

            # Overlap with standard
            threshold_std = importance_std.view(-1).sort().values[n_prune]
            mask_std = (importance_std > threshold_std).float()
            overlap = (mask * mask_std).sum() / mask_std.sum()

            return mse, overlap.item()

        # Standard mask
        threshold_std = importance_std.view(-1).sort().values[n_prune]
        mask_std = (importance_std > threshold_std).float()
        mse_std, _ = evaluate_mask(mask_std, "Standard")
        print(f"\nStandard MSE: {mse_std:.6f}")

        # === QEMS v1: Prune high-error weights ===
        # Low importance_qems = high error = should be pruned
        importance_qems_v1 = -total_error_contrib  # Negative so high error = low importance
        threshold_v1 = importance_qems_v1.view(-1).sort().values[n_prune]
        mask_qems_v1 = (importance_qems_v1 > threshold_v1).float()
        mse_v1, overlap_v1 = evaluate_mask(mask_qems_v1, "QEMS v1")
        print(f"QEMS v1 (prune high error): MSE={mse_v1:.6f} ({(mse_std-mse_v1)/mse_std*100:+.2f}%), overlap={overlap_v1:.1%}")

        # === QEMS v2: Combine importance and error ===
        # Keep weights that are important AND well-quantized
        # importance / (1 + α × error)
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            importance_qems_v2 = importance_std / (1 + alpha * total_error_contrib)
            threshold_v2 = importance_qems_v2.view(-1).sort().values[n_prune]
            mask_qems_v2 = (importance_qems_v2 > threshold_v2).float()
            mse_v2, overlap_v2 = evaluate_mask(mask_qems_v2, f"QEMS v2 α={alpha}")
            print(f"QEMS v2 (α={alpha}): MSE={mse_v2:.6f} ({(mse_std-mse_v2)/mse_std*100:+.2f}%), overlap={overlap_v2:.1%}")

        # === QEMS v3: Importance × (1 - normalized_error) ===
        # Downweight importance for poorly-quantized weights
        normalized_error = (quant_error - quant_error.min()) / (quant_error.max() - quant_error.min() + 1e-8)
        for alpha in [0.5, 1.0]:
            importance_qems_v3 = importance_std * (1 - alpha * normalized_error)
            threshold_v3 = importance_qems_v3.view(-1).sort().values[n_prune]
            mask_qems_v3 = (importance_qems_v3 > threshold_v3).float()
            mse_v3, overlap_v3 = evaluate_mask(mask_qems_v3, f"QEMS v3 α={alpha}")
            print(f"QEMS v3 (α={alpha}): MSE={mse_v3:.6f} ({(mse_std-mse_v3)/mse_std*100:+.2f}%), overlap={overlap_v3:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_qems()
