"""
Hypothesis 37: Simulated Quantization Importance (SQI)

Core insight: We prune based on |W|, but what matters is |Q(W)| after quantization.
Some weights get quantized to small values anyway - pruning them loses little.
Some weights get amplified by quantization grid alignment - they matter more.

Mathematical formulation:
importance_SQI = |Q(W_norm)| × ||X|| / (μ₁ × μ₂)

vs standard:
importance_STD = |W_norm| × ||X|| / (μ₁ × μ₂)

The difference: We use quantized weights instead of raw weights.

Why this might help:
- Weights that quantize to ~0 have low actual contribution after quantization
- Weights that quantize to high values have high actual contribution
- Pruning should account for this transformation
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_sqi():
    print("="*70)
    print("SIMULATED QUANTIZATION IMPORTANCE (SQI) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
        ('k_proj', model.model.layers[0].self_attn.k_proj.weight.data),
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

        act_norms = torch.norm(X, dim=0)

        # === Standard importance (pre-quantization) ===
        importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # === SQI: Quantize first, then compute importance ===
        # Quantize the FULL matrix (no sparsity)
        Q_full, s_full, z_full, _ = quantize_rtn(W_norm, min_max, group_size=group_size)
        Q_grouped = Q_full.view(K, n_groups, group_size)
        W_deq_norm_full = (Q_grouped - z_full) * s_full
        W_deq_full = W_deq_norm_full.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        # SQI importance: Based on dequantized weights
        importance_sqi = W_deq_full.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Alternative: Use the CHANGE caused by quantization
        # Weights that change a lot are "sensitive" to quantization
        quant_change = (W_deq_full - W).abs()
        importance_change = importance_std / (quant_change + 1e-6)  # Keep weights that don't change much

        # Alternative 2: Combine standard with quantized
        importance_combined = importance_std * (1 + W_deq_full.abs() / (W.abs() + 1e-6))

        def evaluate_mask(importance, label):
            threshold = importance.view(-1).sort().values[n_prune]
            mask = (importance > threshold).float()

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

        mse_std, _ = evaluate_mask(importance_std, "Standard")
        mse_sqi, overlap_sqi = evaluate_mask(importance_sqi, "SQI")
        mse_change, overlap_change = evaluate_mask(importance_change, "Change-based")
        mse_combined, overlap_combined = evaluate_mask(importance_combined, "Combined")

        print(f"Standard:      MSE = {mse_std:.6f}")
        print(f"SQI:           MSE = {mse_sqi:.6f} ({(mse_std-mse_sqi)/mse_std*100:+.2f}%), overlap={overlap_sqi:.1%}")
        print(f"Change-based:  MSE = {mse_change:.6f} ({(mse_std-mse_change)/mse_std*100:+.2f}%), overlap={overlap_change:.1%}")
        print(f"Combined:      MSE = {mse_combined:.6f} ({(mse_std-mse_combined)/mse_std*100:+.2f}%), overlap={overlap_combined:.1%}")

        # Analyze the difference between standard and SQI importance
        correlation = torch.corrcoef(torch.stack([
            importance_std.flatten(),
            importance_sqi.flatten()
        ]))[0, 1].item()
        print(f"\nCorrelation(STD, SQI): {correlation:.4f}")

        # What fraction of weights change ranking significantly?
        rank_std = importance_std.flatten().argsort().argsort().float()
        rank_sqi = importance_sqi.flatten().argsort().argsort().float()
        rank_diff = (rank_std - rank_sqi).abs()
        significant_change = (rank_diff > K * N * 0.1).float().mean().item()  # >10% rank change
        print(f"Weights with >10% rank change: {significant_change:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_sqi()
