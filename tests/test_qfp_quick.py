"""
Quick test: Quantization-Friendly Pruning (QFP)

KEY INSIGHT: Not all weights quantize equally well.
- Some weights quantize with low error (their value maps well to grid)
- Some weights quantize with high error (fall between grid points)

Current pruning: Prune low-importance weights
QFP: Prune weights that are BOTH low-importance AND quantize well

Why this might help:
- Weights that quantize well → small error if kept
- Weights that quantize poorly → large error if kept
- If we must prune, prefer weights that would've been accurate anyway
- Save "hard to quantize" weights for the quantization budget

Formula:
importance_qfp = importance_base * (1 + α * quant_error)

Where quant_error = |W - Q_dequant| / |W| for each weight
- High quant_error → weight is hard to quantize → keep it (high importance)
- Low quant_error → weight quantizes well → safe to prune (low importance)
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_qfp():
    print("="*70)
    print("QUANTIZATION-FRIENDLY PRUNING (QFP) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('up_proj', model.model.layers[0].mlp.up_proj.weight.data),
        ('down_proj', model.model.layers[0].mlp.down_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
        ('k_proj', model.model.layers[0].self_attn.k_proj.weight.data),
        ('v_proj', model.model.layers[0].self_attn.v_proj.weight.data),
    ]

    sparsity = 0.35
    nbits = 4
    group_size = 64

    for name, W_orig in test_layers:
        W = W_orig.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]

        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Standard importance (inverse μ)
        act_norms = torch.norm(X, dim=0)
        importance_base = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Compute per-weight quantization error
        # First quantize the full (unpruned) weight matrix
        Q_full, s_full, z_full, _ = quantize_rtn(W_norm, min_max, group_size=group_size)
        Q_grouped = Q_full.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z_full) * s_full
        W_deq_norm_flat = W_deq_norm.view(K, N)

        # Per-weight quantization error (relative)
        quant_error = (W_norm - W_deq_norm_flat).abs() / (W_norm.abs() + 1e-8)

        print(f"\n{name}:")
        print(f"  Quant error - mean: {quant_error.mean():.4f}, std: {quant_error.std():.4f}")
        print(f"  Quant error - range: [{quant_error.min():.4f}, {quant_error.max():.4f}]")

        # Test different QFP formulations
        n_prune = int(K * N * sparsity)

        def evaluate(importance, label):
            threshold = importance.view(-1).sort().values[n_prune]
            mask = (importance > threshold).float()

            W_sparse = W_norm * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            return ((Y_ref - Y_approx) ** 2).mean().item()

        mse_base = evaluate(importance_base, "Standard")

        # QFP v1: Boost importance of hard-to-quantize weights
        alphas = [0.5, 1.0, 2.0, 5.0]
        print(f"  Standard MSE: {mse_base:.6f}")

        for alpha in alphas:
            importance_qfp = importance_base * (1 + alpha * quant_error)
            mse_qfp = evaluate(importance_qfp, f"QFP α={alpha}")
            improvement = (mse_base - mse_qfp) / mse_base * 100
            print(f"  QFP (α={alpha}): {mse_qfp:.6f} ({improvement:+.2f}%)")

        # QFP v2: Inverse - prefer pruning high-error weights
        # (counterintuitive but let's test)
        print("  --- Inverse (prune high-error) ---")
        for alpha in [0.5, 1.0]:
            importance_inv = importance_base / (1 + alpha * quant_error + 1e-6)
            mse_inv = evaluate(importance_inv, f"QFP-inv α={alpha}")
            improvement = (mse_base - mse_inv) / mse_base * 100
            print(f"  QFP-inv (α={alpha}): {mse_inv:.6f} ({improvement:+.2f}%)")

        # QFP v3: Use absolute error instead of relative
        quant_error_abs = (W_norm - W_deq_norm_flat).abs()
        print("  --- Absolute error weighting ---")
        for alpha in [10.0, 100.0]:
            importance_abs = importance_base * (1 + alpha * quant_error_abs)
            mse_abs = evaluate(importance_abs, f"QFP-abs α={alpha}")
            improvement = (mse_base - mse_abs) / mse_base * 100
            print(f"  QFP-abs (α={alpha}): {mse_abs:.6f} ({improvement:+.2f}%)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_qfp()
