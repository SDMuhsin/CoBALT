"""
Quick test: Variance-Aware Importance (VAI)

Current: importance = |W| × ||X|| / (μ₁ × μ₂)

VAI adds activation variance:
importance = |W| × ||X|| × (1 + α × CV(X)) / (μ₁ × μ₂)

Where CV(X) = std(X) / mean(|X|) is coefficient of variation.

Intuition: Columns with high variance have unpredictable importance.
Keep them to handle worst-case inputs.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_vai():
    print("="*70)
    print("VARIANCE-AWARE IMPORTANCE (VAI) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}]")

    # Create realistic activations (multiple samples)
    batch = 256
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Activation statistics
    act_norms = torch.norm(X, dim=0)  # ||X_j||
    act_var = X.var(dim=0)  # Var(X_j) across batch
    act_mean_abs = X.abs().mean(dim=0)  # mean(|X_j|)
    act_cv = (act_var.sqrt() / (act_mean_abs + 1e-6))  # Coefficient of variation

    print(f"\n--- Activation Statistics ---")
    print(f"CV mean: {act_cv.mean():.4f}, std: {act_cv.std():.4f}")
    print(f"CV range: [{act_cv.min():.4f}, {act_cv.max():.4f}]")

    # Standard importance (inverse μ)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # VAI: Add variance weighting
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]

    print(f"\n--- Results (35% sparsity) ---")

    sparsity = 0.35
    n_prune = int(K * N * sparsity)
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]
    n_groups = N // group_size
    Y_ref = X @ W.T

    def eval_importance(importance, name):
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        Y_approx = X @ W_deq.T
        out_mse = ((Y_ref - Y_approx) ** 2).mean().item()
        return out_mse

    out_std = eval_importance(importance_std, "Standard")
    print(f"Standard (α=0): {out_std:.6f}")

    for alpha in alphas[1:]:
        # VAI importance
        variance_factor = 1 + alpha * act_cv
        importance_vai = importance_std * variance_factor.unsqueeze(0)
        out_vai = eval_importance(importance_vai, f"VAI α={alpha}")
        improvement = (out_std - out_vai) / out_std * 100
        print(f"VAI (α={alpha}):    {out_vai:.6f} ({improvement:+.2f}%)")

    # Also test inverse variance (penalize high-variance columns)
    print(f"\n--- Inverse Variance (penalize high-CV columns) ---")
    for alpha in [0.5, 1.0]:
        inv_var_factor = 1 / (1 + alpha * act_cv)
        importance_inv = importance_std * inv_var_factor.unsqueeze(0)
        out_inv = eval_importance(importance_inv, f"InvVar α={alpha}")
        improvement = (out_std - out_inv) / out_std * 100
        print(f"InvVar (α={alpha}): {out_inv:.6f} ({improvement:+.2f}%)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_vai()
