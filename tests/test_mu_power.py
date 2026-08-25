"""
Quick test: μ-Power Scaling

Current importance: |W| × ||X|| / (μ₁ × μ₂)

What if we use different power:
importance = |W| × ||X|| / (μ₁^α × μ₂^α)

- α = 1: Current baseline
- α > 1: Stronger penalty for high-μ weights
- α < 1: Weaker penalty for high-μ weights
- α = 0: Just |W| × ||X|| (ignore μ entirely)

This tests whether the μ^1 scaling is optimal.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_mu_power():
    print("="*70)
    print("μ-POWER SCALING TEST")
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
        mu2_flat = mu2.squeeze()

        act_norms = torch.norm(X, dim=0)

        def evaluate_alpha(alpha):
            # Importance with μ^alpha scaling
            if alpha == 0:
                importance = W.abs() * act_norms.unsqueeze(0)
            else:
                mu_factor = (mu1.unsqueeze(0) * mu2_flat.unsqueeze(1)) ** alpha
                importance = W.abs() * act_norms.unsqueeze(0) / (mu_factor + 1e-6)

            threshold = importance.view(-1).sort().values[n_prune]
            mask = (importance > threshold).float()

            W_sparse = W_norm * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            return ((Y_ref - Y_approx) ** 2).mean().item()

        # Test different alpha values
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        mse_baseline = evaluate_alpha(1.0)  # α=1 is baseline

        print(f"{'α':>6} {'MSE':>12} {'Change':>10}")
        print("-" * 30)

        for alpha in alphas:
            mse = evaluate_alpha(alpha)
            change = (mse_baseline - mse) / mse_baseline * 100
            marker = " *" if alpha == 1.0 else ""
            print(f"{alpha:>6.2f} {mse:>12.6f} {change:>+9.2f}%{marker}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mu_power()
