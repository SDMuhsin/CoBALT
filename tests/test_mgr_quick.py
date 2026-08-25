"""
Quick test: μ-Guided Rounding (MGR)

Idea: Instead of round-to-nearest, bias rounding based on μ factors.
High-μ weights are rounded DOWN to reduce error amplification.

This is like a closed-form AdaRound approximation.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log


def test_mgr():
    print("="*70)
    print("μ-GUIDED ROUNDING (MGR) QUICK TEST")
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

    # Compute Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Quantization setup (4-bit)
    nbits = 4
    group_size = 64
    n_levels = 2 ** nbits
    n_groups = N // group_size

    # Reshape for group-wise quantization
    W_grouped = W_norm.view(K, n_groups, group_size)

    # Compute scales (per-group)
    w_min = W_grouped.min(dim=2, keepdim=True)[0]
    w_max = W_grouped.max(dim=2, keepdim=True)[0]
    scales = (w_max - w_min) / (n_levels - 1)
    scales = scales.clamp(min=1e-6)

    # Standard RTN
    W_scaled = (W_grouped - w_min) / scales
    Q_rtn = W_scaled.round().clamp(0, n_levels - 1)
    W_deq_rtn = Q_rtn * scales + w_min
    W_deq_rtn = W_deq_rtn.view(K, N)

    # Compute μ penalty (per-weight)
    mu_product = mu1.unsqueeze(0) * mu2.unsqueeze(1)  # [K, N]
    mu_mean = mu_product.mean()
    mu_penalty = (mu_product - mu_mean) / mu_product.std()  # Normalized
    mu_penalty_grouped = mu_penalty.view(K, n_groups, group_size)

    # μ-Guided Rounding: Bias high-μ weights DOWN
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5]

    print("\n--- Results ---")
    print(f"{'Lambda':<10} {'MSE':<15} {'vs RTN':<15}")

    mse_rtn = ((W_norm - W_deq_rtn) ** 2).mean().item()
    mse_rtn_deq = ((W - W_deq_rtn * mu2.unsqueeze(1) * mu1.unsqueeze(0)) ** 2).mean().item()
    print(f"{'RTN':<10} {mse_rtn:.6f} (norm) / {mse_rtn_deq:.6f} (orig)")

    for lam in lambdas:
        # Bias: subtract λ × μ_penalty from scaled value before rounding
        # This shifts high-μ weights down
        W_biased = W_scaled - lam * mu_penalty_grouped
        Q_mgr = W_biased.round().clamp(0, n_levels - 1)
        W_deq_mgr = Q_mgr * scales + w_min
        W_deq_mgr = W_deq_mgr.view(K, N)

        mse_mgr = ((W_norm - W_deq_mgr) ** 2).mean().item()
        mse_mgr_deq = ((W - W_deq_mgr * mu2.unsqueeze(1) * mu1.unsqueeze(0)) ** 2).mean().item()

        improvement = (mse_rtn_deq - mse_mgr_deq) / mse_rtn_deq * 100

        print(f"λ={lam:<7} {mse_mgr:.6f} (norm) / {mse_mgr_deq:.6f} (orig) {improvement:+.2f}%")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mgr()
