"""
Quick test: Sparse-Aware Quantization (SAQ)

KEY INSIGHT: Standard quantization computes scale from ALL weights including zeros.
This is suboptimal:
- Zeros don't need precision (they're already exact)
- Including zeros in range calculation wastes bits

SAQ: Compute quantization scale/zero from NON-ZERO weights only.
- Smaller effective range → better precision for surviving weights

This is a QUANTIZATION improvement, not a pruning improvement.
But it only applies when combined with sparsity!
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def quantize_sparse_aware(W, mask, min_max, group_size):
    """Quantize considering only non-zero weights for scale computation."""
    K, N = W.shape
    n_groups = N // group_size
    nbits = int(torch.log2(torch.tensor(min_max[1] + 1)).item())

    W_grouped = W.view(K, n_groups, group_size)
    mask_grouped = mask.view(K, n_groups, group_size)

    Q = torch.zeros_like(W_grouped)
    scales = torch.zeros(K, n_groups, 1, device=W.device, dtype=W.dtype)
    zeros = torch.zeros(K, n_groups, 1, device=W.device, dtype=W.dtype)

    for g in range(n_groups):
        for r in range(K):
            group_weights = W_grouped[r, g]
            group_mask = mask_grouped[r, g]

            # Get non-zero weights
            nonzero_mask = group_mask > 0
            nonzero_weights = group_weights[nonzero_mask]

            if nonzero_weights.numel() == 0:
                # All zeros in this group
                Q[r, g] = 0
                scales[r, g, 0] = 1e-8
                zeros[r, g, 0] = 0
                continue

            # Compute scale from non-zero weights only
            w_min = nonzero_weights.min()
            w_max = nonzero_weights.max()

            scale = (w_max - w_min) / (min_max[1] - min_max[0])
            if scale < 1e-8:
                scale = 1e-8
            zero = min_max[0] - w_min / scale

            # Quantize all weights
            q_vals = torch.clamp(torch.round(group_weights / scale + zero), min_max[0], min_max[1])

            # Set pruned weights to zero (their quantized value)
            q_vals[~nonzero_mask] = torch.round(zero)  # Closest to 0

            Q[r, g] = q_vals
            scales[r, g, 0] = scale
            zeros[r, g, 0] = zero

    return Q.view(K, N), scales, zeros


def test_saq():
    print("="*70)
    print("SPARSE-AWARE QUANTIZATION (SAQ) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('up_proj', model.model.layers[0].mlp.up_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
    ]

    for sparsity in [0.35, 0.50, 0.70]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity:.0%}")
        print("="*70)

        for name, W_orig in test_layers:
            W = W_orig.float()
            K, N = W.shape
            nbits = 4
            group_size = 64
            n_groups = N // group_size
            min_max = [0, 2**nbits - 1]

            batch = 64
            X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
            Y_ref = X @ W.T

            # Sinkhorn
            W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
            mu2 = mu2.squeeze()

            # Importance and mask
            act_norms = torch.norm(X, dim=0)
            importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
            n_prune = int(K * N * sparsity)
            threshold = importance.view(-1).sort().values[n_prune]
            mask = (importance > threshold).float()

            # === Standard quantization ===
            W_sparse = W_norm * mask
            Q_std, s_std, z_std, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g_std = Q_std.view(K, n_groups, group_size)
            W_deq_std = (Q_g_std - z_std) * s_std
            W_deq_std = W_deq_std.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_std = X @ W_deq_std.T
            mse_std = ((Y_ref - Y_std) ** 2).mean().item()

            # === SAQ: Sparse-aware quantization ===
            Q_saq, s_saq, z_saq = quantize_sparse_aware(W_sparse, mask, min_max, group_size)
            Q_g_saq = Q_saq.view(K, n_groups, group_size)
            W_deq_saq = (Q_g_saq - z_saq) * s_saq
            W_deq_saq = W_deq_saq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            # Re-apply mask to ensure zeros
            W_deq_saq = W_deq_saq * mask
            Y_saq = X @ W_deq_saq.T
            mse_saq = ((Y_ref - Y_saq) ** 2).mean().item()

            improvement = (mse_std - mse_saq) / mse_std * 100
            print(f"{name:12s}: Standard={mse_std:.6f}, SAQ={mse_saq:.6f} ({improvement:+.2f}%)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_saq()
