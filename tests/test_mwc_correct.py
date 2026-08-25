"""
Hypothesis 042: MWC with CORRECT OBS formula

Standard OBS: Δw_k = -w_j * H_inv[k,j] / H_inv[j,j]
MWC: Δw_k = -w_j * (μ₁[j]/μ₁[k]) * H_inv[k,j] / H_inv[j,j]

The μ-weighting accounts for the fact that output contribution
is proportional to w × μ, not just w.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_hessian_inverse


def test_mwc_correct():
    print("="*70)
    print("MWC with CORRECT OBS Formula")
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

        batch = 256  # More samples for better Hessian
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        print(f"μ₁ CV: {mu1.std()/mu1.mean():.4f}")

        # Importance and mask
        act_norms = torch.norm(X, dim=0)
        importance = W_norm.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()
        pruned_mask = 1 - mask

        # Compute Hessian inverse
        H_inv = compute_hessian_inverse(X.float(), damping=None)
        H_inv_diag = H_inv.diag()

        def apply_compensation(W_sparse, mask, use_mwc=False):
            """Apply OBS compensation with optional μ-weighting."""
            W_comp = W_sparse.clone()
            pruned_mask = 1 - mask

            for i in range(K):
                pruned_weights = W_norm[i] * pruned_mask[i]  # Use W_norm, not W_sparse

                if use_mwc:
                    # MWC: compensation[k] = sum_j(-w_j * μ₁[j]/μ₁[k] * H_inv[k,j] / H_inv[j,j])
                    # = -H_inv @ (pruned_weights * μ₁ / H_inv_diag) / μ₁
                    weighted_pruned = pruned_weights * mu1 / H_inv_diag
                    compensation = -H_inv @ weighted_pruned / mu1
                else:
                    # Standard: compensation[k] = sum_j(-w_j * H_inv[k,j] / H_inv[j,j])
                    compensation = -H_inv @ (pruned_weights / H_inv_diag)

                W_comp[i] = W_sparse[i] + compensation * mask[i]

            return W_comp

        def evaluate(W_comp, label):
            Q, s, z, _ = quantize_rtn(W_comp, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            mse = ((Y_ref - Y_approx) ** 2).mean().item()
            return mse

        # No compensation
        W_sparse = W_norm * mask
        mse_sparse = evaluate(W_sparse, "Sparse")
        print(f"\nNo compensation: MSE = {mse_sparse:.6f}")

        # Standard OBS
        W_std = apply_compensation(W_sparse, mask, use_mwc=False)
        mse_std = evaluate(W_std, "Standard OBS")
        improvement_std = (mse_sparse - mse_std) / mse_sparse * 100
        print(f"Standard OBS:   MSE = {mse_std:.6f} ({improvement_std:+.2f}% vs sparse)")

        # MWC
        W_mwc = apply_compensation(W_sparse, mask, use_mwc=True)
        mse_mwc = evaluate(W_mwc, "MWC")
        improvement_mwc = (mse_sparse - mse_mwc) / mse_sparse * 100
        improvement_vs_std = (mse_std - mse_mwc) / mse_std * 100
        print(f"MWC:            MSE = {mse_mwc:.6f} ({improvement_mwc:+.2f}% vs sparse, {improvement_vs_std:+.2f}% vs std)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mwc_correct()
