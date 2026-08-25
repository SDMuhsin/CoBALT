"""
Hypothesis 042 CORRECTED: μ-Weighted Compensation (MWC)

Key correction from vetting:
- In row-wise OBS, we compensate w_ik for pruned w_ij (same row i)
- μ_ij = μ₁[j] × μ₂[i], μ_ik = μ₁[k] × μ₂[i]
- Ratio = μ₁[j] / μ₁[k]  (μ₂ cancels!)

So only COLUMN factors (μ₁) matter for row-wise compensation.

Test: Does μ₁-weighted compensation improve over standard?
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_mwc():
    print("="*70)
    print("μ-WEIGHTED COMPENSATION (MWC) - CORRECTED TEST")
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

        print(f"μ₁ stats: mean={mu1.mean():.4f}, std={mu1.std():.4f}, CV={mu1.std()/mu1.mean():.4f}")
        print(f"μ₂ stats: mean={mu2.mean():.4f}, std={mu2.std():.4f}, CV={mu2.std()/mu2.mean():.4f}")

        # Standard importance
        act_norms = torch.norm(X, dim=0)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Create mask
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        # Compute Hessian diagonal approximation per column
        # H_jj ≈ ||X[:,j]||²
        H_diag = (X ** 2).sum(dim=0)  # [N]

        def apply_compensation(W_sparse, mask, use_mwc=False):
            """Apply OBS-style row-wise compensation."""
            W_comp = W_sparse.clone()

            # For each row, compensate for pruned weights
            for i in range(K):
                row_mask = mask[i]  # [N]
                pruned_cols = (row_mask == 0).nonzero().squeeze(-1)
                kept_cols = (row_mask == 1).nonzero().squeeze(-1)

                if len(pruned_cols) == 0 or len(kept_cols) == 0:
                    continue

                # For each pruned weight, distribute compensation to kept weights
                for j in pruned_cols[:min(100, len(pruned_cols))]:  # Limit for speed
                    w_pruned = W_norm[i, j].item()
                    if abs(w_pruned) < 1e-8:
                        continue

                    # Compute compensation for each kept weight
                    # Standard: delta_k = -w_j * H_jk / H_kk
                    # where H_jk ≈ X[:,j] @ X[:,k]
                    X_j = X[:, j]
                    X_kept = X[:, kept_cols]
                    H_jk = X_j @ X_kept  # [len(kept_cols)]
                    H_kk = H_diag[kept_cols]  # [len(kept_cols)]

                    delta = -w_pruned * H_jk / (H_kk + 1e-8)

                    if use_mwc:
                        # MWC: Weight by μ₁[j] / μ₁[k]
                        mu1_j = mu1[j]
                        mu1_k = mu1[kept_cols]
                        mu_ratio = mu1_j / (mu1_k + 1e-8)
                        # Clip for stability
                        mu_ratio = mu_ratio.clamp(0.1, 10.0)
                        delta = delta * mu_ratio

                    W_comp[i, kept_cols] += delta

            return W_comp

        def evaluate(W_comp, label):
            Q, s, z, _ = quantize_rtn(W_comp, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            mse = ((Y_ref - Y_approx) ** 2).mean().item()
            return mse

        # No compensation baseline
        W_sparse = W_norm * mask
        mse_no_comp = evaluate(W_sparse, "No compensation")
        print(f"\nNo compensation: MSE = {mse_no_comp:.6f}")

        # Standard compensation
        W_std_comp = apply_compensation(W_sparse, mask, use_mwc=False)
        mse_std_comp = evaluate(W_std_comp, "Standard OBS")
        improvement_std = (mse_no_comp - mse_std_comp) / mse_no_comp * 100
        print(f"Standard OBS:   MSE = {mse_std_comp:.6f} ({improvement_std:+.2f}% vs no comp)")

        # MWC compensation
        W_mwc_comp = apply_compensation(W_sparse, mask, use_mwc=True)
        mse_mwc_comp = evaluate(W_mwc_comp, "MWC")
        improvement_mwc = (mse_no_comp - mse_mwc_comp) / mse_no_comp * 100
        improvement_vs_std = (mse_std_comp - mse_mwc_comp) / mse_std_comp * 100
        print(f"MWC:            MSE = {mse_mwc_comp:.6f} ({improvement_mwc:+.2f}% vs no comp, {improvement_vs_std:+.2f}% vs std)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mwc()
