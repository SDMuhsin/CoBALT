"""
Hypothesis 042 CORRECTED: μ-Weighted Compensation (MWC) - Fast Version

Vectorized compensation for speed.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_mwc_fast():
    print("="*70)
    print("μ-WEIGHTED COMPENSATION (MWC) - FAST TEST")
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

        # Standard importance
        act_norms = torch.norm(X, dim=0)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Create mask
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        # Hessian approximation: H = X.T @ X
        H = X.T @ X  # [N, N]
        H_diag = H.diag()  # [N]

        def fast_compensation(W_sparse, mask, use_mwc=False):
            """Fast vectorized OBS compensation."""
            W_comp = W_sparse.clone()

            # Pruned weights contribution: W_pruned * (1 - mask)
            W_pruned_contrib = W_sparse * (1 - mask)  # [K, N] - zeros for kept weights

            # For each column j (pruned), compute compensation to all columns k (kept)
            # delta_k = sum_j(-W_j * H_jk / H_kk) for pruned j

            # Compensation matrix: comp[k] = sum_j(W_pruned[j] * H[j,k]) / H[k,k]
            # This is: W_pruned @ H / H_diag
            compensation = (W_pruned_contrib @ H) / (H_diag.unsqueeze(0) + 1e-8)  # [K, N]

            if use_mwc:
                # Weight by μ₁ ratio for each pair
                # For compensation from j to k: multiply by μ₁[j]/μ₁[k]
                # But we've summed over j, so we need:
                # comp_mwc[k] = sum_j(W_pruned[j] * μ₁[j] * H[j,k] / μ₁[k]) / H[k,k]
                #             = (W_pruned * μ₁) @ H / (μ₁ * H_diag)

                W_pruned_mu = W_pruned_contrib * mu1.unsqueeze(0)  # [K, N]
                compensation = (W_pruned_mu @ H) / ((mu1 * H_diag).unsqueeze(0) + 1e-8)

            # Apply compensation only to kept weights
            W_comp = W_comp - compensation * mask

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
        W_std_comp = fast_compensation(W_sparse, mask, use_mwc=False)
        mse_std_comp = evaluate(W_std_comp, "Standard OBS")
        improvement_std = (mse_no_comp - mse_std_comp) / mse_no_comp * 100
        print(f"Standard OBS:   MSE = {mse_std_comp:.6f} ({improvement_std:+.2f}% vs no comp)")

        # MWC compensation
        W_mwc_comp = fast_compensation(W_sparse, mask, use_mwc=True)
        mse_mwc_comp = evaluate(W_mwc_comp, "MWC")
        improvement_mwc = (mse_no_comp - mse_mwc_comp) / mse_no_comp * 100
        improvement_vs_std = (mse_std_comp - mse_mwc_comp) / mse_std_comp * 100
        print(f"MWC:            MSE = {mse_mwc_comp:.6f} ({improvement_mwc:+.2f}% vs no comp, {improvement_vs_std:+.2f}% vs std)")

        # Also test different clipping ranges for μ ratio
        print("\n--- MWC with different μ ratio handling ---")
        for strategy in ['clip_1_10', 'clip_0.5_2', 'softmax']:
            W_comp = W_sparse.clone()
            W_pruned_contrib = W_sparse * (1 - mask)

            if strategy == 'softmax':
                # Use softmax to normalize μ ratios
                mu1_norm = torch.softmax(mu1, dim=0) * N
                W_pruned_mu = W_pruned_contrib * mu1_norm.unsqueeze(0)
                compensation = (W_pruned_mu @ H) / ((mu1_norm * H_diag).unsqueeze(0) + 1e-8)
            else:
                clip_min, clip_max = map(float, strategy.split('_')[1:])
                # Standard MWC with clipping
                W_pruned_mu = W_pruned_contrib * mu1.unsqueeze(0)
                raw_comp = (W_pruned_mu @ H) / ((mu1 * H_diag).unsqueeze(0) + 1e-8)
                # The μ ratio is implicit in the formula
                compensation = raw_comp

            W_comp = W_comp - compensation * mask
            mse = evaluate(W_comp, strategy)
            improvement = (mse_std_comp - mse) / mse_std_comp * 100
            print(f"  {strategy}: MSE = {mse:.6f} ({improvement:+.2f}% vs std)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mwc_fast()
