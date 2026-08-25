"""
Hypothesis 41: Marginal Quantization Impact (MQI)

Core insight: Quantization error depends on GROUP statistics.
Removing a weight changes the group's min/max, affecting scale for ALL weights in group.

MQI: For each weight, compute how much its removal affects group quantization.

Mathematical formulation:
For group g with weights w_1, ..., w_k:
- Current scale: s = (max - min) / (2^n - 1)
- If we remove w_i:
  - New scale: s' = (max' - min') / (2^n - 1) where max', min' exclude w_i
  - Scale change: Δs = |s' - s|
  - Impact on other weights: Δerror ≈ Δs × (k-1)

Weights at group boundaries (min or max) have HIGH impact.
Weights in the middle have LOW impact.

This is different from importance! A high-importance weight at the boundary
could hurt quantization more than a low-importance weight in the middle.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_mqi():
    print("="*70)
    print("MARGINAL QUANTIZATION IMPACT (MQI) TEST")
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

        # Compute MQI: boundary distance for each weight
        W_grouped = W_norm.view(K, n_groups, group_size)

        # For each group, find min and max
        group_min = W_grouped.min(dim=2, keepdim=True).values  # [K, n_groups, 1]
        group_max = W_grouped.max(dim=2, keepdim=True).values  # [K, n_groups, 1]
        group_range = group_max - group_min  # [K, n_groups, 1]

        # Distance to boundary (normalized)
        dist_to_min = (W_grouped - group_min) / (group_range + 1e-8)  # [K, n_groups, group_size]
        dist_to_max = (group_max - W_grouped) / (group_range + 1e-8)  # [K, n_groups, group_size]

        # Boundary score: how close to min or max (0 = at boundary, 1 = in middle)
        boundary_dist = torch.min(dist_to_min, dist_to_max)  # [K, n_groups, group_size]
        boundary_dist = boundary_dist.view(K, N)

        print(f"Boundary distance stats: mean={boundary_dist.mean():.4f}, std={boundary_dist.std():.4f}")
        print(f"Weights at boundary (<0.1): {(boundary_dist < 0.1).float().mean():.1%}")

        # Correlation with importance
        corr = torch.corrcoef(torch.stack([
            importance_std.flatten(),
            boundary_dist.flatten()
        ]))[0, 1].item()
        print(f"Correlation(importance, boundary_dist): {corr:.4f}")

        def evaluate_mask(mask, label):
            W_sparse = W_norm * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            mse = ((Y_ref - Y_approx) ** 2).mean().item()

            threshold_std = importance_std.view(-1).sort().values[n_prune]
            mask_std = (importance_std > threshold_std).float()
            overlap = (mask * mask_std).sum() / mask_std.sum()

            return mse, overlap.item()

        # Standard mask
        threshold_std = importance_std.view(-1).sort().values[n_prune]
        mask_std = (importance_std > threshold_std).float()
        mse_std, _ = evaluate_mask(mask_std, "Standard")
        print(f"\nStandard MSE: {mse_std:.6f}")

        # === MQI v1: Avoid pruning boundary weights ===
        # Multiply importance by boundary_dist (boundary weights get low importance)
        importance_mqi_v1 = importance_std * (boundary_dist + 0.1)  # Add small offset
        threshold_v1 = importance_mqi_v1.view(-1).sort().values[n_prune]
        mask_v1 = (importance_mqi_v1 > threshold_v1).float()
        mse_v1, overlap_v1 = evaluate_mask(mask_v1, "MQI v1")
        print(f"MQI v1 (avoid boundary): MSE={mse_v1:.6f} ({(mse_std-mse_v1)/mse_std*100:+.2f}%), overlap={overlap_v1:.1%}")

        # === MQI v2: Prefer pruning boundary weights ===
        # They reduce scale, which might improve quant for remaining
        importance_mqi_v2 = importance_std / (boundary_dist + 0.1)
        threshold_v2 = importance_mqi_v2.view(-1).sort().values[n_prune]
        mask_v2 = (importance_mqi_v2 > threshold_v2).float()
        mse_v2, overlap_v2 = evaluate_mask(mask_v2, "MQI v2")
        print(f"MQI v2 (prefer boundary): MSE={mse_v2:.6f} ({(mse_std-mse_v2)/mse_std*100:+.2f}%), overlap={overlap_v2:.1%}")

        # === MQI v3: Only for outliers ===
        # Outliers (top/bottom 10% per group) are boundary weights
        outlier_threshold = 0.1
        is_outlier = (dist_to_min < outlier_threshold) | (dist_to_max < outlier_threshold)
        is_outlier = is_outlier.view(K, N).float()
        print(f"Outlier rate: {is_outlier.mean():.1%}")

        # Penalize outlier weights (make them harder to prune)
        importance_mqi_v3 = importance_std * (1 + is_outlier)  # Outliers get 2x importance
        threshold_v3 = importance_mqi_v3.view(-1).sort().values[n_prune]
        mask_v3 = (importance_mqi_v3 > threshold_v3).float()
        mse_v3, overlap_v3 = evaluate_mask(mask_v3, "MQI v3")
        print(f"MQI v3 (protect outliers): MSE={mse_v3:.6f} ({(mse_std-mse_v3)/mse_std*100:+.2f}%), overlap={overlap_v3:.1%}")

        # === MQI v4: Inverse - prune outliers ===
        importance_mqi_v4 = importance_std / (1 + is_outlier)
        threshold_v4 = importance_mqi_v4.view(-1).sort().values[n_prune]
        mask_v4 = (importance_mqi_v4 > threshold_v4).float()
        mse_v4, overlap_v4 = evaluate_mask(mask_v4, "MQI v4")
        print(f"MQI v4 (prune outliers): MSE={mse_v4:.6f} ({(mse_std-mse_v4)/mse_std*100:+.2f}%), overlap={overlap_v4:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mqi()
