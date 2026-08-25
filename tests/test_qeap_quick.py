"""
Quick empirical test of QEAP hypothesis.

Goal: Validate or falsify the vetting concern that quant_benefit affects too few weights.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_activation_norms, compute_importance_scores

def analyze_boundary_weights(W, mu1, mu2, group_size=64):
    """
    Analyze how many weights are at row/group boundaries.
    """
    K, N = W.shape
    W_norm = W / (mu2.view(-1, 1) * mu1.view(1, -1))

    # Per-row analysis (naive QEAP)
    row_max_idx = W_norm.argmax(dim=1)
    row_min_idx = W_norm.argmin(dim=1)

    boundary_count_row = 2 * K  # 2 boundary weights per row
    total_weights = K * N
    boundary_fraction_row = boundary_count_row / total_weights

    print(f"\n=== Boundary Weight Analysis ===")
    print(f"Matrix shape: {K} x {N}")
    print(f"Total weights: {total_weights:,}")
    print(f"\nPer-row analysis:")
    print(f"  Boundary weights: {boundary_count_row:,}")
    print(f"  Fraction affected: {boundary_fraction_row:.4%}")

    # Per-group analysis (correct for SINQ with group_size=64)
    if N % group_size == 0:
        n_groups = N // group_size
        W_norm_grouped = W_norm.view(K, n_groups, group_size)

        boundary_count_group = 2 * K * n_groups  # 2 per group per row
        boundary_fraction_group = boundary_count_group / total_weights

        print(f"\nPer-group analysis (group_size={group_size}):")
        print(f"  Number of groups per row: {n_groups}")
        print(f"  Boundary weights: {boundary_count_group:,}")
        print(f"  Fraction affected: {boundary_fraction_group:.4%}")

    return boundary_fraction_row


def compute_qeap_importance(W, X, mu1, mu2, nbits=4, lambda_quant=1.0, group_size=64):
    """
    Compute QEAP importance scores with per-group boundary analysis.
    """
    K, N = W.shape
    device = W.device

    # Sinkhorn-normalized weights
    W_norm = W / (mu2.view(-1, 1) * mu1.view(1, -1))

    # Standard Wanda importance
    act_norms = torch.norm(X.float(), dim=0)  # [N]
    wanda_importance = W.abs() * act_norms.unsqueeze(0)  # [K, N]

    # Compute quant_benefit per group
    quant_benefit = torch.zeros_like(W)

    if N % group_size != 0:
        # Fallback to per-row if not divisible
        group_size = N

    n_groups = N // group_size

    for g in range(n_groups):
        start, end = g * group_size, (g + 1) * group_size
        W_group = W_norm[:, start:end]  # [K, group_size]

        for i in range(K):
            row = W_group[i, :]

            # Find boundaries in this group
            max_val, max_idx_local = row.max(), row.argmax()
            min_val, min_idx_local = row.min(), row.argmin()

            # Find second max/min
            row_masked_max = row.clone()
            row_masked_max[max_idx_local] = -float('inf')
            second_max = row_masked_max.max()

            row_masked_min = row.clone()
            row_masked_min[min_idx_local] = float('inf')
            second_min = row_masked_min.min()

            # Current range and potential reductions
            current_range = max_val - min_val

            if current_range > 1e-6:
                # Scale reduction if we prune max
                new_range_no_max = second_max - min_val
                scale_red_max = (current_range - new_range_no_max) / (2**nbits - 1)

                # Scale reduction if we prune min
                new_range_no_min = max_val - second_min
                scale_red_min = (current_range - new_range_no_min) / (2**nbits - 1)

                # Expected quantization error factor for this group
                # E[ε²] ∝ s² × group_size × ||X_group||² × μ²
                act_group = act_norms[start:end]
                error_factor = (group_size - 1) * (act_group ** 2).sum() * mu2[i]**2

                # Assign benefits
                max_idx_global = start + max_idx_local.item()
                min_idx_global = start + min_idx_local.item()

                if max_idx_local != min_idx_local:
                    quant_benefit[i, max_idx_global] = scale_red_max * error_factor
                    quant_benefit[i, min_idx_global] = scale_red_min * error_factor

    # QEAP importance: lower = better to prune
    importance_qeap = wanda_importance - lambda_quant * quant_benefit

    return importance_qeap, wanda_importance, quant_benefit


def test_qeap_on_synthetic():
    """
    Test QEAP on synthetic data.
    """
    print("="*60)
    print("TEST: QEAP on Synthetic Data")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    sparsity = 0.35
    nbits = 4
    group_size = 64

    # Create weight matrix with some outliers
    W = torch.randn(K, N, device=device)
    # Add outliers to some positions
    W[10, 50] = 5.0  # Large positive outlier
    W[20, 100] = -4.0  # Large negative outlier

    # Create activations
    X = torch.randn(batch, N, device=device)

    # Sinkhorn normalization
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Analyze boundary weights
    analyze_boundary_weights(W.float(), mu1, mu2, group_size)

    # Compute importance scores
    importance_qeap, importance_wanda, quant_benefit = compute_qeap_importance(
        W.float(), X, mu1, mu2, nbits=nbits, lambda_quant=1.0, group_size=group_size
    )

    print(f"\n=== Importance Score Analysis ===")
    print(f"Wanda importance - range: [{importance_wanda.min():.4f}, {importance_wanda.max():.4f}]")
    print(f"Quant benefit - range: [{quant_benefit.min():.4f}, {quant_benefit.max():.4f}]")
    print(f"Quant benefit - non-zero count: {(quant_benefit != 0).sum().item()}")
    print(f"Quant benefit - non-zero fraction: {(quant_benefit != 0).float().mean():.4%}")
    print(f"QEAP importance - range: [{importance_qeap.min():.4f}, {importance_qeap.max():.4f}]")

    # Compare which weights would be pruned differently
    n_prune = int(K * N * sparsity)

    # Wanda pruning
    flat_wanda = importance_wanda.view(-1)
    _, prune_idx_wanda = torch.topk(flat_wanda, n_prune, largest=False)
    mask_wanda = torch.ones(K * N, device=device)
    mask_wanda[prune_idx_wanda] = 0
    mask_wanda = mask_wanda.view(K, N)

    # QEAP pruning
    flat_qeap = importance_qeap.view(-1)
    _, prune_idx_qeap = torch.topk(flat_qeap, n_prune, largest=False)
    mask_qeap = torch.ones(K * N, device=device)
    mask_qeap[prune_idx_qeap] = 0
    mask_qeap = mask_qeap.view(K, N)

    # Count differences
    different_masks = (mask_wanda != mask_qeap).sum().item()
    print(f"\n=== Pruning Mask Comparison ===")
    print(f"Weights pruned: {n_prune}")
    print(f"Masks differ at: {different_masks} positions")
    print(f"Difference fraction: {different_masks / (K * N):.4%}")

    # Check if QEAP prunes more boundary weights than Wanda
    # Boundary weights have non-zero quant_benefit
    boundary_mask = (quant_benefit != 0)
    boundary_pruned_wanda = ((1 - mask_wanda) * boundary_mask).sum().item()
    boundary_pruned_qeap = ((1 - mask_qeap) * boundary_mask).sum().item()

    print(f"\nBoundary weights pruned by Wanda: {boundary_pruned_wanda}")
    print(f"Boundary weights pruned by QEAP: {boundary_pruned_qeap}")

    # Compute reconstruction error
    # Quantize with each mask
    min_max = [0, 2**nbits - 1]

    # With Wanda mask
    W_masked_wanda = W_norm * mask_wanda
    Q_wanda, s_wanda, z_wanda, _ = quantize_rtn(W_masked_wanda, min_max, group_size=group_size)
    # Dequantize (simplified - ignore mu factors for comparison)
    if len(s_wanda.shape) == 3:
        n_groups = s_wanda.shape[1]
        Q_grouped = Q_wanda.view(K, n_groups, group_size)
        W_deq_wanda = (Q_grouped - z_wanda) * s_wanda
        W_deq_wanda = W_deq_wanda.view(K, N)
    else:
        W_deq_wanda = (Q_wanda - z_wanda) * s_wanda
    W_deq_wanda = W_deq_wanda * mu2 * mu1 * mask_wanda

    # With QEAP mask
    W_masked_qeap = W_norm * mask_qeap
    Q_qeap, s_qeap, z_qeap, _ = quantize_rtn(W_masked_qeap, min_max, group_size=group_size)
    if len(s_qeap.shape) == 3:
        n_groups = s_qeap.shape[1]
        Q_grouped = Q_qeap.view(K, n_groups, group_size)
        W_deq_qeap = (Q_grouped - z_qeap) * s_qeap
        W_deq_qeap = W_deq_qeap.view(K, N)
    else:
        W_deq_qeap = (Q_qeap - z_qeap) * s_qeap
    W_deq_qeap = W_deq_qeap * mu2 * mu1 * mask_qeap

    # Compare MSE
    mse_wanda = ((W - W_deq_wanda) ** 2).mean().item()
    mse_qeap = ((W - W_deq_qeap) ** 2).mean().item()

    print(f"\n=== Reconstruction Error ===")
    print(f"MSE (Wanda): {mse_wanda:.6f}")
    print(f"MSE (QEAP):  {mse_qeap:.6f}")
    print(f"Relative improvement: {(mse_wanda - mse_qeap) / mse_wanda * 100:.2f}%")

    # Compare scales
    if len(s_wanda.shape) == 3:
        scale_wanda_mean = s_wanda.abs().mean().item()
        scale_qeap_mean = s_qeap.abs().mean().item()
    else:
        scale_wanda_mean = s_wanda.abs().mean().item()
        scale_qeap_mean = s_qeap.abs().mean().item()

    print(f"\nMean scale (Wanda): {scale_wanda_mean:.6f}")
    print(f"Mean scale (QEAP):  {scale_qeap_mean:.6f}")
    print(f"Scale reduction: {(scale_wanda_mean - scale_qeap_mean) / scale_wanda_mean * 100:.2f}%")

    return mse_wanda, mse_qeap


def test_qeap_on_real_weights():
    """
    Test QEAP on real Qwen-0.5B weights.
    """
    print("\n" + "="*60)
    print("TEST: QEAP on Real Qwen-0.5B Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("Loading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Get a representative layer weight
        W = model.model.layers[0].mlp.gate_proj.weight.data.float()
        K, N = W.shape
        print(f"Weight shape: {K} x {N}")

        device = W.device

        # Create synthetic activations (we don't have real calibration data)
        batch = 64
        X = torch.randn(batch, N, device=device, dtype=torch.float32)

        # Sinkhorn normalization
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)

        # Analyze boundary weights
        analyze_boundary_weights(W, mu1, mu2, group_size=64)

        # Compute importance scores
        importance_qeap, importance_wanda, quant_benefit = compute_qeap_importance(
            W, X, mu1, mu2, nbits=4, lambda_quant=1.0, group_size=64
        )

        print(f"\n=== Importance Score Analysis ===")
        print(f"Wanda importance - range: [{importance_wanda.min():.4f}, {importance_wanda.max():.4f}]")
        print(f"Quant benefit - non-zero count: {(quant_benefit != 0).sum().item()}")
        print(f"Quant benefit - non-zero fraction: {(quant_benefit != 0).float().mean():.4%}")

        # Compare pruning masks at 35% sparsity
        sparsity = 0.35
        n_prune = int(K * N * sparsity)

        # Wanda pruning
        flat_wanda = importance_wanda.view(-1)
        _, prune_idx_wanda = torch.topk(flat_wanda, n_prune, largest=False)
        mask_wanda = torch.ones(K * N, device=device)
        mask_wanda[prune_idx_wanda] = 0
        mask_wanda = mask_wanda.view(K, N)

        # QEAP pruning (try different lambda values)
        for lam in [0.1, 1.0, 10.0]:
            importance_qeap_lam, _, _ = compute_qeap_importance(
                W, X, mu1, mu2, nbits=4, lambda_quant=lam, group_size=64
            )
            flat_qeap = importance_qeap_lam.view(-1)
            _, prune_idx_qeap = torch.topk(flat_qeap, n_prune, largest=False)
            mask_qeap = torch.ones(K * N, device=device)
            mask_qeap[prune_idx_qeap] = 0
            mask_qeap = mask_qeap.view(K, N)

            different_masks = (mask_wanda != mask_qeap).sum().item()
            print(f"\nλ={lam}: Masks differ at {different_masks} positions ({different_masks / (K*N):.4%})")

        # Clean up
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


if __name__ == '__main__':
    # Run tests
    mse_wanda, mse_qeap = test_qeap_on_synthetic()
    test_qeap_on_real_weights()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    if mse_qeap < mse_wanda:
        print(f"QEAP shows {(mse_wanda - mse_qeap) / mse_wanda * 100:.1f}% improvement over Wanda on synthetic data.")
        print("Consider further development with more rigorous evaluation.")
    else:
        print("QEAP shows NO improvement over Wanda.")
        print("Vetting concern validated: quant_benefit term has limited impact.")
        print("RECOMMENDATION: Pivot to alternative hypothesis.")
