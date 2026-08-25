"""
Quick empirical test of SFC (Sinkhorn Factor Compensation) hypothesis.

Goal: Test if asymmetric factor correction improves reconstruction after pruning.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def sfc_correction(W, W_deq, mask, max_iter=10, verbose=False):
    """
    Compute Sinkhorn Factor Correction.

    Args:
        W: Original weight matrix [K, N]
        W_deq: Dequantized matrix (before mask application) [K, N]
        mask: Sparsity mask [K, N]
        max_iter: Alternating optimization iterations

    Returns:
        gamma1: Column correction factors [N]
        gamma2: Row correction factors [K]
    """
    K, N = W.shape
    A = W * mask  # Target
    B = W_deq * mask  # To be corrected

    # Initialize
    gamma1 = torch.ones(N, device=W.device, dtype=W.dtype)
    gamma2 = torch.ones(K, device=W.device, dtype=W.dtype)

    for it in range(max_iter):
        # Update gamma1 (column factors)
        B_scaled = B * gamma2.view(-1, 1)  # [K, N]
        num = (A * B_scaled).sum(dim=0)  # [N]
        denom = (B_scaled ** 2).sum(dim=0).clamp(min=1e-8)  # [N]
        gamma1_new = num / denom

        # Update gamma2 (row factors)
        B_scaled = B * gamma1_new.view(1, -1)  # [K, N]
        num = (A * B_scaled).sum(dim=1)  # [K]
        denom = (B_scaled ** 2).sum(dim=1).clamp(min=1e-8)  # [K]
        gamma2_new = num / denom

        # Clamp to reasonable range to avoid extreme values
        gamma1 = gamma1_new.clamp(0.5, 2.0)
        gamma2 = gamma2_new.clamp(0.5, 2.0)

        if verbose and it % 3 == 0:
            W_corrected = B * gamma1.view(1, -1) * gamma2.view(-1, 1)
            mse = ((A - W_corrected) ** 2).mean().item()
            print(f"  Iter {it}: MSE = {mse:.6f}")

    return gamma1, gamma2


def apply_sfc(W_deq, mask, gamma1, gamma2):
    """Apply SFC correction to dequantized weights."""
    return W_deq * mask * gamma1.view(1, -1) * gamma2.view(-1, 1)


def create_wanda_mask(W, X, sparsity):
    """Create Wanda-style pruning mask."""
    K, N = W.shape
    act_norms = torch.norm(X.float(), dim=0)  # [N]
    importance = W.abs() * act_norms.unsqueeze(0)

    n_prune = int(K * N * sparsity)
    flat_importance = importance.view(-1)
    _, prune_idx = torch.topk(flat_importance, n_prune, largest=False)

    mask = torch.ones(K * N, device=W.device)
    mask[prune_idx] = 0
    return mask.view(K, N)


def test_sfc_on_synthetic():
    """Test SFC on synthetic data."""
    print("="*60)
    print("TEST: SFC on Synthetic Data")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    sparsity = 0.35
    nbits = 4
    group_size = 64

    # Create weight matrix
    W = torch.randn(K, N, device=device)

    # Create activations
    X = torch.randn(batch, N, device=device)

    # Sinkhorn normalization
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Create pruning mask (Wanda-style)
    mask = create_wanda_mask(W.float(), X, sparsity)

    # Quantize the normalized weights
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Dequantize (without mask for now, to get W_deq)
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    # Apply Sinkhorn factors to get W_deq in original space
    W_deq = W_deq_norm * mu2 * mu1

    # === Baseline: Just apply mask ===
    W_baseline = W_deq * mask
    mse_baseline = ((W - W_baseline) ** 2).mean().item()

    print(f"\n=== Baseline (standard SINQ-Sparse) ===")
    print(f"MSE: {mse_baseline:.6f}")

    # === SFC: Apply factor correction ===
    print(f"\n=== SFC Optimization ===")
    gamma1, gamma2 = sfc_correction(W.float(), W_deq, mask, max_iter=10, verbose=True)

    W_sfc = apply_sfc(W_deq, mask, gamma1, gamma2)
    mse_sfc = ((W - W_sfc) ** 2).mean().item()

    print(f"\n=== Results ===")
    print(f"MSE (Baseline): {mse_baseline:.6f}")
    print(f"MSE (SFC):      {mse_sfc:.6f}")
    print(f"Improvement:    {(mse_baseline - mse_sfc) / mse_baseline * 100:.2f}%")

    print(f"\n=== Correction Factor Analysis ===")
    print(f"gamma1 range: [{gamma1.min():.4f}, {gamma1.max():.4f}]")
    print(f"gamma1 mean:  {gamma1.mean():.4f}")
    print(f"gamma1 std:   {gamma1.std():.4f}")
    print(f"gamma2 range: [{gamma2.min():.4f}, {gamma2.max():.4f}]")
    print(f"gamma2 mean:  {gamma2.mean():.4f}")
    print(f"gamma2 std:   {gamma2.std():.4f}")

    return mse_baseline, mse_sfc


def test_sfc_ablation():
    """Test different SFC variants."""
    print("\n" + "="*60)
    print("TEST: SFC Ablation Study")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    sparsity = 0.35
    nbits = 4
    group_size = 64

    W = torch.randn(K, N, device=device).float()
    X = torch.randn(batch, N, device=device).float()

    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mask = create_wanda_mask(W, X, sparsity)

    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1

    # Baseline
    W_baseline = W_deq * mask
    mse_baseline = ((W - W_baseline) ** 2).mean().item()

    print(f"\nBaseline MSE: {mse_baseline:.6f}")

    # Test different configurations
    configs = [
        ("gamma1 only", True, False),
        ("gamma2 only", False, True),
        ("Both (joint)", True, True),
    ]

    for name, use_g1, use_g2 in configs:
        A = W * mask
        B = W_deq * mask

        gamma1 = torch.ones(N, device=device)
        gamma2 = torch.ones(K, device=device)

        for _ in range(10):
            if use_g1:
                B_scaled = B * gamma2.view(-1, 1)
                num = (A * B_scaled).sum(dim=0)
                denom = (B_scaled ** 2).sum(dim=0).clamp(min=1e-8)
                gamma1 = (num / denom).clamp(0.5, 2.0)

            if use_g2:
                B_scaled = B * gamma1.view(1, -1)
                num = (A * B_scaled).sum(dim=1)
                denom = (B_scaled ** 2).sum(dim=1).clamp(min=1e-8)
                gamma2 = (num / denom).clamp(0.5, 2.0)

        W_corrected = W_deq * mask * gamma1.view(1, -1) * gamma2.view(-1, 1)
        mse = ((W - W_corrected) ** 2).mean().item()
        improvement = (mse_baseline - mse) / mse_baseline * 100
        print(f"{name:20s}: MSE = {mse:.6f} ({improvement:+.2f}%)")


def test_sfc_vs_sparsity():
    """Test SFC across different sparsity levels."""
    print("\n" + "="*60)
    print("TEST: SFC vs Sparsity Level")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    nbits = 4
    group_size = 64

    W = torch.randn(K, N, device=device).float()
    X = torch.randn(batch, N, device=device).float()

    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)

    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1

    print(f"\n{'Sparsity':>10s} | {'Baseline':>12s} | {'SFC':>12s} | {'Improvement':>12s}")
    print("-" * 55)

    for sparsity in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]:
        mask = create_wanda_mask(W, X, sparsity)

        # Baseline
        W_baseline = W_deq * mask
        mse_baseline = ((W - W_baseline) ** 2).mean().item()

        # SFC
        gamma1, gamma2 = sfc_correction(W, W_deq, mask, max_iter=10)
        W_sfc = apply_sfc(W_deq, mask, gamma1, gamma2)
        mse_sfc = ((W - W_sfc) ** 2).mean().item()

        improvement = (mse_baseline - mse_sfc) / mse_baseline * 100
        print(f"{sparsity*100:>9.0f}% | {mse_baseline:>12.6f} | {mse_sfc:>12.6f} | {improvement:>+11.2f}%")


def test_sfc_on_real_weights():
    """Test SFC on real Qwen-0.5B weights."""
    print("\n" + "="*60)
    print("TEST: SFC on Real Qwen-0.5B Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("Loading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Test on a few layers
        layers_to_test = [
            ("layer 0 gate_proj", model.model.layers[0].mlp.gate_proj.weight),
            ("layer 0 down_proj", model.model.layers[0].mlp.down_proj.weight),
            ("layer 5 gate_proj", model.model.layers[5].mlp.gate_proj.weight),
        ]

        for name, weight in layers_to_test:
            W = weight.data.float()
            K, N = W.shape
            device = W.device

            print(f"\n--- {name} [{K}x{N}] ---")

            # Synthetic activations
            batch = 64
            X = torch.randn(batch, N, device=device, dtype=torch.float32)

            # Sinkhorn
            W_norm, mu1, mu2 = sinkhorn_log(W, order=16)

            # Quantize
            nbits = 4
            group_size = 64
            min_max = [0, 2**nbits - 1]
            Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

            if len(scales.shape) == 3:
                n_groups = scales.shape[1]
                Q_grouped = Q.view(K, n_groups, group_size)
                W_deq_norm = (Q_grouped - zeros) * scales
                W_deq_norm = W_deq_norm.view(K, N)
            else:
                W_deq_norm = (Q - zeros) * scales

            W_deq = W_deq_norm * mu2 * mu1

            # Test at 35% sparsity
            sparsity = 0.35
            mask = create_wanda_mask(W, X, sparsity)

            # Baseline
            W_baseline = W_deq * mask
            mse_baseline = ((W - W_baseline) ** 2).mean().item()

            # SFC
            gamma1, gamma2 = sfc_correction(W, W_deq, mask, max_iter=10)
            W_sfc = apply_sfc(W_deq, mask, gamma1, gamma2)
            mse_sfc = ((W - W_sfc) ** 2).mean().item()

            improvement = (mse_baseline - mse_sfc) / mse_baseline * 100
            print(f"Baseline MSE: {mse_baseline:.6f}")
            print(f"SFC MSE:      {mse_sfc:.6f}")
            print(f"Improvement:  {improvement:+.2f}%")
            print(f"gamma1 range: [{gamma1.min():.4f}, {gamma1.max():.4f}]")
            print(f"gamma2 range: [{gamma2.min():.4f}, {gamma2.max():.4f}]")

        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_sfc_on_synthetic()
    test_sfc_ablation()
    test_sfc_vs_sparsity()
    test_sfc_on_real_weights()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("If SFC shows consistent improvement, proceed with full evaluation.")
    print("If improvement is marginal (<1%), the hypothesis may need revision.")
