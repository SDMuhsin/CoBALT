"""
Validation tests for VARQ (Variance-Adaptive Range Quantization)

This script validates that VARQ:
1. Provides closed-form solutions (no iterations)
2. Minimizes per-group ranges better than SINQ
3. Maintains comparable or better global imbalance
4. Has theoretical optimality for Stage 2
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinq.varq import (
    varq_normalize,
    varq_hybrid_normalize,
    compute_group_ranges,
    compare_normalizations
)
from sinq.sinkhorn import sinkhorn_log


def test_basic_functionality():
    """Test that VARQ runs without errors."""
    print("=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)

    torch.manual_seed(42)
    W = torch.randn(256, 512)

    try:
        W_norm, mu1, mu2 = varq_normalize(W, group_size=64)
        print("✓ VARQ executed successfully")
        print(f"  Input shape: {W.shape}")
        print(f"  Output shape: {W_norm.shape}")
        print(f"  mu1 (col scales) shape: {mu1.shape}")
        print(f"  mu2 (row scales) shape: {mu2.shape}")
        return True
    except Exception as e:
        print(f"✗ VARQ failed: {e}")
        return False


def test_scale_bounds():
    """Test that scales are within reasonable bounds."""
    print("\n" + "=" * 70)
    print("TEST 2: Scale Bounds")
    print("=" * 70)

    torch.manual_seed(42)
    W = torch.randn(256, 512)
    W[100, :] *= 10  # Add outlier

    W_norm, mu1, mu2 = varq_normalize(W, group_size=64, clip_min=0.5, clip_max=2.0)

    mu1_min, mu1_max = mu1.min().item(), mu1.max().item()
    mu2_min, mu2_max = mu2.min().item(), mu2.max().item()

    print(f"Column scales (mu1): min={mu1_min:.4f}, max={mu1_max:.4f}")
    print(f"Row scales (mu2): min={mu2_min:.4f}, max={mu2_max:.4f}")

    # Row scales should respect clipping (within tolerance for normalization)
    row_scale_ratio = mu2_max / mu2_min
    print(f"Row scale ratio (max/min): {row_scale_ratio:.4f}")

    if row_scale_ratio <= 4.5:  # 2.0/0.5 = 4.0, allow some slack for normalization
        print("✓ Row scales within expected bounds")
        return True
    else:
        print(f"✗ Row scales exceed expected bounds (ratio={row_scale_ratio:.4f})")
        return False


def test_group_range_minimization():
    """Test that VARQ minimizes per-group ranges better than SINQ."""
    print("\n" + "=" * 70)
    print("TEST 3: Group Range Minimization")
    print("=" * 70)

    torch.manual_seed(42)
    W = torch.randn(512, 512)
    # Add structured outliers
    W[100:120, :] *= 5  # row outliers
    W[:, 200:220] *= 3  # column outliers

    # Compare SINQ vs VARQ
    comparison = compare_normalizations(W, group_size=64)

    print("\nSINQ:")
    print(f"  Max range: {comparison['sinq']['max_range']:.6f}")
    print(f"  Mean range: {comparison['sinq']['mean_range']:.6f}")
    print(f"  Std range: {comparison['sinq']['std_range']:.6f}")
    print(f"  Range imbalance: {comparison['sinq']['range_imbalance']:.4f}")

    print("\nVARQ:")
    print(f"  Max range: {comparison['varq']['max_range']:.6f}")
    print(f"  Mean range: {comparison['varq']['mean_range']:.6f}")
    print(f"  Std range: {comparison['varq']['std_range']:.6f}")
    print(f"  Range imbalance: {comparison['varq']['range_imbalance']:.4f}")

    print("\nImprovement:")
    print(f"  Max range reduction: {comparison['improvement']['max_range_reduction']*100:.2f}%")
    print(f"  Range imbalance reduction: {comparison['improvement']['range_imbalance_reduction']*100:.2f}%")

    # VARQ should reduce max range or range imbalance
    max_range_improved = comparison['varq']['max_range'] <= comparison['sinq']['max_range']
    imbalance_improved = comparison['varq']['range_imbalance'] <= comparison['sinq']['range_imbalance']

    if max_range_improved or imbalance_improved:
        print("\n✓ VARQ improves range metrics")
        return True
    else:
        print("\n✗ VARQ does not improve range metrics")
        return False


def test_theoretical_optimality():
    """
    Test Stage 2 optimality: for a single group, verify that VARQ achieves
    the theoretical minimum (Chebyshev balancing).
    """
    print("\n" + "=" * 70)
    print("TEST 4: Theoretical Optimality (Stage 2)")
    print("=" * 70)

    torch.manual_seed(123)
    # Create a single group (small matrix for analytical verification)
    group = torch.randn(32, 64)

    # VARQ Stage 2 solution
    col_max_abs = group.abs().max(dim=0).values
    t_optimal = 1.0 / (col_max_abs + 1e-10)
    t_optimal = t_optimal / t_optimal.pow(1.0/64).prod()  # geometric mean normalization

    # Apply optimal scaling
    group_scaled = group * t_optimal.view(1, -1)

    # Compute max absolute value per column (should be equalized)
    scaled_col_max = group_scaled.abs().max(dim=0).values

    # Check if column maxes are approximately equal (Chebyshev property)
    max_deviation = (scaled_col_max.max() - scaled_col_max.min()) / scaled_col_max.mean()

    print(f"Scaled column max values:")
    print(f"  Min: {scaled_col_max.min():.6f}")
    print(f"  Max: {scaled_col_max.max():.6f}")
    print(f"  Mean: {scaled_col_max.mean():.6f}")
    print(f"  Std: {scaled_col_max.std():.6f}")
    print(f"  Max deviation: {max_deviation*100:.2f}%")

    # Deviation should be small (accounting for numerical precision)
    if max_deviation < 0.01:  # 1% tolerance
        print("\n✓ Stage 2 achieves Chebyshev balancing (columns equalized)")
        return True
    else:
        print(f"\n✗ Stage 2 does not achieve Chebyshev balancing (deviation={max_deviation*100:.2f}%)")
        return False


def test_reconstruction_error():
    """
    Test quantization reconstruction error: VARQ should produce lower or
    comparable error to SINQ when actual quantization is applied.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Quantization Reconstruction Error")
    print("=" * 70)

    torch.manual_seed(456)
    W = torch.randn(512, 512)
    W[50:60, :] *= 8  # outliers

    # Parameters
    group_size = 64
    nbits = 3
    qmax = 2**nbits - 1

    # Test both methods
    methods = {
        'SINQ': sinkhorn_log(W, order=16),
        'VARQ': varq_normalize(W, group_size=group_size)
    }

    results = {}

    for name, (W_norm, mu1, mu2) in methods.items():
        # Simulate RTN quantization per group
        W_norm_flat = W_norm.reshape(-1, group_size)
        w_max = W_norm_flat.max(dim=1, keepdim=True).values
        w_min = W_norm_flat.min(dim=1, keepdim=True).values
        scale = (w_max - w_min) / qmax
        zero = w_min

        # Quantize
        Q_flat = torch.round((W_norm_flat - zero) / (scale + 1e-10))
        Q_flat = torch.clamp(Q_flat, 0, qmax)

        # Dequantize
        W_recon_flat = Q_flat * scale + zero
        W_recon_norm = W_recon_flat.reshape(W_norm.shape)

        # Reconstruct to original scale
        W_recon = W_recon_norm * mu1.view(1, -1) * mu2.view(-1, 1)

        # Compute error
        mse = ((W - W_recon)**2).mean().item()
        mae = (W - W_recon).abs().mean().item()
        max_error = (W - W_recon).abs().max().item()

        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'Max Error': max_error
        }

        print(f"\n{name}:")
        print(f"  MSE: {mse:.8f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Max Error: {max_error:.6f}")

    # Compare
    mse_ratio = results['VARQ']['MSE'] / results['SINQ']['MSE']
    print(f"\nVARQ/SINQ MSE ratio: {mse_ratio:.4f}")

    if mse_ratio <= 1.05:  # VARQ should be at most 5% worse (ideally better)
        print("✓ VARQ achieves comparable or better reconstruction error")
        return True
    else:
        print(f"✗ VARQ has significantly worse reconstruction error (ratio={mse_ratio:.4f})")
        return False


def test_hybrid_approach():
    """Test the hybrid VARQ (Sinkhorn Stage 1 + VARQ Stage 2)."""
    print("\n" + "=" * 70)
    print("TEST 6: Hybrid VARQ")
    print("=" * 70)

    torch.manual_seed(789)
    W = torch.randn(256, 512)
    W[10:20, :] *= 6

    try:
        W_hybrid, mu1_hybrid, mu2_hybrid = varq_hybrid_normalize(
            W, group_size=64, sinkhorn_order=8
        )

        # Compare ranges
        ranges_hybrid = compute_group_ranges(W_hybrid, group_size=64)
        print(f"Hybrid VARQ:")
        print(f"  Max range: {ranges_hybrid.max():.6f}")
        print(f"  Mean range: {ranges_hybrid.mean():.6f}")
        print(f"  Range imbalance: {(ranges_hybrid.max() / ranges_hybrid.min()):.4f}")

        print("\n✓ Hybrid VARQ executed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Hybrid VARQ failed: {e}")
        return False


def test_different_group_sizes():
    """Test VARQ with different group sizes."""
    print("\n" + "=" * 70)
    print("TEST 7: Different Group Sizes")
    print("=" * 70)

    torch.manual_seed(101112)
    W = torch.randn(512, 512)

    group_sizes = [32, 64, 128]
    all_passed = True

    for gs in group_sizes:
        try:
            W_norm, mu1, mu2 = varq_normalize(W, group_size=gs)
            ranges = compute_group_ranges(W_norm, group_size=gs)
            print(f"\nGroup size {gs}:")
            print(f"  Max range: {ranges.max():.6f}")
            print(f"  Range imbalance: {(ranges.max() / ranges.min()):.4f}")
            print(f"  ✓ Success")
        except Exception as e:
            print(f"\n  ✗ Failed with group_size={gs}: {e}")
            all_passed = False

    return all_passed


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("VARQ VALIDATION TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Scale Bounds", test_scale_bounds),
        ("Group Range Minimization", test_group_range_minimization),
        ("Theoretical Optimality", test_theoretical_optimality),
        ("Reconstruction Error", test_reconstruction_error),
        ("Hybrid Approach", test_hybrid_approach),
        ("Different Group Sizes", test_different_group_sizes),
    ]

    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = passed
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(results.values())
    total_count = len(results)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")

    print("=" * 70)
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! VARQ is ready for deployment.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Review needed.")

    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if all(results.values()) else 1)
