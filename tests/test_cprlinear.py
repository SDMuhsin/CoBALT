#!/usr/bin/env python3
"""
Test and benchmark CPRLinear module.
"""

import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '/workspace/SINQ')

from sinq.cprlinear import CPRLinear

def test_cprlinear_basic():
    """Test basic CPRLinear functionality."""
    print("\n" + "=" * 60)
    print("TEST: CPRLinear Basic Functionality")
    print("=" * 60)

    # Create original linear layer
    in_features = 512
    out_features = 256
    batch_size = 32

    linear = nn.Linear(in_features, out_features, bias=True).cuda().half()

    # Create CPRLinear from the linear layer
    cpr_linear = CPRLinear.from_linear(
        linear,
        high_frac=0.25,
        high_bits=6,
        low_bits=5,
        tile_size=128
    ).cuda()

    print(f"CPRLinear: {cpr_linear}")
    print(f"Average bits: {cpr_linear.compute_avg_bits():.3f}")

    # Test forward pass
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')

    with torch.no_grad():
        y_fp16 = linear(x)
        y_cpr = cpr_linear(x)

    # Compare outputs
    diff = (y_fp16 - y_cpr).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (y_fp16.abs() + 1e-6)).mean().item()

    print(f"\nOutput comparison (FP16 vs CPR):")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Mean relative difference: {rel_diff:.6e}")

    # The difference is expected to be non-zero due to quantization
    # For quantization, we check absolute error is reasonable
    if mean_diff < 0.1:  # Absolute error less than 0.1
        print("PASS: CPRLinear forward pass works correctly")
        return True
    else:
        print("FAIL: CPRLinear output differs too much from FP16")
        return False


def test_cprlinear_dequantize():
    """Test weight dequantization roundtrip."""
    print("\n" + "=" * 60)
    print("TEST: CPRLinear Dequantization")
    print("=" * 60)

    in_features = 512
    out_features = 256

    linear = nn.Linear(in_features, out_features, bias=True).cuda().half()
    original_weight = linear.weight.data.clone()

    cpr_linear = CPRLinear.from_linear(linear).cuda()

    # Dequantize weights
    deq_weight = cpr_linear.dequantize()

    # Compare
    diff = (original_weight - deq_weight).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (original_weight.abs() + 1e-6)).mean().item()

    print(f"Weight comparison (Original vs Dequantized):")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean relative difference: {rel_diff:.6e}")

    # Quantization error should be reasonable
    # For mixed-precision quantization, max absolute error < 0.01 is good
    if max_diff < 0.01:
        print("PASS: Dequantization preserves weight values well")
        return True
    else:
        print("FAIL: Dequantization error too high")
        return False


def benchmark_cprlinear():
    """Benchmark CPRLinear vs FP16 Linear."""
    print("\n" + "=" * 60)
    print("BENCHMARK: CPRLinear vs FP16 Linear")
    print("=" * 60)

    configs = [
        (4096, 4096, "4K x 4K"),
        (4096, 11008, "4K x 11K (MLP up)"),
        (11008, 4096, "11K x 4K (MLP down)"),
    ]

    batch_sizes = [1, 8, 32, 128]
    n_iters = 100
    warmup = 20

    for in_features, out_features, desc in configs:
        print(f"\n{desc}: [{in_features}] -> [{out_features}]")
        print("-" * 50)

        linear = nn.Linear(in_features, out_features, bias=True).cuda().half()
        cpr_linear = CPRLinear.from_linear(linear).cuda()

        print(f"Average bits: {cpr_linear.compute_avg_bits():.3f}")
        print("\n{:>10} {:>12} {:>12} {:>12} {:>12}".format(
            "Batch", "FP16(ms)", "CPR(ms)", "Cached(ms)", "Cached/FP16"
        ))

        for batch in batch_sizes:
            x = torch.randn(batch, in_features, dtype=torch.float16, device='cuda')

            # Warmup FP16
            with torch.no_grad():
                for _ in range(warmup):
                    y = linear(x)
            torch.cuda.synchronize()

            # Benchmark FP16
            start = time.time()
            with torch.no_grad():
                for _ in range(n_iters):
                    y = linear(x)
            torch.cuda.synchronize()
            fp16_time = (time.time() - start) / n_iters * 1000

            # Benchmark CPR (uncached - dequantize each pass)
            cpr_linear.clear_cache()
            with torch.no_grad():
                for _ in range(warmup):
                    y = cpr_linear(x)
            torch.cuda.synchronize()

            start = time.time()
            with torch.no_grad():
                for _ in range(n_iters):
                    y = cpr_linear(x)
            torch.cuda.synchronize()
            cpr_time = (time.time() - start) / n_iters * 1000

            # Benchmark CPR (cached - use cached weights)
            cpr_linear.cache_weights()
            with torch.no_grad():
                for _ in range(warmup):
                    y = cpr_linear(x)
            torch.cuda.synchronize()

            start = time.time()
            with torch.no_grad():
                for _ in range(n_iters):
                    y = cpr_linear(x)
            torch.cuda.synchronize()
            cached_time = (time.time() - start) / n_iters * 1000

            ratio = cached_time / fp16_time

            print("{:>10} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.1%}".format(
                batch, fp16_time, cpr_time, cached_time, ratio
            ))

            # Clear cache after benchmarking
            cpr_linear.clear_cache()


def main():
    print("=" * 60)
    print("CPRLinear Test Suite")
    print("=" * 60)

    results = {}
    results['basic'] = test_cprlinear_basic()
    results['dequantize'] = test_cprlinear_dequantize()

    benchmark_cprlinear()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
