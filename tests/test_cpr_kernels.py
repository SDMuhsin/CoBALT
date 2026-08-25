#!/usr/bin/env python3
"""
Test suite for CPR-SINQ CUDA kernels.

Tests:
1. Bit packing/unpacking correctness
2. Dequantization correctness
3. Full matmul correctness
4. Performance benchmarks
"""

import torch
import numpy as np
import time

# Must import torch before cpr_kernels
import cpr_kernels

def test_pack_unpack_6bit():
    """Test 6-bit packing and unpacking roundtrip."""
    print("\n" + "=" * 60)
    print("TEST: 6-bit Pack/Unpack")
    print("=" * 60)

    # Create test data: random values 0-63
    n_rows, n_cols = 256, 512
    weights = torch.randint(0, 64, (n_rows, n_cols), dtype=torch.int8, device='cuda')

    # Pack
    packed = cpr_kernels.pack_6bit(weights)
    expected_packed_cols = ((n_cols + 3) // 4) * 3
    print(f"Original shape: {weights.shape}")
    print(f"Packed shape: {packed.shape}")
    print(f"Expected packed cols: {expected_packed_cols}")

    assert packed.shape[1] == expected_packed_cols, f"Packed cols mismatch: {packed.shape[1]} vs {expected_packed_cols}"

    # Unpack
    unpacked = cpr_kernels.unpack_6bit(packed, n_cols)
    print(f"Unpacked shape: {unpacked.shape}")

    # Compare
    match = (weights == unpacked).all()
    if match:
        print("PASS: Pack/Unpack roundtrip matches")
    else:
        mismatch = (weights != unpacked).sum().item()
        print(f"FAIL: {mismatch} values mismatch out of {n_rows * n_cols}")
        # Debug: show first mismatch
        for i in range(min(10, n_rows)):
            for j in range(min(10, n_cols)):
                if weights[i, j] != unpacked[i, j]:
                    print(f"  Mismatch at [{i},{j}]: expected {weights[i,j].item()}, got {unpacked[i,j].item()}")

    return match


def test_pack_unpack_5bit():
    """Test 5-bit packing and unpacking roundtrip."""
    print("\n" + "=" * 60)
    print("TEST: 5-bit Pack/Unpack")
    print("=" * 60)

    # Create test data: random values 0-31
    n_rows, n_cols = 256, 512
    weights = torch.randint(0, 32, (n_rows, n_cols), dtype=torch.int8, device='cuda')

    # Pack
    packed = cpr_kernels.pack_5bit(weights)
    expected_packed_cols = ((n_cols + 7) // 8) * 5
    print(f"Original shape: {weights.shape}")
    print(f"Packed shape: {packed.shape}")
    print(f"Expected packed cols: {expected_packed_cols}")

    assert packed.shape[1] == expected_packed_cols, f"Packed cols mismatch"

    # Unpack
    unpacked = cpr_kernels.unpack_5bit(packed, n_cols)
    print(f"Unpacked shape: {unpacked.shape}")

    # Compare
    match = (weights == unpacked).all()
    if match:
        print("PASS: Pack/Unpack roundtrip matches")
    else:
        mismatch = (weights != unpacked).sum().item()
        print(f"FAIL: {mismatch} values mismatch out of {n_rows * n_cols}")

    return match


def test_dequantize_6bit():
    """Test 6-bit dequantization."""
    print("\n" + "=" * 60)
    print("TEST: 6-bit Dequantization")
    print("=" * 60)

    n_rows, n_cols = 256, 512
    tile_size = 128
    n_tiles = (n_cols + tile_size - 1) // tile_size

    # Create quantized weights
    weights_q = torch.randint(0, 64, (n_rows, n_cols), dtype=torch.int8, device='cuda')

    # Create scales and zeros
    scales = torch.randn(n_tiles, n_rows, dtype=torch.float16, device='cuda') * 0.01
    zeros = torch.randint(0, 64, (n_tiles, n_rows), dtype=torch.float16, device='cuda')

    # Pack weights
    packed = cpr_kernels.pack_6bit(weights_q)

    # Dequantize using CUDA kernel
    dequant_cuda = cpr_kernels.dequantize_6bit(packed, scales, zeros, n_cols, tile_size)

    # Dequantize using Python reference
    dequant_ref = torch.zeros(n_rows, n_cols, dtype=torch.float16, device='cuda')
    for t in range(n_tiles):
        col_start = t * tile_size
        col_end = min(col_start + tile_size, n_cols)
        for r in range(n_rows):
            scale = scales[t, r]
            zero = zeros[t, r]
            for c in range(col_start, col_end):
                dequant_ref[r, c] = (weights_q[r, c].float() - zero.float()) * scale.float()

    # Compare
    diff = (dequant_cuda.float() - dequant_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Allow small tolerance for FP16 precision
    if max_diff < 1e-2:
        print("PASS: Dequantization matches reference")
        return True
    else:
        print("FAIL: Dequantization mismatch")
        return False


def test_dequantize_5bit():
    """Test 5-bit dequantization."""
    print("\n" + "=" * 60)
    print("TEST: 5-bit Dequantization")
    print("=" * 60)

    n_rows, n_cols = 256, 512
    tile_size = 128
    n_tiles = (n_cols + tile_size - 1) // tile_size

    # Create quantized weights
    weights_q = torch.randint(0, 32, (n_rows, n_cols), dtype=torch.int8, device='cuda')

    # Create scales and zeros
    scales = torch.randn(n_tiles, n_rows, dtype=torch.float16, device='cuda') * 0.01
    zeros = torch.randint(0, 32, (n_tiles, n_rows), dtype=torch.float16, device='cuda')

    # Pack weights
    packed = cpr_kernels.pack_5bit(weights_q)

    # Dequantize using CUDA kernel
    dequant_cuda = cpr_kernels.dequantize_5bit(packed, scales, zeros, n_cols, tile_size)

    # Dequantize using Python reference
    dequant_ref = torch.zeros(n_rows, n_cols, dtype=torch.float16, device='cuda')
    for t in range(n_tiles):
        col_start = t * tile_size
        col_end = min(col_start + tile_size, n_cols)
        for r in range(n_rows):
            scale = scales[t, r]
            zero = zeros[t, r]
            for c in range(col_start, col_end):
                dequant_ref[r, c] = (weights_q[r, c].float() - zero.float()) * scale.float()

    # Compare
    diff = (dequant_cuda.float() - dequant_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-2:
        print("PASS: Dequantization matches reference")
        return True
    else:
        print("FAIL: Dequantization mismatch")
        return False


def test_cpr_matmul():
    """Test CPR matmul correctness."""
    print("\n" + "=" * 60)
    print("TEST: CPR Matrix Multiplication")
    print("=" * 60)

    batch_size = 32
    in_features = 512
    out_features = 256
    high_frac = 0.25
    tile_size = 128

    n_high_cols = int(high_frac * in_features)
    n_low_cols = in_features - n_high_cols

    print(f"Batch: {batch_size}, In: {in_features}, Out: {out_features}")
    print(f"High cols: {n_high_cols} (6-bit), Low cols: {n_low_cols} (5-bit)")

    # Create input
    X = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')

    # Create column permutation (identity for now)
    col_indices = torch.arange(in_features, dtype=torch.int16, device='cuda')

    # Create quantized weights
    W_high_q = torch.randint(0, 64, (out_features, n_high_cols), dtype=torch.int8, device='cuda')
    W_low_q = torch.randint(0, 32, (out_features, n_low_cols), dtype=torch.int8, device='cuda')

    # Create scales and zeros
    n_tiles_high = (n_high_cols + tile_size - 1) // tile_size
    n_tiles_low = (n_low_cols + tile_size - 1) // tile_size

    scales_high = torch.randn(n_tiles_high, out_features, dtype=torch.float16, device='cuda') * 0.01
    zeros_high = torch.zeros(n_tiles_high, out_features, dtype=torch.float16, device='cuda')
    scales_low = torch.randn(n_tiles_low, out_features, dtype=torch.float16, device='cuda') * 0.01
    zeros_low = torch.zeros(n_tiles_low, out_features, dtype=torch.float16, device='cuda')

    # Pack weights
    W_high_packed = cpr_kernels.pack_6bit(W_high_q)
    W_low_packed = cpr_kernels.pack_5bit(W_low_q)

    print(f"W_high_packed shape: {W_high_packed.shape}")
    print(f"W_low_packed shape: {W_low_packed.shape}")

    # CPR matmul
    Y_cuda = cpr_kernels.cpr_matmul(
        X, W_high_packed, W_low_packed,
        scales_high, zeros_high, scales_low, zeros_low,
        col_indices, n_high_cols, n_low_cols, tile_size
    )

    # Reference: dequantize and matmul
    # Dequantize high
    W_high_deq = torch.zeros(out_features, n_high_cols, dtype=torch.float16, device='cuda')
    for t in range(n_tiles_high):
        col_start = t * tile_size
        col_end = min(col_start + tile_size, n_high_cols)
        for n in range(out_features):
            for c in range(col_start, col_end):
                W_high_deq[n, c] = (W_high_q[n, c].float() - zeros_high[t, n].float()) * scales_high[t, n].float()

    # Dequantize low
    W_low_deq = torch.zeros(out_features, n_low_cols, dtype=torch.float16, device='cuda')
    for t in range(n_tiles_low):
        col_start = t * tile_size
        col_end = min(col_start + tile_size, n_low_cols)
        for n in range(out_features):
            for c in range(col_start, col_end):
                W_low_deq[n, c] = (W_low_q[n, c].float() - zeros_low[t, n].float()) * scales_low[t, n].float()

    # Combine weights (no permutation for this test)
    W_full = torch.cat([W_high_deq, W_low_deq], dim=1)

    # Reference matmul
    Y_ref = torch.matmul(X, W_full.t())

    # Compare
    diff = (Y_cuda.float() - Y_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (Y_ref.float().abs() + 1e-6)).mean().item()

    print(f"Output shape: {Y_cuda.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Mean relative difference: {rel_diff:.6e}")

    # Allow tolerance for accumulated FP16 errors
    if rel_diff < 0.01:  # 1% relative error
        print("PASS: CPR matmul matches reference")
        return True
    else:
        print("FAIL: CPR matmul mismatch")
        return False


def benchmark_pack_unpack():
    """Benchmark packing/unpacking throughput."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Packing/Unpacking Throughput")
    print("=" * 60)

    n_rows, n_cols = 4096, 4096
    n_iters = 100

    weights_6bit = torch.randint(0, 64, (n_rows, n_cols), dtype=torch.int8, device='cuda')
    weights_5bit = torch.randint(0, 32, (n_rows, n_cols), dtype=torch.int8, device='cuda')

    # Warmup
    for _ in range(10):
        _ = cpr_kernels.pack_6bit(weights_6bit)
        _ = cpr_kernels.pack_5bit(weights_5bit)

    torch.cuda.synchronize()

    # Benchmark 6-bit pack
    start = time.time()
    for _ in range(n_iters):
        packed = cpr_kernels.pack_6bit(weights_6bit)
    torch.cuda.synchronize()
    time_6bit_pack = (time.time() - start) / n_iters

    # Benchmark 5-bit pack
    start = time.time()
    for _ in range(n_iters):
        packed = cpr_kernels.pack_5bit(weights_5bit)
    torch.cuda.synchronize()
    time_5bit_pack = (time.time() - start) / n_iters

    # Benchmark unpack
    packed_6 = cpr_kernels.pack_6bit(weights_6bit)
    packed_5 = cpr_kernels.pack_5bit(weights_5bit)

    start = time.time()
    for _ in range(n_iters):
        _ = cpr_kernels.unpack_6bit(packed_6, n_cols)
    torch.cuda.synchronize()
    time_6bit_unpack = (time.time() - start) / n_iters

    start = time.time()
    for _ in range(n_iters):
        _ = cpr_kernels.unpack_5bit(packed_5, n_cols)
    torch.cuda.synchronize()
    time_5bit_unpack = (time.time() - start) / n_iters

    data_size_mb = n_rows * n_cols / 1e6

    print(f"Matrix size: {n_rows} x {n_cols}")
    print(f"6-bit pack:   {time_6bit_pack*1000:.3f} ms ({data_size_mb/time_6bit_pack:.1f} MB/s)")
    print(f"6-bit unpack: {time_6bit_unpack*1000:.3f} ms ({data_size_mb/time_6bit_unpack:.1f} MB/s)")
    print(f"5-bit pack:   {time_5bit_pack*1000:.3f} ms ({data_size_mb/time_5bit_pack:.1f} MB/s)")
    print(f"5-bit unpack: {time_5bit_unpack*1000:.3f} ms ({data_size_mb/time_5bit_unpack:.1f} MB/s)")


def main():
    print("=" * 60)
    print("CPR-SINQ CUDA Kernel Tests")
    print("=" * 60)

    # Print CUDA info
    cpr_kernels.check_cuda()

    results = {}

    # Run tests
    results['pack_6bit'] = test_pack_unpack_6bit()
    results['pack_5bit'] = test_pack_unpack_5bit()
    results['dequant_6bit'] = test_dequantize_6bit()
    results['dequant_5bit'] = test_dequantize_5bit()
    results['cpr_matmul'] = test_cpr_matmul()

    # Run benchmarks
    benchmark_pack_unpack()

    # Summary
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
    import sys
    success = main()
    sys.exit(0 if success else 1)
