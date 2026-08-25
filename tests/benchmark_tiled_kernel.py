#!/usr/bin/env python3
"""
Benchmark the tiled CPR matmul kernel.
"""

import torch
import time
import cpr_kernels


def benchmark_tiled_vs_naive(batch_sizes, in_features, out_features, high_frac=0.25,
                              n_iters=100, warmup=20, tile_size=128):
    """Compare tiled kernel against naive and cuBLAS."""

    n_high_cols = int(high_frac * in_features)
    n_low_cols = in_features - n_high_cols

    print(f"\nMatrix: batch x [{in_features}] @ [{out_features}, {in_features}]^T")
    print(f"High cols: {n_high_cols} (6-bit), Low cols: {n_low_cols} (5-bit)")
    print("\n{:>10} {:>12} {:>12} {:>12} {:>12}".format(
        "Batch", "cuBLAS(ms)", "Naive(ms)", "Tiled(ms)", "Tiled/cuBLAS"
    ))
    print("-" * 60)

    for batch in batch_sizes:
        X = torch.randn(batch, in_features, dtype=torch.float16, device='cuda')
        W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device='cuda')

        # Create column permutation
        col_indices = torch.arange(in_features, dtype=torch.int16, device='cuda')

        # Create quantized weights
        W_high_q = torch.randint(0, 64, (out_features, n_high_cols), dtype=torch.int8, device='cuda')
        W_low_q = torch.randint(0, 32, (out_features, n_low_cols), dtype=torch.int8, device='cuda')

        # Pack weights
        W_high_packed = cpr_kernels.pack_6bit(W_high_q)
        W_low_packed = cpr_kernels.pack_5bit(W_low_q)

        # Create scales and zeros
        n_tiles_high = (n_high_cols + tile_size - 1) // tile_size
        n_tiles_low = (n_low_cols + tile_size - 1) // tile_size

        scales_high = torch.randn(n_tiles_high, out_features, dtype=torch.float16, device='cuda') * 0.01
        zeros_high = torch.zeros(n_tiles_high, out_features, dtype=torch.float16, device='cuda')
        scales_low = torch.randn(n_tiles_low, out_features, dtype=torch.float16, device='cuda') * 0.01
        zeros_low = torch.zeros(n_tiles_low, out_features, dtype=torch.float16, device='cuda')

        # Warmup cuBLAS
        for _ in range(warmup):
            Y = torch.matmul(X, W_fp16.t())
        torch.cuda.synchronize()

        # Benchmark cuBLAS
        start = time.time()
        for _ in range(n_iters):
            Y = torch.matmul(X, W_fp16.t())
        torch.cuda.synchronize()
        cublas_time = (time.time() - start) / n_iters * 1000

        # Warmup naive
        for _ in range(min(5, warmup)):
            Y = cpr_kernels.cpr_matmul(
                X, W_high_packed, W_low_packed,
                scales_high, zeros_high, scales_low, zeros_low,
                col_indices, n_high_cols, n_low_cols, tile_size
            )
        torch.cuda.synchronize()

        # Benchmark naive (fewer iterations since it's slow)
        naive_iters = min(10, n_iters)
        start = time.time()
        for _ in range(naive_iters):
            Y = cpr_kernels.cpr_matmul(
                X, W_high_packed, W_low_packed,
                scales_high, zeros_high, scales_low, zeros_low,
                col_indices, n_high_cols, n_low_cols, tile_size
            )
        torch.cuda.synchronize()
        naive_time = (time.time() - start) / naive_iters * 1000

        # Warmup tiled
        for _ in range(warmup):
            Y = cpr_kernels.cpr_matmul_tiled(
                X, W_high_packed, W_low_packed,
                scales_high, zeros_high, scales_low, zeros_low,
                col_indices, n_high_cols, n_low_cols, tile_size
            )
        torch.cuda.synchronize()

        # Benchmark tiled
        start = time.time()
        for _ in range(n_iters):
            Y = cpr_kernels.cpr_matmul_tiled(
                X, W_high_packed, W_low_packed,
                scales_high, zeros_high, scales_low, zeros_low,
                col_indices, n_high_cols, n_low_cols, tile_size
            )
        torch.cuda.synchronize()
        tiled_time = (time.time() - start) / n_iters * 1000

        ratio = cublas_time / tiled_time

        print("{:>10} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.1%}".format(
            batch, cublas_time, naive_time, tiled_time, ratio
        ))


def test_tiled_correctness():
    """Verify tiled kernel gives correct results."""
    print("\n" + "=" * 60)
    print("CORRECTNESS TEST: Tiled vs Naive")
    print("=" * 60)

    batch = 32
    in_features = 512
    out_features = 256
    high_frac = 0.25
    tile_size = 128

    n_high_cols = int(high_frac * in_features)
    n_low_cols = in_features - n_high_cols

    X = torch.randn(batch, in_features, dtype=torch.float16, device='cuda')
    col_indices = torch.arange(in_features, dtype=torch.int16, device='cuda')

    W_high_q = torch.randint(0, 64, (out_features, n_high_cols), dtype=torch.int8, device='cuda')
    W_low_q = torch.randint(0, 32, (out_features, n_low_cols), dtype=torch.int8, device='cuda')

    W_high_packed = cpr_kernels.pack_6bit(W_high_q)
    W_low_packed = cpr_kernels.pack_5bit(W_low_q)

    n_tiles_high = (n_high_cols + tile_size - 1) // tile_size
    n_tiles_low = (n_low_cols + tile_size - 1) // tile_size

    scales_high = torch.randn(n_tiles_high, out_features, dtype=torch.float16, device='cuda') * 0.01
    zeros_high = torch.zeros(n_tiles_high, out_features, dtype=torch.float16, device='cuda')
    scales_low = torch.randn(n_tiles_low, out_features, dtype=torch.float16, device='cuda') * 0.01
    zeros_low = torch.zeros(n_tiles_low, out_features, dtype=torch.float16, device='cuda')

    # Compute with naive kernel
    Y_naive = cpr_kernels.cpr_matmul(
        X, W_high_packed, W_low_packed,
        scales_high, zeros_high, scales_low, zeros_low,
        col_indices, n_high_cols, n_low_cols, tile_size
    )

    # Compute with tiled kernel
    Y_tiled = cpr_kernels.cpr_matmul_tiled(
        X, W_high_packed, W_low_packed,
        scales_high, zeros_high, scales_low, zeros_low,
        col_indices, n_high_cols, n_low_cols, tile_size
    )

    # Compare
    diff = (Y_naive.float() - Y_tiled.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (Y_naive.float().abs() + 1e-6)).mean().item()

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Mean relative difference: {rel_diff:.6e}")

    if rel_diff < 0.01:
        print("PASS: Tiled kernel matches naive kernel")
        return True
    else:
        print("FAIL: Tiled kernel mismatch")
        return False


def main():
    print("=" * 60)
    print("CPR-SINQ TILED KERNEL BENCHMARK")
    print("=" * 60)

    cpr_kernels.check_cuda()

    # Test correctness first
    correct = test_tiled_correctness()
    if not correct:
        print("Aborting benchmarks due to correctness failure")
        return

    # Benchmark
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    batch_sizes = [1, 8, 32, 128]

    # Small matrices
    benchmark_tiled_vs_naive(batch_sizes, 2048, 2048)

    # Medium matrices
    benchmark_tiled_vs_naive(batch_sizes, 4096, 4096)


if __name__ == "__main__":
    main()
