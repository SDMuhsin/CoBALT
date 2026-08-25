#!/usr/bin/env python3
"""
Benchmark CPR-SINQ matmul kernel against cuBLAS FP16.

This is the critical benchmark to verify we meet the throughput target:
- Target: >= 80% of cuBLAS FP16 throughput
"""

import torch
import time
import numpy as np
import cpr_kernels

def benchmark_cublas_fp16(batch_sizes, in_features, out_features, n_iters=100, warmup=20):
    """Benchmark cuBLAS FP16 matmul: Y = X @ W^T"""
    results = []

    for batch in batch_sizes:
        X = torch.randn(batch, in_features, dtype=torch.float16, device='cuda')
        W = torch.randn(out_features, in_features, dtype=torch.float16, device='cuda')

        # Warmup
        for _ in range(warmup):
            Y = torch.matmul(X, W.t())
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(n_iters):
            Y = torch.matmul(X, W.t())
        torch.cuda.synchronize()
        elapsed = time.time() - start

        time_ms = elapsed / n_iters * 1000
        flops = 2 * batch * in_features * out_features
        tflops = flops / (elapsed / n_iters) / 1e12

        results.append({
            'batch': batch,
            'time_ms': time_ms,
            'tflops': tflops,
        })

    return results


def benchmark_cpr_matmul(batch_sizes, in_features, out_features, high_frac=0.25,
                         n_iters=100, warmup=20, tile_size=128):
    """Benchmark CPR quantized matmul."""
    results = []

    n_high_cols = int(high_frac * in_features)
    n_low_cols = in_features - n_high_cols

    for batch in batch_sizes:
        X = torch.randn(batch, in_features, dtype=torch.float16, device='cuda')

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

        # Warmup
        for _ in range(warmup):
            Y = cpr_kernels.cpr_matmul(
                X, W_high_packed, W_low_packed,
                scales_high, zeros_high, scales_low, zeros_low,
                col_indices, n_high_cols, n_low_cols, tile_size
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(n_iters):
            Y = cpr_kernels.cpr_matmul(
                X, W_high_packed, W_low_packed,
                scales_high, zeros_high, scales_low, zeros_low,
                col_indices, n_high_cols, n_low_cols, tile_size
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        time_ms = elapsed / n_iters * 1000
        flops = 2 * batch * in_features * out_features
        tflops = flops / (elapsed / n_iters) / 1e12

        results.append({
            'batch': batch,
            'time_ms': time_ms,
            'tflops': tflops,
        })

    return results


def benchmark_dequant_then_matmul(batch_sizes, in_features, out_features, high_frac=0.25,
                                   n_iters=100, warmup=20, tile_size=128):
    """Benchmark dequantize + cuBLAS matmul approach."""
    results = []

    n_high_cols = int(high_frac * in_features)
    n_low_cols = in_features - n_high_cols

    for batch in batch_sizes:
        X = torch.randn(batch, in_features, dtype=torch.float16, device='cuda')

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

        # Warmup
        for _ in range(warmup):
            W_high_deq = cpr_kernels.dequantize_6bit(W_high_packed, scales_high, zeros_high, n_high_cols, tile_size)
            W_low_deq = cpr_kernels.dequantize_5bit(W_low_packed, scales_low, zeros_low, n_low_cols, tile_size)
            W = torch.cat([W_high_deq, W_low_deq], dim=1)
            Y = torch.matmul(X, W.t())
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(n_iters):
            W_high_deq = cpr_kernels.dequantize_6bit(W_high_packed, scales_high, zeros_high, n_high_cols, tile_size)
            W_low_deq = cpr_kernels.dequantize_5bit(W_low_packed, scales_low, zeros_low, n_low_cols, tile_size)
            W = torch.cat([W_high_deq, W_low_deq], dim=1)
            Y = torch.matmul(X, W.t())
        torch.cuda.synchronize()
        elapsed = time.time() - start

        time_ms = elapsed / n_iters * 1000
        flops = 2 * batch * in_features * out_features
        tflops = flops / (elapsed / n_iters) / 1e12

        results.append({
            'batch': batch,
            'time_ms': time_ms,
            'tflops': tflops,
        })

    return results


def main():
    print("=" * 70)
    print("CPR-SINQ MATMUL BENCHMARK")
    print("=" * 70)

    # Print device info
    cpr_kernels.check_cuda()

    # Test configurations (typical transformer layer sizes)
    configs = [
        # (in_features, out_features, description)
        (2048, 2048, "2K x 2K (small)"),
        (4096, 4096, "4K x 4K (medium)"),
        (4096, 11008, "4K x 11K (MLP up)"),
        (11008, 4096, "11K x 4K (MLP down)"),
    ]

    batch_sizes = [1, 8, 32, 128]

    for in_features, out_features, desc in configs:
        print("\n" + "=" * 70)
        print(f"Configuration: {desc}")
        print(f"Matrix: [{in_features}] x [{out_features}]^T")
        print("=" * 70)

        print("\n{:>10} {:>12} {:>12} {:>12} {:>10} {:>10}".format(
            "Batch", "cuBLAS(ms)", "CPR(ms)", "Deq+MM(ms)", "CPR/cuBLAS", "Target"
        ))
        print("-" * 70)

        cublas_results = benchmark_cublas_fp16(batch_sizes, in_features, out_features)
        cpr_results = benchmark_cpr_matmul(batch_sizes, in_features, out_features)
        deq_results = benchmark_dequant_then_matmul(batch_sizes, in_features, out_features)

        for i, batch in enumerate(batch_sizes):
            cublas_time = cublas_results[i]['time_ms']
            cpr_time = cpr_results[i]['time_ms']
            deq_time = deq_results[i]['time_ms']
            ratio = cublas_time / cpr_time  # Higher is better for CPR
            meets_target = ratio >= 0.8

            print("{:>10} {:>12.3f} {:>12.3f} {:>12.3f} {:>10.1%} {:>10}".format(
                batch, cublas_time, cpr_time, deq_time,
                ratio, "OK" if meets_target else "SLOW"
            ))

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Current naive kernel performance analysis:
- The naive kernel processes one output element per thread
- No shared memory tiling or memory coalescing optimization
- Full weight dequantization on every access (inefficient)

Expected speedup opportunities:
1. Shared memory tiling for weight reuse
2. Vectorized memory loads (float4)
3. Pre-computing dequantization to shared memory
4. Using Tensor Cores for the matmul (requires specific data layout)

Alternative approach (Dequant + cuBLAS):
- Dequantize weights to FP16 once
- Use highly optimized cuBLAS for matmul
- May be competitive for large batch sizes
- Memory overhead: need to store full FP16 weights temporarily
""")


if __name__ == "__main__":
    main()
