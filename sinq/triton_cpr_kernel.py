"""Triton-based CPR fused matmul kernel.

This implements a fused dequantize + matmul kernel for CPR quantization.
The weights are stored compressed and dequantized on-the-fly during matmul.
"""
import torch
import triton
import triton.language as tl


def get_autotune_config():
    """Autotuning configurations for the CPR matmul kernel."""
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]


# =============================================================================
# First: Standard FP16 matmul with Triton (baseline)
# =============================================================================

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_fp16(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Standard FP16 matmul kernel using Triton."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP16 matmul using Triton."""
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    matmul_kernel_fp16[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# =============================================================================
# Second: CPR Dequantize + Matmul Fused Kernel
# =============================================================================

@triton.jit
def dequantize_6bit_tile(
    packed_ptr,
    scales_ptr,
    zeros_ptr,
    n_rows: int,
    n_cols: int,
    packed_stride: int,
    offs_row,
    offs_col,
    BLOCK_K: tl.constexpr,
):
    """
    Dequantize a tile of 6-bit packed weights to FP16.

    6-bit packing: 4 values packed into 3 bytes
    Values 0-63 -> dequantized using scale and zero point

    Returns: [BLOCK_K, BLOCK_N] tile of FP16 weights
    """
    # For simplicity, we'll implement 8-bit first (simpler packing)
    # TODO: Implement proper 6-bit unpacking
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def cpr_matmul_kernel_int8(
    # Input activation (FP16)
    x_ptr,
    # Quantized weights (INT8)
    w_ptr,
    # Scales and zeros for dequantization
    scales_ptr, zeros_ptr,
    # Output (FP16)
    y_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    # Quantization params
    group_size,  # Quantization group size
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    CPR fused dequantize + matmul kernel.

    Y = X @ W^T where W is stored as INT8 and dequantized on-the-fly.

    W is [K, N] stored as INT8
    scales is [K // group_size, N]
    zeros is [K // group_size, N]
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Input pointers
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    # Weight pointers (INT8)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_tiles):
        k = k_idx * BLOCK_K

        # Load input tile (FP16)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load weight tile (INT8)
        w_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)

        # Load scale and zero for this K range
        # Assuming group_size divides K, each K tile has its own scale/zero
        scale_idx = k // group_size
        scale_ptrs = scales_ptr + scale_idx * N + offs_n
        zero_ptrs = zeros_ptr + scale_idx * N + offs_n

        # Broadcast to [BLOCK_K, BLOCK_N]
        scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0)[None, :]
        zeros = tl.load(zero_ptrs, mask=offs_n < N, other=0.0)[None, :]

        # Dequantize: w_fp16 = (w_int8 - zero) * scale
        w_fp16 = (w_int8.to(tl.float16) - zeros) * scales

        # Matmul
        accumulator = tl.dot(x, w_fp16, accumulator)

        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Store output
    y = accumulator.to(tl.float16)
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def triton_cpr_matmul_int8(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    CPR fused dequantize + matmul using Triton.

    Args:
        x: Input activation [M, K] FP16
        w: Quantized weights [K, N] INT8
        scales: Per-group scales [K // group_size, N] FP16
        zeros: Per-group zeros [K // group_size, N] FP16
        group_size: Quantization group size

    Returns:
        y: Output [M, N] FP16
    """
    assert x.dim() == 2 and w.dim() == 2
    M, K = x.shape
    K2, N = w.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    cpr_matmul_kernel_int8[grid](
        x, w, scales, zeros, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        group_size,
    )
    return y


# =============================================================================
# Testing
# =============================================================================

def test_triton_fp16():
    """Test FP16 Triton matmul."""
    import time

    torch.manual_seed(42)
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(5):
        _ = triton_matmul_fp16(A, B)
    torch.cuda.synchronize()

    # Correctness
    C_ref = torch.matmul(A, B)
    C_triton = triton_matmul_fp16(A, B)
    torch.cuda.synchronize()

    max_err = (C_ref - C_triton).abs().max().item()
    mean_err = (C_ref - C_triton).abs().mean().item()
    print(f'Triton FP16 correctness: max_err={max_err:.6f}, mean_err={mean_err:.6f}')

    # Benchmark
    def benchmark(fn, warmup=10, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / iters * 1000
        tflops = 2 * M * N * K / (avg_ms / 1000) / 1e12
        return avg_ms, tflops

    cublas_ms, cublas_tflops = benchmark(lambda: torch.matmul(A, B))
    triton_ms, triton_tflops = benchmark(lambda: triton_matmul_fp16(A, B))

    print(f'cuBLAS: {cublas_ms:.3f}ms  {cublas_tflops:.1f} TFLOPS')
    print(f'Triton: {triton_ms:.3f}ms  {triton_tflops:.1f} TFLOPS  ({100*triton_tflops/cublas_tflops:.1f}%)')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def cpr_matmul_kernel_int8_v2(
    # Input activation (FP16)
    x_ptr,
    # Quantized weights (INT8)
    w_ptr,
    # Scales for dequantization [num_groups, N]
    scales_ptr,
    # Output (FP16)
    y_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    stride_sk, stride_sn,
    # Quantization params
    group_size,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized CPR fused dequantize + matmul kernel.

    Uses asymmetric quantization with zero=0 (symmetric quantization).
    Dequantize: w_fp16 = w_int8 * scale
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Input pointers
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    # Weight pointers (INT8)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_tiles):
        k = k_idx * BLOCK_K

        # Load input tile (FP16)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load weight tile (INT8)
        w_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)

        # Load scale for this K range
        scale_idx = k // group_size
        scale_ptrs = scales_ptr + scale_idx * stride_sk + offs_n * stride_sn
        scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0)

        # Dequantize: w_fp16 = w_int8 * scale (symmetric quantization)
        w_fp16 = w_int8.to(tl.float16) * scales[None, :]

        # Matmul
        accumulator = tl.dot(x, w_fp16, accumulator)

        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Store output
    y = accumulator.to(tl.float16)
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def triton_cpr_matmul_int8_v2(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Optimized CPR fused dequantize + matmul using Triton.
    Uses symmetric quantization (zero=0).
    """
    assert x.dim() == 2 and w.dim() == 2
    M, K = x.shape
    K2, N = w.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    cpr_matmul_kernel_int8_v2[grid](
        x, w, scales, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        scales.stride(0), scales.stride(1),
        group_size,
    )
    return y


def test_triton_int8_dequant():
    """Test INT8 dequantize + matmul kernel."""
    import time

    torch.manual_seed(42)
    M, N, K = 4096, 4096, 4096
    group_size = 128

    # Create input
    x = torch.randn(M, K, dtype=torch.float16, device='cuda')

    # Create quantized weights
    w_fp16 = torch.randn(K, N, dtype=torch.float16, device='cuda')

    # Simulate quantization
    num_groups = K // group_size
    w_reshaped = w_fp16.view(num_groups, group_size, N)
    scales = w_reshaped.abs().amax(dim=1, keepdim=True) / 127.0
    scales = scales.squeeze(1).half()  # [num_groups, N]
    zeros = torch.zeros_like(scales)

    # Quantize
    w_int8 = torch.round(w_reshaped / scales[:, None, :]).clamp(-128, 127).to(torch.int8)
    w_int8 = w_int8.view(K, N)

    # Reference: dequantize then matmul
    w_dequant = w_int8.view(num_groups, group_size, N).float()
    w_dequant = (w_dequant * scales[:, None, :].float()).view(K, N).half()
    y_ref = torch.matmul(x, w_dequant)

    # Warmup
    for _ in range(5):
        _ = triton_cpr_matmul_int8(x, w_int8, scales, zeros, group_size)
    torch.cuda.synchronize()

    # Test fused kernel
    y_fused = triton_cpr_matmul_int8(x, w_int8, scales, zeros, group_size)
    torch.cuda.synchronize()

    max_err = (y_ref - y_fused).abs().max().item()
    mean_err = (y_ref - y_fused).abs().mean().item()
    print(f'Triton INT8 correctness: max_err={max_err:.6f}, mean_err={mean_err:.6f}')

    # Benchmark
    def benchmark(fn, warmup=10, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / iters * 1000
        tflops = 2 * M * N * K / (avg_ms / 1000) / 1e12
        return avg_ms, tflops

    cublas_ms, cublas_tflops = benchmark(lambda: torch.matmul(x, w_fp16))
    fused_ms, fused_tflops = benchmark(lambda: triton_cpr_matmul_int8(x, w_int8, scales, zeros, group_size))

    print(f'cuBLAS FP16:  {cublas_ms:.3f}ms  {cublas_tflops:.1f} TFLOPS')
    print(f'Triton INT8:  {fused_ms:.3f}ms  {fused_tflops:.1f} TFLOPS  ({100*fused_tflops/cublas_tflops:.1f}%)')


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Triton FP16 matmul")
    print("=" * 60)
    test_triton_fp16()
    print()

    print("=" * 60)
    print("Testing Triton INT8 dequantize + matmul")
    print("=" * 60)
    test_triton_int8_dequant()
