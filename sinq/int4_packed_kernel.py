"""
INT4 Packed Quantization with Fused Triton Kernel

This implements ACTUAL 4-bit packed storage:
- 2 INT4 values packed per byte (real 75% weight memory reduction)
- Triton kernel that unpacks during matmul
- Per-group scales for quality preservation

Goal: Real measurable memory reduction, accept speed tradeoff.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple


# =============================================================================
# 4-bit Packing Utilities
# =============================================================================

def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack INT8 tensor (values in -8..7 range) to INT4 packed format.
    Two 4-bit values per byte along the FIRST dimension (K).

    Args:
        tensor: [K, N] INT8 tensor with values in -8..7 range

    Returns:
        [K//2, N] UINT8 tensor with packed values
    """
    assert tensor.dtype == torch.int8
    K, N = tensor.shape
    assert K % 2 == 0, f"K must be even, got {K}"

    # Reshape to [K//2, 2, N]
    tensor = tensor.view(K // 2, 2, N)

    # Convert to unsigned (add 8 to shift from -8..7 to 0..15)
    tensor_u = (tensor + 8).to(torch.uint8)

    # Pack: low nibble from index 0, high nibble from index 1
    # tensor_u[:, 0, :] are even K indices, tensor_u[:, 1, :] are odd K indices
    packed = (tensor_u[:, 0, :] & 0x0F) | ((tensor_u[:, 1, :] & 0x0F) << 4)

    return packed  # [K//2, N]


def unpack_int4(packed: torch.Tensor, original_k: int) -> torch.Tensor:
    """
    Unpack INT4 packed format back to INT8.

    Args:
        packed: [K//2, N] UINT8 tensor
        original_k: Original K dimension size

    Returns:
        [K, N] INT8 tensor with values in -8..7 range
    """
    packed_k, N = packed.shape
    assert packed_k == original_k // 2

    # Extract low and high nibbles
    low = (packed & 0x0F).to(torch.int8) - 8  # Convert back to signed, [K//2, N]
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # [K//2, N]

    # Stack and reshape to interleave: [K//2, 2, N] -> [K, N]
    result = torch.stack([low, high], dim=1).view(original_k, N)
    return result


# =============================================================================
# Triton Kernel for INT4 Packed Matmul
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def int4_packed_matmul_kernel(
    # Input [M, K] FP16
    x_ptr,
    # Packed weights [K//2, N] UINT8 (2 values per byte)
    w_packed_ptr,
    # Scales [num_groups, N] FP16
    scales_ptr,
    # Output [M, N] FP16
    y_ptr,
    # Dimensions
    M, N, K,
    # Quantization
    group_size,
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,  # Note: stride_wk is for packed dimension (K//2)
    stride_ym, stride_yn,
    stride_sk, stride_sn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused INT4 unpack + dequantize + matmul kernel.

    Weights are stored packed: 2 INT4 values per byte.
    Low nibble = even K index, high nibble = odd K index.
    Unpacking: value = (packed >> shift) & 0x0F - 8
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # For packed weights, we need to handle K in pairs
    # BLOCK_K must be even
    offs_k = tl.arange(0, BLOCK_K)

    # Input pointers
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # Packed weight pointers - K//2 rows
    # offs_k_packed indexes into packed dimension
    offs_k_packed = offs_k // 2
    w_ptrs = w_packed_ptr + offs_k_packed[:, None] * stride_wk + offs_n[None, :] * stride_wn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)

    for k_idx in range(num_k_iters):
        k = k_idx * BLOCK_K

        # Load input tile
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load packed weights [BLOCK_K//2, BLOCK_N]
        # Each byte contains 2 values
        k_packed = k // 2
        w_mask_packed = ((offs_k // 2)[:, None] < (K - k + 1) // 2) & (offs_n[None, :] < N)
        w_packed = tl.load(w_ptrs, mask=w_mask_packed, other=0)

        # Unpack: determine if each K index is even (low nibble) or odd (high nibble)
        is_odd = (offs_k % 2) == 1

        # Extract values: low nibble for even indices, high nibble for odd
        w_low = (w_packed & 0x0F).to(tl.int8) - 8  # Low nibble
        w_high = ((w_packed >> 4) & 0x0F).to(tl.int8) - 8  # High nibble

        # Select based on odd/even - this is a broadcast operation
        # We need to duplicate rows to get proper alignment
        # Actually, we loaded with offs_k_packed = offs_k // 2
        # So w_packed[i, j] contains values for k=2*i and k=2*i+1
        # We need to reshape/broadcast properly

        # For proper handling, we expand the packed weights
        # w_unpacked[k, n] = low[k//2, n] if k%2==0 else high[k//2, n]
        w_int8 = tl.where(is_odd[:, None], w_high, w_low)

        # Load scales for this K block
        scale_idx = k // group_size
        scale_ptrs = scales_ptr + scale_idx * stride_sk + offs_n * stride_sn
        scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0)

        # Dequantize
        w_fp16 = w_int8.to(tl.float16) * scales[None, :]

        # Matmul
        accumulator = tl.dot(x, w_fp16, accumulator)

        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += (BLOCK_K // 2) * stride_wk

    # Store output
    y = accumulator.to(tl.float16)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def int4_packed_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    INT4 packed matmul.

    Args:
        x: Input [M, K] FP16
        w_packed: Packed weights [K//2, N] UINT8
        scales: Scales [num_groups, N] FP16
        K: Original K dimension (before packing)
        group_size: Quantization group size

    Returns:
        Output [M, N] FP16
    """
    M, K_input = x.shape
    assert K_input == K, f"K mismatch: {K_input} vs {K}"
    K_packed, N = w_packed.shape
    assert K_packed == K // 2, f"Packed K mismatch: {K_packed} vs {K // 2}"

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    int4_packed_matmul_kernel[grid](
        x, w_packed, scales, y,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        y.stride(0), y.stride(1),
        scales.stride(0), scales.stride(1),
    )

    return y


# =============================================================================
# INT4PackedLinear: Drop-in replacement for nn.Linear
# =============================================================================

class INT4PackedLinear(nn.Module):
    """
    Linear layer with actual 4-bit packed weight storage.

    Memory usage:
    - FP16: 2 bytes per weight
    - INT4: 0.5 bytes per weight (4x reduction on weights)
    - Plus scales: ~1% overhead

    Total: ~75% memory reduction on weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = compute_dtype

        # Ensure in_features is even (for packing)
        assert in_features % 2 == 0, f"in_features must be even, got {in_features}"

        # Number of groups
        self.num_groups = (in_features + group_size - 1) // group_size

        if device is None:
            device = 'cuda'
        self._device = torch.device(device)

        # Packed weights: [in_features // 2, out_features] UINT8
        # This is the ACTUAL storage - 0.5 bytes per weight
        self.register_buffer('weight_packed',
            torch.zeros(in_features // 2, out_features, dtype=torch.uint8, device=self._device))

        # Per-group scales: [num_groups, out_features] FP16
        self.register_buffer('scales',
            torch.ones(self.num_groups, out_features, dtype=compute_dtype, device=self._device))

        # Bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=compute_dtype, device=self._device))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
    ) -> 'INT4PackedLinear':
        """Create from existing nn.Linear."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            compute_dtype=compute_dtype,
            device=linear.weight.device,
        )

        layer.quantize_weights(linear.weight.data)

        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(compute_dtype)

        return layer

    def quantize_weights(self, weight: torch.Tensor):
        """
        Quantize weights to INT4 packed format.

        Args:
            weight: [out_features, in_features] FP16/FP32
        """
        device = weight.device
        W = weight.float()  # [N, K]
        N, K = W.shape

        # Transpose to [K, N] for our kernel layout
        W_t = W.t()  # [K, N]

        # Pad K to multiple of group_size
        padded_k = self.num_groups * self.group_size
        if padded_k > K:
            W_t = torch.nn.functional.pad(W_t, (0, 0, 0, padded_k - K))

        # Reshape for group quantization: [num_groups, group_size, N]
        W_grouped = W_t.view(self.num_groups, self.group_size, N)

        # Per-group scales (symmetric quantization to INT4: -8 to 7)
        max_abs = W_grouped.abs().amax(dim=1)  # [num_groups, N]
        scales = (max_abs / 7.0).clamp(min=1e-8)  # 7 is max positive INT4
        self.scales.copy_(scales.to(self.compute_dtype))

        # Quantize
        scales_expanded = scales.unsqueeze(1)  # [num_groups, 1, N]
        W_q = torch.round(W_grouped / scales_expanded)
        W_q = W_q.clamp(-8, 7).to(torch.int8)

        # Reshape back to [K, N]
        W_q = W_q.view(padded_k, N)[:K, :]

        # Pack to INT4: 2 values per byte
        # W_q is [K, N], pack along K dimension
        packed = pack_int4(W_q)  # [K//2, N]
        self.weight_packed.copy_(packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with INT4 unpacking."""
        x = x.to(self.compute_dtype)

        # Handle 3D input
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x = x.view(batch * seq_len, -1)

        # Use fused kernel
        out = int4_packed_matmul(
            x,
            self.weight_packed,
            self.scales,
            self.in_features,
            self.group_size,
        )

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Restore shape
        if len(original_shape) == 3:
            out = out.view(batch, seq_len, -1)

        return out

    def memory_bytes(self) -> int:
        """Return actual memory in bytes."""
        weight_bytes = self.weight_packed.numel() * 1  # UINT8 = 1 byte, but holds 2 values
        scale_bytes = self.scales.numel() * 2  # FP16
        bias_bytes = self.bias.numel() * 2 if self.bias is not None else 0
        return weight_bytes + scale_bytes + bias_bytes

    def memory_bytes_fp16_equivalent(self) -> int:
        """Return what FP16 would use."""
        weight_bytes = self.in_features * self.out_features * 2  # FP16
        bias_bytes = self.out_features * 2 if self.bias is not None else 0
        return weight_bytes + bias_bytes

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bits=4, group_size={self.group_size}')


# =============================================================================
# Testing
# =============================================================================

def test_packing():
    """Test INT4 packing/unpacking."""
    print("Testing INT4 packing...")

    # Create test tensor
    K, N = 128, 64
    original = torch.randint(-8, 8, (K, N), dtype=torch.int8, device='cuda')

    # Pack
    packed = pack_int4(original)
    print(f"Original shape: {original.shape}, Packed shape: {packed.shape}")
    assert packed.shape == (K // 2, N)

    # Unpack
    unpacked = unpack_int4(packed, K)
    assert unpacked.shape == original.shape

    # Verify
    if torch.equal(original, unpacked):
        print("PASS: Packing/unpacking is lossless")
    else:
        diff = (original != unpacked).sum().item()
        print(f"FAIL: {diff} values differ")

    return torch.equal(original, unpacked)


def test_int4_linear():
    """Test INT4PackedLinear."""
    import time

    print("\n" + "=" * 70)
    print("Testing INT4PackedLinear")
    print("=" * 70)

    torch.manual_seed(42)

    in_features, out_features = 4096, 4096
    batch_size = 8

    # Create reference layer
    linear = nn.Linear(in_features, out_features, dtype=torch.float16, device='cuda')

    # Create INT4 layer
    int4_layer = INT4PackedLinear.from_linear(linear, group_size=128)

    # Test input
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')

    # Correctness
    with torch.no_grad():
        y_fp16 = linear(x)
        y_int4 = int4_layer(x)

    max_err = (y_fp16 - y_int4).abs().max().item()
    mean_err = (y_fp16 - y_int4).abs().mean().item()
    rel_err = mean_err / y_fp16.abs().mean().item()

    print(f"Max error: {max_err:.4f}")
    print(f"Mean error: {mean_err:.4f}")
    print(f"Relative error: {rel_err:.2%}")

    # Memory comparison
    fp16_bytes = linear.weight.numel() * 2 + (linear.bias.numel() * 2 if linear.bias is not None else 0)
    int4_bytes = int4_layer.memory_bytes()

    print(f"\nMemory:")
    print(f"  FP16: {fp16_bytes / 1e6:.2f} MB")
    print(f"  INT4: {int4_bytes / 1e6:.2f} MB")
    print(f"  Reduction: {(1 - int4_bytes / fp16_bytes) * 100:.1f}%")

    # Speed benchmark
    def benchmark(fn, warmup=10, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1000

    with torch.no_grad():
        fp16_ms = benchmark(lambda: linear(x))
        int4_ms = benchmark(lambda: int4_layer(x))

    print(f"\nSpeed:")
    print(f"  FP16: {fp16_ms:.3f} ms")
    print(f"  INT4: {int4_ms:.3f} ms")
    print(f"  INT4 vs FP16: {fp16_ms / int4_ms * 100:.1f}%")


def test_various_shapes():
    """Test on various layer shapes."""
    import time

    print("\n" + "=" * 70)
    print("INT4 Packed - Various Shapes")
    print("=" * 70)

    shapes = [
        (4096, 4096, "Attention Q/K/V/O"),
        (4096, 11008, "MLP up/gate"),
        (11008, 4096, "MLP down"),
        (4096, 32000, "LM head"),
    ]

    batch_size = 1

    print(f"\n{'Shape':<25} | {'FP16 (ms)':<10} | {'INT4 (ms)':<10} | {'Speed %':<10} | {'Mem %':<10}")
    print("-" * 75)

    for in_f, out_f, name in shapes:
        torch.manual_seed(42)

        linear = nn.Linear(in_f, out_f, dtype=torch.float16, device='cuda')
        int4_layer = INT4PackedLinear.from_linear(linear, group_size=128)

        x = torch.randn(batch_size, in_f, dtype=torch.float16, device='cuda')

        def benchmark(fn, warmup=10, iters=100):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) / iters * 1000

        with torch.no_grad():
            fp16_ms = benchmark(lambda: linear(x))
            int4_ms = benchmark(lambda: int4_layer(x))

        speed_pct = fp16_ms / int4_ms * 100
        mem_pct = int4_layer.memory_bytes() / int4_layer.memory_bytes_fp16_equivalent() * 100

        print(f"{name:<25} | {fp16_ms:<10.3f} | {int4_ms:<10.3f} | {speed_pct:<10.1f} | {mem_pct:<10.1f}")

        del linear, int4_layer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test_packing()
    test_int4_linear()
    test_various_shapes()
