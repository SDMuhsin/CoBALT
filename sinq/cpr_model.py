"""
CPR Model Quantization: Full pipeline for quantizing LLMs with CPR.

This module provides:
1. Model-level CPR quantization using INT8 with per-group scales
2. Integration with Triton fused dequant+matmul kernels
3. Memory-efficient inference with quantized weights
4. Multi-precision CPR: 25% columns at INT8, 75% at configurable lower bits (3-8)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm
import gc

# Import Triton kernels
try:
    from .triton_cpr_kernel import triton_cpr_matmul_int8_v2
    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False
    print("Warning: Triton CPR kernels not available")

# Import SINQ for multi-precision quantization
from .sinkhorn import sinkhorn_log


class CPRLinearFused(nn.Module):
    """
    CPR Linear layer using Triton fused dequant+matmul.

    Uses INT8 quantization with per-group scales (symmetric).
    Memory layout:
    - weight_int8: [in_features, out_features] INT8 (kernel-friendly)
    - scales: [num_groups, out_features] FP16

    Shape-aware kernel routing:
    - Uses fused Triton kernel for favorable shapes (out_features > in_features)
    - Falls back to dequantize+cuBLAS for unfavorable shapes
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        force_fused: Optional[bool] = None,  # None = auto, True = always fused, False = always dequant
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = compute_dtype

        # Calculate number of groups
        self.num_groups = (in_features + group_size - 1) // group_size

        if device is None:
            device = 'cuda'
        self._device = torch.device(device)

        # Quantized weights stored in kernel-friendly layout: [in_features, out_features] = [K, N]
        # This avoids transpose in forward pass!
        self.register_buffer('weight_int8',
            torch.zeros(in_features, out_features, dtype=torch.int8, device=self._device))

        # Per-group scales [num_groups, out_features] = [K//group, N]
        self.register_buffer('scales',
            torch.ones(self.num_groups, out_features, dtype=compute_dtype, device=self._device))

        # Cache for dequantized weights (lazy initialization)
        self._weight_dequant_cache: Optional[torch.Tensor] = None

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype, device=self._device))
        else:
            self.register_parameter('bias', None)

        # Use Triton fused kernel?
        self.use_triton = _HAVE_TRITON

        # Shape-aware kernel selection
        # Fused kernel is faster when N > K (output wider than input)
        # This includes MLP up/gate projections and LM head
        if force_fused is not None:
            self._use_fused = force_fused
        else:
            # Auto-detect: use fused for "favorable" shapes
            # Favorable = large output dimension (memory-bound)
            self._use_fused = out_features >= in_features * 1.5 or out_features >= 8192

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        force_fused: Optional[bool] = None,
    ) -> 'CPRLinearFused':
        """
        Create CPRLinearFused from an existing nn.Linear layer.

        Args:
            linear: Source nn.Linear layer
            group_size: Quantization group size
            compute_dtype: Compute dtype (FP16 recommended)
            force_fused: None=auto, True=always fused kernel, False=always dequant+cuBLAS
        """
        cpr = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            compute_dtype=compute_dtype,
            device=linear.weight.device,
            force_fused=force_fused,
        )

        # Quantize weights
        cpr.quantize_weights(linear.weight.data)

        # Copy bias
        if linear.bias is not None:
            cpr.bias.data = linear.bias.data.to(compute_dtype)

        return cpr

    def quantize_weights(self, weight: torch.Tensor):
        """
        Quantize weights to INT8 with per-group scales.

        Args:
            weight: [out_features, in_features] FP16/FP32 weights

        The Triton kernel expects W: [K, N] and scales: [K//group, N]
        where K=in_features and N=out_features.

        So we need to:
        1. Transpose weight to [in_features, out_features] = [K, N]
        2. Quantize along K dimension (groups of rows)
        3. Store transposed weight and scales in correct format
        """
        device = weight.device
        out_features, in_features = weight.shape

        # Transpose to [K, N] = [in_features, out_features]
        weight_kn = weight.t().float()  # [K, N]

        # Pad K to multiple of group_size if needed
        K, N = weight_kn.shape
        padded_k = self.num_groups * self.group_size
        if padded_k > K:
            weight_kn = torch.nn.functional.pad(weight_kn, (0, 0, 0, padded_k - K))

        # Reshape for group quantization: [num_groups, group_size, N]
        weight_grouped = weight_kn.view(self.num_groups, self.group_size, N)

        # Compute per-group scales (symmetric quantization)
        # scale = max(abs(group)) / 127, computed along group_size dimension
        max_abs = weight_grouped.abs().amax(dim=1)  # [num_groups, N]
        scales = (max_abs / 127.0).clamp(min=1e-8)  # [num_groups, N]

        # Store scales: [num_groups, out_features]
        self.scales.copy_(scales.to(self.compute_dtype))

        # Quantize: q = round(w / scale)
        scales_expanded = scales.unsqueeze(1)  # [num_groups, 1, N]
        weight_q = torch.round(weight_grouped / scales_expanded)
        weight_q = weight_q.clamp(-128, 127).to(torch.int8)

        # Reshape back to [K, N] - already in kernel-friendly layout!
        weight_q = weight_q.view(padded_k, N)[:K, :]  # [K, N] = [in_features, out_features]

        self.weight_int8.copy_(weight_q)

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights to FP16 for debugging/verification."""
        # weight_int8 is already [K, N] = [in_features, out_features]
        K, N = self.weight_int8.shape
        weight_kn = self.weight_int8.float()  # [K, N]

        # Pad K for group computation
        padded_k = self.num_groups * self.group_size
        if padded_k > K:
            weight_kn = torch.nn.functional.pad(weight_kn, (0, 0, 0, padded_k - K))

        # Reshape: [num_groups, group_size, N]
        weight_grouped = weight_kn.view(self.num_groups, self.group_size, N)

        # Dequantize: w = q * scale
        scales = self.scales.unsqueeze(1)  # [num_groups, 1, N]
        weight_deq = (weight_grouped * scales).view(padded_k, N)[:K, :]  # [K, N]

        # Transpose to [N, K] = [out_features, in_features] for nn.Linear compatibility
        return weight_deq.t().to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with shape-aware kernel selection.

        Uses fused Triton kernel for favorable shapes (large N),
        falls back to dequantize+cuBLAS for unfavorable shapes.

        Args:
            x: Input [batch, seq_len, in_features] or [batch*seq_len, in_features]

        Returns:
            Output [batch, seq_len, out_features] or [batch*seq_len, out_features]
        """
        x = x.to(self.compute_dtype)

        # Handle 3D input
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x = x.view(batch * seq_len, -1)

        # Shape-aware kernel selection
        if self.use_triton and x.is_cuda and self._use_fused:
            # Use fused Triton kernel for favorable shapes
            # weight_int8 is already [K, N] = [in_features, out_features]
            out = triton_cpr_matmul_int8_v2(
                x,  # [M, K]
                self.weight_int8,  # [K, N]
                self.scales,  # [num_groups, N]
                self.group_size,
            )
        else:
            # Use dequantize + cuBLAS for unfavorable shapes
            # Dequantize on the fly (no caching to save memory)
            W = self.dequantize()
            out = torch.nn.functional.linear(x, W)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Restore 3D shape
        if len(original_shape) == 3:
            out = out.view(batch, seq_len, -1)

        return out

    def clear_dequant_cache(self):
        """Clear the dequantized weight cache to save memory."""
        self._weight_dequant_cache = None

    def extra_repr(self) -> str:
        kernel = 'fused' if self._use_fused else 'dequant'
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'group_size={self.group_size}, bits=8, kernel={kernel}')

    def memory_footprint(self) -> int:
        """Return memory footprint in bytes."""
        weight_bytes = self.weight_int8.numel() * 1  # INT8 = 1 byte
        scale_bytes = self.scales.numel() * 2  # FP16 = 2 bytes
        bias_bytes = self.bias.numel() * 2 if self.bias is not None else 0
        return weight_bytes + scale_bytes + bias_bytes


class CPRLinearMultiPrecision(nn.Module):
    """
    Multi-Precision CPR Linear layer supporting 3-8 bit quantization.

    Key innovation: Mixed-precision quantization where:
    - 25% of columns (most sensitive) are kept at INT8
    - 75% of columns are quantized at lower precision (configurable 3-8 bits)

    Sensitivity is determined by Sinkhorn's μ₁ scaling factors - columns with
    higher μ₁ values are more sensitive to quantization error.

    Memory layout:
    - high_prec_weight: [in_features, n_high_cols] INT8
    - high_prec_scales: [num_groups, n_high_cols] FP16
    - low_prec_weight: [in_features, n_low_cols] INT8/INT16 (depending on bits)
    - low_prec_scales: [n_low_cols, 1] FP16 (per-row scales from SINQ)
    - low_prec_scales2: [1, n_low_cols] FP16 (per-column μ₁ scales)
    - low_prec_zeros: [n_low_cols, 1] FP16
    - col_indices: [out_features] INT32 (reordering indices)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        low_bits: int = 4,
        high_prec_ratio: float = 0.25,
        compute_dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.low_bits = low_bits
        self.high_prec_ratio = high_prec_ratio
        self.compute_dtype = compute_dtype

        # Calculate column splits
        self.n_high_cols = int(out_features * high_prec_ratio)
        self.n_low_cols = out_features - self.n_high_cols

        # Number of groups for high-precision columns
        self.num_groups = (in_features + group_size - 1) // group_size

        if device is None:
            device = 'cuda'
        self._device = torch.device(device)

        # High-precision (INT8) weights and scales
        self.register_buffer('high_prec_weight',
            torch.zeros(in_features, self.n_high_cols, dtype=torch.int8, device=self._device))
        self.register_buffer('high_prec_scales',
            torch.ones(self.num_groups, self.n_high_cols, dtype=compute_dtype, device=self._device))

        # Low-precision weights (using SINQ-style asymmetric quantization)
        # Use int16 for 8-bit to avoid overflow, int8 for lower
        low_dtype = torch.int16 if low_bits == 8 else torch.int8
        self.register_buffer('low_prec_weight',
            torch.zeros(in_features, self.n_low_cols, dtype=low_dtype, device=self._device))
        # Per-row scales (s1 from SINQ)
        self.register_buffer('low_prec_scales',
            torch.ones(in_features, 1, dtype=compute_dtype, device=self._device))
        # Per-column scales (μ₁ from Sinkhorn)
        self.register_buffer('low_prec_scales2',
            torch.ones(1, self.n_low_cols, dtype=compute_dtype, device=self._device))
        # Per-row zero points
        self.register_buffer('low_prec_zeros',
            torch.zeros(in_features, 1, dtype=compute_dtype, device=self._device))

        # Column reordering indices (to restore original order)
        self.register_buffer('col_indices',
            torch.arange(out_features, dtype=torch.int64, device=self._device))

        # Inverse indices for output reordering
        self.register_buffer('col_indices_inv',
            torch.arange(out_features, dtype=torch.int64, device=self._device))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype, device=self._device))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        low_bits: int = 4,
        high_prec_ratio: float = 0.25,
        compute_dtype: torch.dtype = torch.float16,
    ) -> 'CPRLinearMultiPrecision':
        """
        Create CPRLinearMultiPrecision from an existing nn.Linear layer.

        Uses Sinkhorn's μ₁ values to identify sensitive columns.
        """
        cpr = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            low_bits=low_bits,
            high_prec_ratio=high_prec_ratio,
            compute_dtype=compute_dtype,
            device=linear.weight.device,
        )

        # Quantize weights with column sensitivity detection
        cpr.quantize_weights(linear.weight.data)

        # Copy bias (will be reordered in quantize_weights)
        if linear.bias is not None:
            cpr.bias.data = linear.bias.data.to(compute_dtype)

        return cpr

    def quantize_weights(self, weight: torch.Tensor):
        """
        Quantize weights with mixed precision based on Sinkhorn sensitivity.

        Args:
            weight: [out_features, in_features] FP16/FP32 weights
        """
        device = weight.device
        out_features, in_features = weight.shape

        # Step 1: Apply Sinkhorn to get sensitivity scores (μ₁)
        # Sinkhorn works on [rows, cols], so transpose: [in_features, out_features]
        W_t = weight.t().float()  # [in_features, out_features]

        # Get Sinkhorn-normalized matrix and scales
        W_norm, mu1, mu2 = sinkhorn_log(W_t, order=16)
        # mu1: [out_features] - per-column sensitivity (1D)
        # mu2: [in_features, 1] - per-row sensitivity

        # Step 2: Identify high-sensitivity columns (top 25% by μ₁)
        # Higher μ₁ means the column has larger values and is more sensitive
        _, sorted_indices = torch.sort(mu1, descending=True)

        high_cols = sorted_indices[:self.n_high_cols]  # Most sensitive
        low_cols = sorted_indices[self.n_high_cols:]   # Less sensitive

        # Store reordering indices (high_prec columns first, then low_prec)
        self.col_indices.copy_(torch.cat([high_cols, low_cols]))

        # Compute inverse indices for output reordering
        inv_indices = torch.zeros_like(self.col_indices)
        inv_indices[self.col_indices] = torch.arange(out_features, device=device)
        self.col_indices_inv.copy_(inv_indices)

        # Step 3: Quantize high-precision columns (INT8 symmetric)
        W_high = W_t[:, high_cols]  # [in_features, n_high_cols]
        self._quantize_high_precision(W_high)

        # Step 4: Quantize low-precision columns (SINQ-style asymmetric)
        W_low = W_t[:, low_cols]  # [in_features, n_low_cols]
        # Note: We'll re-compute Sinkhorn on the subset in _quantize_low_precision
        self._quantize_low_precision(W_low)

    def _quantize_high_precision(self, weight: torch.Tensor):
        """
        Quantize high-sensitivity columns with INT8 symmetric per-group quantization.

        Args:
            weight: [in_features, n_high_cols] weights
        """
        K, N = weight.shape

        # Pad K to multiple of group_size
        padded_k = self.num_groups * self.group_size
        if padded_k > K:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, padded_k - K))

        # Reshape for group quantization: [num_groups, group_size, N]
        weight_grouped = weight.view(self.num_groups, self.group_size, N)

        # Compute per-group scales (symmetric)
        max_abs = weight_grouped.abs().amax(dim=1)  # [num_groups, N]
        scales = (max_abs / 127.0).clamp(min=1e-8)

        self.high_prec_scales.copy_(scales.to(self.compute_dtype))

        # Quantize
        scales_expanded = scales.unsqueeze(1)
        weight_q = torch.round(weight_grouped / scales_expanded)
        weight_q = weight_q.clamp(-128, 127).to(torch.int8)

        # Reshape and store
        weight_q = weight_q.view(padded_k, N)[:K, :]
        self.high_prec_weight.copy_(weight_q)

    def _quantize_low_precision(self, weight: torch.Tensor):
        """
        Quantize low-sensitivity columns with SINQ-style asymmetric quantization.

        Uses the dual-scale approach:
        W_deq = ((Q - z) * s1) * s2

        Args:
            weight: [in_features, n_low_cols] weights
        """
        K, N = weight.shape

        # Apply Sinkhorn to get normalized weights and scales
        W_norm, mu1_local, mu2_local = sinkhorn_log(weight, order=16)
        # mu1_local: [n_low_cols] (1D)
        # mu2_local: [in_features, 1]

        # Compute min/max for asymmetric quantization
        max_int = (1 << self.low_bits) - 1  # e.g., 15 for 4-bit
        min_int = 0

        max_val = W_norm.amax(dim=1, keepdim=True)
        min_val = W_norm.amin(dim=1, keepdim=True)

        # Compute scales and zeros
        scales = (max_val - min_val).clamp(min=1e-4) / max_int
        zeros = -torch.round(min_val / scales)

        # Quantize
        q = torch.clamp(torch.round(W_norm / scales + zeros), min_int, max_int)

        # Choose dtype based on bits
        if max_int > 127:
            q = q.to(torch.int16)
        else:
            q = q.to(torch.int8)

        # Store quantized values and scales
        self.low_prec_weight.copy_(q)
        self.low_prec_scales.copy_((scales * mu2_local).to(self.compute_dtype))
        # mu1_local is 1D [n_low_cols], reshape to [1, n_low_cols]
        self.low_prec_scales2.copy_(mu1_local.unsqueeze(0).to(self.compute_dtype))
        self.low_prec_zeros.copy_(zeros.to(self.compute_dtype))

    def _dequantize_high(self) -> torch.Tensor:
        """Dequantize high-precision columns to FP16."""
        K, N = self.high_prec_weight.shape
        weight_kn = self.high_prec_weight.float()

        # Pad for group computation
        padded_k = self.num_groups * self.group_size
        if padded_k > K:
            weight_kn = torch.nn.functional.pad(weight_kn, (0, 0, 0, padded_k - K))

        # Reshape and dequantize
        weight_grouped = weight_kn.view(self.num_groups, self.group_size, N)
        scales = self.high_prec_scales.unsqueeze(1)
        weight_deq = (weight_grouped * scales).view(padded_k, N)[:K, :]

        return weight_deq.to(self.compute_dtype)

    def _dequantize_low(self) -> torch.Tensor:
        """Dequantize low-precision columns to FP16 using SINQ formula."""
        # W_deq = ((Q - z) * s1) * s2
        q = self.low_prec_weight.float()
        z = self.low_prec_zeros
        s1 = self.low_prec_scales
        s2 = self.low_prec_scales2

        weight_deq = ((q - z) * s1) * s2
        return weight_deq.to(self.compute_dtype)

    def dequantize(self) -> torch.Tensor:
        """Dequantize all weights and restore original column order."""
        # Dequantize both parts
        W_high = self._dequantize_high()  # [in_features, n_high_cols]
        W_low = self._dequantize_low()    # [in_features, n_low_cols]

        # Concatenate and reorder
        W_reordered = torch.cat([W_high, W_low], dim=1)  # [in_features, out_features]

        # Restore original column order using inverse indices
        W_original = W_reordered[:, self.col_indices_inv]

        # Transpose to [out_features, in_features] for nn.Linear compatibility
        return W_original.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mixed-precision dequantization.

        Args:
            x: Input [batch, seq_len, in_features] or [batch*seq_len, in_features]

        Returns:
            Output [batch, seq_len, out_features] or [batch*seq_len, out_features]
        """
        x = x.to(self.compute_dtype)

        # Handle 3D input
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x = x.view(batch * seq_len, -1)

        # Dequantize weights
        W = self.dequantize()  # [out_features, in_features]

        # Linear operation
        out = torch.nn.functional.linear(x, W)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Restore 3D shape
        if len(original_shape) == 3:
            out = out.view(batch, seq_len, -1)

        return out

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'group_size={self.group_size}, low_bits={self.low_bits}, '
                f'high_prec_ratio={self.high_prec_ratio}')

    def memory_footprint(self) -> int:
        """Return memory footprint in bytes."""
        # High-precision: INT8 weights + FP16 scales
        high_weight_bytes = self.high_prec_weight.numel() * 1
        high_scale_bytes = self.high_prec_scales.numel() * 2

        # Low-precision: INT8/16 weights + FP16 scales/zeros
        low_weight_bytes = self.low_prec_weight.numel() * self.low_prec_weight.element_size()
        low_scale_bytes = self.low_prec_scales.numel() * 2
        low_scale2_bytes = self.low_prec_scales2.numel() * 2
        low_zero_bytes = self.low_prec_zeros.numel() * 2

        # Indices
        index_bytes = (self.col_indices.numel() + self.col_indices_inv.numel()) * 8

        # Bias
        bias_bytes = self.bias.numel() * 2 if self.bias is not None else 0

        return (high_weight_bytes + high_scale_bytes +
                low_weight_bytes + low_scale_bytes + low_scale2_bytes + low_zero_bytes +
                index_bytes + bias_bytes)


def quantize_model_cpr(
    model: nn.Module,
    group_size: int = 128,
    compute_dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    skip_layers: List[str] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Quantize all linear layers in a model using CPR INT8 quantization.

    Args:
        model: HuggingFace model to quantize
        group_size: Group size for per-group quantization
        compute_dtype: Computation dtype (FP16 recommended)
        device: Target device
        skip_layers: Layer name patterns to skip (e.g., ['lm_head'])
        verbose: Print progress

    Returns:
        Quantized model with CPRLinearFused layers
    """
    if skip_layers is None:
        skip_layers = ['lm_head']  # Usually keep LM head in FP16

    # Move model to device
    model = model.to(device)

    # Collect linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if should skip
            skip = any(pattern in name for pattern in skip_layers)
            if not skip:
                linear_layers.append((name, module))

    if verbose:
        print(f"Found {len(linear_layers)} linear layers to quantize")

    # Quantize each layer
    total_original_bytes = 0
    total_quantized_bytes = 0

    pbar = tqdm(linear_layers, desc="Quantizing", disable=not verbose)
    for name, module in pbar:
        pbar.set_postfix_str(f"{name}")

        # Calculate original size
        orig_bytes = module.weight.numel() * 2  # FP16
        if module.bias is not None:
            orig_bytes += module.bias.numel() * 2
        total_original_bytes += orig_bytes

        # Create quantized layer
        cpr_layer = CPRLinearFused.from_linear(
            module,
            group_size=group_size,
            compute_dtype=compute_dtype,
        )

        # Calculate quantized size
        total_quantized_bytes += cpr_layer.memory_footprint()

        # Replace in model
        _replace_module(model, name, cpr_layer)

        # Free original weights
        del module

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        reduction = (1 - total_quantized_bytes / total_original_bytes) * 100
        print(f"\nMemory reduction: {total_original_bytes/1e6:.1f}MB -> {total_quantized_bytes/1e6:.1f}MB ({reduction:.1f}% savings)")

    # Mark as quantized
    model.cpr_quantized = True
    model.cpr_config = {
        'group_size': group_size,
        'compute_dtype': str(compute_dtype),
    }

    return model


def _replace_module(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a module by name path (e.g., 'model.layers.0.self_attn.q_proj')."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def get_model_memory(model: nn.Module) -> Tuple[int, int]:
    """
    Get model memory usage.

    Returns:
        (weight_bytes, total_bytes)
    """
    weight_bytes = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        bytes_per_elem = param.element_size()
        param_bytes = param.numel() * bytes_per_elem
        total_bytes += param_bytes
        if 'weight' in name:
            weight_bytes += param_bytes

    for name, buffer in model.named_buffers():
        bytes_per_elem = buffer.element_size()
        buffer_bytes = buffer.numel() * bytes_per_elem
        total_bytes += buffer_bytes
        if 'weight' in name:
            weight_bytes += buffer_bytes

    return weight_bytes, total_bytes


# Quick test
if __name__ == '__main__':
    import time

    print("Testing CPRLinearFused...")

    # Create test layer
    in_features, out_features = 4096, 4096
    linear = nn.Linear(in_features, out_features, dtype=torch.float16, device='cuda')

    # Quantize
    cpr = CPRLinearFused.from_linear(linear, group_size=128)

    # Test correctness
    x = torch.randn(8, in_features, dtype=torch.float16, device='cuda')

    with torch.no_grad():
        y_fp16 = linear(x)
        y_cpr = cpr(x)

    max_err = (y_fp16 - y_cpr).abs().max().item()
    print(f"Max error vs FP16: {max_err:.6f}")

    # Test speed
    def benchmark(fn, warmup=10, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1000

    fp16_ms = benchmark(lambda: linear(x))
    cpr_ms = benchmark(lambda: cpr(x))

    print(f"FP16: {fp16_ms:.3f}ms")
    print(f"CPR:  {cpr_ms:.3f}ms ({100*fp16_ms/cpr_ms:.1f}% of FP16 speed)")

    # Memory
    fp16_bytes = linear.weight.numel() * 2 + linear.bias.numel() * 2
    cpr_bytes = cpr.memory_footprint()
    print(f"Memory: {fp16_bytes/1e6:.2f}MB -> {cpr_bytes/1e6:.2f}MB ({100*(1-cpr_bytes/fp16_bytes):.1f}% reduction)")
