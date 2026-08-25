"""
CPRLinear: Column-Precision Reordering Linear Layer

This module implements a PyTorch linear layer with CPR quantization,
providing GPU-efficient mixed-precision quantization.

CPR Quantization:
- Identifies high-error columns based on quantization error analysis
- Assigns higher precision (6-bit) to high-error columns
- Assigns lower precision (5-bit) to remaining columns
- Reorders columns for contiguous memory access

Memory Layout:
- W_high: [out_features, n_high_cols] packed 6-bit
- W_low: [out_features, n_low_cols] packed 5-bit
- col_indices: [in_features] permutation for column reordering
- scales/zeros: per-tile quantization parameters

Inference Strategy:
- Dequantize weights using CUDA kernels
- Use cuBLAS for matrix multiplication
- This achieves practical throughput for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import warnings

# Try to import CUDA kernels
try:
    import cpr_kernels
    _HAVE_CPR_KERNELS = True
except ImportError:
    _HAVE_CPR_KERNELS = False
    warnings.warn("CPR CUDA kernels not found. Install with: cd csrc && pip install -e .")


class CPRLinear(nn.Module):
    """
    Linear layer with CPR (Column-Precision Reordering) quantization.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias
        high_frac: Fraction of columns to quantize at high precision (default: 0.25)
        high_bits: Bit-width for high-precision columns (default: 6)
        low_bits: Bit-width for low-precision columns (default: 5)
        tile_size: Tile size for per-tile quantization (default: 128)
        compute_dtype: Dtype for computation (default: torch.float16)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        high_frac: float = 0.25,
        high_bits: int = 6,
        low_bits: int = 5,
        tile_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.high_frac = high_frac
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.tile_size = tile_size
        self.compute_dtype = compute_dtype

        # Calculate column counts
        self.n_high_cols = int(high_frac * in_features)
        self.n_low_cols = in_features - self.n_high_cols

        # Calculate packed sizes
        # 6-bit: 4 values -> 3 bytes
        self.packed_high_cols = ((self.n_high_cols + 3) // 4) * 3
        # 5-bit: 8 values -> 5 bytes
        self.packed_low_cols = ((self.n_low_cols + 7) // 8) * 5

        # Calculate tile counts
        self.n_tiles_high = (self.n_high_cols + tile_size - 1) // tile_size
        self.n_tiles_low = (self.n_low_cols + tile_size - 1) // tile_size

        # Resolve device
        if device is None:
            device = 'cpu'
        self._device = torch.device(device)

        # Register buffers for quantized weights
        self.register_buffer('W_high_packed',
            torch.zeros(out_features, self.packed_high_cols, dtype=torch.uint8, device=self._device))
        self.register_buffer('W_low_packed',
            torch.zeros(out_features, self.packed_low_cols, dtype=torch.uint8, device=self._device))

        # Scales and zeros (per tile)
        self.register_buffer('scales_high',
            torch.ones(self.n_tiles_high, out_features, dtype=compute_dtype, device=self._device))
        self.register_buffer('zeros_high',
            torch.zeros(self.n_tiles_high, out_features, dtype=compute_dtype, device=self._device))
        self.register_buffer('scales_low',
            torch.ones(self.n_tiles_low, out_features, dtype=compute_dtype, device=self._device))
        self.register_buffer('zeros_low',
            torch.zeros(self.n_tiles_low, out_features, dtype=compute_dtype, device=self._device))

        # Column permutation indices
        self.register_buffer('col_indices',
            torch.arange(in_features, dtype=torch.int16, device=self._device))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype, device=self._device))
        else:
            self.register_parameter('bias', None)

        # Cache for dequantized weights (optional, for repeated inference)
        self._weight_cache = None

        # Track if weights are packed (True) or unpacked (False)
        self._packed = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        high_frac: float = 0.25,
        high_bits: int = 6,
        low_bits: int = 5,
        tile_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
    ) -> 'CPRLinear':
        """
        Create CPRLinear from an existing nn.Linear layer.

        Args:
            linear: Source linear layer
            high_frac: Fraction of high-precision columns
            high_bits: Bit-width for high-precision
            low_bits: Bit-width for low-precision
            tile_size: Tile size for quantization
            compute_dtype: Computation dtype

        Returns:
            Quantized CPRLinear layer
        """
        cpr_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            high_frac=high_frac,
            high_bits=high_bits,
            low_bits=low_bits,
            tile_size=tile_size,
            compute_dtype=compute_dtype,
            device=linear.weight.device,
        )

        # Quantize weights
        cpr_linear.quantize_weights(linear.weight.data)

        # Copy bias
        if linear.bias is not None:
            cpr_linear.bias.data = linear.bias.data.to(compute_dtype)

        return cpr_linear

    def quantize_weights(self, weight: torch.Tensor):
        """
        Quantize weight matrix using CPR.

        Args:
            weight: Weight matrix [out_features, in_features]
        """
        device = weight.device
        W = weight.float()

        # Step 1: Compute per-column quantization errors at low precision
        col_errors = self._compute_column_errors(W, self.low_bits)

        # Step 2: Identify high-error columns
        _, high_indices = torch.topk(col_errors, self.n_high_cols)
        high_mask = torch.zeros(self.in_features, dtype=torch.bool, device=device)
        high_mask[high_indices] = True
        low_indices = (~high_mask).nonzero(as_tuple=True)[0]

        # Step 3: Create permutation
        col_indices = torch.cat([high_indices, low_indices])
        self.col_indices.copy_(col_indices.to(torch.int16))

        # Step 4: Reorder columns
        W_perm = W[:, col_indices]
        W_high = W_perm[:, :self.n_high_cols]
        W_low = W_perm[:, self.n_high_cols:]

        # Step 5: Quantize each region
        self._quantize_region(W_high, self.high_bits, 'high')
        self._quantize_region(W_low, self.low_bits, 'low')

    def _compute_column_errors(self, W: torch.Tensor, nbits: int) -> torch.Tensor:
        """Compute per-column quantization error."""
        n_rows, n_cols = W.shape
        col_errors = torch.zeros(n_cols, device=W.device)

        for t in range((n_cols + self.tile_size - 1) // self.tile_size):
            c_start = t * self.tile_size
            c_end = min(c_start + self.tile_size, n_cols)
            tile = W[:, c_start:c_end]

            # Simple min-max quantization for error estimation
            min_val = tile.min(dim=0, keepdim=True)[0]
            max_val = tile.max(dim=0, keepdim=True)[0]
            scale = (max_val - min_val) / (2**nbits - 1)
            scale = torch.clamp(scale, min=1e-8)

            q = torch.round((tile - min_val) / scale)
            q = torch.clamp(q, 0, 2**nbits - 1)
            deq = q * scale + min_val

            error = ((tile - deq) ** 2).sum(dim=0)
            col_errors[c_start:c_end] = error

        return col_errors

    def _quantize_region(self, W: torch.Tensor, nbits: int, region: str):
        """Quantize a weight region (high or low)."""
        device = W.device
        n_rows, n_cols = W.shape

        if region == 'high':
            n_tiles = self.n_tiles_high
            scales = self.scales_high
            zeros = self.zeros_high
        else:
            n_tiles = self.n_tiles_low
            scales = self.scales_low
            zeros = self.zeros_low

        # Quantize tile by tile
        W_q_list = []
        for t in range(n_tiles):
            c_start = t * self.tile_size
            c_end = min(c_start + self.tile_size, n_cols)
            tile = W[:, c_start:c_end]

            # Per-row min-max quantization
            min_val = tile.min(dim=1, keepdim=True)[0]
            max_val = tile.max(dim=1, keepdim=True)[0]
            scale = (max_val - min_val) / (2**nbits - 1)
            scale = torch.clamp(scale, min=1e-8)

            # q = (tile - min_val) / scale
            # tile = q * scale + min_val
            # We want: tile = (q - zero) * scale
            # So: q * scale + min_val = (q - zero) * scale
            #     q * scale + min_val = q * scale - zero * scale
            #     min_val = -zero * scale
            #     zero = -min_val / scale

            q = torch.round((tile - min_val) / scale)
            q = torch.clamp(q, 0, 2**nbits - 1)

            # Store scales and zeros
            # Dequantization: tile = (q - zero) * scale
            scales[t] = scale.squeeze(1).to(self.compute_dtype)
            zeros[t] = (-min_val.squeeze(1) / scale.squeeze(1)).to(self.compute_dtype)

            W_q_list.append(q.to(torch.int8))

        # Concatenate and pack
        W_q = torch.cat(W_q_list, dim=1)

        if _HAVE_CPR_KERNELS and device.type == 'cuda':
            if region == 'high':
                self.W_high_packed.copy_(cpr_kernels.pack_6bit(W_q))
            else:
                self.W_low_packed.copy_(cpr_kernels.pack_5bit(W_q))
            self._packed = True
        else:
            # Fallback: store unpacked (for CPU or when kernels unavailable)
            if region == 'high':
                self.W_high_packed = W_q.to(torch.uint8)
            else:
                self.W_low_packed = W_q.to(torch.uint8)
            self._packed = False

    def dequantize(self) -> torch.Tensor:
        """
        Dequantize weights to full precision.

        Returns:
            Dequantized weight matrix [out_features, in_features]
        """
        device = self.W_high_packed.device

        if _HAVE_CPR_KERNELS and device.type == 'cuda' and self._packed:
            # Use CUDA kernels for dequantization
            W_high = cpr_kernels.dequantize_6bit(
                self.W_high_packed, self.scales_high, self.zeros_high,
                self.n_high_cols, self.tile_size
            )
            W_low = cpr_kernels.dequantize_5bit(
                self.W_low_packed, self.scales_low, self.zeros_low,
                self.n_low_cols, self.tile_size
            )
        else:
            # Fallback to PyTorch dequantization (unpacked weights)
            W_high = self._dequantize_pytorch(
                self.W_high_packed, self.scales_high, self.zeros_high,
                self.n_high_cols, self.high_bits
            )
            W_low = self._dequantize_pytorch(
                self.W_low_packed, self.scales_low, self.zeros_low,
                self.n_low_cols, self.low_bits
            )

        # Concatenate
        W_perm = torch.cat([W_high, W_low], dim=1)

        # Inverse permute
        W = torch.zeros(self.out_features, self.in_features,
                       dtype=self.compute_dtype, device=device)
        W[:, self.col_indices.long()] = W_perm

        return W

    def _dequantize_pytorch(
        self,
        W_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        n_cols: int,
        nbits: int
    ) -> torch.Tensor:
        """PyTorch fallback for dequantization."""
        # This is a simplified version - in practice, would need proper unpacking
        W_q = W_packed.float()
        n_tiles = (n_cols + self.tile_size - 1) // self.tile_size

        W = torch.zeros(self.out_features, n_cols, dtype=self.compute_dtype,
                       device=W_packed.device)

        for t in range(n_tiles):
            c_start = t * self.tile_size
            c_end = min(c_start + self.tile_size, n_cols)
            scale = scales[t].unsqueeze(1)
            zero = zeros[t].unsqueeze(1)
            W[:, c_start:c_end] = (W_q[:, c_start:c_end] - zero) * scale

        return W

    def cache_weights(self):
        """Cache dequantized weights for faster inference."""
        if self._weight_cache is None:
            self._weight_cache = self.dequantize()

    def clear_cache(self):
        """Clear weight cache to save memory."""
        self._weight_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CPR quantized weights.

        Args:
            x: Input tensor [batch, in_features]

        Returns:
            Output tensor [batch, out_features]
        """
        x = x.to(self.compute_dtype)

        # Use cached weights if available, otherwise dequantize
        if self._weight_cache is not None:
            W = self._weight_cache
        else:
            W = self.dequantize()

        # Matrix multiplication using cuBLAS
        out = F.linear(x, W, self.bias)

        return out

    def compute_avg_bits(self) -> float:
        """Compute average bits per weight."""
        total_bits = (self.n_high_cols * self.high_bits +
                     self.n_low_cols * self.low_bits)
        return total_bits / self.in_features

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'high_frac={self.high_frac}, high_bits={self.high_bits}, '
                f'low_bits={self.low_bits}, avg_bits={self.compute_avg_bits():.3f}')

    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        """
        Export CPR quantized state for saving.

        Returns dict with:
          - W_high_packed: Packed 6-bit weights
          - W_low_packed: Packed 5-bit weights
          - scales_high, zeros_high: High-precision quantization params
          - scales_low, zeros_low: Low-precision quantization params
          - col_indices: Column permutation
          - bias: Bias tensor (or None)
          - meta: Metadata dict for reconstruction
        """
        sd = {}
        # Packed weights
        sd["W_high_packed"] = self.W_high_packed.detach().cpu()
        sd["W_low_packed"] = self.W_low_packed.detach().cpu()

        # Scales and zeros
        sd["scales_high"] = self.scales_high.detach().cpu()
        sd["zeros_high"] = self.zeros_high.detach().cpu()
        sd["scales_low"] = self.scales_low.detach().cpu()
        sd["zeros_low"] = self.zeros_low.detach().cpu()

        # Column permutation
        sd["col_indices"] = self.col_indices.detach().cpu()

        # Bias
        if self.bias is not None:
            sd["bias"] = self.bias.detach().cpu()

        # Metadata for reconstruction
        sd["meta"] = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "high_frac": self.high_frac,
            "high_bits": self.high_bits,
            "low_bits": self.low_bits,
            "tile_size": self.tile_size,
            "n_high_cols": self.n_high_cols,
            "n_low_cols": self.n_low_cols,
            "compute_dtype": str(self.compute_dtype),
            "packed": self._packed,
        }

        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load CPR quantized state from saved dict.

        Args:
            state_dict: Dict from state_dict() method
            strict: If True, raise error on missing keys

        Returns:
            _IncompatibleKeys object
        """
        from torch.nn.modules.module import _IncompatibleKeys

        meta = state_dict["meta"]
        device = self.W_high_packed.device

        # Verify dimensions match
        if meta["in_features"] != self.in_features or meta["out_features"] != self.out_features:
            raise RuntimeError(
                f"Shape mismatch: expected ({self.in_features}, {self.out_features}), "
                f"got ({meta['in_features']}, {meta['out_features']})"
            )

        # Load packed flag first
        self._packed = meta.get("packed", True)

        # Load packed weights - handle size mismatch for unpacked weights
        loaded_high = state_dict["W_high_packed"].to(device)
        loaded_low = state_dict["W_low_packed"].to(device)

        if loaded_high.shape == self.W_high_packed.shape:
            self.W_high_packed.copy_(loaded_high)
        else:
            # Unpacked weights - replace the buffer entirely
            self.W_high_packed = loaded_high

        if loaded_low.shape == self.W_low_packed.shape:
            self.W_low_packed.copy_(loaded_low)
        else:
            # Unpacked weights - replace the buffer entirely
            self.W_low_packed = loaded_low

        # Load scales and zeros
        self.scales_high.copy_(state_dict["scales_high"].to(device))
        self.zeros_high.copy_(state_dict["zeros_high"].to(device))
        self.scales_low.copy_(state_dict["scales_low"].to(device))
        self.zeros_low.copy_(state_dict["zeros_low"].to(device))

        # Load column permutation
        self.col_indices.copy_(state_dict["col_indices"].to(device))

        # Load bias
        if "bias" in state_dict and self.bias is not None:
            self.bias.data.copy_(state_dict["bias"].to(device, dtype=self.compute_dtype))

        # Clear any cached weights
        self._weight_cache = None

        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        device: str = "cuda",
        compute_dtype: torch.dtype = torch.float16,
    ) -> 'CPRLinear':
        """
        Create CPRLinear instance from saved state dict.

        Args:
            state_dict: Dict from state_dict() method
            device: Target device
            compute_dtype: Computation dtype

        Returns:
            CPRLinear instance with loaded weights
        """
        meta = state_dict["meta"]

        # Create instance with correct dimensions
        instance = cls(
            in_features=meta["in_features"],
            out_features=meta["out_features"],
            bias="bias" in state_dict,
            high_frac=meta["high_frac"],
            high_bits=meta["high_bits"],
            low_bits=meta["low_bits"],
            tile_size=meta["tile_size"],
            compute_dtype=compute_dtype,
            device=device,
        )

        # Move to device and load state
        instance = instance.to(device)
        instance.load_state_dict(state_dict)

        return instance


def cpr_quant_config(
    high_frac: float = 0.25,
    high_bits: int = 6,
    low_bits: int = 5,
    tile_size: int = 128,
) -> dict:
    """
    Create CPR quantization configuration.

    Args:
        high_frac: Fraction of columns at high precision (default: 0.25)
        high_bits: Bit-width for high-precision columns (default: 6)
        low_bits: Bit-width for low-precision columns (default: 5)
        tile_size: Tile size for per-tile quantization (default: 128)

    Returns:
        Quantization config dict for CPR
    """
    avg_bits = high_frac * high_bits + (1 - high_frac) * low_bits

    return {
        "method": "cpr",
        "high_frac": high_frac,
        "high_bits": high_bits,
        "low_bits": low_bits,
        "tile_size": tile_size,
        "avg_bits": avg_bits,
    }
