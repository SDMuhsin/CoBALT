// CPR-SINQ: Column-Precision Reordering CUDA Kernels
// Header file with declarations

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cpr {

// ============================================================================
// Bit Packing/Unpacking Functions
// ============================================================================

// Pack FP16 weights to 6-bit representation
// Input: weights [n_rows, n_cols] as int8 (values 0-63)
// Output: packed [n_rows, ceil(n_cols * 6 / 8)] as uint8
torch::Tensor pack_6bit(torch::Tensor weights);

// Unpack 6-bit weights to int8
// Input: packed [n_rows, packed_cols] as uint8
// Output: weights [n_rows, n_cols] as int8 (values 0-63)
torch::Tensor unpack_6bit(torch::Tensor packed, int64_t n_cols);

// Pack weights to 5-bit representation
// Input: weights [n_rows, n_cols] as int8 (values 0-31)
// Output: packed [n_rows, ceil(n_cols * 5 / 8)] as uint8
torch::Tensor pack_5bit(torch::Tensor weights);

// Unpack 5-bit weights to int8
// Input: packed [n_rows, packed_cols] as uint8
// Output: weights [n_rows, n_cols] as int8 (values 0-31)
torch::Tensor unpack_5bit(torch::Tensor packed, int64_t n_cols);

// ============================================================================
// Dequantization Functions
// ============================================================================

// Dequantize packed 6-bit weights to FP16
// Input: packed weights, scales [n_tiles, n_rows], zeros [n_tiles, n_rows]
// Output: dequantized weights [n_rows, n_cols] as FP16
torch::Tensor dequantize_6bit(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t n_cols,
    int64_t tile_size
);

// Dequantize packed 5-bit weights to FP16
torch::Tensor dequantize_5bit(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t n_cols,
    int64_t tile_size
);

// ============================================================================
// CPR Matrix Multiplication
// ============================================================================

// Full CPR matmul: Y = X @ W^T where W is CPR-quantized
// X: [batch, in_features] FP16
// W_high: packed 6-bit weights [n_high_cols_packed, out_features]
// W_low: packed 5-bit weights [n_low_cols_packed, out_features]
// col_indices: permutation indices [in_features]
// Returns: Y [batch, out_features] FP16
torch::Tensor cpr_matmul(
    torch::Tensor X,
    torch::Tensor W_high_packed,
    torch::Tensor W_low_packed,
    torch::Tensor scales_high,
    torch::Tensor zeros_high,
    torch::Tensor scales_low,
    torch::Tensor zeros_low,
    torch::Tensor col_indices,
    int64_t n_high_cols,
    int64_t n_low_cols,
    int64_t tile_size
);

// Fused CPR linear layer: Y = X @ W^T + bias
torch::Tensor cpr_linear(
    torch::Tensor X,
    torch::Tensor W_high_packed,
    torch::Tensor W_low_packed,
    torch::Tensor scales_high,
    torch::Tensor zeros_high,
    torch::Tensor scales_low,
    torch::Tensor zeros_low,
    torch::Tensor col_indices,
    torch::Tensor bias,  // optional, can be empty
    int64_t n_high_cols,
    int64_t n_low_cols,
    int64_t tile_size
);

// ============================================================================
// Utility Functions
// ============================================================================

// Check CUDA availability and print device info
void check_cuda();

// Get optimal tile size for given dimensions
int64_t get_optimal_tile_size(int64_t n_rows, int64_t n_cols);

// ============================================================================
// Optimized Tiled Kernels
// ============================================================================

// Tiled CPR matmul with shared memory staging
torch::Tensor cpr_matmul_tiled(
    torch::Tensor X,
    torch::Tensor W_high_packed,
    torch::Tensor W_low_packed,
    torch::Tensor scales_high,
    torch::Tensor zeros_high,
    torch::Tensor scales_low,
    torch::Tensor zeros_low,
    torch::Tensor col_indices,
    int64_t n_high_cols,
    int64_t n_low_cols,
    int64_t tile_size
);

// ============================================================================
// MMA (Tensor Core) Kernels
// ============================================================================

// Simple MMA-based matmul for FP16 (baseline for comparison)
// A: [M, K], B: [K, N] -> C: [M, N]
torch::Tensor mma_matmul_simple(
    torch::Tensor A,
    torch::Tensor B
);

// Tiled WMMA matmul with shared memory
torch::Tensor wmma_matmul_tiled(
    torch::Tensor A,
    torch::Tensor B
);

// Pipelined WMMA matmul with double buffering
torch::Tensor wmma_matmul_pipelined(
    torch::Tensor A,
    torch::Tensor B
);

// Optimized WMMA with async copy and deeper pipeline
torch::Tensor wmma_matmul_optimized(
    torch::Tensor A,
    torch::Tensor B
);

// Heavy register blocking WMMA
torch::Tensor wmma_matmul_heavy(
    torch::Tensor A,
    torch::Tensor B
);

// WMMA v2 with vectorized loads and larger K tile
torch::Tensor wmma_matmul_v2(
    torch::Tensor A,
    torch::Tensor B
);

// Warp-specialized WMMA with bank conflict avoidance
torch::Tensor wmma_matmul_warp_special(
    torch::Tensor A,
    torch::Tensor B
);

// Small tile WMMA for higher occupancy
torch::Tensor wmma_matmul_small_tile(
    torch::Tensor A,
    torch::Tensor B
);

} // namespace cpr
