// CPR-SINQ: Dequantization CUDA Kernels
//
// Dequantization formula: W_fp = (W_q - zero) * scale
// Where scale and zero are per-tile (tile_size columns)

#include "cpr.h"
#include <ATen/cuda/CUDAContext.h>

namespace cpr {

// ============================================================================
// Fused Unpack + Dequantize 6-bit Kernel
// ============================================================================

__global__ void dequantize_6bit_kernel(
    const uint8_t* __restrict__ packed,
    const half* __restrict__ scales,     // [n_tiles, n_rows]
    const half* __restrict__ zeros,      // [n_tiles, n_rows]
    half* __restrict__ output,           // [n_rows, n_cols]
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols,
    int64_t tile_size,
    int64_t n_tiles
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 3) return;

    int64_t in_base = pack_idx * 3;
    int64_t col_base = pack_idx * 4;

    // Load 3 bytes
    uint8_t b0 = packed[row * packed_cols + in_base + 0];
    uint8_t b1 = packed[row * packed_cols + in_base + 1];
    uint8_t b2 = packed[row * packed_cols + in_base + 2];

    // Unpack to 4 values
    int8_t v0 = (b0 >> 2) & 0x3F;
    int8_t v1 = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F);
    int8_t v2 = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03);
    int8_t v3 = b2 & 0x3F;

    // Dequantize each value
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int64_t col = col_base + i;
        if (col < n_cols) {
            int64_t tile_idx = col / tile_size;
            half scale = scales[tile_idx * n_rows + row];
            half zero = zeros[tile_idx * n_rows + row];

            int8_t val = (i == 0) ? v0 : ((i == 1) ? v1 : ((i == 2) ? v2 : v3));
            half dequant = __hmul(__hsub(__float2half((float)val), zero), scale);
            output[row * n_cols + col] = dequant;
        }
    }
}

// ============================================================================
// Fused Unpack + Dequantize 5-bit Kernel
// ============================================================================

__global__ void dequantize_5bit_kernel(
    const uint8_t* __restrict__ packed,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half* __restrict__ output,
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols,
    int64_t tile_size,
    int64_t n_tiles
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 5) return;

    int64_t in_base = pack_idx * 5;
    int64_t col_base = pack_idx * 8;

    // Load 5 bytes
    uint8_t b0 = packed[row * packed_cols + in_base + 0];
    uint8_t b1 = packed[row * packed_cols + in_base + 1];
    uint8_t b2 = packed[row * packed_cols + in_base + 2];
    uint8_t b3 = packed[row * packed_cols + in_base + 3];
    uint8_t b4 = packed[row * packed_cols + in_base + 4];

    // Unpack to 8 values
    int8_t v[8];
    v[0] = (b0 >> 3) & 0x1F;
    v[1] = ((b0 & 0x07) << 2) | ((b1 >> 6) & 0x03);
    v[2] = (b1 >> 1) & 0x1F;
    v[3] = ((b1 & 0x01) << 4) | ((b2 >> 4) & 0x0F);
    v[4] = ((b2 & 0x0F) << 1) | ((b3 >> 7) & 0x01);
    v[5] = (b3 >> 2) & 0x1F;
    v[6] = ((b3 & 0x03) << 3) | ((b4 >> 5) & 0x07);
    v[7] = b4 & 0x1F;

    // Dequantize each value
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t col = col_base + i;
        if (col < n_cols) {
            int64_t tile_idx = col / tile_size;
            half scale = scales[tile_idx * n_rows + row];
            half zero = zeros[tile_idx * n_rows + row];

            half dequant = __hmul(__hsub(__float2half((float)v[i]), zero), scale);
            output[row * n_cols + col] = dequant;
        }
    }
}

// ============================================================================
// Optimized Dequantize with Vectorized Stores (half2)
// ============================================================================

__global__ void dequantize_6bit_vectorized_kernel(
    const uint8_t* __restrict__ packed,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half2* __restrict__ output,          // Output as half2 for coalesced writes
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols,
    int64_t tile_size,
    int64_t n_tiles
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 3) return;

    int64_t in_base = pack_idx * 3;
    int64_t col_base = pack_idx * 4;
    int64_t out_half2_base = (row * n_cols + col_base) / 2;

    // Load 3 bytes
    uint8_t b0 = packed[row * packed_cols + in_base + 0];
    uint8_t b1 = packed[row * packed_cols + in_base + 1];
    uint8_t b2 = packed[row * packed_cols + in_base + 2];

    // Unpack to 4 values
    float v0 = (float)((b0 >> 2) & 0x3F);
    float v1 = (float)(((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F));
    float v2 = (float)(((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03));
    float v3 = (float)(b2 & 0x3F);

    // Get tile indices (assuming all 4 values in same tile for simplicity)
    // For proper handling, check tile boundaries
    int64_t tile_idx = col_base / tile_size;
    half scale = scales[tile_idx * n_rows + row];
    half zero = zeros[tile_idx * n_rows + row];

    float scale_f = __half2float(scale);
    float zero_f = __half2float(zero);

    // Dequantize
    half d0 = __float2half((v0 - zero_f) * scale_f);
    half d1 = __float2half((v1 - zero_f) * scale_f);
    half d2 = __float2half((v2 - zero_f) * scale_f);
    half d3 = __float2half((v3 - zero_f) * scale_f);

    // Write as half2 (2 writes instead of 4)
    if (col_base + 1 < n_cols) {
        output[out_half2_base] = __halves2half2(d0, d1);
    }
    if (col_base + 3 < n_cols) {
        output[out_half2_base + 1] = __halves2half2(d2, d3);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

torch::Tensor dequantize_6bit(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t n_cols,
    int64_t tile_size
) {
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");
    TORCH_CHECK(zeros.is_cuda(), "zeros must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be float16");

    int64_t n_rows = packed.size(0);
    int64_t packed_cols = packed.size(1);
    int64_t n_tiles = (n_cols + tile_size - 1) / tile_size;

    auto output = torch::zeros({n_rows, n_cols},
        torch::TensorOptions().dtype(torch::kFloat16).device(packed.device()));

    int64_t n_pack_groups = packed_cols / 3;
    const int threads = 256;
    dim3 blocks((n_pack_groups + threads - 1) / threads, n_rows);

    dequantize_6bit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        n_rows, n_cols, packed_cols, tile_size, n_tiles
    );

    return output;
}

torch::Tensor dequantize_5bit(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t n_cols,
    int64_t tile_size
) {
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");
    TORCH_CHECK(zeros.is_cuda(), "zeros must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be float16");

    int64_t n_rows = packed.size(0);
    int64_t packed_cols = packed.size(1);
    int64_t n_tiles = (n_cols + tile_size - 1) / tile_size;

    auto output = torch::zeros({n_rows, n_cols},
        torch::TensorOptions().dtype(torch::kFloat16).device(packed.device()));

    int64_t n_pack_groups = packed_cols / 5;
    const int threads = 256;
    dim3 blocks((n_pack_groups + threads - 1) / threads, n_rows);

    dequantize_5bit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        n_rows, n_cols, packed_cols, tile_size, n_tiles
    );

    return output;
}

} // namespace cpr
