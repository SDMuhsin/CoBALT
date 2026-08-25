// CPR-SINQ: Bit Packing/Unpacking CUDA Kernels
//
// Packing schemes:
// - 6-bit: 4 values packed into 3 bytes (24 bits)
// - 5-bit: 8 values packed into 5 bytes (40 bits)

#include "cpr.h"
#include <ATen/cuda/CUDAContext.h>

namespace cpr {

// ============================================================================
// 6-bit Packing: 4 values -> 3 bytes
// Layout: [v0:6][v1:6][v2:6][v3:6] -> [b0][b1][b2]
// b0 = v0[5:0] << 2 | v1[5:4]
// b1 = v1[3:0] << 4 | v2[5:2]
// b2 = v2[1:0] << 6 | v3[5:0]
// ============================================================================

__global__ void pack_6bit_kernel(
    const int8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 3) return;

    int64_t col_base = pack_idx * 4;  // 4 values per pack group
    int64_t out_base = pack_idx * 3;  // 3 bytes per pack group

    // Load 4 values (with bounds checking)
    uint8_t v0 = (col_base + 0 < n_cols) ? input[row * n_cols + col_base + 0] & 0x3F : 0;
    uint8_t v1 = (col_base + 1 < n_cols) ? input[row * n_cols + col_base + 1] & 0x3F : 0;
    uint8_t v2 = (col_base + 2 < n_cols) ? input[row * n_cols + col_base + 2] & 0x3F : 0;
    uint8_t v3 = (col_base + 3 < n_cols) ? input[row * n_cols + col_base + 3] & 0x3F : 0;

    // Pack into 3 bytes
    output[row * packed_cols + out_base + 0] = (v0 << 2) | (v1 >> 4);
    output[row * packed_cols + out_base + 1] = (v1 << 4) | (v2 >> 2);
    output[row * packed_cols + out_base + 2] = (v2 << 6) | v3;
}

__global__ void unpack_6bit_kernel(
    const uint8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 3) return;

    int64_t in_base = pack_idx * 3;
    int64_t col_base = pack_idx * 4;

    // Load 3 bytes
    uint8_t b0 = input[row * packed_cols + in_base + 0];
    uint8_t b1 = input[row * packed_cols + in_base + 1];
    uint8_t b2 = input[row * packed_cols + in_base + 2];

    // Unpack to 4 values
    uint8_t v0 = (b0 >> 2) & 0x3F;
    uint8_t v1 = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F);
    uint8_t v2 = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03);
    uint8_t v3 = b2 & 0x3F;

    // Store with bounds checking
    if (col_base + 0 < n_cols) output[row * n_cols + col_base + 0] = v0;
    if (col_base + 1 < n_cols) output[row * n_cols + col_base + 1] = v1;
    if (col_base + 2 < n_cols) output[row * n_cols + col_base + 2] = v2;
    if (col_base + 3 < n_cols) output[row * n_cols + col_base + 3] = v3;
}

// ============================================================================
// 5-bit Packing: 8 values -> 5 bytes
// Layout: [v0:5][v1:5][v2:5][v3:5][v4:5][v5:5][v6:5][v7:5] -> [b0][b1][b2][b3][b4]
// b0 = v0[4:0] << 3 | v1[4:2]
// b1 = v1[1:0] << 6 | v2[4:0] << 1 | v3[4]
// b2 = v3[3:0] << 4 | v4[4:1]
// b3 = v4[0] << 7 | v5[4:0] << 2 | v6[4:3]
// b4 = v6[2:0] << 5 | v7[4:0]
// ============================================================================

__global__ void pack_5bit_kernel(
    const int8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 5) return;

    int64_t col_base = pack_idx * 8;  // 8 values per pack group
    int64_t out_base = pack_idx * 5;  // 5 bytes per pack group

    // Load 8 values
    uint8_t v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = (col_base + i < n_cols) ? input[row * n_cols + col_base + i] & 0x1F : 0;
    }

    // Pack into 5 bytes
    output[row * packed_cols + out_base + 0] = (v[0] << 3) | (v[1] >> 2);
    output[row * packed_cols + out_base + 1] = (v[1] << 6) | (v[2] << 1) | (v[3] >> 4);
    output[row * packed_cols + out_base + 2] = (v[3] << 4) | (v[4] >> 1);
    output[row * packed_cols + out_base + 3] = (v[4] << 7) | (v[5] << 2) | (v[6] >> 3);
    output[row * packed_cols + out_base + 4] = (v[6] << 5) | v[7];
}

__global__ void unpack_5bit_kernel(
    const uint8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t n_rows,
    int64_t n_cols,
    int64_t packed_cols
) {
    int64_t row = blockIdx.y;
    int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows || pack_idx >= packed_cols / 5) return;

    int64_t in_base = pack_idx * 5;
    int64_t col_base = pack_idx * 8;

    // Load 5 bytes
    uint8_t b0 = input[row * packed_cols + in_base + 0];
    uint8_t b1 = input[row * packed_cols + in_base + 1];
    uint8_t b2 = input[row * packed_cols + in_base + 2];
    uint8_t b3 = input[row * packed_cols + in_base + 3];
    uint8_t b4 = input[row * packed_cols + in_base + 4];

    // Unpack to 8 values
    uint8_t v0 = (b0 >> 3) & 0x1F;
    uint8_t v1 = ((b0 & 0x07) << 2) | ((b1 >> 6) & 0x03);
    uint8_t v2 = (b1 >> 1) & 0x1F;
    uint8_t v3 = ((b1 & 0x01) << 4) | ((b2 >> 4) & 0x0F);
    uint8_t v4 = ((b2 & 0x0F) << 1) | ((b3 >> 7) & 0x01);
    uint8_t v5 = (b3 >> 2) & 0x1F;
    uint8_t v6 = ((b3 & 0x03) << 3) | ((b4 >> 5) & 0x07);
    uint8_t v7 = b4 & 0x1F;

    // Store with bounds checking
    if (col_base + 0 < n_cols) output[row * n_cols + col_base + 0] = v0;
    if (col_base + 1 < n_cols) output[row * n_cols + col_base + 1] = v1;
    if (col_base + 2 < n_cols) output[row * n_cols + col_base + 2] = v2;
    if (col_base + 3 < n_cols) output[row * n_cols + col_base + 3] = v3;
    if (col_base + 4 < n_cols) output[row * n_cols + col_base + 4] = v4;
    if (col_base + 5 < n_cols) output[row * n_cols + col_base + 5] = v5;
    if (col_base + 6 < n_cols) output[row * n_cols + col_base + 6] = v6;
    if (col_base + 7 < n_cols) output[row * n_cols + col_base + 7] = v7;
}

// ============================================================================
// Host Functions
// ============================================================================

torch::Tensor pack_6bit(torch::Tensor weights) {
    TORCH_CHECK(weights.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weights.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weights.dtype() == torch::kInt8, "Input must be int8");

    int64_t n_rows = weights.size(0);
    int64_t n_cols = weights.size(1);

    // Calculate packed size: ceil(n_cols / 4) * 3 bytes
    int64_t n_pack_groups = (n_cols + 3) / 4;
    int64_t packed_cols = n_pack_groups * 3;

    auto output = torch::zeros({n_rows, packed_cols},
        torch::TensorOptions().dtype(torch::kUInt8).device(weights.device()));

    const int threads = 256;
    dim3 blocks((n_pack_groups + threads - 1) / threads, n_rows);

    pack_6bit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        weights.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        n_rows, n_cols, packed_cols
    );

    return output;
}

torch::Tensor unpack_6bit(torch::Tensor packed, int64_t n_cols) {
    TORCH_CHECK(packed.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(packed.dim() == 2, "Input must be 2D");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "Input must be uint8");

    int64_t n_rows = packed.size(0);
    int64_t packed_cols = packed.size(1);

    auto output = torch::zeros({n_rows, n_cols},
        torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    int64_t n_pack_groups = packed_cols / 3;
    const int threads = 256;
    dim3 blocks((n_pack_groups + threads - 1) / threads, n_rows);

    unpack_6bit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        n_rows, n_cols, packed_cols
    );

    return output;
}

torch::Tensor pack_5bit(torch::Tensor weights) {
    TORCH_CHECK(weights.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weights.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weights.dtype() == torch::kInt8, "Input must be int8");

    int64_t n_rows = weights.size(0);
    int64_t n_cols = weights.size(1);

    // Calculate packed size: ceil(n_cols / 8) * 5 bytes
    int64_t n_pack_groups = (n_cols + 7) / 8;
    int64_t packed_cols = n_pack_groups * 5;

    auto output = torch::zeros({n_rows, packed_cols},
        torch::TensorOptions().dtype(torch::kUInt8).device(weights.device()));

    const int threads = 256;
    dim3 blocks((n_pack_groups + threads - 1) / threads, n_rows);

    pack_5bit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        weights.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        n_rows, n_cols, packed_cols
    );

    return output;
}

torch::Tensor unpack_5bit(torch::Tensor packed, int64_t n_cols) {
    TORCH_CHECK(packed.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(packed.dim() == 2, "Input must be 2D");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "Input must be uint8");

    int64_t n_rows = packed.size(0);
    int64_t packed_cols = packed.size(1);

    auto output = torch::zeros({n_rows, n_cols},
        torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    int64_t n_pack_groups = packed_cols / 5;
    const int threads = 256;
    dim3 blocks((n_pack_groups + threads - 1) / threads, n_rows);

    unpack_5bit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        n_rows, n_cols, packed_cols
    );

    return output;
}

} // namespace cpr
