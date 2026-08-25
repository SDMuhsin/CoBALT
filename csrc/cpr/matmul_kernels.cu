// CPR-SINQ: Matrix Multiplication CUDA Kernels
//
// Implements Y = X @ W^T for CPR-quantized weights
// Strategy: Two separate dequant+matmul for high and low precision regions

#include "cpr.h"
#include <ATen/cuda/CUDAContext.h>

namespace cpr {

// ============================================================================
// Constants and Configuration
// ============================================================================

// Tile sizes for shared memory tiling
constexpr int TILE_M = 64;  // Rows of output per block
constexpr int TILE_N = 64;  // Columns of output per block
constexpr int TILE_K = 32;  // Reduction dimension per iteration

// Block dimensions
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Naive Matmul with On-the-fly Dequantization (for correctness testing)
// Y[m,n] = sum_k X[m,k] * W[n,k]  (W stored as [out_features, in_features])
// ============================================================================

__global__ void cpr_matmul_naive_kernel(
    const half* __restrict__ X,           // [batch, in_features]
    const uint8_t* __restrict__ W_high,   // [out_features, packed_high_cols]
    const uint8_t* __restrict__ W_low,    // [out_features, packed_low_cols]
    const half* __restrict__ scales_high, // [n_tiles_high, out_features]
    const half* __restrict__ zeros_high,
    const half* __restrict__ scales_low,  // [n_tiles_low, out_features]
    const half* __restrict__ zeros_low,
    const int16_t* __restrict__ col_indices, // [in_features]
    half* __restrict__ Y,                 // [batch, out_features]
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    int64_t n_high_cols,
    int64_t n_low_cols,
    int64_t tile_size,
    int64_t packed_high_cols,
    int64_t packed_low_cols
) {
    int64_t m = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;  // output feature index

    if (m >= batch_size || n >= out_features) return;

    float acc = 0.0f;

    // Process high-precision columns (6-bit)
    for (int64_t k = 0; k < n_high_cols; k++) {
        // Get original column index
        int64_t orig_col = col_indices[k];

        // Load activation
        float x_val = __half2float(X[m * in_features + orig_col]);

        // Unpack 6-bit weight (simplified - assuming aligned access)
        int64_t pack_group = k / 4;
        int64_t pack_offset = k % 4;
        int64_t byte_idx = pack_group * 3;

        uint8_t b0 = W_high[n * packed_high_cols + byte_idx + 0];
        uint8_t b1 = W_high[n * packed_high_cols + byte_idx + 1];
        uint8_t b2 = W_high[n * packed_high_cols + byte_idx + 2];

        int8_t w_q;
        if (pack_offset == 0) w_q = (b0 >> 2) & 0x3F;
        else if (pack_offset == 1) w_q = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F);
        else if (pack_offset == 2) w_q = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03);
        else w_q = b2 & 0x3F;

        // Dequantize
        int64_t tile_idx = k / tile_size;
        float scale = __half2float(scales_high[tile_idx * out_features + n]);
        float zero = __half2float(zeros_high[tile_idx * out_features + n]);
        float w_val = ((float)w_q - zero) * scale;

        acc += x_val * w_val;
    }

    // Process low-precision columns (5-bit)
    for (int64_t k = 0; k < n_low_cols; k++) {
        // Get original column index
        int64_t orig_col = col_indices[n_high_cols + k];

        // Load activation
        float x_val = __half2float(X[m * in_features + orig_col]);

        // Unpack 5-bit weight
        int64_t pack_group = k / 8;
        int64_t pack_offset = k % 8;
        int64_t byte_idx = pack_group * 5;

        uint8_t b0 = W_low[n * packed_low_cols + byte_idx + 0];
        uint8_t b1 = W_low[n * packed_low_cols + byte_idx + 1];
        uint8_t b2 = W_low[n * packed_low_cols + byte_idx + 2];
        uint8_t b3 = W_low[n * packed_low_cols + byte_idx + 3];
        uint8_t b4 = W_low[n * packed_low_cols + byte_idx + 4];

        int8_t w_q;
        switch (pack_offset) {
            case 0: w_q = (b0 >> 3) & 0x1F; break;
            case 1: w_q = ((b0 & 0x07) << 2) | ((b1 >> 6) & 0x03); break;
            case 2: w_q = (b1 >> 1) & 0x1F; break;
            case 3: w_q = ((b1 & 0x01) << 4) | ((b2 >> 4) & 0x0F); break;
            case 4: w_q = ((b2 & 0x0F) << 1) | ((b3 >> 7) & 0x01); break;
            case 5: w_q = (b3 >> 2) & 0x1F; break;
            case 6: w_q = ((b3 & 0x03) << 3) | ((b4 >> 5) & 0x07); break;
            case 7: w_q = b4 & 0x1F; break;
            default: w_q = 0;
        }

        // Dequantize
        int64_t tile_idx = k / tile_size;
        float scale = __half2float(scales_low[tile_idx * out_features + n]);
        float zero = __half2float(zeros_low[tile_idx * out_features + n]);
        float w_val = ((float)w_q - zero) * scale;

        acc += x_val * w_val;
    }

    Y[m * out_features + n] = __float2half(acc);
}

// ============================================================================
// Tiled Matmul with Shared Memory (Phase 4 optimization)
// ============================================================================

// Shared memory tile for activations and dequantized weights
// This kernel pre-dequantizes a tile of weights into shared memory,
// then performs tile-wise matmul

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int BLOCK_TILE_K>
__global__ void cpr_matmul_tiled_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W_deq,  // Pre-dequantized for now
    half* __restrict__ Y,
    int64_t M,  // batch
    int64_t N,  // out_features
    int64_t K   // in_features
) {
    // Shared memory for tiles
    __shared__ half X_shared[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ half W_shared[BLOCK_TILE_K][BLOCK_TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = bx * BLOCK_TILE_M + ty;
    int col = by * BLOCK_TILE_N + tx;

    float acc = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K; t++) {
        // Cooperative loading of X tile
        int x_row = row;
        int x_col = t * BLOCK_TILE_K + tx;
        if (x_row < M && x_col < K) {
            X_shared[ty][tx] = X[x_row * K + x_col];
        } else {
            X_shared[ty][tx] = __float2half(0.0f);
        }

        // Cooperative loading of W tile (W is [N, K], we want [K, N] tile)
        int w_row = t * BLOCK_TILE_K + ty;
        int w_col = col;
        if (w_row < K && w_col < N) {
            W_shared[ty][tx] = W_deq[w_col * K + w_row];  // Transpose access
        } else {
            W_shared[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; k++) {
            acc += __half2float(X_shared[ty][k]) * __half2float(W_shared[k][tx]);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

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
) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat16, "X must be float16");
    TORCH_CHECK(col_indices.dtype() == torch::kInt16, "col_indices must be int16");

    int64_t batch_size = X.size(0);
    int64_t in_features = X.size(1);
    int64_t out_features = W_high_packed.size(0);

    int64_t packed_high_cols = W_high_packed.size(1);
    int64_t packed_low_cols = W_low_packed.size(1);

    auto Y = torch::zeros({batch_size, out_features},
        torch::TensorOptions().dtype(torch::kFloat16).device(X.device()));

    // Use naive kernel for now (will optimize later)
    dim3 block(16, 16);
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (out_features + block.y - 1) / block.y
    );

    cpr_matmul_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        W_high_packed.data_ptr<uint8_t>(),
        W_low_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_high.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros_high.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_low.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros_low.data_ptr<at::Half>()),
        col_indices.data_ptr<int16_t>(),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        batch_size, in_features, out_features,
        n_high_cols, n_low_cols, tile_size,
        packed_high_cols, packed_low_cols
    );

    return Y;
}

torch::Tensor cpr_linear(
    torch::Tensor X,
    torch::Tensor W_high_packed,
    torch::Tensor W_low_packed,
    torch::Tensor scales_high,
    torch::Tensor zeros_high,
    torch::Tensor scales_low,
    torch::Tensor zeros_low,
    torch::Tensor col_indices,
    torch::Tensor bias,
    int64_t n_high_cols,
    int64_t n_low_cols,
    int64_t tile_size
) {
    auto Y = cpr_matmul(
        X, W_high_packed, W_low_packed,
        scales_high, zeros_high, scales_low, zeros_low,
        col_indices, n_high_cols, n_low_cols, tile_size
    );

    if (bias.numel() > 0) {
        Y = Y + bias;
    }

    return Y;
}

void check_cuda() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
}

int64_t get_optimal_tile_size(int64_t n_rows, int64_t n_cols) {
    // Default tile size that balances shared memory usage and parallelism
    return 128;
}

} // namespace cpr
