// CPR-SINQ: Optimized Tiled Matrix Multiplication
//
// This implements a tiled matmul with shared memory staging.
// The approach:
// 1. Load weight tiles into shared memory and dequantize there
// 2. Load activation tiles into shared memory
// 3. Perform tiled matrix multiply
//
// For even better performance, consider Tensor Core variants.

#include "cpr.h"
#include <ATen/cuda/CUDAContext.h>

namespace cpr {

// ============================================================================
// Tiled matmul with on-the-fly dequantization
// Block configuration optimized for A100/A40
// ============================================================================

// Tile sizes
constexpr int BM = 128;  // Tile rows (batch dimension)
constexpr int BN = 128;  // Tile cols (output features)
constexpr int BK = 32;   // Reduction tile size

// Thread block dimensions
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

// Values computed per thread
constexpr int TM = BM / THREADS_Y;  // 8 rows per thread
constexpr int TN = BN / THREADS_X;  // 8 cols per thread

__device__ __forceinline__ void unpack_6bit_values(
    uint8_t b0, uint8_t b1, uint8_t b2,
    int& v0, int& v1, int& v2, int& v3
) {
    v0 = (b0 >> 2) & 0x3F;
    v1 = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F);
    v2 = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03);
    v3 = b2 & 0x3F;
}

__device__ __forceinline__ void unpack_5bit_values(
    uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4,
    int* v  // Array of 8 values
) {
    v[0] = (b0 >> 3) & 0x1F;
    v[1] = ((b0 & 0x07) << 2) | ((b1 >> 6) & 0x03);
    v[2] = (b1 >> 1) & 0x1F;
    v[3] = ((b1 & 0x01) << 4) | ((b2 >> 4) & 0x0F);
    v[4] = ((b2 & 0x0F) << 1) | ((b3 >> 7) & 0x01);
    v[5] = (b3 >> 2) & 0x1F;
    v[6] = ((b3 & 0x03) << 3) | ((b4 >> 5) & 0x07);
    v[7] = b4 & 0x1F;
}

// ============================================================================
// Optimized kernel using two-phase approach:
// Phase 1: Dequantize weights (can be done once and cached)
// Phase 2: Standard tiled matmul
// ============================================================================

// For inference, we pre-dequantize and use standard matmul
// This avoids the complexity of fused kernel while maintaining correctness

// ============================================================================
// Alternative: Batched dequantization for memory efficiency
// Dequantize BK columns at a time, then multiply
// ============================================================================

template<int TILE_K>
__global__ void cpr_matmul_streaming_kernel(
    const half* __restrict__ X,           // [M, K]
    const uint8_t* __restrict__ W_high,   // [N, packed_high]
    const uint8_t* __restrict__ W_low,    // [N, packed_low]
    const half* __restrict__ scales_high, // [n_tiles_high, N]
    const half* __restrict__ zeros_high,
    const half* __restrict__ scales_low,
    const half* __restrict__ zeros_low,
    const int16_t* __restrict__ col_indices,
    half* __restrict__ Y,                 // [M, N]
    int M, int N, int K,
    int n_high, int n_low,
    int tile_size,
    int packed_high_cols, int packed_low_cols
) {
    // This kernel processes a TILE_K chunk of K at a time
    // reducing memory requirements for weight dequantization

    extern __shared__ half smem[];

    // Partition shared memory
    // W_tile: [TILE_K, BN] for dequantized weights
    // X_tile: [BM, TILE_K] for activations
    half* W_tile = smem;  // TILE_K * BN
    half* X_tile = smem + TILE_K * BN;  // BM * TILE_K

    int bm = blockIdx.x;  // Block row
    int bn = blockIdx.y;  // Block col

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    // Register accumulators
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    int row_start = bm * BM;
    int col_start = bn * BN;

    // Iterate over K dimension in TILE_K chunks
    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        int k_end = min(k_start + TILE_K, K);
        int k_len = k_end - k_start;

        // Cooperative load of X tile
        for (int idx = tid; idx < BM * k_len; idx += blockDim.x * blockDim.y) {
            int local_m = idx / k_len;
            int local_k = idx % k_len;
            int global_m = row_start + local_m;
            int global_k = k_start + local_k;

            if (global_m < M && global_k < K) {
                X_tile[local_m * TILE_K + local_k] = X[global_m * K + global_k];
            } else {
                X_tile[local_m * TILE_K + local_k] = __float2half(0.0f);
            }
        }

        // Cooperative load and dequantize W tile
        // Each thread handles multiple (k, n) pairs
        for (int idx = tid; idx < k_len * BN; idx += blockDim.x * blockDim.y) {
            int local_k = idx / BN;
            int local_n = idx % BN;
            int global_k = k_start + local_k;
            int global_n = col_start + local_n;

            half w_val = __float2half(0.0f);

            if (global_n < N && global_k < K) {
                // Determine if this K index is in high or low precision region
                int perm_k = col_indices[global_k];

                if (perm_k < n_high) {
                    // 6-bit high precision
                    int pack_group = perm_k / 4;
                    int pack_offset = perm_k % 4;
                    int byte_idx = pack_group * 3;

                    uint8_t b0 = W_high[global_n * packed_high_cols + byte_idx + 0];
                    uint8_t b1 = W_high[global_n * packed_high_cols + byte_idx + 1];
                    uint8_t b2 = W_high[global_n * packed_high_cols + byte_idx + 2];

                    int v0, v1, v2, v3;
                    unpack_6bit_values(b0, b1, b2, v0, v1, v2, v3);
                    int w_q = (pack_offset == 0) ? v0 :
                              (pack_offset == 1) ? v1 :
                              (pack_offset == 2) ? v2 : v3;

                    int tile_idx = perm_k / tile_size;
                    float scale = __half2float(scales_high[tile_idx * N + global_n]);
                    float zero = __half2float(zeros_high[tile_idx * N + global_n]);
                    w_val = __float2half(((float)w_q - zero) * scale);
                } else {
                    // 5-bit low precision
                    int low_k = perm_k - n_high;
                    int pack_group = low_k / 8;
                    int pack_offset = low_k % 8;
                    int byte_idx = pack_group * 5;

                    uint8_t b0 = W_low[global_n * packed_low_cols + byte_idx + 0];
                    uint8_t b1 = W_low[global_n * packed_low_cols + byte_idx + 1];
                    uint8_t b2 = W_low[global_n * packed_low_cols + byte_idx + 2];
                    uint8_t b3 = W_low[global_n * packed_low_cols + byte_idx + 3];
                    uint8_t b4 = W_low[global_n * packed_low_cols + byte_idx + 4];

                    int v[8];
                    unpack_5bit_values(b0, b1, b2, b3, b4, v);
                    int w_q = v[pack_offset];

                    int tile_idx = low_k / tile_size;
                    float scale = __half2float(scales_low[tile_idx * N + global_n]);
                    float zero = __half2float(zeros_low[tile_idx * N + global_n]);
                    w_val = __float2half(((float)w_q - zero) * scale);
                }
            }

            W_tile[local_k * BN + local_n] = w_val;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int k = 0; k < k_len; k++) {
            // Load X values for this thread's rows
            half x_vals[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                int local_m = ty * TM + i;
                x_vals[i] = X_tile[local_m * TILE_K + k];
            }

            // Load W values for this thread's cols
            half w_vals[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int local_n = tx * TN + j;
                w_vals[j] = W_tile[k * BN + local_n];
            }

            // Outer product
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += __half2float(x_vals[i]) * __half2float(w_vals[j]);
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int global_m = row_start + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int global_n = col_start + tx * TN + j;
            if (global_m < M && global_n < N) {
                Y[global_m * N + global_n] = __float2half(acc[i][j]);
            }
        }
    }
}

// Host function for streaming kernel
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
) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat16, "X must be float16");

    int64_t M = X.size(0);
    int64_t K = X.size(1);
    int64_t N = W_high_packed.size(0);

    int64_t packed_high_cols = W_high_packed.size(1);
    int64_t packed_low_cols = W_low_packed.size(1);

    auto Y = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat16).device(X.device()));

    // Grid and block configuration
    constexpr int TILE_K = 32;
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    // Shared memory size
    size_t smem_size = (TILE_K * BN + BM * TILE_K) * sizeof(half);

    cpr_matmul_streaming_kernel<TILE_K><<<grid, block, smem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        W_high_packed.data_ptr<uint8_t>(),
        W_low_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_high.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros_high.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_low.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros_low.data_ptr<at::Half>()),
        col_indices.data_ptr<int16_t>(),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        M, N, K,
        n_high_cols, n_low_cols,
        tile_size,
        packed_high_cols, packed_low_cols
    );

    return Y;
}

} // namespace cpr
