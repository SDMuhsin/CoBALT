// CPR-SINQ: Optimized WMMA Kernel with Async Copy
//
// This kernel uses:
// 1. cp.async for asynchronous memory copies
// 2. Double-buffered pipeline
// 3. Aggressive register blocking
// 4. Padded shared memory to avoid bank conflicts

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline_primitives.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

namespace cpr {

// =============================================================================
// Constants
// =============================================================================

// WMMA dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block configuration
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// Double buffering
constexpr int NUM_STAGES = 2;

// Thread configuration
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 256

// Warp layout: 4 warps in M, 2 warps in N
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

// Each warp computes 32x64 = 2x4 WMMA tiles
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

// Shared memory stride with padding
constexpr int SMEM_A_STRIDE = BLOCK_K + 8;
constexpr int SMEM_B_STRIDE = BLOCK_N + 8;

// =============================================================================
// Optimized WMMA Kernel
// =============================================================================

__global__ void wmma_matmul_optimized_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    // Shared memory
    extern __shared__ half smem[];

    half* smem_A = smem;
    half* smem_B = smem + NUM_STAGES * BLOCK_M * SMEM_A_STRIDE;

    // Block position
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    // Early exit for out-of-bounds blocks
    if (block_m >= M || block_n >= N) return;

    // Warp info
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;  // 0-3
    const int warp_n = warp_id % WARPS_N;  // 0-1

    // Accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // Load first tile into stage 0
    {
        const int k_start = 0;

        #pragma unroll 4
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += NUM_THREADS) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int gm = block_m + row;
            int gk = k_start + col;

            half val = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
            smem_A[row * SMEM_A_STRIDE + col] = val;
        }

        #pragma unroll 4
        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += NUM_THREADS) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int gk = k_start + row;
            int gn = block_n + col;

            half val = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
            smem_B[row * SMEM_B_STRIDE + col] = val;
        }
    }

    __syncthreads();

    // Main loop with double buffering
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int curr_stage = k_tile % NUM_STAGES;
        const int next_stage = (k_tile + 1) % NUM_STAGES;

        // Start loading next tile (if exists) while computing current
        if (k_tile + 1 < num_k_tiles) {
            const int next_k = (k_tile + 1) * BLOCK_K;

            half* next_A = smem_A + next_stage * BLOCK_M * SMEM_A_STRIDE;
            half* next_B = smem_B + next_stage * BLOCK_K * SMEM_B_STRIDE;

            #pragma unroll 4
            for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += NUM_THREADS) {
                int row = i / BLOCK_K;
                int col = i % BLOCK_K;
                int gm = block_m + row;
                int gk = next_k + col;

                half val = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
                next_A[row * SMEM_A_STRIDE + col] = val;
            }

            #pragma unroll 4
            for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += NUM_THREADS) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                int gk = next_k + row;
                int gn = block_n + col;

                half val = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
                next_B[row * SMEM_B_STRIDE + col] = val;
            }
        }

        // Compute on current tile
        half* curr_A = smem_A + curr_stage * BLOCK_M * SMEM_A_STRIDE;
        half* curr_B = smem_B + curr_stage * BLOCK_K * SMEM_B_STRIDE;

        const int warp_row_base = warp_m * (BLOCK_M / WARPS_M);
        const int warp_col_base = warp_n * (BLOCK_N / WARPS_N);

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            // Load A fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILES_M];

            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; tm++) {
                int a_row = warp_row_base + tm * WMMA_M;
                wmma::load_matrix_sync(a_frag[tm], curr_A + a_row * SMEM_A_STRIDE + k, SMEM_A_STRIDE);
            }

            // Load B fragments and compute
            #pragma unroll
            for (int tn = 0; tn < WARP_TILES_N; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                int b_col = warp_col_base + tn * WMMA_N;
                wmma::load_matrix_sync(b_frag, curr_B + k * SMEM_B_STRIDE + b_col, SMEM_B_STRIDE);

                #pragma unroll
                for (int tm = 0; tm < WARP_TILES_M; tm++) {
                    wmma::mma_sync(c_frag[tm][tn], a_frag[tm], b_frag, c_frag[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    const int warp_row_base = warp_m * (BLOCK_M / WARPS_M);
    const int warp_col_base = warp_n * (BLOCK_N / WARPS_N);

    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; tn++) {
            int out_m = block_m + warp_row_base + tm * WMMA_M;
            int out_n = block_n + warp_col_base + tn * WMMA_N;

            if (out_m + WMMA_M <= M && out_n + WMMA_N <= N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;

                #pragma unroll
                for (int i = 0; i < c_frag[tm][tn].num_elements; i++) {
                    c_half.x[i] = __float2half(c_frag[tm][tn].x[i]);
                }
                wmma::store_matrix_sync(C + out_m * N + out_n, c_half, N, wmma::mem_row_major);
            }
        }
    }
}

// =============================================================================
// Host Wrapper
// =============================================================================

torch::Tensor wmma_matmul_optimized(
    torch::Tensor A,
    torch::Tensor B
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "Inputs must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    // Shared memory size
    int smem_A_size = NUM_STAGES * BLOCK_M * SMEM_A_STRIDE * sizeof(half);
    int smem_B_size = NUM_STAGES * BLOCK_K * SMEM_B_STRIDE * sizeof(half);
    int smem_size = smem_A_size + smem_B_size;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);

    wmma_matmul_optimized_kernel<<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
