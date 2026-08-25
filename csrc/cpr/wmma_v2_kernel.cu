// CPR-SINQ: WMMA v2 Kernel
//
// Key insight: The pipelined kernel at 33% is bottlenecked by:
// 1. Memory load latency not fully hidden
// 2. __syncthreads barriers
//
// This version tries:
// 1. Larger K dimension per tile (more compute per sync)
// 2. Vectorized loads (load4 for coalesced access)
// 3. More warps per block for faster loads

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

namespace cpr {

// =============================================================================
// Constants
// =============================================================================

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Larger K tile for more compute per memory access
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;  // 4x WMMA_K

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 256

// Warp layout: 4x2 = 8 warps
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

// Each warp: 32x64 = 2x4 tiles
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

constexpr int SMEM_A_STRIDE = BLOCK_K;
constexpr int SMEM_B_STRIDE = BLOCK_N;

// =============================================================================
// WMMA v2 Kernel
// =============================================================================

__global__ void wmma_matmul_v2_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Dynamic shared memory - double buffered
    extern __shared__ half smem[];
    half* smem_A_base = smem;
    half* smem_B_base = smem + 2 * BLOCK_M * BLOCK_K;

    // Helper macros to index into the flattened arrays
    #define SMEM_A(stage, row, col) smem_A_base[(stage) * BLOCK_M * BLOCK_K + (row) * BLOCK_K + (col)]
    #define SMEM_B(stage, row, col) smem_B_base[(stage) * BLOCK_K * BLOCK_N + (row) * BLOCK_N + (col)]
    #define SMEM_A_PTR(stage, row, col) (&smem_A_base[(stage) * BLOCK_M * BLOCK_K + (row) * BLOCK_K + (col)])
    #define SMEM_B_PTR(stage, row, col) (&smem_B_base[(stage) * BLOCK_K * BLOCK_N + (row) * BLOCK_N + (col)])

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= N) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    // Accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    int write_stage = 0;
    int read_stage = 0;

    // Load first tile
    {
        // Use vectorized loads where possible
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K / 4; i += NUM_THREADS) {
            int idx = i * 4;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int gm = block_m + row;
            int gk = col;

            if (gm < M && gk + 3 < K) {
                // Load 4 halfs at once
                float2 val = *reinterpret_cast<const float2*>(A + gm * K + gk);
                *reinterpret_cast<float2*>(SMEM_A_PTR(write_stage, row, col)) = val;
            } else {
                // Scalar fallback with bounds check
                for (int j = 0; j < 4; j++) {
                    half val = (gm < M && gk + j < K) ? A[gm * K + gk + j] : __float2half(0.0f);
                    SMEM_A(write_stage, row, col + j) = val;
                }
            }
        }

        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N / 4; i += NUM_THREADS) {
            int idx = i * 4;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int gk = row;
            int gn = block_n + col;

            if (gk < K && gn + 3 < N) {
                float2 val = *reinterpret_cast<const float2*>(B + gk * N + gn);
                *reinterpret_cast<float2*>(SMEM_B_PTR(write_stage, row, col)) = val;
            } else {
                for (int j = 0; j < 4; j++) {
                    half val = (gk < K && gn + j < N) ? B[gk * N + gn + j] : __float2half(0.0f);
                    SMEM_B(write_stage, row, col + j) = val;
                }
            }
        }
    }

    __syncthreads();
    write_stage ^= 1;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        read_stage = k_tile % 2;
        write_stage = (k_tile + 1) % 2;

        // Start loading next tile if available
        if (k_tile + 1 < num_k_tiles) {
            int next_k = (k_tile + 1) * BLOCK_K;

            for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K / 4; i += NUM_THREADS) {
                int idx = i * 4;
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                int gm = block_m + row;
                int gk = next_k + col;

                if (gm < M && gk + 3 < K) {
                    float2 val = *reinterpret_cast<const float2*>(A + gm * K + gk);
                    *reinterpret_cast<float2*>(SMEM_A_PTR(write_stage, row, col)) = val;
                } else {
                    for (int j = 0; j < 4; j++) {
                        half val = (gm < M && gk + j < K) ? A[gm * K + gk + j] : __float2half(0.0f);
                        SMEM_A(write_stage, row, col + j) = val;
                    }
                }
            }

            for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N / 4; i += NUM_THREADS) {
                int idx = i * 4;
                int row = idx / BLOCK_N;
                int col = idx % BLOCK_N;
                int gk = next_k + row;
                int gn = block_n + col;

                if (gk < K && gn + 3 < N) {
                    float2 val = *reinterpret_cast<const float2*>(B + gk * N + gn);
                    *reinterpret_cast<float2*>(SMEM_B_PTR(write_stage, row, col)) = val;
                } else {
                    for (int j = 0; j < 4; j++) {
                        half val = (gk < K && gn + j < N) ? B[gk * N + gn + j] : __float2half(0.0f);
                        SMEM_B(write_stage, row, col + j) = val;
                    }
                }
            }
        }

        // Compute on current tile - process all 4 K sub-tiles
        const int warp_row_base = warp_m * (BLOCK_M / WARPS_M);
        const int warp_col_base = warp_n * (BLOCK_N / WARPS_N);

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILES_M];

            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; tm++) {
                int a_row = warp_row_base + tm * WMMA_M;
                wmma::load_matrix_sync(a_frag[tm], SMEM_A_PTR(read_stage, a_row, k), BLOCK_K);
            }

            #pragma unroll
            for (int tn = 0; tn < WARP_TILES_N; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                int b_col = warp_col_base + tn * WMMA_N;
                wmma::load_matrix_sync(b_frag, SMEM_B_PTR(read_stage, k, b_col), BLOCK_N);

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

torch::Tensor wmma_matmul_v2(
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

    // Calculate shared memory size (needs 64KB for double-buffered A and B)
    int smem_size = 2 * BLOCK_M * BLOCK_K * sizeof(half) +
                    2 * BLOCK_K * BLOCK_N * sizeof(half);

    // Configure kernel to use extended shared memory
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(wmma_matmul_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        configured = true;
    }

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);

    wmma_matmul_v2_kernel<<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
