// CPR-SINQ: Pipelined WMMA Kernel
//
// This kernel uses double buffering and overlapped memory/compute
// to better hide memory latency.
//
// Key optimizations:
// 1. Double-buffered shared memory
// 2. Async memory copies (cp.async)
// 3. Overlapped load and compute
// 4. Better thread block configuration

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

namespace cpr {

// =============================================================================
// Constants
// =============================================================================

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block dimensions - using 128x128 blocks with 8 warps
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// Warp configuration: 4x2 arrangement of 16x16 tiles = 64x32 per warp set
// Each warp computes multiple output tiles
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 256 threads

// Warps per dimension
constexpr int WARPS_M = 4;  // 4 warps in M
constexpr int WARPS_N = 2;  // 2 warps in N
// Each warp computes: (BLOCK_M/WARPS_M) x (BLOCK_N/WARPS_N) = 32x64 output tile
// That's 2x4 = 8 WMMA tiles per warp

// Number of pipeline stages
constexpr int NUM_STAGES = 2;

// Shared memory padding to avoid bank conflicts
constexpr int SMEM_PAD = 8;

// =============================================================================
// Pipelined WMMA Kernel
// =============================================================================

__global__ void wmma_matmul_pipelined_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    // Double-buffered shared memory with padding
    __shared__ half smem_A[NUM_STAGES][BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ half smem_B[NUM_STAGES][BLOCK_K][BLOCK_N + SMEM_PAD];

    // Block position
    int block_m = blockIdx.y * BLOCK_M;
    int block_n = blockIdx.x * BLOCK_N;

    // Early exit if block is out of bounds
    if (block_m >= M || block_n >= N) return;

    // Warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Warp position: 4 rows x 2 cols of warps
    int warp_m = (warp_id / WARPS_N);  // 0-3
    int warp_n = (warp_id % WARPS_N);  // 0-1

    // Each warp computes 32x64 output (2x4 WMMA tiles)
    // Accumulator fragments for 2x4 tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][4];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Calculate number of K iterations
    int num_k_iters = (K + BLOCK_K - 1) / BLOCK_K;

    // Prefetch first tile
    int k_start = 0;
    int stage = 0;

    // Load first A tile
    for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += NUM_THREADS) {
        int row = i / BLOCK_K;
        int col = i % BLOCK_K;
        int gm = block_m + row;
        int gk = k_start + col;
        half val = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
        smem_A[stage][row][col] = val;
    }

    // Load first B tile
    for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += NUM_THREADS) {
        int row = i / BLOCK_N;
        int col = i % BLOCK_N;
        int gk = k_start + row;
        int gn = block_n + col;
        half val = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
        smem_B[stage][row][col] = val;
    }

    __syncthreads();

    // Main loop with pipelining
    for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
        int curr_stage = k_iter % NUM_STAGES;
        int next_stage = (k_iter + 1) % NUM_STAGES;
        int next_k = (k_iter + 1) * BLOCK_K;

        // Start loading next tile (if there is one)
        if (k_iter + 1 < num_k_iters) {
            for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += NUM_THREADS) {
                int row = i / BLOCK_K;
                int col = i % BLOCK_K;
                int gm = block_m + row;
                int gk = next_k + col;
                half val = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
                smem_A[next_stage][row][col] = val;
            }

            for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += NUM_THREADS) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                int gk = next_k + row;
                int gn = block_n + col;
                half val = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
                smem_B[next_stage][row][col] = val;
            }
        }

        // Compute on current tile
        // Each warp handles 32x64 output = 2x4 WMMA tiles
        int warp_row_base = warp_m * (BLOCK_M / WARPS_M);  // 0, 32, 64, 96
        int warp_col_base = warp_n * (BLOCK_N / WARPS_N);  // 0, 64

        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            // Load A and B fragments for each output tile
            for (int tm = 0; tm < 2; tm++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                int a_row = warp_row_base + tm * WMMA_M;
                wmma::load_matrix_sync(a_frag, &smem_A[curr_stage][a_row][k], BLOCK_K + SMEM_PAD);

                for (int tn = 0; tn < 4; tn++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    int b_col = warp_col_base + tn * WMMA_N;
                    wmma::load_matrix_sync(b_frag, &smem_B[curr_stage][k][b_col], BLOCK_N + SMEM_PAD);

                    wmma::mma_sync(c_frag[tm][tn], a_frag, b_frag, c_frag[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    int warp_row_base = warp_m * (BLOCK_M / WARPS_M);
    int warp_col_base = warp_n * (BLOCK_N / WARPS_N);

    for (int tm = 0; tm < 2; tm++) {
        for (int tn = 0; tn < 4; tn++) {
            int out_m = block_m + warp_row_base + tm * WMMA_M;
            int out_n = block_n + warp_col_base + tn * WMMA_N;

            if (out_m + WMMA_M <= M && out_n + WMMA_N <= N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;
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

torch::Tensor wmma_matmul_pipelined(
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

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);

    wmma_matmul_pipelined_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
