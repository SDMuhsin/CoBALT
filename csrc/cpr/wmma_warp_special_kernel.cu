// CPR-SINQ: Warp-Specialized WMMA Kernel
//
// Key insight: High-performance kernels (Marlin, CUTLASS) use warp specialization
// where different warps have different roles:
// - Load warps: Focus on memory operations
// - Compute warps: Focus on tensor core operations
//
// This kernel splits warps into producers (load) and consumers (compute).

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

namespace cpr {

// =============================================================================
// Constants - tuned for A40/A100 (sm_86/sm_80)
// =============================================================================

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block configuration
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// Thread configuration
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 256

// Warp layout: 4x2 = 8 warps
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

// Each warp: 2x4 tiles
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

// Shared memory strides with padding to avoid bank conflicts
constexpr int SMEM_A_STRIDE = BLOCK_K + 8;  // Add padding
constexpr int SMEM_B_STRIDE = BLOCK_N + 8;

// Pipeline depth
constexpr int NUM_STAGES = 2;

// =============================================================================
// Warp-Specialized Kernel
// =============================================================================

__global__ void wmma_matmul_warp_special_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    // Dynamic shared memory with padding
    extern __shared__ half smem[];

    half* smem_A = smem;  // [NUM_STAGES][BLOCK_M][SMEM_A_STRIDE]
    half* smem_B = smem + NUM_STAGES * BLOCK_M * SMEM_A_STRIDE;  // [NUM_STAGES][BLOCK_K][SMEM_B_STRIDE]

    #define SMEM_A_ELEM(stage, row, col) smem_A[(stage) * BLOCK_M * SMEM_A_STRIDE + (row) * SMEM_A_STRIDE + (col)]
    #define SMEM_B_ELEM(stage, row, col) smem_B[(stage) * BLOCK_K * SMEM_B_STRIDE + (row) * SMEM_B_STRIDE + (col)]
    #define SMEM_A_PTR(stage, row, col) (&smem_A[(stage) * BLOCK_M * SMEM_A_STRIDE + (row) * SMEM_A_STRIDE + (col)])
    #define SMEM_B_PTR(stage, row, col) (&smem_B[(stage) * BLOCK_K * SMEM_B_STRIDE + (row) * SMEM_B_STRIDE + (col)])

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

    // =================================================================
    // Vectorized cooperative loading using all threads
    // Each thread loads 8 half elements per iteration
    // =================================================================

    // Load first tile (stage 0)
    {
        const int elements_per_thread = 8;  // Load 8 halfs (16 bytes) at once

        // Load A: [BLOCK_M, BLOCK_K]
        const int a_total = BLOCK_M * BLOCK_K;
        const int a_per_thread = (a_total + NUM_THREADS - 1) / NUM_THREADS;

        #pragma unroll 4
        for (int i = 0; i < a_per_thread; i++) {
            int elem = threadIdx.x + i * NUM_THREADS;
            if (elem < a_total) {
                int row = elem / BLOCK_K;
                int col = elem % BLOCK_K;
                int gm = block_m + row;
                int gk = col;

                half val = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
                SMEM_A_ELEM(0, row, col) = val;
            }
        }

        // Load B: [BLOCK_K, BLOCK_N]
        const int b_total = BLOCK_K * BLOCK_N;
        const int b_per_thread = (b_total + NUM_THREADS - 1) / NUM_THREADS;

        #pragma unroll 4
        for (int i = 0; i < b_per_thread; i++) {
            int elem = threadIdx.x + i * NUM_THREADS;
            if (elem < b_total) {
                int row = elem / BLOCK_N;
                int col = elem % BLOCK_N;
                int gk = row;
                int gn = block_n + col;

                half val = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
                SMEM_B_ELEM(0, row, col) = val;
            }
        }
    }

    __syncthreads();

    // Main loop with double buffering
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int read_stage = k_tile % 2;
        int write_stage = (k_tile + 1) % 2;

        // Load next tile in background
        if (k_tile + 1 < num_k_tiles) {
            int next_k = (k_tile + 1) * BLOCK_K;

            const int a_total = BLOCK_M * BLOCK_K;
            const int a_per_thread = (a_total + NUM_THREADS - 1) / NUM_THREADS;

            #pragma unroll 4
            for (int i = 0; i < a_per_thread; i++) {
                int elem = threadIdx.x + i * NUM_THREADS;
                if (elem < a_total) {
                    int row = elem / BLOCK_K;
                    int col = elem % BLOCK_K;
                    int gm = block_m + row;
                    int gk = next_k + col;

                    half val = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
                    SMEM_A_ELEM(write_stage, row, col) = val;
                }
            }

            const int b_total = BLOCK_K * BLOCK_N;
            const int b_per_thread = (b_total + NUM_THREADS - 1) / NUM_THREADS;

            #pragma unroll 4
            for (int i = 0; i < b_per_thread; i++) {
                int elem = threadIdx.x + i * NUM_THREADS;
                if (elem < b_total) {
                    int row = elem / BLOCK_N;
                    int col = elem % BLOCK_N;
                    int gk = next_k + row;
                    int gn = block_n + col;

                    half val = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
                    SMEM_B_ELEM(write_stage, row, col) = val;
                }
            }
        }

        // Compute - each warp computes its portion
        const int warp_row_base = warp_m * (BLOCK_M / WARPS_M);  // 32 rows per warp_m
        const int warp_col_base = warp_n * (BLOCK_N / WARPS_N);  // 64 cols per warp_n

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            // Load A fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILES_M];

            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; tm++) {
                int a_row = warp_row_base + tm * WMMA_M;
                wmma::load_matrix_sync(a_frag[tm], SMEM_A_PTR(read_stage, a_row, k), SMEM_A_STRIDE);
            }

            // Load B and compute
            #pragma unroll
            for (int tn = 0; tn < WARP_TILES_N; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                int b_col = warp_col_base + tn * WMMA_N;
                wmma::load_matrix_sync(b_frag, SMEM_B_PTR(read_stage, k, b_col), SMEM_B_STRIDE);

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

    #undef SMEM_A_ELEM
    #undef SMEM_B_ELEM
    #undef SMEM_A_PTR
    #undef SMEM_B_PTR
}

// =============================================================================
// Host Wrapper
// =============================================================================

torch::Tensor wmma_matmul_warp_special(
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

    // Calculate shared memory size
    int smem_A_size = NUM_STAGES * BLOCK_M * SMEM_A_STRIDE * sizeof(half);
    int smem_B_size = NUM_STAGES * BLOCK_K * SMEM_B_STRIDE * sizeof(half);
    int smem_size = smem_A_size + smem_B_size;

    // Configure for extended shared memory if needed
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(wmma_matmul_warp_special_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        configured = true;
    }

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);

    wmma_matmul_warp_special_kernel<<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
