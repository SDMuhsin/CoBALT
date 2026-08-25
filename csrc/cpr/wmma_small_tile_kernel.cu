// CPR-SINQ: Small Tile WMMA Kernel for Higher Occupancy
//
// Key insight: Our large tiles (128x128) use 37KB shared memory, limiting
// occupancy to 2 blocks per SM (33%). By using smaller tiles, we can
// achieve higher occupancy and better latency hiding.

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

namespace cpr {

// =============================================================================
// Constants - Smaller tiles for higher occupancy
// =============================================================================

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Smaller block for higher occupancy
// Target: ~16KB shared memory per block -> 6 blocks per SM -> 100% occupancy
constexpr int BLOCK_M = 64;   // Was 128
constexpr int BLOCK_N = 64;   // Was 128
constexpr int BLOCK_K = 32;

// Thread configuration
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;  // Reduced from 8
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 128

// Warp layout: 2x2 = 4 warps
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;

// Each warp: 2x2 tiles = 32x32 output
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 2;

// Shared memory strides with padding
constexpr int SMEM_A_STRIDE = BLOCK_K + 8;
constexpr int SMEM_B_STRIDE = BLOCK_N + 8;

constexpr int NUM_STAGES = 2;

// =============================================================================
// Small Tile Kernel
// =============================================================================

__global__ void wmma_matmul_small_tile_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ half smem[];

    half* smem_A = smem;
    half* smem_B = smem + NUM_STAGES * BLOCK_M * SMEM_A_STRIDE;

    #define SMEM_A_ELEM(stage, row, col) smem_A[(stage) * BLOCK_M * SMEM_A_STRIDE + (row) * SMEM_A_STRIDE + (col)]
    #define SMEM_B_ELEM(stage, row, col) smem_B[(stage) * BLOCK_K * SMEM_B_STRIDE + (row) * SMEM_B_STRIDE + (col)]
    #define SMEM_A_PTR(stage, row, col) (&smem_A[(stage) * BLOCK_M * SMEM_A_STRIDE + (row) * SMEM_A_STRIDE + (col)])
    #define SMEM_B_PTR(stage, row, col) (&smem_B[(stage) * BLOCK_K * SMEM_B_STRIDE + (row) * SMEM_B_STRIDE + (col)])

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= N) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    // Accumulators - 2x2 = 4 fragments per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // Load first tile
    {
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

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int read_stage = k_tile % 2;
        int write_stage = (k_tile + 1) % 2;

        // Load next tile
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

        // Compute
        const int warp_row_base = warp_m * (BLOCK_M / WARPS_M);
        const int warp_col_base = warp_n * (BLOCK_N / WARPS_N);

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILES_M];

            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; tm++) {
                int a_row = warp_row_base + tm * WMMA_M;
                wmma::load_matrix_sync(a_frag[tm], SMEM_A_PTR(read_stage, a_row, k), SMEM_A_STRIDE);
            }

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

torch::Tensor wmma_matmul_small_tile(
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

    // Calculate shared memory
    int smem_A_size = NUM_STAGES * BLOCK_M * SMEM_A_STRIDE * sizeof(half);
    int smem_B_size = NUM_STAGES * BLOCK_K * SMEM_B_STRIDE * sizeof(half);
    int smem_size = smem_A_size + smem_B_size;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);

    wmma_matmul_small_tile_kernel<<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
