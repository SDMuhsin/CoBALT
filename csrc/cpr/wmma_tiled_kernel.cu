// CPR-SINQ: Tiled WMMA Kernel with Shared Memory
//
// This kernel uses shared memory tiling to improve data reuse.
// Multiple warps per thread block share data loaded from global memory.
//
// Compute: C = A @ B where A is [M, K], B is [K, N], C is [M, N]
// All matrices are row-major.

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
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

// Block dimensions for output tiles
constexpr int BLOCK_M = 64;   // Output rows per block (4 WMMA tiles)
constexpr int BLOCK_N = 64;   // Output cols per block (4 WMMA tiles)
constexpr int BLOCK_K = 32;   // K dimension chunk per iteration

// Warp configuration
constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = BLOCK_M / WMMA_M;  // 4 warps in M dimension
constexpr int WARPS_N = BLOCK_N / WMMA_N;  // 4 warps in N dimension
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 16 warps = 512 threads

// =============================================================================
// Tiled WMMA Kernel
// =============================================================================

__global__ void wmma_matmul_tiled_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    // Shared memory for input tiles
    __shared__ half smem_A[BLOCK_M][BLOCK_K];
    __shared__ half smem_B[BLOCK_K][BLOCK_N];

    // Block and warp indices
    int block_m = blockIdx.y * BLOCK_M;
    int block_n = blockIdx.x * BLOCK_N;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Warp position within block (2D arrangement)
    int warp_m = (warp_id / WARPS_N) * WMMA_M;
    int warp_n = (warp_id % WARPS_N) * WMMA_N;

    // Skip if block is out of bounds
    if (block_m >= M || block_n >= N) return;

    // Declare accumulator fragments - one per warp output tile
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Main loop over K dimension
    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Cooperative load of A tile into shared memory
        // A tile: [BLOCK_M, BLOCK_K] from A[block_m:block_m+BLOCK_M, k_start:k_start+BLOCK_K]
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_m = block_m + row;
            int global_k = k_start + col;

            if (global_m < M && global_k < K) {
                smem_A[row][col] = A[global_m * K + global_k];
            } else {
                smem_A[row][col] = __float2half(0.0f);
            }
        }

        // Cooperative load of B tile into shared memory
        // B tile: [BLOCK_K, BLOCK_N] from B[k_start:k_start+BLOCK_K, block_n:block_n+BLOCK_N]
        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_k = k_start + row;
            int global_n = block_n + col;

            if (global_k < K && global_n < N) {
                smem_B[row][col] = B[global_k * N + global_n];
            } else {
                smem_B[row][col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Each warp performs WMMA on its portion of the tile
        // Loop over BLOCK_K in WMMA_K steps
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            // Declare input fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

            // Load A fragment from shared memory
            // A fragment: smem_A[warp_m:warp_m+16, k:k+16]
            wmma::load_matrix_sync(a_frag, &smem_A[warp_m][k], BLOCK_K);

            // Load B fragment from shared memory
            // B fragment: smem_B[k:k+16, warp_n:warp_n+16]
            wmma::load_matrix_sync(b_frag, &smem_B[k][warp_n], BLOCK_N);

            // Accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // Store result
    int out_m = block_m + warp_m;
    int out_n = block_n + warp_n;

    if (out_m + WMMA_M <= M && out_n + WMMA_N <= N) {
        // Convert FP32 accumulator to FP16
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_half;
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag_half.x[i] = __float2half(c_frag.x[i]);
        }
        wmma::store_matrix_sync(C + out_m * N + out_n, c_frag_half, N, wmma::mem_row_major);
    }
}

// =============================================================================
// Host Wrapper
// =============================================================================

torch::Tensor wmma_matmul_tiled(
    torch::Tensor A,  // [M, K]
    torch::Tensor B   // [K, N]
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "Inputs must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor
    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    // Grid dimensions
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 512 threads

    wmma_matmul_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
