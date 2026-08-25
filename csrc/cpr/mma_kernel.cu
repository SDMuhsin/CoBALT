// CPR-SINQ: MMA-based Fused Kernel
//
// This kernel uses Tensor Core WMMA instructions for matrix multiplication.
// Phase 3.1: Simple WMMA-based matmul for FP16 (baseline for verification)
//
// Computing: C = A @ B where A is [M, K], B is [K, N], C is [M, N]
// All matrices are row-major in memory.
//
// WMMA operations:
// - matrix_a with row_major: loads M×K tile from row-major memory
// - matrix_b with row_major: loads K×N tile from row-major memory
// - The MMA computes: C[m,n] += sum_k(A[m,k] * B[k,n])

#include "cpr.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

namespace cpr {

// =============================================================================
// Constants and Types
// =============================================================================

// WMMA tile dimensions (m16n16k16 is standard for Ampere)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp size
constexpr int WARP_SIZE = 32;

// =============================================================================
// Simple WMMA Kernel
// =============================================================================

// C = A @ B where A is [M, K], B is [K, N], C is [M, N]
// All matrices are row-major in memory
__global__ void wmma_matmul_simple_kernel(
    const half* __restrict__ A,  // [M, K] row-major
    const half* __restrict__ B,  // [K, N] row-major
    half* __restrict__ C,        // [M, N] row-major
    int M, int N, int K
) {
    // Each warp computes one 16x16 output tile
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    // Calculate which output tile this warp handles
    int num_warps_n = (N + WMMA_N - 1) / WMMA_N;
    int warp_m = (warp_id / num_warps_n) * WMMA_M;
    int warp_n = (warp_id % num_warps_n) * WMMA_N;

    // Skip if out of bounds
    if (warp_m >= M || warp_n >= N) return;

    // Skip partial tiles at boundaries (for simplicity)
    if (warp_m + WMMA_M > M || warp_n + WMMA_N > N) return;

    // Declare fragments
    // For C = A @ B where all are row-major:
    // - A is row-major, so use row_major layout
    // - B is row-major, so use row_major layout
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Skip partial K tiles
        if (k + WMMA_K > K) break;

        // Load A fragment: A[warp_m:warp_m+16, k:k+16]
        // A is [M, K] row-major, stride is K
        wmma::load_matrix_sync(a_frag, A + warp_m * K + k, K);

        // Load B fragment: B[k:k+16, warp_n:warp_n+16]
        // B is [K, N] row-major, stride is N
        wmma::load_matrix_sync(b_frag, B + k * N + warp_n, N);

        // Perform WMMA: C += A @ B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result - convert FP32 accumulator to FP16
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_half;

    // Convert each element from float to half
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag_half.x[i] = __float2half(c_frag.x[i]);
    }

    // Store to C[warp_m:warp_m+16, warp_n:warp_n+16]
    // C is [M, N] row-major, stride is N
    wmma::store_matrix_sync(C + warp_m * N + warp_n, c_frag_half, N, wmma::mem_row_major);
}

// =============================================================================
// Host Wrapper
// =============================================================================

torch::Tensor mma_matmul_simple(
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

    // Calculate grid dimensions
    // Each warp handles one 16x16 tile
    int warps_m = (M + WMMA_M - 1) / WMMA_M;
    int warps_n = (N + WMMA_N - 1) / WMMA_N;
    int total_warps = warps_m * warps_n;

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    wmma_matmul_simple_kernel<<<num_blocks, threads_per_block, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

} // namespace cpr
