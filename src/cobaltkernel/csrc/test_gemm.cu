// test_gemm.cu -- correctness + roofline/benchmark harness for cbk::gemm_tile.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "cobalt_format.h"
#include "cobalt_gemm.cuh"

// CBK_TEST_BLK: 0 = DENSE4/DENSE8 harness (unchanged); 4 / 6 = build the harness against
// cbk::gemm_tile's CoBALT-16:32 BLK1632_4 / BLK1632_6 path.  One value per extension build
// (the tile is templated on it), and a REDUCED config list, so the compile stays short.
#ifndef CBK_TEST_BLK
#define CBK_TEST_BLK 0
#endif

namespace {

struct GDesc { int K, N, layout, cs_pairs, ldX, ldY, M; };

__device__ __forceinline__ cbk::Mat build(const GDesc& d, const uint8_t* b, long long od,
                                          long long osc, long long oz) {
  cbk::Mat m;
  m.K = d.K; m.N = d.N; m.G = d.N / cbk::GROUP; m.layout = d.layout;
  m.data = b + od; m.row_off = nullptr;
  m.scale = (const __half*)(b + osc); m.zero = b + oz;
  return m;
}

template <int MT, int EPI, int WM, int S, int BR, int KT, int PF, int MINB>
__global__ __launch_bounds__(256, MINB) void gemm_kernel(GDesc d, const uint8_t* __restrict__ base,
                                                   long long od, long long osc, long long oz,
                                                   const __half* cs,
                                                   const __nv_bfloat16* __restrict__ X,
                                                   const __nv_bfloat16* nw, const float* rr,
                                                   void* Y) {
  extern __shared__ uint8_t smem[];
  const cbk::Mat w = build(d, base, od, osc, oz);
  const int nrt = (d.K + BR - 1) / BR;
  const int rt = blockIdx.x % nrt, mt = blockIdx.x / nrt;
  const int row0 = rt * BR;
  cbk::gemm_tile<MT, EPI, WM, S, BR, KT, PF, CBK_TEST_BLK>(w, cs, d.cs_pairs, row0,
                                         min(BR, d.K - row0), X,
                                         d.ldX, d.M, mt * MT, nw, rr, Y, d.ldY, smem);
}

template <int MT, int WM, int S, int BR, int KT, int PF, int MINB>
void launch_epi(int epi, GDesc d, const uint8_t* base, long long od, long long osc,
                long long oz, const __half* cs, const __nv_bfloat16* X,
                const __nv_bfloat16* nw, const float* rr, void* Y, int iters) {
  const int nrt = (d.K + BR - 1) / BR;
  const int nmt = (d.M + MT - 1) / MT;
  const int d8 = (d.layout == cbk::LAYOUT_DENSE8) || (CBK_TEST_BLK == 6);
  const int sh = cbk::gemm_smem_bytes<MT, S, BR, KT>(d8);
  dim3 g(nrt * nmt);
  auto go = [&](auto tag) {
    constexpr int E = decltype(tag)::value;
    if (sh > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<MT, E, WM, S, BR, KT, PF, MINB>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, sh));
    for (int i = 0; i < iters; ++i)
      gemm_kernel<MT, E, WM, S, BR, KT, PF, MINB><<<g, 256, sh>>>(d, base, od, osc, oz, cs, X, nw, rr, Y);
  };
  if (epi == 0) go(std::integral_constant<int, 0>{});
  else if (epi == 1) go(std::integral_constant<int, 1>{});
  else go(std::integral_constant<int, 2>{});
  C10_CUDA_CHECK(cudaGetLastError());
}

// (MT, WM, S, BR, KT) configurations compiled into the harness.  Keep CBK_CFGS in sync
// with CONFIGS in src/cobaltkernel/test_gemm.py.
#if CBK_TEST_BLK
// Coverage for the BLK16_32 tile path: MT in {16,32,64,128}, BR in {64,128,256},
// KT in {64,128,256}, WM in {1,2}, minb in {1,2}, including the shipping prefill tile
// (128,1,2,256,128) and the batched-decode tiles (32,1,4,{64,128},256).
#define CBK_CFGS(F)                                                                       \
  F(16, 1, 6, 64, 64, 1, 1)    F(32, 1, 4, 64, 64, 1, 1)   F(32, 1, 4, 128, 128, 1, 1)    \
  F(32, 1, 4, 64, 256, 1, 1)   F(32, 1, 4, 128, 256, 1, 2)                                \
  F(64, 1, 3, 128, 128, 1, 1)  F(64, 1, 3, 256, 128, 1, 1) F(64, 2, 3, 128, 128, 1, 2)    \
  F(128, 1, 2, 128, 128, 1, 1) F(128, 2, 2, 256, 128, 1, 1) F(128, 1, 2, 256, 128, 1, 1)
#else
#define CBK_CFGS(F)                                                                       \
  F(32, 1, 4, 64, 64, 1, 1)    F(32, 1, 4, 128, 128, 1, 1) F(32, 1, 4, 256, 128, 1, 1)     \
  F(32, 1, 4, 64, 256, 1, 1)   F(32, 1, 4, 64, 256, 2, 1)                                  \
  F(32, 1, 4, 128, 256, 1, 1)  F(32, 1, 4, 128, 256, 2, 1)                                 \
  F(16, 1, 6, 64, 64, 1, 1)                                                                \
  F(64, 1, 4, 64, 64, 1, 1)    F(64, 1, 3, 128, 128, 1, 1) F(64, 1, 3, 256, 128, 1, 1)     \
  F(64, 1, 3, 256, 128, 2, 1)                                                              \
  F(128, 1, 2, 128, 64, 1, 1)  F(128, 1, 2, 128, 128, 1, 1) F(128, 1, 2, 128, 128, 2, 1)   \
  F(128, 1, 2, 128, 128, 3, 1) F(128, 2, 2, 256, 128, 1, 1) F(128, 1, 2, 256, 128, 1, 1)   \
  /* --- round 4: OCCUPANCY configs, __launch_bounds__(256, MINB) --- */                   \
  F(64, 1, 3, 128, 128, 1, 2)  F(64, 2, 3, 128, 128, 1, 2) F(64, 2, 3, 128, 128, 1, 1)     \
  F(128, 1, 2, 64, 128, 1, 2)  F(128, 2, 2, 128, 128, 1, 2) F(128, 2, 2, 128, 128, 1, 1)   \
  F(64, 2, 3, 256, 128, 1, 2)  F(64, 2, 3, 256, 128, 1, 1)                                 \
  F(128, 2, 2, 256, 128, 1, 2)                                                             \
  F(64, 1, 3, 64, 128, 1, 3)   F(32, 1, 4, 128, 256, 1, 2) F(32, 1, 4, 64, 256, 1, 3)      \
  F(128, 2, 2, 128, 256, 1, 1)
#endif

void dispatch(int MT, int wm, int S, int BR, int KT, int pf, int mb, int epi, GDesc d,
              const uint8_t* base,
              long long od, long long osc, long long oz, const __half* cs,
              const __nv_bfloat16* X, const __nv_bfloat16* nw, const float* rr, void* Y,
              int iters) {
#define CBK_D(mt, w, st, br, kt, pfd, mnb)                                                 \
  if (MT == mt && wm == w && S == st && BR == br && KT == kt && pf == pfd && mb == mnb) {  \
    launch_epi<mt, w, st, br, kt, pfd, mnb>(epi, d, base, od, osc, oz, cs, X, nw, rr, Y,   \
                                            iters);                                        \
    return;                                                                                \
  }
  CBK_CFGS(CBK_D)
#undef CBK_D
  TORCH_CHECK(false, "unsupported cfg ", MT, ",", wm, ",", S, ",", BR, ",", KT, ",", pf, ",", mb);
}

// ---------------------------------------------------------------- mma roofline
__global__ __launch_bounds__(256) void mma_peak_kernel(int iters, float* sink) {
  uint32_t a[4], b[2];
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  const uint32_t s = threadIdx.x * 2654435761u + 1u;
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = s + i;
  b[0] = s ^ 0x3f803f80u; b[1] = s ^ 0x3f003f00u;
  float e[4] = {0.f, 0.f, 0.f, 0.f}, f[4] = {0.f, 0.f, 0.f, 0.f}, g[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < iters; ++i) {
    cbk::detail::mma16816(d, a, b);
    cbk::detail::mma16816(e, a, b);
    cbk::detail::mma16816(f, a, b);
    cbk::detail::mma16816(g, a, b);
    a[0] ^= (uint32_t)i;                 // keep the loop alive, 1 ALU op per 4 mma
  }
  sink[threadIdx.x & 3] = d[0] + e[1] + f[2] + g[3];
}

}  // namespace

double mma_peak(int64_t blocks, int64_t iters, int64_t reps) {
  auto o = torch::zeros({4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  mma_peak_kernel<<<blocks, 256>>>(iters, o.data_ptr<float>());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int r = 0; r < reps; ++r) mma_peak_kernel<<<blocks, 256>>>(iters, o.data_ptr<float>());
  cudaEventRecord(e1);
  C10_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  // 4 mma per iteration per warp, 8 warps per block, 16*8*16*2 flops per mma
  double flops = (double)blocks * 8.0 * (double)iters * 4.0 * 4096.0 * (double)reps;
  return flops / (ms * 1e-3) / 1e12;
}

torch::Tensor gemm(torch::Tensor blob, torch::Tensor X, torch::Tensor col_scale, int64_t K,
                   int64_t N, int64_t layout, int64_t od, int64_t osc, int64_t oz,
                   int64_t M, int64_t MT, int64_t epi, int64_t cs_pairs, int64_t wm,
                   int64_t S, int64_t BR, int64_t KT, int64_t PF, int64_t MINB) {
  TORCH_CHECK(blob.is_cuda() && X.is_cuda());
  const int outw = (epi == 2) ? (int)K / 2 : (int)K;
  auto opt = torch::dtype(epi == 0 ? torch::kFloat32 : torch::kBFloat16).device(blob.device());
  auto Y = torch::zeros({M, outw}, opt);
  GDesc d{(int)K, (int)N, (int)layout, (int)cs_pairs, (int)N, outw, (int)M};
  dispatch((int)MT, (int)wm, (int)S, (int)BR, (int)KT, (int)PF, (int)MINB, (int)epi, d, blob.data_ptr<uint8_t>(),
           od, osc, oz, col_scale.numel() ? (const __half*)col_scale.data_ptr() : nullptr,
           (const __nv_bfloat16*)X.data_ptr(), nullptr, nullptr, Y.data_ptr(), 1);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return Y;
}

double bench_gemm(torch::Tensor blob, torch::Tensor X, torch::Tensor col_scale, int64_t K,
                  int64_t N, int64_t layout, int64_t od, int64_t osc, int64_t oz, int64_t M,
                  int64_t MT, int64_t epi, int64_t cs_pairs, int64_t wm, int64_t S,
                  int64_t BR, int64_t KT, int64_t PF, int64_t MINB, int64_t warmup,
                  int64_t iters) {
  const int outw = (epi == 2) ? (int)K / 2 : (int)K;
  auto opt = torch::dtype(epi == 0 ? torch::kFloat32 : torch::kBFloat16).device(blob.device());
  auto Y = torch::zeros({M, outw}, opt);
  GDesc d{(int)K, (int)N, (int)layout, (int)cs_pairs, (int)N, outw, (int)M};
  const __half* cs = col_scale.numel() ? (const __half*)col_scale.data_ptr() : nullptr;
  dispatch((int)MT, (int)wm, (int)S, (int)BR, (int)KT, (int)PF, (int)MINB, (int)epi, d, blob.data_ptr<uint8_t>(),
           od, osc, oz, cs, (const __nv_bfloat16*)X.data_ptr(), nullptr, nullptr,
           Y.data_ptr(), (int)warmup);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  dispatch((int)MT, (int)wm, (int)S, (int)BR, (int)KT, (int)PF, (int)MINB, (int)epi, d, blob.data_ptr<uint8_t>(),
           od, osc, oz, cs, (const __nv_bfloat16*)X.data_ptr(), nullptr, nullptr,
           Y.data_ptr(), (int)iters);
  cudaEventRecord(e1);
  C10_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (double)ms / (double)iters;
}

int64_t smem_bytes(int64_t MT, int64_t S, int64_t BR, int64_t KT, int64_t d8) {
#define CBK_S(mt, w, st, br, kt, pfd, mnb)                                                \
  if (MT == mt && S == st && BR == br && KT == kt)                                        \
    return cbk::gemm_smem_bytes<mt, st, br, kt>((int)d8);
  CBK_CFGS(CBK_S)
#undef CBK_S
  return -1;
}


// Per-config occupancy report: registers, spill bytes, smem, blocks/SM (EPI_BF16 build).
std::vector<int64_t> cfg_info(int64_t MT, int64_t wm, int64_t S, int64_t BR, int64_t KT,
                              int64_t PF, int64_t MINB, int64_t d8) {
  std::vector<int64_t> r;
#define CBK_I(mt, w, st, br, kt, pfd, mnb)                                                \
  if (MT == mt && wm == w && S == st && BR == br && KT == kt && PF == pfd &&              \
      MINB == mnb) {                                                                      \
    const int sh = cbk::gemm_smem_bytes<mt, st, br, kt>((int)d8);                         \
    auto f = gemm_kernel<mt, 1, w, st, br, kt, pfd, mnb>;                                 \
    if (sh > 48 * 1024)                                                                   \
      C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                          sh));                                           \
    cudaFuncAttributes at;                                                                \
    C10_CUDA_CHECK(cudaFuncGetAttributes(&at, f));                                        \
    int nb = 0;                                                                           \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 256, sh));       \
    r = {at.numRegs, (int64_t)at.localSizeBytes, sh, nb};                                 \
    return r;                                                                             \
  }
  CBK_CFGS(CBK_I)
#undef CBK_I
  return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm);
  m.def("bench_gemm", &bench_gemm);
  m.def("mma_peak", &mma_peak);
  m.def("smem_bytes", &smem_bytes);
  m.def("cfg_info", &cfg_info);
}
