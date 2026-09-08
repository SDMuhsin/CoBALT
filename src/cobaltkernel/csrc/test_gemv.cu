// test_gemv.cu -- unit-test + bandwidth benchmark harness for cbk::gemv_rows / cbk::dequant_row.
// Built with torch.utils.cpp_extension.load (see ENV.md); driver = src/cobaltkernel/test_gemv.py
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "cobalt_format.h"
#include "cobalt_gemv.cuh"

namespace {

struct Desc {
  int K, N, layout;
  long long stride;                 // bytes between blob copies
  long long o_data, o_ro, o_sc, o_z;
};

__device__ __forceinline__ cbk::Mat build(const Desc& d, const uint8_t* base, int copy) {
  const uint8_t* b = base + (size_t)copy * (size_t)d.stride;
  cbk::Mat m;
  m.K = d.K; m.N = d.N; m.G = d.N / cbk::GROUP; m.layout = d.layout;
  m.data    = b + d.o_data;
  m.row_off = cbk::layout_is_sparse(d.layout) ? (const uint32_t*)(b + d.o_ro) : nullptr;
  m.scale   = (const __half*)(b + d.o_sc);
  m.zero    = b + d.o_z;
  return m;
}

template<int M, bool SMEM>
__global__ __launch_bounds__(256) void gemv_kernel(Desc d, const uint8_t* __restrict__ base,
                                                   const __nv_bfloat16* __restrict__ xg,
                                                   float* __restrict__ out, int n_items) {
  extern __shared__ uint8_t smem[];
  const int N = d.N;
  const __nv_bfloat16* x = xg;
  size_t soff = 0;
  if (SMEM) {
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    for (int i = threadIdx.x; i < M * N; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();
    x = xs;
    soff = (size_t)M * N * 2;
  }
  uint8_t* scratch = smem + soff + (size_t)(threadIdx.x >> 5) * cbk::scratch_bytes(0);
  const int warps = blockDim.x >> 5;
  const int stride = gridDim.x * warps;
  for (int it = blockIdx.x * warps + (threadIdx.x >> 5); it < n_items; it += stride) {
    const int row = it % d.K;
    cbk::Mat m = build(d, base, it / d.K);
    float acc[M];
    cbk::gemv_rows<M>(m, row, x, acc, scratch);
    if ((threadIdx.x & 31) == 0) {
      #pragma unroll
      for (int i = 0; i < M; ++i) out[(size_t)row * M + i] = acc[i];
    }
  }
}

__global__ void dequant_kernel(Desc d, const uint8_t* base, int row0, int nrows,
                               __nv_bfloat16* out) {
  const int warps = blockDim.x >> 5;
  const int w = blockIdx.x * warps + (threadIdx.x >> 5);
  if (w >= nrows) return;
  cbk::Mat m = build(d, base, 0);
  cbk::dequant_row(m, row0 + w, out + (size_t)w * d.N);
}

template<int M>
void launch(const Desc& d, const uint8_t* base, const __nv_bfloat16* x, float* out,
            int n_items, int blocks, bool smem) {
  const int thr = 256, warps = thr / 32;
  size_t sh = (size_t)warps * cbk::scratch_bytes(0);
  if (smem) {
    sh += (size_t)M * d.N * 2;
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<M, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    gemv_kernel<M, true><<<blocks, thr, sh>>>(d, base, x, out, n_items);
  } else {
    gemv_kernel<M, false><<<blocks, thr, sh>>>(d, base, x, out, n_items);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void dispatch(int M, const Desc& d, const uint8_t* base, const __nv_bfloat16* x, float* out,
              int n_items, int blocks, bool smem) {
  switch (M) {
    case 1: launch<1>(d, base, x, out, n_items, blocks, smem); break;
    case 2: launch<2>(d, base, x, out, n_items, blocks, smem); break;
    case 4: launch<4>(d, base, x, out, n_items, blocks, smem); break;
    case 8: launch<8>(d, base, x, out, n_items, blocks, smem); break;
    default: TORCH_CHECK(false, "M must be 1,2,4 or 8");
  }
}

// ---------------------------------------------------------------- MULTI-ROW harness
// Mirrors the decode megakernel's gemv_phase_r: a warp owns R CONSECUTIVE rows, x lives
// in GLOBAL memory in the INTERLEAVED layout x[col*M + m], PF granules are prefetched.
// LAY is a compile-time layout so the DENSE4 / BLK16_32 routines can be compared
// head-to-head in one process.
// CBK_TEST_MINB mirrors the megakernel's __launch_bounds__(256, CBK_MINB) register cap
// (MINB=2 => 512 threads/SM => 128 registers/thread).  Default 1 = no second bound.
#ifndef CBK_TEST_MINB
#define CBK_TEST_MINB 1
#endif
template<int M, int R, int NX, int PF, int LAY, bool FT>
__global__ __launch_bounds__(256, CBK_TEST_MINB) void gemv_multi_kernel(
    Desc d, const uint8_t* __restrict__ base, const __nv_bfloat16* __restrict__ xa,
    const __nv_bfloat16* __restrict__ xb, float* __restrict__ out,
    int ngrp_total, int ngrp_per) {
  cbk::detail::prmt_smem_stage();          // no-op unless CBK_BLK1632_LUT==2
  __syncthreads();
  const int lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int gw = blockIdx.x * warps + (threadIdx.x >> 5), GW = gridDim.x * warps;
  const size_t rs = cbk::dense_row_stride(LAY, d.N);
  const int G = d.N / cbk::GROUP;
  for (int g = gw; g < ngrp_total; g += GW) {
    const int copy = g / ngrp_per, r0 = (g % ngrp_per) * R;
    cbk::Mat m = build(d, base, copy);
    float acc[R][M];
    if constexpr (LAY == cbk::LAYOUT_DENSE4) {
      cbk::detail::gemv_dense4_multi<M, R, NX, PF>(
          m.data + (size_t)r0 * rs, rs, m.scale + (size_t)r0 * G,
          m.zero + (size_t)r0 * G, G, d.N, xa, xb, acc);
    } else {
      cbk::detail::gemv_blk1632_multi<M, R, NX, PF, (LAY == cbk::LAYOUT_BLK1632_6 ? 6 : 4), FT>(
          m.data + (size_t)r0 * rs, rs, m.scale + (size_t)r0 * G,
          m.zero + (size_t)r0 * G, G, d.N, xa, xb, acc);
    }
    if (lane == 0) {
      #pragma unroll
      for (int i = 0; i < R; ++i)
        #pragma unroll
        for (int mm = 0; mm < M; ++mm) out[(size_t)(r0 + i) * M + mm] = acc[i][mm];
    }
  }
}

template<int M, int R, int NX, int LAY>
void launch_multi_l(const Desc& d, const uint8_t* base, const __nv_bfloat16* xa,
                    const __nv_bfloat16* xb, float* out, int ngt, int ngp, int blocks,
                    bool flattail) {
  constexpr int PF = (R == 4) ? 2 : (R == 2) ? 4 : 8;
  if (flattail)
    gemv_multi_kernel<M, R, NX, PF, LAY, true><<<blocks, 256>>>(d, base, xa, xb, out, ngt, ngp);
  else
    gemv_multi_kernel<M, R, NX, PF, LAY, false><<<blocks, 256>>>(d, base, xa, xb, out, ngt, ngp);
  C10_CUDA_CHECK(cudaGetLastError());
}

template<int M, int R, int NX>
void launch_multi_r(const Desc& d, const uint8_t* base, const __nv_bfloat16* xa,
                    const __nv_bfloat16* xb, float* out, int ngt, int ngp, int blocks,
                    bool ft) {
  switch (d.layout) {
    case cbk::LAYOUT_DENSE4:
      launch_multi_l<M, R, NX, cbk::LAYOUT_DENSE4>(d, base, xa, xb, out, ngt, ngp, blocks, ft); break;
    case cbk::LAYOUT_BLK1632_4:
      launch_multi_l<M, R, NX, cbk::LAYOUT_BLK1632_4>(d, base, xa, xb, out, ngt, ngp, blocks, ft); break;
    case cbk::LAYOUT_BLK1632_6:
      launch_multi_l<M, R, NX, cbk::LAYOUT_BLK1632_6>(d, base, xa, xb, out, ngt, ngp, blocks, ft); break;
    default: TORCH_CHECK(false, "multi harness: layout must be DENSE4 or BLK1632_*");
  }
}

void dispatch_multi(int M, int R, int NX, const Desc& d, const uint8_t* base,
                    const __nv_bfloat16* xa, const __nv_bfloat16* xb, float* out,
                    int ngt, int ngp, int blocks, bool ft) {
  TORCH_CHECK(M == 1, "multi harness is built for M=1");
  if (NX == 1) {
    if (R == 4)      launch_multi_r<1, 4, 1>(d, base, xa, xb, out, ngt, ngp, blocks, ft);
    else if (R == 2) launch_multi_r<1, 2, 1>(d, base, xa, xb, out, ngt, ngp, blocks, ft);
    else             launch_multi_r<1, 1, 1>(d, base, xa, xb, out, ngt, ngp, blocks, ft);
  } else {
    if (R == 4)      launch_multi_r<1, 4, 2>(d, base, xa, xb, out, ngt, ngp, blocks, ft);
    else if (R == 2) launch_multi_r<1, 2, 2>(d, base, xa, xb, out, ngt, ngp, blocks, ft);
    else             TORCH_CHECK(false, "NX=2 needs R even");
  }
}

Desc mk(int64_t K, int64_t N, int64_t layout, int64_t stride,
        int64_t od, int64_t oro, int64_t osc, int64_t oz) {
  Desc d; d.K = (int)K; d.N = (int)N; d.layout = (int)layout; d.stride = stride;
  d.o_data = od; d.o_ro = oro; d.o_sc = osc; d.o_z = oz; return d;
}

} // namespace

torch::Tensor gemv(torch::Tensor blob, torch::Tensor x, int64_t K, int64_t N, int64_t layout,
                   int64_t stride, int64_t od, int64_t oro, int64_t osc, int64_t oz,
                   int64_t M, int64_t n_copies, int64_t blocks, bool smem) {
  TORCH_CHECK(blob.is_cuda() && blob.scalar_type() == torch::kUInt8);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16);
  auto out = torch::zeros({K, M}, torch::dtype(torch::kFloat32).device(blob.device()));
  Desc d = mk(K, N, layout, stride, od, oro, osc, oz);
  dispatch((int)M, d, blob.data_ptr<uint8_t>(),
           reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
           out.data_ptr<float>(), (int)(n_copies * K), (int)blocks, smem);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

double bench(torch::Tensor blob, torch::Tensor x, int64_t K, int64_t N, int64_t layout,
             int64_t stride, int64_t od, int64_t oro, int64_t osc, int64_t oz,
             int64_t M, int64_t n_copies, int64_t blocks, bool smem,
             int64_t warmup, int64_t iters) {
  auto out = torch::zeros({K, M}, torch::dtype(torch::kFloat32).device(blob.device()));
  Desc d = mk(K, N, layout, stride, od, oro, osc, oz);
  const uint8_t* base = blob.data_ptr<uint8_t>();
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  float* op = out.data_ptr<float>();
  int items = (int)(n_copies * K);
  for (int i = 0; i < warmup; ++i) dispatch((int)M, d, base, xp, op, items, (int)blocks, smem);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) dispatch((int)M, d, base, xp, op, items, (int)blocks, smem);
  cudaEventRecord(e1);
  C10_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (double)ms / (double)iters;
}

torch::Tensor dequant_rows(torch::Tensor blob, int64_t K, int64_t N, int64_t layout,
                           int64_t od, int64_t oro, int64_t osc, int64_t oz,
                           int64_t row0, int64_t nrows) {
  auto out = torch::zeros({nrows, N}, torch::dtype(torch::kBFloat16).device(blob.device()));
  Desc d = mk(K, N, layout, 0, od, oro, osc, oz);
  int thr = 256, warps = 8;
  int blocks = (int)((nrows + warps - 1) / warps);
  dequant_kernel<<<blocks, thr>>>(d, blob.data_ptr<uint8_t>(), (int)row0, (int)nrows,
                                  reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

static void multi_args(int64_t K, int64_t R, int64_t n_copies, int* ngt, int* ngp) {
  TORCH_CHECK(K % R == 0, "multi harness needs K % R == 0");
  *ngp = (int)(K / R);
  *ngt = (int)(n_copies * (K / R));
}

torch::Tensor gemv_multi(torch::Tensor blob, torch::Tensor x, torch::Tensor xb2,
                         int64_t K, int64_t N, int64_t layout, int64_t stride,
                         int64_t od, int64_t oro, int64_t osc, int64_t oz,
                         int64_t M, int64_t R, int64_t NX, int64_t n_copies, int64_t blocks,
                         bool flattail) {
  auto out = torch::zeros({K, M}, torch::dtype(torch::kFloat32).device(blob.device()));
  Desc d = mk(K, N, layout, stride, od, oro, osc, oz);
  int ngt, ngp; multi_args(K, R, n_copies, &ngt, &ngp);
  dispatch_multi((int)M, (int)R, (int)NX, d, blob.data_ptr<uint8_t>(),
                 reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                 reinterpret_cast<const __nv_bfloat16*>(xb2.data_ptr()),
                 out.data_ptr<float>(), ngt, ngp, (int)blocks, flattail);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

double bench_multi(torch::Tensor blob, torch::Tensor x, torch::Tensor xb2,
                   int64_t K, int64_t N, int64_t layout, int64_t stride,
                   int64_t od, int64_t oro, int64_t osc, int64_t oz,
                   int64_t M, int64_t R, int64_t NX, int64_t n_copies, int64_t blocks,
                   bool flattail, int64_t warmup, int64_t iters) {
  auto out = torch::zeros({K, M}, torch::dtype(torch::kFloat32).device(blob.device()));
  Desc d = mk(K, N, layout, stride, od, oro, osc, oz);
  int ngt, ngp; multi_args(K, R, n_copies, &ngt, &ngp);
  const uint8_t* base = blob.data_ptr<uint8_t>();
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const __nv_bfloat16* xq = reinterpret_cast<const __nv_bfloat16*>(xb2.data_ptr());
  float* op = out.data_ptr<float>();
  for (int i = 0; i < warmup; ++i)
    dispatch_multi((int)M, (int)R, (int)NX, d, base, xp, xq, op, ngt, ngp, (int)blocks, flattail);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i)
    dispatch_multi((int)M, (int)R, (int)NX, d, base, xp, xq, op, ngt, ngp, (int)blocks, flattail);
  cudaEventRecord(e1);
  C10_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (double)ms / (double)iters;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv);
  m.def("bench", &bench);
  m.def("gemv_multi", &gemv_multi);
  m.def("bench_multi", &bench_multi);
  m.def("dequant_rows", &dequant_rows);
}
