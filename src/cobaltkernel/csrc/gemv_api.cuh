// gemv_api.cuh -- the ONE weight-access layer the megakernel uses.
//
// It dispatches on MatDesc::layout:
//   CBK_LAYOUT_PLAIN_BF16 (-1)  -> local bf16 GEMV (numerics reference / A-B arm)
//   cbk::Layout 0..4            -> cbk::gemv_rows<M>() over a
//                                  CBK1 packed cbk::Mat (see FORMAT.md sec.5)
//
// CONTRACT (matches A's, FORMAT.md sec.5):
//  * warp-collective, all 32 lanes, identical args;
//  * `x` holds M activation vectors of the *view's* length, ALREADY multiplied by
//    the matrix's column scale, interleaved as x[j*M + m] (-DCBK_X_INTERLEAVED);
//  * `out[M]` is ASSIGNED (not accumulated) and valid in every lane;
//  * `scratch` is per-warp scratch, cbk::scratch_bytes(N) bytes (currently unused).
//
// Column slicing: A's Mat carries no column offset, so a K-chunk is expressed as a
// ONE-ROW matrix whose data/scale/zero pointers are pre-offset to (row, c0) and whose
// N is the chunk length.  Exact for the fixed-stride layouts (DENSE4/DENSE8/BF16);
// the SPARSE layouts cannot be sliced (variable row offsets + a per-1024-window
// prefix scan), so for them the host must stage the whole row (len == N).
#pragma once
#include "cobalt_format.h"
#include "cobalt_gemv.cuh"
#include "megakernel.cuh"

namespace cbk {

__host__ __device__ inline bool layout_sliceable(long long L) {
  return L == CBK_LAYOUT_PLAIN_BF16 || L == LAYOUT_DENSE4 || L == LAYOUT_DENSE8 ||
         L == LAYOUT_BF16;
}

// ---------------------------------------------------------------- plain bf16
template <int M>
__device__ __forceinline__ void
gemv_plain_bf16(const __nv_bfloat16* __restrict__ r, int len,
                const __nv_bfloat16* __restrict__ x, float* out) {
  const int lane = threadIdx.x & 31;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  // U independent 16-byte weight loads in flight: the megakernel runs at only
  // 2-6 blocks/SM, so MLP has to come from ILP inside the warp.
  constexpr int U = 4;
  for (int j0 = lane * 8; j0 < len; j0 += 32 * 8 * U) {
    uint4 wv[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int j = j0 + u * 256;
      wv[u] = (j < len) ? *reinterpret_cast<const uint4*>(r + j) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int j = j0 + u * 256;
      if (j >= len) continue;
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[u]);
      // x is interleaved, so the 8*M activations for columns j..j+7 are contiguous
      uint4 xv[M];
      const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)j * M);
#pragma unroll
      for (int q = 0; q < M; ++q) xv[q] = xp[q];
      const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(xv);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const float wf = __bfloat162float(wb[t]);
#pragma unroll
        for (int m = 0; m < M; ++m)
          acc[m] = fmaf(wf, __bfloat162float(xb[t * M + m]), acc[m]);
      }
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], d);
#pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}

// ---------------------------------------------------------------- dispatch
template <int M>
__device__ __forceinline__ void
gemv_view(const MatDesc& d, int row, int c0, int len,
          const __nv_bfloat16* __restrict__ x, float* out, uint8_t* scratch) {
  if (d.layout == CBK_LAYOUT_PLAIN_BF16) {
    gemv_plain_bf16<M>(reinterpret_cast<const __nv_bfloat16*>(d.data) +
                           (size_t)row * d.N + c0,
                       len, x, out);
    return;
  }
  Mat m;
  m.layout = (int)d.layout;
  m.data = d.data;
  m.row_off = d.row_off;
  m.scale = d.scale;
  m.zero = d.zero;
  if (len == (int)d.N) {
    m.K = (int)d.K; m.N = (int)d.N; m.G = (int)d.G;
    gemv_rows<M>(m, row, x, out, scratch);
    return;
  }
  size_t coff;
  switch ((int)d.layout) {
    case LAYOUT_DENSE4: coff = (size_t)(c0 >> 1); break;
    case LAYOUT_DENSE8: coff = (size_t)c0; break;
    default:            coff = (size_t)c0 * 2; break;   // LAYOUT_BF16
  }
  m.data = d.data + (size_t)row * dense_row_stride((int)d.layout, (int)d.N) + coff;
  m.row_off = nullptr;
  m.scale = d.scale + (size_t)row * d.G + (c0 >> 7);
  m.zero = d.zero + (size_t)row * d.G + (c0 >> 7);
  m.K = 1; m.N = len; m.G = len >> 7;
  gemv_rows<M>(m, 0, x, out, scratch);
}

// ---------------------------------------------------------------- single element
// (embedding lookup; dense layouts only)
__device__ __forceinline__ float mat_elem(const MatDesc& d, int row, int col) {
  if (d.layout == CBK_LAYOUT_PLAIN_BF16 || d.layout == LAYOUT_BF16)
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(d.data)[(size_t)row * d.N + col]);
  const int g = col >> 7;
  const float sf = __half2float(d.scale[(size_t)row * d.G + g]);
  const float zf = (float)d.zero[(size_t)row * d.G + g];
  float q = 0.f;
  if (d.layout == LAYOUT_DENSE4) {
    const uint8_t b = d.data[(size_t)row * ((size_t)d.N >> 1) + (col >> 1)];
    q = (float)((col & 1) ? (b >> 4) : (b & 0xF));
  } else if (d.layout == LAYOUT_DENSE8) {
    q = (float)d.data[(size_t)row * (size_t)d.N + col];
  }
  const float cs = d.col_scale ? __half2float(d.col_scale[col]) : 1.f;
  return (q - zf) * sf * cs;
}

// ---------------------------------------------------------------- chunking
// Chunk length depends ONLY on N (never on M), so a sequence decoded at M=4 is
// bit-identical to the same sequence decoded at M=1.
__host__ __device__ inline int chunk_len(int N) {
  // A remainder tail (not an equal split): the packed decoders consume 1024 columns
  // per warp iteration, so 2048+512 costs exactly the same 3 iterations as one
  // unsliced 2560-column call, whereas an equal 1280+1280 split would cost 4.
  return N < CBK_CHUNK_MAX ? N : CBK_CHUNK_MAX;
}

}  // namespace cbk
