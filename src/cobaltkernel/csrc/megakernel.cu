// megakernel.cu -- persistent single-launch cooperative CUDA kernel for the
// Gemma3 text decode step (batch M in {1,2,4,8}, one token per sequence per launch).
//
// Spec: docs/KERNELS.md   Oracle: src/cobaltkernel/ref_gemma3.py (executable)
//
// v2.  Two structural changes over v1:
//
//  1. GEMV phases no longer stage the activation into shared memory a 1024/2048-column
//     chunk at a time.  A tiny "prep" phase writes the normalised, column-scaled x'
//     ONCE per matrix into a global buffer (10.5 KB at hidden=5376); the GEMV warps
//     read x' straight from global (L1-resident -- every warp hits) with uint4 loads.
//     That removes the per-chunk __syncthreads pairs, removes the repeated re-read of
//     h by every block, frees shared memory, and lets a warp keep PF*R independent
//     16-byte weight loads in flight (cbk::detail::gemv_dense4_multi).  o_proj's and
//     down_proj's column scales are folded into the WRITE of their input (attention
//     output / GeGLU output), so those two phases need no prep at all.
//     q/k/v are one FUSED row space (FORMAT.md sec.2.4b) -> one GEMV, one barrier.
//
//  2. Attention is one memory round trip.  A block owns (sequence, kv-head, key-split)
//     and serves BOTH q heads of the GQA pair, so each KV byte is read once per layer.
//     Inside a warp a chunk of 32 keys is processed by giving every LANE one key: the
//     lane streams that key's whole row as uint4 and computes the q.k dot in registers
//     (q broadcast from smem) -- no per-key shuffle reduction.  The p.V product then
//     switches to lane-owns-dims (VPL contiguous dims per lane) and loops the 32 keys
//     with p broadcast by __shfl.  QK-norm + RoPE + the KV append moved into their own
//     (very cheap) phase so the walk has no per-key special case.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>

#include "megakernel.cuh"
#include "gemv_api.cuh"

namespace cg = cooperative_groups;
using cbk::Args;
using cbk::LayerW;
using cbk::MatDesc;

// CBK_ATTN_DIAG -- DIAGNOSTIC ONLY, NUMERICALLY WRONG.  1 = skip the q.k
// dot, 2 = skip the p.V accumulation, 3 = skip both.  Prices the two halves of the walk.
#ifndef CBK_ATTN_DIAG
#define CBK_ATTN_DIAG 0
#endif
// CBK_KVONCE: with CBK_FUSEPREP the k/v cache APPEND is executed
// by EVERY key-split block of a (sequence, kv-head) -- `split` blocks writing byte-identical
// data, plus `split` copies of the k head's norm+RoPE.  Only the block whose key range
// actually contains `pos` needs it, and that block writes it before its own __syncthreads,
// so the other splits (which only read positions < their k0 <= pos) can skip it entirely.
// BIT-IDENTICAL by construction: the same bytes, written once instead of `split` times.
#ifndef CBK_KVONCE
#define CBK_KVONCE 1
#endif
// Keys per p.V load batch in the attention walk (COBALT_PVB).  See phase_attn.
#ifndef CBK_PVB
#define CBK_PVB 4
#endif

// pick_R() tolerance numerator over 10: take the LARGEST row-group R whose row-time
// cost is within CBK_RTOL/10 of the best.  11 = the shipping v2 rule (+10%); 10 makes
// the choice strictly minimal-row-time (COBALT_RTOL).
#ifndef CBK_RTOL
#define CBK_RTOL 11
#endif

// CoBALT-16:32 arm (COBALT_BLK1632, FORMAT.md sec.13).  0 = OFF, the shipping DENSE4
// control: the BLK16_32 multi-row routines are not instantiated at all, so the DENSE4
// binary is unchanged.  4 / 6 additionally instantiate gemv_blk1632_multi at that code
// width, which is what the decode phases then use for a BLK1632_{4,6} artifact.  One
// width per build keeps the compile time at ~2x, not ~3x, and keeps the control build
// provably untouched.
#ifndef CBK_BLK1632_ARM
#define CBK_BLK1632_ARM 0
#endif
// The BLK16_32 warp-tail fix (cobalt_gemv.cuh).  1 = on (default).  COBALT_BLK1632_FT=0
// builds the un-flattened control.
#ifndef CBK_BLK1632_FT
#define CBK_BLK1632_FT 1
#endif
// Override pick_R's gate/up pair-group RP (COBALT_GATEUP_RP).  0 = pick_R decides (the
// shipping DENSE4 behaviour, unchanged).  MEASURED: with a second activation stream
// (NX==2) the BLK16_32 b=4 decoder is issue-bound and PREFERS RP=1 (gateup 83.7 % of
// ceiling) over pick_R's RP=2 (77.6 %); DENSE4 and b=6 prefer RP=2.
#ifndef CBK_GATEUP_RP
#define CBK_GATEUP_RP 0
#endif
// Multiplier on the BLK16_32 granule-prefetch depth PF (COBALT_BLK1632_PFX, default 1).
// The decode-step GEMV phases run only 1-2 loop iterations per warp, so they are
// LATENCY- not throughput-bound; a deeper prefetch puts more independent loads in flight
// per iteration at the cost of live registers under the MINB launch bound.
#ifndef CBK_BLK1632_PFX
#define CBK_BLK1632_PFX 1
#endif
// Gate/up as TWO NX=1 passes instead of one NX=2 pass, for BLK16_32 only
// (COBALT_GATEUP_SPLIT).  MEASURED: a second activation stream costs the BLK16_32 decoder
// 18-25 % of wall clock on every shape while costing DENSE4 ~0 %, so the fused NX=2 form
// throws away the layout's edge on the single biggest matrix in the model.
#ifndef CBK_GATEUP_SPLIT
#define CBK_GATEUP_SPLIT 0
#endif

// Fold the QK-norm + RoPE + KV-append phase INTO the attention phase (COBALT_FUSEPREP).
// It deletes one grid.sync and one whole (tiny but barrier-priced) phase per layer.
// Every block that owns a key-split of (m, kv-head) recomputes that head's k/v and
// writes the SAME bytes to the cache, and computes its KG query vectors straight into
// smem instead of writing them back into a.qkv (an in-place write would race between
// the splits; smem cannot).  Arithmetic is unchanged, so results are bit-identical.
#ifndef CBK_FUSEPREP
#define CBK_FUSEPREP 1
#endif

#define F(x) __bfloat162float(x)
#define BF(x) __float2bfloat16(x)

__device__ __forceinline__ float rb(float v) { return F(BF(v)); }
__device__ __forceinline__ int ceildiv(int a, int b) { return (a + b - 1) / b; }
__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// ------------------------------------------------------- sub-phase profiling
// Built only with -DCBK_PROF (COBALT_PROF=1).  Block 0 / thread 0 accumulates
// clock64() deltas for the ATTENTION sub-phases into the tail of a.timings, so the
// flat per-layer attention cost can be split into prep / sq / walk / combine and
// compared with the grid.sync-delimited phase time (the difference is barrier wait).
#ifdef CBK_PROF
#define PROF_BASE(a) ((a).n_layers * 11 + 8)
#define PROF_T(a) long long _pt = ((a).timings && blockIdx.x == 0 && threadIdx.x == 0) ? clock64() : 0
#define PROF_ADD(a, slot)                                                    \
  do {                                                                       \
    if ((a).timings && blockIdx.x == 0 && threadIdx.x == 0) {                \
      long long _now = clock64();                                            \
      (a).timings[PROF_BASE(a) + (slot)] += _now - _pt;                      \
      _pt = _now;                                                            \
    }                                                                        \
  } while (0)
#else
#define PROF_T(a) ((void)0)
#define PROF_ADD(a, slot) ((void)0)
#endif

__device__ __forceinline__ float block_reduce_sum(float v, float* s_red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  if (lane == 0) s_red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (lane < CBK_WARPS) ? s_red[lane] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0) s_red[CBK_WARPS] = t;
  }
  __syncthreads();
  const float r = s_red[CBK_WARPS];
  __syncthreads();
  return r;
}

// rr[m] = rsqrt(mean(src[m]^2) + eps), computed redundantly by every block.  Used for
// the two vectors (obuf, dbuf) that are produced by a GEMV and therefore have no
// per-block partial sums; h and h2 use rms_from_partials instead.
template <int M>
__device__ __forceinline__ void block_rms(const __nv_bfloat16* src, int N, float eps,
                                          float* s_red, float* rr) {
  for (int m = 0; m < M; ++m) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += CBK_THREADS) {
      const float v = F(src[(size_t)m * N + i]);
      ss += v * v;
    }
    ss = block_reduce_sum(ss, s_red) / (float)N;
    rr[m] = rsqrtf(ss + eps);
  }
}

// Same value, from the per-block partial sums the producing phase already wrote.
template <int M>
__device__ __forceinline__ void rms_from_partials(const float* part, int N, float eps,
                                                  float* s_red, float* rr) {
  for (int m = 0; m < M; ++m) {
    float s = 0.f;
    for (int b = threadIdx.x; b < gridDim.x; b += CBK_THREADS) s += part[(size_t)b * M + m];
    s = block_reduce_sum(s, s_red) / (float)N;
    rr[m] = rsqrtf(s + eps);
  }
}

// ------------------------------------------------------------- row-group sizing
// A GEMV phase costs ceil(ceil(K/R)/GW) * R row-times, where GW = warps in the grid.
// R also sets how many independent weight loads a warp has in flight, so on a tie the
// LARGER R wins.
__device__ __forceinline__ int pick_R(int K, int GW, int rmax) {
  int bc = 1 << 30;
#pragma unroll
  for (int r = 1; r <= 4; r <<= 1) {
    if (r > rmax) break;
    bc = min(bc, ceildiv(ceildiv(K, r), GW) * r);
  }
  // R also divides the activation traffic: at 4 bits x' is 4x the weight bytes of one
  // row, so R rows sharing one x' stream cuts the L1/L2 re-read by R.  Take the largest
  // R whose tail is within 10% of the best (measured: lm_head 1146 -> 950 us).
  int best = 1;
#pragma unroll
  for (int r = 1; r <= 4; r <<= 1) {
    if (r > rmax) break;
    if (ceildiv(ceildiv(K, r), GW) * r * 10 <= bc * CBK_RTOL) best = r;
  }
  return best;
}

// x'[c][j][m] = bf16( bf16(src[m][j] * rr[m] * (1+nw[j])) * cs[c][j] ), NC variants.
template <int M>
__device__ __forceinline__ void write_xprime(const __nv_bfloat16* src, int N,
                                             const __nv_bfloat16* nw, const float* rr,
                                             int NC, const __half* cs,
                                             __nv_bfloat16* dst) {
  const int i0 = blockIdx.x * CBK_THREADS + threadIdx.x;
  const int istep = gridDim.x * CBK_THREADS;
  for (int j = i0; j < N; j += istep) {
    const float g = nw ? (1.f + F(nw[j])) : 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = F(src[(size_t)m * N + j]);
      if (nw) v = rb(v * rr[m] * g);
      for (int c = 0; c < NC; ++c)
        dst[(size_t)c * N * M + (size_t)j * M + m] =
            cs ? BF(v * __half2float(cs[(size_t)c * N + j])) : BF(v);
    }
  }
}

// ------------------------------------------------------------------ GEMV driver
// Rows are handed to WARPS in groups of R CONSECUTIVE rows, so the group shares one
// activation vector and one uint4 x-load stream.  Groups are warp-cyclic over the whole
// grid; the phase uses no shared memory and no __syncthreads.
template <int M, int R, int NX, int PF, class XF, class OutFn>
__device__ __forceinline__ void gemv_phase_r(const MatDesc& d, int K, XF xf,
                                             OutFn out_fn) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int ngrp = ceildiv(K, R);
  for (int g = gw; g < ngrp; g += GW) {
    const int r0 = g * R;
    const __nv_bfloat16 *xa = nullptr, *xb = nullptr;
    xf(r0, &xa, &xb);
    float acc[R][M];
    if (d.layout == cbk::LAYOUT_DENSE4 && r0 + R <= K) {
      cbk::detail::gemv_dense4_multi<M, R, NX, PF>(
          d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G,
          d.zero + (size_t)r0 * G, G, N, xa, xb, acc);
#if CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6
    } else if (cbk::layout_is_blk1632((int)d.layout) && r0 + R <= K) {
      cbk::detail::gemv_blk1632_multi<M, R, NX, PF * CBK_BLK1632_PFX, CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT>(
          d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G,
          d.zero + (size_t)r0 * G, G, N, xa, xb, acc);
#endif
    } else {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const int row = r0 + i;
        if (row < K)
          cbk::gemv_view<M>(d, row, 0, N, (NX == 2 && (i & 1)) ? xb : xa, acc[i], nullptr);
        else
#pragma unroll
          for (int m = 0; m < M; ++m) acc[i][m] = 0.f;
      }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int row = r0 + i;
      if (row < K)
#pragma unroll
        for (int m = 0; m < M; ++m)
          if (lane == m) out_fn(row, m, acc[i][m]);
    }
  }
}

template <int M, class XF, class OutFn>
__device__ __forceinline__ void gemv_phase(const MatDesc& d, int K, XF xf, OutFn f) {
  const int GW = gridDim.x * CBK_WARPS;
  const int R = pick_R(K, GW, 4);
  if (R >= 4)      gemv_phase_r<M, 4, 1, 2>(d, K, xf, f);
  else if (R == 2) gemv_phase_r<M, 2, 1, 4>(d, K, xf, f);
  else             gemv_phase_r<M, 1, 1, 8>(d, K, xf, f);
}

// Fused gate/up: R PAIRS (2R consecutive rows, gate = even, up = odd) per warp group,
// so the GeGLU closes in registers and the two branches share one weight-load stream.
template <int M, int RP, int PF, class OutFn>
__device__ __forceinline__ void gemv_pairs_r(const MatDesc& d, int npairs,
                                             const __nv_bfloat16* xg,
                                             const __nv_bfloat16* xu, OutFn out_fn) {
  constexpr int R = 2 * RP;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int ngrp = ceildiv(npairs, RP);
  for (int g = gw; g < ngrp; g += GW) {
    const int p0 = g * RP, r0 = 2 * p0;
    float acc[R][M];
    if (d.layout == cbk::LAYOUT_DENSE4 && r0 + R <= 2 * npairs) {
      cbk::detail::gemv_dense4_multi<M, R, 2, PF>(
          d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G,
          d.zero + (size_t)r0 * G, G, N, xg, xu, acc);
#if CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6
    } else if (cbk::layout_is_blk1632((int)d.layout) && r0 + R <= 2 * npairs) {
      cbk::detail::gemv_blk1632_multi<M, R, 2, PF * CBK_BLK1632_PFX, CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT>(
          d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G,
          d.zero + (size_t)r0 * G, G, N, xg, xu, acc);
#endif
    } else {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if (r0 + i < 2 * npairs)
          cbk::gemv_view<M>(d, r0 + i, 0, N, (i & 1) ? xu : xg, acc[i], nullptr);
        else
#pragma unroll
          for (int m = 0; m < M; ++m) acc[i][m] = 0.f;
      }
    }
#pragma unroll
    for (int i = 0; i < RP; ++i) {
      const int p = p0 + i;
      if (p < npairs)
#pragma unroll
        for (int m = 0; m < M; ++m)
          if (lane == m) out_fn(p, m, acc[2 * i][m], acc[2 * i + 1][m]);
    }
  }
}

#if CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6
// Fused gate/up for BLK16_32 as TWO NX=1 passes over the SAME blob.  A warp group owns RP
// gate rows (the even rows, walked at stride 2*rstride) and the RP matching up rows (odd).
// Activation traffic is IDENTICAL to the NX=2 form -- one xg read and one xu read per
// granule per 2*RP rows -- but only ONE activation staging array is live in the inner loop,
// which is what the measured NX=2 penalty actually is.  The GeGLU still closes in registers,
// so there is no extra phase and no extra buffer.
template <int M, int RP, int PF, class OutFn>
__device__ __forceinline__ void gemv_pairs_split_r(const MatDesc& d, int npairs,
                                                   const __nv_bfloat16* xg,
                                                   const __nv_bfloat16* xu, OutFn out_fn) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int ngrp = ceildiv(npairs, RP);
  for (int g = gw; g < ngrp; g += GW) {
    const int p0 = g * RP, r0 = 2 * p0;
    float ag[RP][M], au[RP][M];
    if (p0 + RP <= npairs) {
      // stride-2 row walk: pass 2*rs as the row stride and 2*G as the scale/zero row stride
      cbk::detail::gemv_blk1632_multi<M, RP, 1, PF, CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT>(
          d.data + (size_t)r0 * rs, 2 * rs, d.scale + (size_t)r0 * G,
          d.zero + (size_t)r0 * G, 2 * G, N, xg, xg, ag);
      cbk::detail::gemv_blk1632_multi<M, RP, 1, PF, CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT>(
          d.data + (size_t)(r0 + 1) * rs, 2 * rs, d.scale + (size_t)(r0 + 1) * G,
          d.zero + (size_t)(r0 + 1) * G, 2 * G, N, xu, xu, au);
    } else {
#pragma unroll
      for (int i = 0; i < RP; ++i) {
        if (p0 + i < npairs) {
          cbk::gemv_view<M>(d, r0 + 2 * i, 0, N, xg, ag[i], nullptr);
          cbk::gemv_view<M>(d, r0 + 2 * i + 1, 0, N, xu, au[i], nullptr);
        } else {
#pragma unroll
          for (int m = 0; m < M; ++m) { ag[i][m] = 0.f; au[i][m] = 0.f; }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < RP; ++i) {
      const int p = p0 + i;
      if (p < npairs)
#pragma unroll
        for (int m = 0; m < M; ++m)
          if (lane == m) out_fn(p, m, ag[i][m], au[i][m]);
    }
  }
}
#endif

// ------------------------------------------------------------------- residuals
// L == 0        : h = embed[token] * embed_scale
// 0 < L <= n_L  : h = h2 + post_feedforward_layernorm(dbuf)
// Each block also accumulates sum(h^2) over its slice into part[block][m]; the next
// phase turns those partials into rr without re-reading h.
template <int M>
__device__ __forceinline__ void phase_h(const Args& a, const LayerW* Wprev, int L,
                                        const float* rrd, float* s_red, float* part) {
  const int hid = a.hidden;
  const int i0 = blockIdx.x * CBK_THREADS + threadIdx.x;
  const int istep = gridDim.x * CBK_THREADS;
  float ss[M];
#pragma unroll
  for (int m = 0; m < M; ++m) ss[m] = 0.f;
  if (L == 0) {
    for (int m = 0; m < M; ++m) {
      const int tok = a.tok_v[m];
      for (int i = i0; i < hid; i += istep) {
        const __nv_bfloat16 b = BF(rb(cbk::mat_elem(a.embed, tok, i)) * a.embed_scale);
        a.h[(size_t)m * hid + i] = b;
        ss[m] += F(b) * F(b);
      }
    }
  } else {
    const __nv_bfloat16* w = Wprev->post_ff_ln;
    for (int m = 0; m < M; ++m)
      for (int i = i0; i < hid; i += istep) {
        const __nv_bfloat16 b =
            BF(F(a.h2[(size_t)m * hid + i]) +
               rb(F(a.dbuf[(size_t)m * hid + i]) * rrd[m] * (1.f + F(w[i]))));
        a.h[(size_t)m * hid + i] = b;
        ss[m] += F(b) * F(b);
      }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float t = block_reduce_sum(ss[m], s_red);
    if (threadIdx.x == 0) part[(size_t)blockIdx.x * M + m] = t;
  }
  if (a.dbg_h)
    for (int m = 0; m < M; ++m)
      for (int i = i0; i < hid; i += istep)
        a.dbg_h[((size_t)L * M + m) * hid + i] = a.h[(size_t)m * hid + i];
}

// h2 = h + post_attention_layernorm(obuf)  (+ the sum(h2^2) partials)
template <int M>
__device__ __forceinline__ void phase_h2(const Args& a, const LayerW& W,
                                         const float* rro, float* s_red, float* part) {
  const int hid = a.hidden;
  const int i0 = blockIdx.x * CBK_THREADS + threadIdx.x;
  const int istep = gridDim.x * CBK_THREADS;
  float ss[M];
#pragma unroll
  for (int m = 0; m < M; ++m) ss[m] = 0.f;
  for (int m = 0; m < M; ++m)
    for (int i = i0; i < hid; i += istep) {
      const __nv_bfloat16 b =
          BF(F(a.h[(size_t)m * hid + i]) +
             rb(F(a.obuf[(size_t)m * hid + i]) * rro[m] * (1.f + F(W.post_attn_ln[i]))));
      a.h2[(size_t)m * hid + i] = b;
      ss[m] += F(b) * F(b);
    }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float t = block_reduce_sum(ss[m], s_red);
    if (threadIdx.x == 0) part[(size_t)blockIdx.x * M + m] = t;
  }
}

// --------------------------------------------------- attention: one head's prep
// q_norm/k_norm + RoPE for ONE (sequence, head) by ONE warp.  `base` points at the
// head's raw row in a.qkv; lane l owns dims {l, l+32, ...}.  Returns the result in
// y[0..DPL) (lane-strided).  head_dim/32 must be even (the NeoX half-split partner
// then lives in the same lane).
__device__ __forceinline__ void head_norm_rope(const __nv_bfloat16* base, int D,
                                               const __nv_bfloat16* nw, const float* cs,
                                               const float* sn, float eps, float* y) {
  const int lane = threadIdx.x & 31, DPL = D / 32, HD = D / 2;
  float r[CBK_MAXDPL];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < CBK_MAXDPL; ++i)
    if (i < DPL) { r[i] = F(base[lane + 32 * i]); ss += r[i] * r[i]; }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float rn = rsqrtf(ss / (float)D + eps);
#pragma unroll
  for (int i = 0; i < CBK_MAXDPL; ++i)
    if (i < DPL) r[i] = rb(r[i] * rn * (1.f + F(nw[lane + 32 * i])));
#pragma unroll
  for (int i = 0; i < CBK_MAXDPL; ++i)
    if (i < DPL) {
      const int d = lane + 32 * i;
      const int j = (d < HD) ? d : (d - HD);
      const int ip = (d < HD) ? (i + DPL / 2) : (i - DPL / 2);
      const float sgn = (d < HD) ? -1.f : 1.f;
      y[i] = rb(rb(r[i] * cs[j]) + rb(sgn * r[ip] * sn[j]));
    }
}

// ----------------------------------------------------------- attention: prepare
// One WARP per (sequence, q-head) applies q_norm + RoPE in place in a.qkv; one warp per
// (sequence, kv-head) applies k_norm + RoPE and appends k/v to the cache.  Lane l owns
// head dims {l, l+32, ...}; head_dim/2 is a multiple of 32, so the NeoX half-split RoPE
// partner (d -/+ D/2) is index i -/+ DPL/2 in the SAME lane.
template <int M>
__device__ __forceinline__ void phase_attn_prep(const Args& a, const LayerW& W, int L) {
  PROF_T(a);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, DPL = D / 32, H = a.n_heads, HD = D / 2;
  const size_t roff = (size_t)((W.is_sliding ? 0 : 1) * M) * HD;
  const int units = M * (H + a.n_kv);
  const int GW = gridDim.x * CBK_WARPS;
  for (int u = blockIdx.x * CBK_WARPS + warp; u < units; u += GW) {
    const bool isq = (u < M * H);
    const int t = isq ? u : (u - M * H);
    const int hh = isq ? (t % H) : (t % a.n_kv);
    const int m = isq ? (t / H) : (t / a.n_kv);
    const float* cs = a.rope_cs + roff + (size_t)m * HD;
    const float* sn = a.rope_sn + roff + (size_t)m * HD;
    __nv_bfloat16* base =
        a.qkv + (size_t)m * a.nqkv + (isq ? hh * D : (a.nq_dim + hh * D));
    float y[CBK_MAXDPL];
    head_norm_rope(base, D, isq ? W.q_norm : W.k_norm, cs, sn, a.eps, y);
    if (isq) {
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < DPL) base[lane + 32 * i] = BF(y[i]);
    } else {
      const size_t kb = (((size_t)L * M + m) * a.n_kv + hh) * (size_t)a.max_ctx * D +
                        (size_t)a.pos_v[m] * D;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < DPL) {
          a.kcache[kb + lane + 32 * i] = BF(y[i]);
          a.vcache[kb + lane + 32 * i] = base[a.nkv_dim + lane + 32 * i];
        }
    }
  }
  PROF_ADD(a, 0);
}

// ------------------------------------------------------- attention: the key walk
// Combine the block's warps' online-softmax states for one head through smem.
// acc is laid out lane-owns-CONTIGUOUS-dims: lane l owns dims [l*VPL, l*VPL+VPL).
__device__ __forceinline__ void block_combine(float* sacc, float* smx, float* sl, int D,
                                              int VPL, const float* acc, float mx,
                                              float lsum, float* aa, float& gm, float& l) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < CBK_MAXDPL; ++i)
    if (i < VPL) sacc[warp * D + lane * VPL + i] = acc[i];
  if (lane == 0) { smx[warp] = mx; sl[warp] = lsum; }
  __syncthreads();
  gm = -1e30f; l = 0.f;
#pragma unroll
  for (int i = 0; i < CBK_MAXDPL; ++i) aa[i] = 0.f;
  if (warp == 0) {
    for (int w = 0; w < CBK_WARPS; ++w) {
      const float pm = smx[w], nm = fmaxf(gm, pm);
      const float co = __expf(gm - nm), cn = __expf(pm - nm);
      l = l * co + sl[w] * cn;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < VPL) aa[i] = aa[i] * co + sacc[w * D + lane * VPL + i] * cn;
      gm = nm;
    }
  }
}

// One BLOCK per (sequence, kv-head, key-split); the block serves ALL KG query heads of
// that GQA group, so every K/V byte is read exactly once per layer.
template <int M, int KG>
__device__ __forceinline__ void phase_attn(const Args& a, const LayerW& W, int L,
                                           __nv_bfloat16* s_x) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, VPL = D / 32, H = a.n_heads, SPB = a.split;
  float* sacc = (float*)s_x;                              // CBK_WARPS * D floats
  float* smx = sacc + CBK_WARPS * D;                      // CBK_WARPS
  float* sl = smx + CBK_WARPS;                            // CBK_WARPS
  __nv_bfloat16* sq = (__nv_bfloat16*)(sl + CBK_WARPS);   // KG * D bf16
  const int units = M * a.n_kv * SPB;
  PROF_T(a);
  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    const int sp = u % SPB, t = u / SPB;
    const int kvh = t % a.n_kv, m = t / a.n_kv;
    const int pos = a.pos_v[m];
    const int lo = W.is_sliding ? max(0, pos - a.sliding_window + 1) : 0;
    __syncthreads();
#if CBK_FUSEPREP
#if CBK_KVONCE
    // the one split whose key range contains `pos` -- the only block that must append k/v
    const int per_ = ceildiv(max(0, pos + 1 - lo), SPB);
    const int k0_ = lo + sp * per_;
    const bool own_new = (k0_ <= pos) && (pos < k0_ + per_);
#else
    const bool own_new = true;
#endif
    if (warp < KG || (warp == KG && own_new)) {
      const int DPL = D / 32, HD = D / 2;
      const size_t roff = (size_t)((W.is_sliding ? 0 : 1) * M) * HD;
      const float* cs = a.rope_cs + roff + (size_t)m * HD;
      const float* sn = a.rope_sn + roff + (size_t)m * HD;
      const bool isq = (warp < KG);
      const int hh = isq ? (kvh * KG + warp) : kvh;
      const __nv_bfloat16* base =
          a.qkv + (size_t)m * a.nqkv + (isq ? hh * D : (a.nq_dim + hh * D));
      float y[CBK_MAXDPL];
      head_norm_rope(base, D, isq ? W.q_norm : W.k_norm, cs, sn, a.eps, y);
      if (isq) {
#pragma unroll
        for (int i = 0; i < CBK_MAXDPL; ++i)
          if (i < DPL) sq[warp * D + lane + 32 * i] = BF(y[i]);
      } else {
        const size_t kb = (((size_t)L * M + m) * a.n_kv + hh) * (size_t)a.max_ctx * D +
                          (size_t)pos * D;
#pragma unroll
        for (int i = 0; i < CBK_MAXDPL; ++i)
          if (i < DPL) {
            a.kcache[kb + lane + 32 * i] = BF(y[i]);
            a.vcache[kb + lane + 32 * i] = base[a.nkv_dim + lane + 32 * i];
          }
      }
    }
#else
    for (int i = threadIdx.x; i < KG * D; i += CBK_THREADS)
      sq[i] = a.qkv[(size_t)m * a.nqkv + (size_t)(kvh * KG) * D + i];
#endif
    __syncthreads();
    PROF_ADD(a, 1);

    const size_t kbase = (((size_t)L * M + m) * a.n_kv + kvh) * (size_t)a.max_ctx * D;
    const int S = pos + 1 - lo;
    const int per = ceildiv(S, SPB);
    const int k0 = lo + sp * per, k1 = min(pos + 1, k0 + per);
    float acc[KG][CBK_MAXDPL], mx[KG], lsum[KG];
#pragma unroll
    for (int g = 0; g < KG; ++g) {
      mx[g] = -1e30f; lsum[g] = 0.f;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i) acc[g][i] = 0.f;
    }
    const int NR = D >> 3;
    const int nch = ceildiv(max(0, k1 - k0), 32);
    for (int c = warp; c < nch; c += CBK_WARPS) {
      const int cb = k0 + c * 32;
      const int ce = min(k1, cb + 32);
      const int kk = min(cb + lane, ce - 1);
      const bool ok = (cb + lane) < ce;
      // ---- q.k : this LANE owns key kk and streams its whole row as uint4, so the dot
      // closes in registers with no shuffle reduction (q is a smem broadcast).
      // MEASURED NEGATIVE alternative: lane-owns-contiguous-dims (fully coalesced loads,
      // 5 shuffles per key per head) costs 43.4 us/layer vs 30.4 here on the 27B -- the
      // extra shuffles and the live q registers outweigh the better coalescing.
      float s[KG];
#pragma unroll
      for (int g = 0; g < KG; ++g) s[g] = 0.f;
      {
        const uint4* kp = reinterpret_cast<const uint4*>(a.kcache + kbase + (size_t)kk * D);
        const uint4* qp = reinterpret_cast<const uint4*>(sq);
#if CBK_ATTN_DIAG & 1
#pragma unroll 1
        for (int r = 0; r < 0; ++r) {          // DIAGNOSTIC: q.k deleted
          const uint4 kv4 = kp[r];
#else
#pragma unroll 8
        for (int r = 0; r < NR; ++r) {
          const uint4 kv4 = kp[r];
#endif
          const __nv_bfloat16* kh = reinterpret_cast<const __nv_bfloat16*>(&kv4);
#pragma unroll
          for (int g = 0; g < KG; ++g) {
            const uint4 qv4 = qp[g * NR + r];
            const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qv4);
#pragma unroll
            for (int j = 0; j < 8; ++j) s[g] = fmaf(F(qh[j]), F(kh[j]), s[g]);
          }
        }
      }
      float p[KG];
#pragma unroll
      for (int g = 0; g < KG; ++g) {
        s[g] = ok ? (s[g] * a.attn_scale) : -1e30f;
        float cm = s[g];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, off));
        const float nm = fmaxf(mx[g], cm);
        const float corr = __expf(mx[g] - nm);
        p[g] = __expf(s[g] - nm);
        float ps = p[g];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
        lsum[g] = lsum[g] * corr + ps;
        mx[g] = nm;
#pragma unroll
        for (int i = 0; i < CBK_MAXDPL; ++i)
          if (i < VPL) acc[g][i] *= corr;
      }
      // ---- p.V : lane-owns-dims, VPL CONTIGUOUS dims per lane.
      // The 32 keys are consumed in batches of CBK_PVB whose LOADS ARE ISSUED FIRST,
      // so the warp has CBK_PVB independent 8/16-B loads in flight.  With one load
      // per key (v2) the `if (j >= nv) break` stopped the compiler hoisting them and
      // the batch became a serial chain of 32 dependent memory latencies -- that,
      // not the KV traffic, was the context-INDEPENDENT ~25 us/layer attention cost.
      // The clamp `min(jb+t, nv-1)` keeps every load in range so none is predicated
      // off; the FMA order over j is unchanged, so the result is bit-identical.
      {
        const int nv = ce - cb;
        const __nv_bfloat16* vbase = a.vcache + kbase + lane * VPL;
#pragma unroll 1
        for (int jb = 0; jb < 32; jb += CBK_PVB) {
          if (jb >= nv) break;
          uint4 raw[CBK_PVB];
#pragma unroll
          for (int t = 0; t < CBK_PVB; ++t) {
            const __nv_bfloat16* vp = vbase + (size_t)(cb + min(jb + t, nv - 1)) * D;
            if (VPL == 4)
              *reinterpret_cast<uint2*>(&raw[t]) = *reinterpret_cast<const uint2*>(vp);
            else if (VPL == 8)
              raw[t] = *reinterpret_cast<const uint4*>(vp);
            else {
              __nv_bfloat16* rh = reinterpret_cast<__nv_bfloat16*>(&raw[t]);
#pragma unroll
              for (int i = 0; i < CBK_MAXDPL; ++i)
                if (i < VPL) rh[i] = vp[i];
            }
          }
#pragma unroll
          for (int t = 0; t < CBK_PVB; ++t) {
            const int j = jb + t;
            if (j >= nv) break;
            const __nv_bfloat16* vh = reinterpret_cast<const __nv_bfloat16*>(&raw[t]);
            float vv[CBK_MAXDPL];
#pragma unroll
            for (int i = 0; i < CBK_MAXDPL; ++i)
              if (i < VPL) vv[i] = F(vh[i]);
#pragma unroll
            for (int g = 0; g < KG; ++g) {
              const float pv = __shfl_sync(0xffffffffu, p[g], j);
#pragma unroll
              for (int i = 0; i < CBK_MAXDPL; ++i)
                if (i < VPL && !(CBK_ATTN_DIAG & 2))   // DIAGNOSTIC: p.V deleted
                  acc[g][i] = fmaf(pv, vv[i], acc[g][i]);
            }
          }
        }
      }
    }
    // ---- combine the block's warps, then emit
    PROF_ADD(a, 2);
#pragma unroll
    for (int g = 0; g < KG; ++g) {
      float aa[CBK_MAXDPL], gm, l;
      block_combine(sacc, smx, sl, D, VPL, acc[g], mx[g], lsum[g], aa, gm, l);
      const int h = kvh * KG + g;
      if (warp == 0) {
        if (SPB == 1) {
          const float invl = 1.f / l;
          const int c0 = h * D + lane * VPL;
          // attn_out is o_proj's activation and is read STRAIGHT from global by the
          // GEMV, so it must use the interleaved x[col*M + m] layout.
#pragma unroll
          for (int i = 0; i < CBK_MAXDPL; ++i)
            if (i < VPL) {
              const float ov = rb(aa[i] * invl);
              a.attn_out[(size_t)(c0 + i) * M + m] =
                  W.o.col_scale ? BF(ov * __half2float(W.o.col_scale[c0 + i])) : BF(ov);
            }
        } else {
          float* pt = a.partials + ((size_t)(m * H + h) * SPB + sp) * (D + 2);
#pragma unroll
          for (int i = 0; i < CBK_MAXDPL; ++i)
            if (i < VPL) pt[lane * VPL + i] = aa[i];
          if (lane == 0) { pt[D] = gm; pt[D + 1] = l; }
        }
      }
    }
    __syncthreads();
    PROF_ADD(a, 3);
  }
}

// Cross-block reduce: one BLOCK per (sequence, head), the warps split the SPB partials.
template <int M>
__device__ __forceinline__ void phase_attn_reduce(const Args& a, const LayerW& W,
                                                  __nv_bfloat16* s_x) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, VPL = D / 32, H = a.n_heads, SPB = a.split;
  float* sacc = (float*)s_x;
  float* smx = sacc + CBK_WARPS * D;
  float* sl = smx + CBK_WARPS;
  const int units = M * H;
  PROF_T(a);
  const int pw = ceildiv(SPB, CBK_WARPS);
  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    const int h = u % H, m = u / H;
    const float* pb = a.partials + (size_t)(m * H + h) * SPB * (D + 2);
    float acc[CBK_MAXDPL], mx = -1e30f, lsum = 0.f;
#pragma unroll
    for (int i = 0; i < CBK_MAXDPL; ++i) acc[i] = 0.f;
    const int e = min(SPB, (warp + 1) * pw);
    for (int sp = warp * pw; sp < e; ++sp) {
      const float* p = pb + (size_t)sp * (D + 2);
      const float pm = p[D], nm = fmaxf(mx, pm);
      const float co = __expf(mx - nm), cn = __expf(pm - nm);
      lsum = lsum * co + p[D + 1] * cn;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < VPL) acc[i] = acc[i] * co + p[lane * VPL + i] * cn;
      mx = nm;
    }
    float aa[CBK_MAXDPL], gm, l;
    block_combine(sacc, smx, sl, D, VPL, acc, mx, lsum, aa, gm, l);
    if (warp == 0) {
      const float invl = 1.f / l;
      const int c0 = h * D + lane * VPL;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < VPL) {
          const float ov = rb(aa[i] * invl);
          a.attn_out[(size_t)(c0 + i) * M + m] =
              W.o.col_scale ? BF(ov * __half2float(W.o.col_scale[c0 + i])) : BF(ov);
        }
    }
    __syncthreads();
    PROF_ADD(a, 4);
  }
}

// ------------------------------------------------------------------ the kernel
#ifndef CBK_XSYNC
#define CBK_XSYNC 0
#endif
// 11 grid.sync phases per layer; see PHASES in runner.py for the timing map.
template <int M, int KG, int MINB>
__global__ void __launch_bounds__(CBK_THREADS, MINB) megakernel(Args a) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ __nv_bfloat16 s_x[];
  __shared__ float s_red[CBK_WARPS + 1];
  __shared__ float s_av[CBK_THREADS];
  __shared__ int s_ai[CBK_THREADS];
  const int hid = a.hidden;
  const int nq = a.nq_dim, nk = a.nkv_dim;
  float* part = a.rsums;
  int ts = 0;
#define STAMP()                                                     \
  do {                                                              \
    if (a.timings && blockIdx.x == 0 && threadIdx.x == 0)           \
      a.timings[ts] = clock64();                                    \
    ++ts;                                                           \
  } while (0)

  STAMP();
  cbk::detail::prmt_smem_stage();   // no-op unless CBK_BLK1632_LUT==2 (one copy per block)
  __syncthreads();
  {  // RoPE cos/sin for this step's positions, once.
    const int HD = a.head_dim / 2;
    for (int t = blockIdx.x * CBK_THREADS + threadIdx.x; t < 2 * M * HD;
         t += gridDim.x * CBK_THREADS) {
      const int j = t % HD, r = t / HD;
      const int m = r % M, lt = r / M;
      const float ang = (float)a.pos_v[m] * (lt ? a.inv_global[j] : a.inv_local[j]);
      a.rope_cs[t] = rb(cosf(ang));
      a.rope_sn[t] = rb(sinf(ang));
    }
  }
  float rr[M], rrd[M];
  for (int L = 0; L < a.n_layers; ++L) {
    const LayerW& W = a.layers[L];
    // 1. residual tail of the previous layer -> h
    if (L > 0) block_rms<M>(a.dbuf, hid, a.eps, s_red, rrd);
    phase_h<M>(a, L > 0 ? &a.layers[L - 1] : nullptr, L, rrd, s_red, part);
    grid.sync(); STAMP();
    // 2. input_layernorm + the column-scaled copies of x for q/k/v
    rms_from_partials<M>(part, hid, a.eps, s_red, rr);
    write_xprime<M>(a.h, hid, W.in_ln, rr, W.qkv.col_scale ? 3 : 1, W.qkv.col_scale,
                    a.xbuf);
    grid.sync(); STAMP();
    // 3. fused qkv GEMV
    {
      PROF_T(a);
      __nv_bfloat16* out = a.qkv;
      const int ld = a.nqkv;
      const __nv_bfloat16* xb = a.xbuf;
      const int hd = hid * M;
      const bool cs = (W.qkv.col_scale != nullptr);
      gemv_phase<M>(
          W.qkv, (int)W.qkv.K,
          [=] __device__(int r0, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
            const int t = cs ? cbk::qkv_cs_row(r0, nq, nk) : 0;
            *pa = xb + (size_t)t * hd; *pb = *pa;
          },
          [=] __device__(int r, int m, float v) { out[(size_t)m * ld + r] = BF(v); });
      PROF_ADD(a, 5);
    }
    grid.sync(); STAMP();
    // 4. q/k norm + RoPE + KV append (folded into phase 5 when CBK_FUSEPREP)
#if !CBK_FUSEPREP
    phase_attn_prep<M>(a, W, L);
    grid.sync();
#endif
    STAMP();
    // 5. attention
    phase_attn<M, KG>(a, W, L, s_x);
    grid.sync(); STAMP();
    if (a.split > 1) {
      phase_attn_reduce<M>(a, W, s_x);
      grid.sync();
    }
    STAMP();
    // 6. o_proj (its column scale was folded into the attention output write)
    {
      PROF_T(a);
      __nv_bfloat16* out = a.obuf;
      const __nv_bfloat16* xo = a.attn_out;
      gemv_phase<M>(
          W.o, (int)W.o.K,
          [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
            *pa = xo; *pb = xo;
          },
          [=] __device__(int r, int m, float v) { out[(size_t)m * hid + r] = BF(v); });
      PROF_ADD(a, 6);
    }
    grid.sync(); STAMP();
    // 7. h2 = h + post_attention_layernorm(o)
    {
      float rro[M];
      block_rms<M>(a.obuf, hid, a.eps, s_red, rro);
      phase_h2<M>(a, W, rro, s_red, part);
    }
    grid.sync(); STAMP();
    // 8. pre_feedforward_layernorm + the column-scaled copies of x for gate/up
    rms_from_partials<M>(part, hid, a.eps, s_red, rr);
    write_xprime<M>(a.h2, hid, W.pre_ff_ln, rr, W.gateup.col_scale ? 2 : 1,
                    W.gateup.col_scale, a.xbuf);
    grid.sync(); STAMP();
    // 9. gate/up + GeGLU; down_proj's column scale is folded into the write
    {
      PROF_T(a);
      __nv_bfloat16* out = a.act;
      const int ld = a.inter;
      const __nv_bfloat16* xg = a.xbuf;
      const __nv_bfloat16* xu = a.xbuf + (W.gateup.col_scale ? (size_t)hid * M : 0);
      const __half* dcs = W.down.col_scale;
      // a.act is down_proj's activation, read straight from global -> interleaved.
      auto of = [=] __device__(int p, int m, float gv, float uv) {
        const float av = rb(gelu_tanh(rb(gv))) * rb(uv);
        out[(size_t)p * M + m] = dcs ? BF(rb(av) * __half2float(dcs[p])) : BF(rb(av));
      };
      (void)ld;
      const int RP = (CBK_GATEUP_RP != 0) ? CBK_GATEUP_RP
                                          : pick_R(a.inter, gridDim.x * CBK_WARPS, 2);
      (void)RP;
#if (CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6) && CBK_GATEUP_SPLIT
      if (cbk::layout_is_blk1632((int)W.gateup.layout)) {
#if CBK_GATEUP_RP == 4
        gemv_pairs_split_r<M, 4, 2>(W.gateup, a.inter, xg, xu, of);
#elif CBK_GATEUP_RP == 1
        gemv_pairs_split_r<M, 1, 8>(W.gateup, a.inter, xg, xu, of);
#else
        gemv_pairs_split_r<M, 2, 4>(W.gateup, a.inter, xg, xu, of);
#endif
      } else
#endif
#if CBK_GATEUP_RP == 4
      gemv_pairs_r<M, 4, 2>(W.gateup, a.inter, xg, xu, of);
#else
      if (RP >= 2) gemv_pairs_r<M, 2, 2>(W.gateup, a.inter, xg, xu, of);
      else         gemv_pairs_r<M, 1, 4>(W.gateup, a.inter, xg, xu, of);
#endif
      PROF_ADD(a, 7);
    }
    grid.sync(); STAMP();
    // 10. down_proj (column scale folded into a.act)
    {
      PROF_T(a);
      __nv_bfloat16* out = a.dbuf;
      const __nv_bfloat16* xd = a.act;
      gemv_phase<M>(
          W.down, (int)W.down.K,
          [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
            *pa = xd; *pb = xd;
          },
          [=] __device__(int r, int m, float v) { out[(size_t)m * hid + r] = BF(v); });
      PROF_ADD(a, 8);
    }
    grid.sync(); STAMP();
#if CBK_XSYNC
    // DIAGNOSTIC: CBK_XSYNC extra cooperative barriers per layer, doing no
    // work.  The slope of decode time vs CBK_XSYNC is the MEASURED price of one
    // grid.sync in this kernel, i.e. what deleting two barriers per layer by folding
    // the RMS into the GEMV would be worth.
#pragma unroll 1
    for (int xs = 0; xs < CBK_XSYNC; ++xs) grid.sync();
#endif
  }
  // tail: h, final norm, lm_head
  block_rms<M>(a.dbuf, hid, a.eps, s_red, rrd);
  phase_h<M>(a, &a.layers[a.n_layers - 1], a.n_layers, rrd, s_red, part);
  grid.sync(); STAMP();
  rms_from_partials<M>(part, hid, a.eps, s_red, rr);
  write_xprime<M>(a.h, hid, a.final_norm, rr, 1, nullptr, a.xbuf);
  grid.sync(); STAMP();
  {
    PROF_T(a);
    float* out = a.logits;
    const int V = a.vocab;
    const __nv_bfloat16* xb = a.xbuf;
    gemv_phase<M>(
        a.embed, V,
        [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
          *pa = xb; *pb = xb;
        },
        [=] __device__(int r, int m, float v) { out[(size_t)m * V + r] = v; });
    PROF_ADD(a, 9);
  }
  grid.sync(); STAMP();
  {  // argmax: per-block partial, then block 0 reduces
    const int V = a.vocab;
    for (int m = 0; m < M; ++m) {
      float best = -1e30f;
      int bi = 0;
      const float* lg = a.logits + (size_t)m * V;
      for (int i = blockIdx.x * CBK_THREADS + threadIdx.x; i < V;
           i += gridDim.x * CBK_THREADS) {
        const float v = lg[i];
        if (v > best) { best = v; bi = i; }
      }
      s_av[threadIdx.x] = best;
      s_ai[threadIdx.x] = bi;
      __syncthreads();
      for (int s = CBK_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && s_av[threadIdx.x + s] > s_av[threadIdx.x]) {
          s_av[threadIdx.x] = s_av[threadIdx.x + s];
          s_ai[threadIdx.x] = s_ai[threadIdx.x + s];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        a.amax_val[(size_t)blockIdx.x * M + m] = s_av[0];
        a.amax_idx[(size_t)blockIdx.x * M + m] = s_ai[0];
      }
      __syncthreads();
    }
  }
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x < M) {
    const int m = threadIdx.x;
    float best = -1e30f;
    int bi = 0;
    for (int b = 0; b < gridDim.x; ++b) {
      const float v = a.amax_val[(size_t)b * M + m];
      if (v > best) { best = v; bi = a.amax_idx[(size_t)b * M + m]; }
    }
    a.amax_val[(size_t)gridDim.x * M + m] = best;
    a.amax_idx[(size_t)gridDim.x * M + m] = bi;
  }
  STAMP();
#undef STAMP
}

// --------------------------------------------------------------- host launcher
namespace cbk {

// MINB (min blocks/SM = the register budget) is a COMPILE-TIME knob: one kernel per
// build, selected by -DCBK_MINB=<n> from runner.py (COBALT_MINB).  Instantiating the
// whole sweep in one translation unit costs ~10 min of cicc, so it is not worth it.
// MEASURED (27B, 2g): MINB 4 -> 64 regs, 1296 B spill stores, 37.2 tok/s at 336 blocks;
// MINB 3 -> 80 regs, 342 B spills, 38.5 at 188; MINB 2 -> 128 regs, 16 B spills,
// 40.8 tok/s at 188 blocks.  The spill traffic, not the occupancy, was the limiter.
#ifndef CBK_MINB
#define CBK_MINB 2
#endif
void mega_set_minb(int) {}

// KG = kv_group.  Every Gemma3 checkpoint we target has KG == 2; other ratios fall back
// to the KG == 1 instantiation, M == 1 only (smoke).
static void* pick_kernel(int M, int KG) {
  if (KG != 2) return (M == 1) ? (void*)megakernel<1, 1, CBK_MINB> : nullptr;
  switch (M) {
    case 1: return (void*)megakernel<1, 2, CBK_MINB>;
#ifndef CBK_M1ONLY
    case 2: return (void*)megakernel<2, 2, 2>;
    case 4: return (void*)megakernel<4, 2, 2>;
    case 8: return (void*)megakernel<8, 2, 1>;
#endif
    default: return nullptr;
  }
}

int mega_plan(int M, int KG, int smem_bytes, int* blocks_out) {
  void* k = pick_kernel(M, KG);
  if (!k) return -1;
  if (smem_bytes > 48 * 1024)
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, CBK_THREADS, smem_bytes) !=
      cudaSuccess)
    return -2;
  int dev = 0;
  cudaGetDevice(&dev);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  *blocks_out = nb * sms;
  return nb;
}

int mega_launch(Args a, int M, int KG, int blocks, int smem_bytes, cudaStream_t stream) {
  void* k = pick_kernel(M, KG);
  if (!k) return -1;
  void* args[] = {(void*)&a};
  cudaError_t e = cudaLaunchCooperativeKernel(k, dim3(blocks), dim3(CBK_THREADS), args,
                                             smem_bytes, stream);
  if (e != cudaSuccess) {
    printf("[cobaltkernel] launch failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace cbk
