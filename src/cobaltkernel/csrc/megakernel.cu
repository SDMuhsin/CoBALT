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
#include "cobalt_gemv_mma.cuh"
#include "cobalt_gemv_stream.cuh"

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
// COBALT_ATTN_COMBINE1: combine ALL KG heads' warp states in ONE smem pass (2 __syncthreads
// instead of 2*KG; the KG combines run on KG warps in parallel).  MEASURED on BioMistral-7B
// (1g, PROF build): the per-head combine was 7.7 of attention's 32.7 us/layer.  Same combine
// order over warps -> bit-identical results.  Needs KG x the sacc/smx/sl smem (runner sizes it).
#ifndef CBK_ATTN_COMBINE1
#define CBK_ATTN_COMBINE1 0
#endif
// COBALT_FUSE_RESID: for layers WITHOUT post-norms (llama family: post_attn_ln / post_ff_ln null) the
// residual adds h2 = h + o and h = h2 + down are done in the o_proj / down_proj GEMV epilogues and the
// two residual phases (and their grid.syncs) are skipped; the following norm phase takes the RMS from a
// block-wide read of the row instead of the per-block partials.  Same arithmetic (bf16(h + bf16(v))).
// Gemma3 layers have post-norms and take the unfused path regardless (bit-identical by construction).
#ifndef CBK_FUSE_RESID
#define CBK_FUSE_RESID 0
#endif
// COBALT_XSMEM: stage a GEMV phase's activation x' (NC column-scaled copies, [col][M] interleaved) in
// dynamic shared memory once per block, so the row groups read x' from smem instead of re-reading it
// through L1 (hidden 4096: ~230 MB/layer of x' re-reads for gate|up alone).  Falls back to global
// when the phase's x' does not fit a.xsmem_bytes.  Same arithmetic, same bytes -> bit-identical.
#ifndef CBK_XSMEM
#define CBK_XSMEM 0
#endif
// CBK_FUSE_RESID == 2: as 1, but the RMS partial sums of the fused residual are accumulated in the GEMV
// epilogue (shared-memory atomics, one global write per block) so the following norm phase uses
// rms_from_partials instead of re-reading the row (the re-read cost more than the deleted phase).
// COBALT_ATTN_KUNROLL: K-row uint4 loads kept in flight per key in the q.k dot (8 = shipped).
#ifndef CBK_ATTN_KUNROLL
#define CBK_ATTN_KUNROLL 8
#endif
#define CBK_STR_(x) #x
#define CBK_PRAGMA_UNROLL(n) _Pragma(CBK_STR_(unroll n))   // `#pragma unroll MACRO` is not expanded by nvcc

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
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
// Per-decode-step state (token, position, attention key-split).  Lives in __shared__ and is
// passed by const reference so `Args a` (the kernel parameter block) is never written: a
// written-to param struct is spilled to local memory and every phase pays for it.
struct Step { int tok[CBK_MAXM]; int pos[CBK_MAXM]; int split; };
// RMSNorm weight application, both conventions (arch.py): gemma multiplies by (1+w) in
// fp32 and rounds once; llama rounds the normalised value to bf16 FIRST, then w * x.
__device__ __forceinline__ float norm_apply(float v, float rr, float w, int plus_one) {
  return plus_one ? rb(v * rr * (1.f + w)) : rb(rb(v * rr) * w);
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
// R candidates: the shipped {1,2,4} plus 3/5/6/8 (CBK_RMAX >= 5 enables them).  Non-power-of-2 R
// lets a small-K phase fit ONE wave at full occupancy instead of paying a whole second wave for a
// few leftover groups (e.g. qkv K=6144 at 1504 warps: R=4 -> 1536 groups = 2 waves; R=5 -> 1 wave).
// R never changes a row's arithmetic (each row is accumulated per lane over its own granules and
// shuffle-reduced the same way), so any R is bit-identical to any other.
// the shipped set {1,2,4} is always eligible; 3/5/6/8 only when rmax >= 5 (so the default schedule is unchanged)
// pow2_only: the fused q|k|v matrix selects its column-scale row PER ROW GROUP (xf(r0)), so a group
// must never straddle the q/k/v boundaries -> R must divide gcd(nq_dim, nkv_dim) (powers of two here).
__device__ __forceinline__ bool r_ok(int r, int rmax, bool pow2_only) {
  if (r == 1 || r == 2 || r == 4) return true;
  if (r == 8) return rmax >= 5;
  return !pow2_only && rmax >= 5 && (r == 3 || r == 5 || r == 6);
}
__device__ __forceinline__ int pick_R(int K, int GW, int rmax, bool pow2_only = false) {
  int bc = 1 << 30;
#pragma unroll
  for (int r = 1; r <= 8; ++r) {
    if (r > rmax || !r_ok(r, rmax, pow2_only)) continue;
    bc = min(bc, ceildiv(ceildiv(K, r), GW) * r);
  }
  // R also divides the activation traffic: at 4 bits x' is 4x the weight bytes of one
  // row, so R rows sharing one x' stream cuts the L1/L2 re-read by R.  Take the largest
  // R whose tail is within 10% of the best (measured: lm_head 1146 -> 950 us).
  int best = 1;
#pragma unroll
  for (int r = 1; r <= 8; ++r) {
    if (r > rmax || !r_ok(r, rmax, pow2_only)) continue;
    if (ceildiv(ceildiv(K, r), GW) * r * 10 <= bc * CBK_RTOL) best = r;
  }
  return best;
}

// x'[c][j][m] = bf16( bf16(src[m][j] * rr[m] * (1+nw[j])) * cs[c][j] ), NC variants.
template <int M>
__device__ __forceinline__ void write_xprime(const __nv_bfloat16* src, int N,
                                             const __nv_bfloat16* nw, const float* rr,
                                             int NC, const __half* cs,
                                             __nv_bfloat16* dst, int plus_one) {
  const int i0 = blockIdx.x * CBK_THREADS + threadIdx.x;
  const int istep = gridDim.x * CBK_THREADS;
  for (int j = i0; j < N; j += istep) {
    const float g = nw ? F(nw[j]) : 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = F(src[(size_t)m * N + j]);
      if (nw) v = norm_apply(v, rr[m], g, plus_one);
      for (int c = 0; c < NC; ++c)
        dst[(size_t)c * N * M + (size_t)j * M + m] =
            cs ? BF(v * __half2float(cs[(size_t)c * N + j])) : BF(v);
    }
  }
}

// Block-cooperative copy of n bf16 (n % 8 == 0) from global into shared memory (uint4 granules).
__device__ __forceinline__ void stage_x(const __nv_bfloat16* __restrict__ src, int n, __nv_bfloat16* dst) {
  const uint4* s4 = reinterpret_cast<const uint4*>(src);
  uint4* d4 = reinterpret_cast<uint4*>(dst);
  for (int i = threadIdx.x; i < (n >> 3); i += CBK_THREADS) d4[i] = __ldcg(s4 + i);
}
#define CBK_XSTAGE(ptr, nelem)                                                         \
  (CBK_XSMEM && ((int)(nelem) * 2 <= a.xsmem_bytes)                                   \
       ? (stage_x((ptr), (int)(nelem), s_x), __syncthreads(), (const __nv_bfloat16*)s_x) \
       : (ptr))

// ------------------------------------------------------------------ GEMV driver
// Rows are handed to WARPS in groups of R CONSECUTIVE rows, so the group shares one
// activation vector and one uint4 x-load stream.  Groups are warp-cyclic over the whole
// grid; the phase uses no shared memory and no __syncthreads.
#ifndef CBK_BLK1632_PFH
#define CBK_BLK1632_PFH 1
#endif
#ifndef CBK_RMAX
#define CBK_RMAX 4
#endif
#ifndef CBK_CSPLIT
#define CBK_CSPLIT 1
#endif
template <int M, int R, int NX, int PF, int CS, class XF, class OutFn>
__device__ __forceinline__ void gemv_phase_r(const MatDesc& d, int K, XF xf,
                                             OutFn out_fn, float* s_part) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int ngrp = ceildiv(K, R);
  if constexpr (CS == 1) {
    // ---- the shipped walk: one WARP per row-group, full rows ----
    const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
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
        cbk::detail::gemv_blk1632_multi<M, R, NX, (PF * CBK_BLK1632_PFX / CBK_BLK1632_PFH > 0 ? PF * CBK_BLK1632_PFX / CBK_BLK1632_PFH : 1), CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT>(
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
  } else {
    // ---- COLUMN-SPLIT walk (CBK_CSPLIT = CS): CS consecutive warps of a block share one
    // row-group, each streams a 1/CS column slice (half the serial chain per warp on the
    // latency-bound small-K phases), partials are summed through shared memory.  The
    // group loop is BLOCK-uniform so the __syncthreads below are safe.
    constexpr int GPB = CBK_WARPS / CS;                   // row-groups per block
    const int cs = warp % CS, gl = warp / CS;
    float* mine = s_part + (size_t)warp * (R * M);
    for (int gb = blockIdx.x * GPB; gb < ngrp; gb += gridDim.x * GPB) {
      const int g = gb + gl;
      const int r0 = g * R;
      float acc[R][M];
#pragma unroll
      for (int i = 0; i < R; ++i)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[i][m] = 0.f;
      if (g < ngrp) {
        const __nv_bfloat16 *xa = nullptr, *xb = nullptr;
        xf(r0, &xa, &xb);
        if (d.layout == cbk::LAYOUT_DENSE4 && r0 + R <= K) {
          cbk::detail::gemv_dense4_multi<M, R, NX, PF, CS>(
              d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G,
              d.zero + (size_t)r0 * G, G, N, xa, xb, acc, cs);
#if CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6
        } else if (cbk::layout_is_blk1632((int)d.layout) && r0 + R <= K) {
          cbk::detail::gemv_blk1632_multi<M, R, NX, (PF * CBK_BLK1632_PFX / CBK_BLK1632_PFH > 0 ? PF * CBK_BLK1632_PFX / CBK_BLK1632_PFH : 1), CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT, CS>(
              d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G,
              d.zero + (size_t)r0 * G, G, N, xa, xb, acc, cs);
#endif
        } else if (cs == 0) {                             // generic path: slice 0 does full rows
#pragma unroll
          for (int i = 0; i < R; ++i) {
            const int row = r0 + i;
            if (row < K)
              cbk::gemv_view<M>(d, row, 0, N, (NX == 2 && (i & 1)) ? xb : xa, acc[i], nullptr);
          }
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
          for (int m = 0; m < M; ++m) mine[i * M + m] = acc[i][m];
      }
      __syncthreads();
      if (g < ngrp && cs == 0) {
        const float* base = s_part + (size_t)(warp) * (R * M);   // warps warp..warp+CS-1
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const int row = r0 + i;
          if (row < K)
#pragma unroll
            for (int m = 0; m < M; ++m)
              if (lane == m) {
                float tot = 0.f;
#pragma unroll
                for (int c = 0; c < CS; ++c) tot += base[(size_t)c * (R * M) + i * M + m];
                out_fn(row, m, tot);
              }
        }
      }
      __syncthreads();
    }
  }
}

#ifndef CBK_CHUNK
#define CBK_CHUNK 0   // bitmask of phases walked with contiguous equal-work chunking (1 qkv 2 o 4 gate|up 8 down 16 lm_head)
#endif
// Contiguous equal-work chunking of the row walk (CBK_CHUNK).  The phase's work is the
// (row-group, 32-column granule) grid; warp w owns items [w*C, (w+1)*C), C = ceil(items / warps),
// so every warp streams the same bytes (+-1 granule) instead of ceil(groups / warps) whole
// row-groups (down_proj: 1024 groups on 1280 warps left 20 % of the warps idle; gate|up: 5.6
// groups per warp paid a 6th round).  A row-group split between warps is summed through f32
// atomics and finished by the last-arriving warp; the contributor count follows from the
// arithmetic.  fin(r0, v[R]) is called by ALL lanes with the full sums of rows r0..r0+R-1.
template <int M, int R, int NX, int PF, class XF, class Fin>
__device__ __forceinline__ void gemv_phase_chunk(const MatDesc& d, int K, XF xf, Fin fin,
                                                 float* __restrict__ mpart, unsigned* __restrict__ mcnt) {
  static_assert(M == 1, "the chunked walk is the M == 1 decode path");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int NT = N >> 5;
  const int ngrp = K / R;                       // caller guarantees K % R == 0
  const long items = (long)ngrp * NT;
  const long C = (items + GW - 1) / GW;
  long pos = (long)gw * C;
  const long end = min(items, pos + C);
  while (pos < end) {
    const int g = (int)(pos / NT);
    const int t0 = (int)(pos - (long)g * NT);
    const int t1 = (int)min((long)NT, t0 + (end - pos));
    const int r0 = g * R;
    const __nv_bfloat16 *xa = nullptr, *xb = nullptr;
    xf(r0, &xa, &xb);
    float acc[R][M];
#if CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6
    cbk::detail::gemv_blk1632_multi<M, R, NX, PF, CBK_BLK1632_ARM, false>(
        d.data + (size_t)r0 * rs, rs, d.scale + (size_t)r0 * G, d.zero + (size_t)r0 * G, G, N,
        xa, xb, acc, 0, t0, t1);
#else
    (void)xa; (void)xb; (void)t0; (void)t1;
#pragma unroll
    for (int i = 0; i < R; ++i) acc[i][0] = 0.f;
#endif
    float v[R];
#pragma unroll
    for (int i = 0; i < R; ++i) v[i] = acc[i][0];
    if (t0 == 0 && t1 == NT) {
      fin(r0, v);
    } else {
      const long gb = (long)g * NT, ge = gb + NT - 1;
      const unsigned contrib = (unsigned)(ge / C - gb / C + 1);
      float mine = 0.f;
#pragma unroll
      for (int i = 0; i < R; ++i) if (lane == i) mine = v[i];
      if (lane < R) atomicAdd(mpart + r0 + lane, mine);
      __threadfence();
      unsigned old = 0u;
      if (lane == 0) old = atomicAdd(mcnt + g, 1u);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old == contrib - 1u) {
        __threadfence();
        if (lane < R) mine = atomicExch(mpart + r0 + lane, 0.f);
        if (lane == 0) mcnt[g] = 0u;
#pragma unroll
        for (int i = 0; i < R; ++i) v[i] = __shfl_sync(0xffffffffu, mine, i);
        fin(r0, v);
      }
    }
    pos += (t1 - t0);
  }
}

// PH: the phase's bit in CBK_MMA_PHASES (1 qkv, 2 o_proj, 8 down, 16 lm_head); PF its MMA prefetch depth.
template <int M, int PH, int PF, int UPW, class XF, class OutFn>
__device__ __forceinline__ void gemv_phase(float* s_part, const MatDesc& d, int K, XF xf, OutFn f,
                                           bool pow2_only, float* mpart, unsigned* mcnt) {
#if CBK_BLK1632_MMA && (CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6)
  // tensor-core decode: M == 1, 16-row tiles, 128-column groups (the fused q|k|v boundaries are
  // multiples of 16 rows, so a tile never straddles a column-scale row).  Measured: it wins on the
  // phases that run L2-resident (qkv, o_proj) and loses to the row-streaming loop on the
  // DRAM-bound ones, hence the per-phase mask.
  if ((CBK_MMA_PHASES & PH) && M == 1 && cbk::layout_is_blk1632((int)d.layout) && (K & 15) == 0 &&
      (d.N & 127) == 0) {
    if (CBK_CHUNK & PH)
      cbk::detail::gemv_phase_mma_chunk<CBK_BLK1632_ARM, 1, PF>(d, K, xf,
          [=] __device__(int row0, int lane, float v) { if (lane < 16) f(row0 + lane, 0, v); },
          mpart, mcnt);
    else
      cbk::detail::gemv_phase_mma<CBK_BLK1632_ARM, 1, PF, UPW>(d, K, xf,
          [=] __device__(int row0, int lane, float v) { if (lane < 16) f(row0 + lane, 0, v); },
          mpart, mcnt);
    return;
  }
#endif
#if CBK_STREAM && CBK_BLK1632_ARM == 4
  if constexpr (M == 1)
  if ((CBK_STREAM & PH) && cbk::layout_is_blk1632((int)d.layout) && (K & 3) == 0 &&
      (d.N & (32 * CBK_STREAM_PF - 1)) == 0) {
    cbk::detail::gemv_phase_stream<4, 1, CBK_STREAM_S, CBK_STREAM_PF, 4>(d, K, xf,
        [=] __device__(int r0, const float* v) {
          if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) f(r0 + i, 0, v[i]);
          }
        }, mpart, mcnt, reinterpret_cast<uint8_t*>(s_part));
    return;
  }
#endif
#if (CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6)
  if constexpr (M == 1)
  if ((CBK_CHUNK & PH) && cbk::layout_is_blk1632((int)d.layout) && (K & 3) == 0) {
    gemv_phase_chunk<M, 4, 1, 2>(d, K, xf,
        [=] __device__(int r0, const float* v) {
          if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) f(r0 + i, 0, v[i]);
          }
        }, mpart, mcnt);
    return;
  }
#endif
  (void)mpart; (void)mcnt;
  constexpr int CS = CBK_CSPLIT;
  const int GW = gridDim.x * CBK_WARPS / CS;
  const int R = pick_R(K, GW, CBK_RMAX, pow2_only);
#if CBK_RMAX >= 5
  // register budget (the KG=4 kernel sits at the 128-register MINB=2 bound): keep R*PF <= 9 uint4 of
  // weights live per lane -- R=6/PF=2 (12) spilled and tripled down_proj's time.
  if (R >= 8)      gemv_phase_r<M, 8, 1, 1, CS>(d, K, xf, f, s_part);   // 8 loads in flight
  else if (R == 6) gemv_phase_r<M, 6, 1, 1, CS>(d, K, xf, f, s_part);   // 6
  else if (R == 5) gemv_phase_r<M, 5, 1, 1, CS>(d, K, xf, f, s_part);   // 5
  else if (R == 4) gemv_phase_r<M, 4, 1, 2, CS>(d, K, xf, f, s_part);   // 8 (shipped)
  else if (R == 3) gemv_phase_r<M, 3, 1, 3, CS>(d, K, xf, f, s_part);   // 9
  else if (R == 2) gemv_phase_r<M, 2, 1, 4, CS>(d, K, xf, f, s_part);
  else             gemv_phase_r<M, 1, 1, 8, CS>(d, K, xf, f, s_part);
#else
  if (R >= 4)      gemv_phase_r<M, 4, 1, 2, CS>(d, K, xf, f, s_part);
  else if (R == 2) gemv_phase_r<M, 2, 1, 4, CS>(d, K, xf, f, s_part);
  else             gemv_phase_r<M, 1, 1, 8, CS>(d, K, xf, f, s_part);
#endif
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
      cbk::detail::gemv_blk1632_multi<M, R, 2, (PF * CBK_BLK1632_PFX / CBK_BLK1632_PFH > 0 ? PF * CBK_BLK1632_PFX / CBK_BLK1632_PFH : 1), CBK_BLK1632_ARM, (bool)CBK_BLK1632_FT>(
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
__device__ __forceinline__ void phase_h(const Args& a, const Step& S, const LayerW* Wprev, int L,
                                        const float* rrd, float* s_red, float* part) {
  const int hid = a.hidden;
  const int i0 = blockIdx.x * CBK_THREADS + threadIdx.x;
  const int istep = gridDim.x * CBK_THREADS;
  float ss[M];
#pragma unroll
  for (int m = 0; m < M; ++m) ss[m] = 0.f;
  if (L == 0) {
    for (int m = 0; m < M; ++m) {
      const int tok = S.tok[m];
      for (int i = i0; i < hid; i += istep) {
        const __nv_bfloat16 b = BF(rb(cbk::mat_elem(a.embed, tok, i)) * a.embed_scale);
        a.h[(size_t)m * hid + i] = b;
        ss[m] += F(b) * F(b);
      }
    }
  } else {
    const __nv_bfloat16* w = Wprev->post_ff_ln;   // null = plain residual add (llama)
    const int po = a.norm_plus_one;
    for (int m = 0; m < M; ++m)
      for (int i = i0; i < hid; i += istep) {
        const float d = F(a.dbuf[(size_t)m * hid + i]);
        const __nv_bfloat16 b =
            BF(F(a.h2[(size_t)m * hid + i]) + (w ? norm_apply(d, rrd[m], F(w[i]), po) : d));
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
  const __nv_bfloat16* w = W.post_attn_ln;        // null = plain residual add (llama)
  const int po = a.norm_plus_one;
  for (int m = 0; m < M; ++m)
    for (int i = i0; i < hid; i += istep) {
      const float o = F(a.obuf[(size_t)m * hid + i]);
      const __nv_bfloat16 b =
          BF(F(a.h[(size_t)m * hid + i]) + (w ? norm_apply(o, rro[m], F(w[i]), po) : o));
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
                                               const float* sn, float eps, float* y,
                                               int plus_one) {
  const int lane = threadIdx.x & 31, DPL = D / 32, HD = D / 2;
  float r[CBK_MAXDPL];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < CBK_MAXDPL; ++i)
    if (i < DPL) { r[i] = F(base[lane + 32 * i]); ss += r[i] * r[i]; }
  if (nw) {   // QK-norm (gemma3 / qwen3); null = RoPE only (llama / mistral)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float rn = rsqrtf(ss / (float)D + eps);
#pragma unroll
    for (int i = 0; i < CBK_MAXDPL; ++i)
      if (i < DPL) r[i] = norm_apply(r[i], rn, F(nw[lane + 32 * i]), plus_one);
  }
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
__device__ __forceinline__ void phase_attn_prep(const Args& a, const Step& S, const LayerW& W, int L) {
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
    head_norm_rope(base, D, isq ? W.q_norm : W.k_norm, cs, sn, a.eps, y, a.norm_plus_one);
    if (isq) {
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < DPL) base[lane + 32 * i] = BF(y[i]);
    } else {
      const size_t kb = (((size_t)L * M + m) * a.n_kv + hh) * (size_t)a.max_ctx * D +
                        (size_t)S.pos[m] * D;
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
__device__ __forceinline__ void phase_attn(const Args& a, const Step& S, const LayerW& W, int L,
                                           __nv_bfloat16* s_x) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, VPL = D / 32, H = a.n_heads, SPB = S.split;
#if CBK_ATTN_COMBINE1
  constexpr int NSLOT = KG * CBK_WARPS;                   // one state slot per (head, warp)
#else
  constexpr int NSLOT = CBK_WARPS;
#endif
  float* sacc = (float*)s_x;                              // NSLOT * D floats
  float* smx = sacc + NSLOT * D;                          // NSLOT
  float* sl = smx + NSLOT;                                // NSLOT
  __nv_bfloat16* sq = (__nv_bfloat16*)(sl + NSLOT);       // KG * D bf16
  const int units = M * a.n_kv * SPB;
  PROF_T(a);
  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    const int sp = u % SPB, t = u / SPB;
    const int kvh = t % a.n_kv, m = t / a.n_kv;
    const int pos = S.pos[m];
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
      head_norm_rope(base, D, isq ? W.q_norm : W.k_norm, cs, sn, a.eps, y, a.norm_plus_one);
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
        CBK_PRAGMA_UNROLL(CBK_ATTN_KUNROLL)
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
#if CBK_ATTN_COMBINE1
    __syncthreads();
#pragma unroll
    for (int g = 0; g < KG; ++g) {
      const int slot = g * CBK_WARPS + warp;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i)
        if (i < VPL) sacc[slot * D + lane * VPL + i] = acc[g][i];
      if (lane == 0) { smx[slot] = mx[g]; sl[slot] = lsum[g]; }
    }
    __syncthreads();
    if (warp < KG) {               // warp g combines head g over the 8 warps, same order as before
      const int g = warp;
      float aa[CBK_MAXDPL], gm = -1e30f, l = 0.f;
#pragma unroll
      for (int i = 0; i < CBK_MAXDPL; ++i) aa[i] = 0.f;
      for (int w = 0; w < CBK_WARPS; ++w) {
        const int slot = g * CBK_WARPS + w;
        const float pm = smx[slot], nm = fmaxf(gm, pm);
        const float co = __expf(gm - nm), cn = __expf(pm - nm);
        l = l * co + sl[slot] * cn;
#pragma unroll
        for (int i = 0; i < CBK_MAXDPL; ++i)
          if (i < VPL) aa[i] = aa[i] * co + sacc[slot * D + lane * VPL + i] * cn;
        gm = nm;
      }
      const int h = kvh * KG + g;
      if (SPB == 1) {
        const float invl = 1.f / l;
        const int c0 = h * D + lane * VPL;
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
#else
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
#endif
    __syncthreads();
    PROF_ADD(a, 3);
  }
}

// Cross-block reduce: one BLOCK per (sequence, head), the warps split the SPB partials.
template <int M>
__device__ __forceinline__ void phase_attn_reduce(const Args& a, const Step& S, const LayerW& W,
                                                  __nv_bfloat16* s_x) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, VPL = D / 32, H = a.n_heads, SPB = S.split;
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

// ------------------------------------------------- attention on tensor cores (M == 1)
#ifndef CBK_ATTN_TC
#define CBK_ATTN_TC 0
#endif
#ifndef CBK_ATTN_TC_MOVM
#define CBK_ATTN_TC_MOVM 1     // V^T fragments via movmatrix (4-byte loads) instead of 2-byte gathers
#endif
#ifndef CBK_ATTN_TC_VPRE
#define CBK_ATTN_TC_VPRE 0     // issue the chunk's V loads before the S mma (one latency per chunk)
#endif
#ifndef CBK_ATTN_TC_QROPE
#define CBK_ATTN_TC_QROPE 0    // RoPE of q applied in the A fragments (no sq staging); measured +0.6 % step on 2g -> off
#endif
#ifndef CBK_ATTN_TC_HALF
#define CBK_ATTN_TC_HALF 1     // one-round merge, O/l stored as fp16 (half the smem)
#endif
#ifndef CBK_ATTN_TC_TREE
#define CBK_ATTN_TC_TREE 0     // register tree merge of the warps' states (half the smem)
#endif
#ifndef CBK_ATTN_TC_REDUCE
#define CBK_ATTN_TC_REDUCE 1   // with CBK_ATTN_TC: the last split block of a kv-head reduces (no attn_reduce phase)
#endif
#if CBK_ATTN_TC
// S = Q.K^T as mma.m16n8k16 bf16 (A = the KG query heads of the GQA group padded to 16 rows, B = 8 keys
// of the K cache), the online softmax on S's C fragment, then O^T = V^T.P^T (A = V^T read from the
// row-major V cache, B = P^T, which is S's C fragment re-packed to bf16 -- no shuffle).  The mma's
// k index is a permutation of the real dims / keys, which is free as long as A and B agree:
//   S:  lane (g, t) holds dims 16ks + 4t .. 4t+3 (one 8-byte load of Q and of key g per k-step);
//       B column n = g <-> key kb + 4(g>>1) + (g&1) + 2*tau for S tile tau, so that the C fragment
//       of lane (g, t) holds head g at keys kb + 4t .. 4t+3 (c0, c1 of tile 0, then of tile 1);
//   PV: k-slots (2t, 2t+1, 2t+8, 2t+9) <-> keys kb + 4t .. 4t+3, n = head, m = dims 16mt + g (+8).
// One warp walks 16 keys per chunk; a block's key split is walked in chunks of 16 x CBK_WARPS and
// the warps' online-softmax states are merged through smem exactly as before (same partials).
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);     // .x (low half) = lo
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t pack_u16x2(const __nv_bfloat16* lo, const __nv_bfloat16* hi) {
  return (uint32_t)(*reinterpret_cast<const unsigned short*>(lo)) |
         ((uint32_t)(*reinterpret_cast<const unsigned short*>(hi)) << 16);
}
template <int M, int KG>
__device__ __forceinline__ void phase_attn_tc(const Args& a, const Step& S, const LayerW& W, int L,
                                              __nv_bfloat16* s_x) {
  static_assert(KG >= 1 && KG <= 8, "the GQA group is the mma's n dimension");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int D = a.head_dim, DT = D >> 4, H = a.n_heads, SPB = S.split;
#if CBK_ATTN_TC_TREE
  constexpr int NSL = CBK_WARPS / 2;                      // tree merge: 4 state slots; sq aliases slot 0 (dead after the walk)
  float* sacc = (float*)s_x;                              // [NSL][KG][D]
  float* smx = sacc + NSL * KG * D;                       // [NSL][KG]
  float* sl = smx + NSL * KG;                             // [NSL][KG]
  __nv_bfloat16* sq = (__nv_bfloat16*)s_x;                // [KG][D]
#elif CBK_ATTN_TC_HALF
  // one-round merge with the warps' O/l stored as fp16 (8 KB at KG=4, D=128); sq aliases the slot buffer
  __half* sacc16 = (__half*)s_x;                          // [WARPS][KG][D]
  float* smx = (float*)(sacc16 + CBK_WARPS * KG * D);     // [WARPS][KG]
  float* sl = smx + CBK_WARPS * KG;                       // [WARPS][KG]
  __nv_bfloat16* sq = (__nv_bfloat16*)s_x;                // [KG][D]
#else
  float* sacc = (float*)s_x;                              // [WARPS][KG][D]
  float* smx = sacc + CBK_WARPS * KG * D;                 // [WARPS][KG]
  float* sl = smx + CBK_WARPS * KG;                       // [WARPS][KG]
  __nv_bfloat16* sq = (__nv_bfloat16*)(sl + CBK_WARPS * KG);   // [KG][D]
#endif
  const int units = M * a.n_kv * SPB;     // one unit per (sequence, kv-head, key split), as phase_attn
  PROF_T(a);
  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    const int sp = u % SPB, tt = u / SPB;
    const int kvh = tt % a.n_kv, m = tt / a.n_kv;
    const int pos = S.pos[m];
    const int lo = W.is_sliding ? max(0, pos - a.sliding_window + 1) : 0;
    __syncthreads();
#if CBK_ATTN_TC_QROPE
    const bool qreg = (W.q_norm == nullptr);      // no QK-norm: RoPE the query inside the A fragments, no sq staging
#else
    const bool qreg = false;
#endif
    {   // q RoPE (+norm) into sq, k RoPE + K/V append by the split that owns `pos`
      const int per_ = ceildiv(max(0, pos + 1 - lo), SPB);
      const int k0_ = lo + sp * per_;
      const bool own_new = (k0_ <= pos) && (pos < k0_ + per_);
      if ((warp < KG && !qreg) || (warp == KG && own_new)) {
        const int DPL = D / 32, HD = D / 2;
        const size_t roff = (size_t)((W.is_sliding ? 0 : 1) * M) * HD;
        const float* cs = a.rope_cs + roff + (size_t)m * HD;
        const float* sn = a.rope_sn + roff + (size_t)m * HD;
        const bool isq = (warp < KG);
        const int hh = isq ? (kvh * KG + warp) : kvh;
        const __nv_bfloat16* base =
            a.qkv + (size_t)m * a.nqkv + (isq ? hh * D : (a.nq_dim + hh * D));
        float y[CBK_MAXDPL];
        head_norm_rope(base, D, isq ? W.q_norm : W.k_norm, cs, sn, a.eps, y, a.norm_plus_one);
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
    }
    __syncthreads();
    PROF_ADD(a, 1);
    const size_t kbase = (((size_t)L * M + m) * a.n_kv + kvh) * (size_t)a.max_ctx * D;
    const int Sn = pos + 1 - lo;
    const int per = ceildiv(Sn, SPB);
    const int k0 = lo + sp * per, k1 = min(pos + 1, k0 + per);
    // Q as A fragments (rows g < KG; the other 12 rows are zero): dims 16ks + 4t .. 4t+3
    uint32_t qa[CBK_MAXDPL * 2][2];
#pragma unroll
    for (int ks = 0; ks < CBK_MAXDPL * 2; ++ks) { qa[ks][0] = 0u; qa[ks][1] = 0u; }
    if (qreg) {
      // lane (g, t) owns dims 16ks + 4t .. 4t+3 of head g; the RoPE partner of dim d is d +- D/2, i.e. the same
      // lane slot at k-step ks +- DT/2 -- so head_norm_rope's arithmetic (rb(rb(r*cs) + rb(sgn*rp*sn))) runs in place
      if (g < KG) {
        const int HD = D / 2;
        const size_t roff = (size_t)((W.is_sliding ? 0 : 1) * M) * HD;
        const float* cs = a.rope_cs + roff + (size_t)m * HD;
        const float* sn = a.rope_sn + roff + (size_t)m * HD;
        const __nv_bfloat16* qb = a.qkv + (size_t)m * a.nqkv + (size_t)(kvh * KG + g) * D + 4 * t;
        // one RoPE pair of k-steps (ks, ks + DT/2) at a time: 8 floats live, nothing survives into the walk
#pragma unroll
        for (int ks = 0; ks < CBK_MAXDPL; ++ks)
          if (ks < DT / 2) {
            const int ksp = ks + DT / 2;
            const uint2 va = *reinterpret_cast<const uint2*>(qb + 16 * ks);
            const uint2 vb = *reinterpret_cast<const uint2*>(qb + 16 * ksp);
            const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(&va);
            const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&vb);
            const int j0 = 16 * ks + 4 * t;
            float ya[4], yb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float ra = F(ha[i]), rb_ = F(hb[i]);
              const float c = cs[j0 + i], sv = sn[j0 + i];
              ya[i] = rb(rb(ra * c) + rb(-rb_ * sv));     // low half:  r*cos - partner*sin
              yb[i] = rb(rb(rb_ * c) + rb(ra * sv));      // high half: r*cos + partner*sin
            }
            qa[ks][0] = pack_bf16x2(ya[0], ya[1]);  qa[ks][1] = pack_bf16x2(ya[2], ya[3]);
            qa[ksp][0] = pack_bf16x2(yb[0], yb[1]); qa[ksp][1] = pack_bf16x2(yb[2], yb[3]);
          }
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < CBK_MAXDPL * 2; ++ks)
        if (ks < DT && g < KG) {
          const uint2 v = *reinterpret_cast<const uint2*>(sq + g * D + 16 * ks + 4 * t);
          qa[ks][0] = v.x; qa[ks][1] = v.y;
        }
    }
    float oacc[CBK_MAXDPL * 2][4];
#pragma unroll
    for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt) { oacc[mt][0] = 0.f; oacc[mt][1] = 0.f; oacc[mt][2] = 0.f; oacc[mt][3] = 0.f; }
    float mrun = -1e30f, lrun = 0.f;
    const int nch = ceildiv(max(0, k1 - k0), 16);
    for (int c = warp; c < nch; c += CBK_WARPS) {
      const int kb = k0 + 16 * c;
      // ---- S = Q.K^T for 16 keys: two n8 tiles, DT k-steps each
#if CBK_ATTN_TC_MOVM
      const int kA = min(kb + g, pos), kB = min(kb + 8 + g, pos);          // tile 0: keys kb+g, tile 1: kb+8+g
#else
      const int kA = min(kb + 4 * (g >> 1) + (g & 1), pos), kB = min(kA + 2, pos);
#endif
      const __nv_bfloat16* kr0 = a.kcache + kbase + (size_t)kA * D + 4 * t;
      const __nv_bfloat16* kr1 = a.kcache + kbase + (size_t)kB * D + 4 * t;
      float C0[4] = {0.f, 0.f, 0.f, 0.f}, C1[4] = {0.f, 0.f, 0.f, 0.f};
      // every K load of the chunk (and, with CBK_ATTN_TC_VPRE, every V load) is issued before the first
      // mma, so the chunk pays ONE memory latency instead of a K round trip, the softmax, then a V one
#if CBK_ATTN_TC_MOVM && CBK_ATTN_TC_VPRE
      uint32_t vraw[CBK_MAXDPL * 2][4];
      {
        const __nv_bfloat16* vpA = a.vcache + kbase + (size_t)min(kb + g, pos) * D + 2 * t;
        const __nv_bfloat16* vpB = a.vcache + kbase + (size_t)min(kb + 8 + g, pos) * D + 2 * t;
#pragma unroll
        for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt)
          if (mt < DT) {
            vraw[mt][0] = *reinterpret_cast<const uint32_t*>(vpA + 16 * mt);
            vraw[mt][1] = *reinterpret_cast<const uint32_t*>(vpA + 16 * mt + 8);
            vraw[mt][2] = *reinterpret_cast<const uint32_t*>(vpB + 16 * mt);
            vraw[mt][3] = *reinterpret_cast<const uint32_t*>(vpB + 16 * mt + 8);
          }
      }
#endif
#pragma unroll
      for (int ks = 0; ks < CBK_MAXDPL * 2; ++ks) {
        if (ks < DT) {
          const uint2 kv0 = *reinterpret_cast<const uint2*>(kr0 + 16 * ks);
          const uint2 kv1 = *reinterpret_cast<const uint2*>(kr1 + 16 * ks);
          const uint32_t A[4] = {qa[ks][0], 0u, qa[ks][1], 0u};
          const uint32_t B0[2] = {kv0.x, kv0.y}, B1[2] = {kv1.x, kv1.y};
          cbk::detail::mma_bf16_16816(C0, A, B0);
          cbk::detail::mma_bf16_16816(C1, A, B1);
        }
      }
      // lane (g, t): head g at keys kb + 4t + {0,1} (C0) and {2,3} (C1); C[2], C[3] are the padding rows
      float sv[4] = {C0[0], C0[1], C1[0], C1[1]};
      float cm = -1e30f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#if CBK_ATTN_TC_MOVM
        const int key = kb + 2 * t + (i & 1) + 8 * (i >> 1);              // keys 2t, 2t+1, 8+2t, 9+2t
#else
        const int key = kb + 4 * t + i;
#endif
        sv[i] = (key < k1) ? sv[i] * a.attn_scale : -1e30f;
        cm = fmaxf(cm, sv[i]);
      }
      cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 1));
      cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
      const float nm = fmaxf(mrun, cm);
      const float corr = __expf(mrun - nm);
      float p[4], ps = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { p[i] = __expf(sv[i] - nm); ps += p[i]; }
      ps += __shfl_xor_sync(0xffffffffu, ps, 1);
      ps += __shfl_xor_sync(0xffffffffu, ps, 2);
      lrun = lrun * corr + ps;
      mrun = nm;
      // O^T accumulators hold heads 2t, 2t+1 (c0/c2, c1/c3): their corr lives in lanes 4*(2t), 4*(2t+1)
      const float ca = __shfl_sync(0xffffffffu, corr, (2 * t) << 2);
      const float cb = __shfl_sync(0xffffffffu, corr, (2 * t + 1) << 2);
#pragma unroll
      for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt)
        if (mt < DT) { oacc[mt][0] *= ca; oacc[mt][1] *= cb; oacc[mt][2] *= ca; oacc[mt][3] *= cb; }
      // ---- O^T += V^T . P^T : B = P^T (k-slots 2t,2t+1 <-> keys 4t,4t+1; 2t+8,2t+9 <-> 4t+2,4t+3)
      const uint32_t PB[2] = {pack_bf16x2(p[0], p[1]), pack_bf16x2(p[2], p[3])};
#if CBK_ATTN_TC_MOVM
      // V^T A-fragments by movmatrix: lane (g, t) loads V[key kb + 8*tau + g][dims 8*dl + 2t .. 2t+1] (one 4-byte
      // load, 8 keys x 16 B per warp instruction) and the 8x8 transpose hands it V^T[dim 8*dl + g][keys 2t, 2t+1]
      const __nv_bfloat16* vrA = a.vcache + kbase + (size_t)min(kb + g, pos) * D + 2 * t;
      const __nv_bfloat16* vrB = a.vcache + kbase + (size_t)min(kb + 8 + g, pos) * D + 2 * t;
#pragma unroll
      for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt) {
        if (mt < DT) {
          uint32_t A[4];
#if CBK_ATTN_TC_VPRE
          const uint32_t r0 = vraw[mt][0], r1 = vraw[mt][1], r2 = vraw[mt][2], r3 = vraw[mt][3];
          (void)vrA; (void)vrB;
#else
          const uint32_t r0 = *reinterpret_cast<const uint32_t*>(vrA + 16 * mt);
          const uint32_t r1 = *reinterpret_cast<const uint32_t*>(vrA + 16 * mt + 8);
          const uint32_t r2 = *reinterpret_cast<const uint32_t*>(vrB + 16 * mt);
          const uint32_t r3 = *reinterpret_cast<const uint32_t*>(vrB + 16 * mt + 8);
#endif
          asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(A[0]) : "r"(r0));
          asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(A[1]) : "r"(r1));
          asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(A[2]) : "r"(r2));
          asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(A[3]) : "r"(r3));
          cbk::detail::mma_bf16_16816(oacc[mt], A, PB);
        }
      }
#else
      const __nv_bfloat16* vr0 = a.vcache + kbase + (size_t)min(kb + 4 * t, pos) * D;
      const __nv_bfloat16* vr1 = a.vcache + kbase + (size_t)min(kb + 4 * t + 1, pos) * D;
      const __nv_bfloat16* vr2 = a.vcache + kbase + (size_t)min(kb + 4 * t + 2, pos) * D;
      const __nv_bfloat16* vr3 = a.vcache + kbase + (size_t)min(kb + 4 * t + 3, pos) * D;
#pragma unroll
      for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt) {
        if (mt < DT) {
          const int d0 = 16 * mt + g, d1 = d0 + 8;
          const uint32_t A[4] = {pack_u16x2(vr0 + d0, vr1 + d0), pack_u16x2(vr0 + d1, vr1 + d1),
                                 pack_u16x2(vr2 + d0, vr3 + d0), pack_u16x2(vr2 + d1, vr3 + d1)};
          cbk::detail::mma_bf16_16816(oacc[mt], A, PB);
        }
      }
#endif
    }
#if CBK_ATTN_TC_TREE
    // ---- merge the warps' states as a 3-level tree in registers (smem: 4 slots x KG x D f32 = 8 KB at
    // D=128/KG=4 instead of 8 slots; the dynamic smem decides the L1/smem carve-out on this part --
    // measured: the shipped build with its dynamic smem inflated to 18 KB loses 3 % on every GEMV phase).
    // Slot layout is the C-fragment layout itself, so a merge is a per-lane load + fma.
    {
      float* tb = sacc;                                     // [4][KG][D]  (aliases sq: dead after the walk)
      float* tm = smx;                                      // [4][KG]
      float* tl = sl;                                       // [4][KG]
#pragma unroll 1
      for (int lvl = CBK_WARPS / 2; lvl >= 1; lvl >>= 1) {
        __syncthreads();
        if (warp >= lvl && warp < 2 * lvl) {
          const int slot = warp - lvl;
#pragma unroll
          for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt)
            if (mt < DT) {
              const int d0 = 16 * mt + g;
              if (2 * t < KG) {
                tb[(slot * KG + 2 * t) * D + d0] = oacc[mt][0];
                tb[(slot * KG + 2 * t) * D + d0 + 8] = oacc[mt][2];
              }
              if (2 * t + 1 < KG) {
                tb[(slot * KG + 2 * t + 1) * D + d0] = oacc[mt][1];
                tb[(slot * KG + 2 * t + 1) * D + d0 + 8] = oacc[mt][3];
              }
            }
          if (t == 0 && g < KG) { tm[slot * KG + g] = mrun; tl[slot * KG + g] = lrun; }
        }
        __syncthreads();
        if (warp < lvl) {
          const int slot = warp;
          const float pm = (g < KG) ? tm[slot * KG + g] : -1e30f;
          const float pl = (g < KG) ? tl[slot * KG + g] : 0.f;
          const float nm = fmaxf(mrun, pm);
          const float co = __expf(mrun - nm), cn = __expf(pm - nm);
          lrun = lrun * co + pl * cn;
          mrun = nm;
          const float ca = __shfl_sync(0xffffffffu, co, (2 * t) << 2), cna = __shfl_sync(0xffffffffu, cn, (2 * t) << 2);
          const float cb = __shfl_sync(0xffffffffu, co, (2 * t + 1) << 2), cnb = __shfl_sync(0xffffffffu, cn, (2 * t + 1) << 2);
#pragma unroll
          for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt)
            if (mt < DT) {
              const int d0 = 16 * mt + g;
              const float o0 = (2 * t < KG) ? tb[(slot * KG + 2 * t) * D + d0] : 0.f;
              const float o2 = (2 * t < KG) ? tb[(slot * KG + 2 * t) * D + d0 + 8] : 0.f;
              const float o1 = (2 * t + 1 < KG) ? tb[(slot * KG + 2 * t + 1) * D + d0] : 0.f;
              const float o3 = (2 * t + 1 < KG) ? tb[(slot * KG + 2 * t + 1) * D + d0 + 8] : 0.f;
              oacc[mt][0] = oacc[mt][0] * ca + o0 * cna;
              oacc[mt][2] = oacc[mt][2] * ca + o2 * cna;
              oacc[mt][1] = oacc[mt][1] * cb + o1 * cnb;
              oacc[mt][3] = oacc[mt][3] * cb + o3 * cnb;
            }
        }
      }
    }
    PROF_ADD(a, 2);
    if (warp == 0) {                                        // warp 0 holds the block's state
      const float invl = 1.f / lrun;
      const float ia = __shfl_sync(0xffffffffu, invl, (2 * t) << 2), ib = __shfl_sync(0xffffffffu, invl, (2 * t + 1) << 2);
#pragma unroll
      for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt)
        if (mt < DT) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int hg = 2 * t + (e & 1), d = 16 * mt + g + 8 * (e >> 1);
            if (hg < KG) {
              const int h = kvh * KG + hg;
              if (SPB == 1) {
                const float ov = rb(oacc[mt][e] * ((e & 1) ? ib : ia));
                const int c0 = h * D + d;
                a.attn_out[(size_t)c0 * M + m] =
                    W.o.col_scale ? BF(ov * __half2float(W.o.col_scale[c0])) : BF(ov);
              } else {
                float* pt = a.partials + ((size_t)(m * H + h) * SPB + sp) * (D + 2);
                pt[d] = oacc[mt][e];
              }
            }
          }
        }
      if (SPB > 1 && t == 0 && g < KG) {
        float* pt = a.partials + ((size_t)(m * H + kvh * KG + g) * SPB + sp) * (D + 2);
        pt[D] = mrun; pt[D + 1] = lrun;
      }
    }
    __syncthreads();
    PROF_ADD(a, 3);
#elif CBK_ATTN_TC_HALF
    // ---- merge the warps' states (one round): each warp stores O/l as fp16 (|O/l| <= max|V|), m and l as f32
    __syncthreads();
    {
      const float invl = (lrun > 0.f) ? 1.f / lrun : 0.f;
      const float ia = __shfl_sync(0xffffffffu, invl, (2 * t) << 2), ib = __shfl_sync(0xffffffffu, invl, (2 * t + 1) << 2);
#pragma unroll
      for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt) {
        if (mt < DT) {
          const int d0 = 16 * mt + g;
          if (2 * t < KG) {
            sacc16[(warp * KG + 2 * t) * D + d0] = __float2half(oacc[mt][0] * ia);
            sacc16[(warp * KG + 2 * t) * D + d0 + 8] = __float2half(oacc[mt][2] * ia);
          }
          if (2 * t + 1 < KG) {
            sacc16[(warp * KG + 2 * t + 1) * D + d0] = __float2half(oacc[mt][1] * ib);
            sacc16[(warp * KG + 2 * t + 1) * D + d0 + 8] = __float2half(oacc[mt][3] * ib);
          }
        }
      }
    }
    if (t == 0 && g < KG) { smx[warp * KG + g] = mrun; sl[warp * KG + g] = lrun; }
    __syncthreads();
    PROF_ADD(a, 2);
    for (int o = threadIdx.x; o < KG * D; o += CBK_THREADS) {
      const int hg = o / D, d = o - hg * D;
      float gm = -1e30f, l = 0.f, acc = 0.f;
      for (int w = 0; w < CBK_WARPS; ++w) {
        const float pm = smx[w * KG + hg], nm = fmaxf(gm, pm);
        const float co = __expf(gm - nm), cn = __expf(pm - nm);
        const float pl = sl[w * KG + hg];
        l = l * co + pl * cn;
        acc = acc * co + __half2float(sacc16[(w * KG + hg) * D + d]) * (pl * cn);
        gm = nm;
      }
      const int h = kvh * KG + hg;
      if (SPB == 1) {
        const float ov = rb(acc / l);
        const int c0 = h * D + d;
        a.attn_out[(size_t)c0 * M + m] =
            W.o.col_scale ? BF(ov * __half2float(W.o.col_scale[c0])) : BF(ov);
      } else {
        float* pt = a.partials + ((size_t)(m * H + h) * SPB + sp) * (D + 2);
        pt[d] = acc;
        if (d == 0) { pt[D] = gm; pt[D + 1] = l; }
      }
    }
    __syncthreads();
    PROF_ADD(a, 3);
#else
    // ---- merge the warps' states: lane (g, t) holds heads 2t, 2t+1 at dims 16mt + g (+8)
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < CBK_MAXDPL * 2; ++mt) {
      if (mt < DT) {
        const int d0 = 16 * mt + g;
        if (2 * t < KG) {
          sacc[(warp * KG + 2 * t) * D + d0] = oacc[mt][0];
          sacc[(warp * KG + 2 * t) * D + d0 + 8] = oacc[mt][2];
        }
        if (2 * t + 1 < KG) {
          sacc[(warp * KG + 2 * t + 1) * D + d0] = oacc[mt][1];
          sacc[(warp * KG + 2 * t + 1) * D + d0 + 8] = oacc[mt][3];
        }
      }
    }
    if (t == 0 && g < KG) { smx[warp * KG + g] = mrun; sl[warp * KG + g] = lrun; }
    __syncthreads();
    PROF_ADD(a, 2);
    for (int o = threadIdx.x; o < KG * D; o += CBK_THREADS) {
      const int hg = o / D, d = o - hg * D;
      float gm = -1e30f, l = 0.f, acc = 0.f;
      for (int w = 0; w < CBK_WARPS; ++w) {
        const float pm = smx[w * KG + hg], nm = fmaxf(gm, pm);
        const float co = __expf(gm - nm), cn = __expf(pm - nm);
        l = l * co + sl[w * KG + hg] * cn;
        acc = acc * co + sacc[(w * KG + hg) * D + d] * cn;
        gm = nm;
      }
      const int h = kvh * KG + hg;
      if (SPB == 1) {
        const float ov = rb(acc / l);
        const int c0 = h * D + d;
        a.attn_out[(size_t)c0 * M + m] =
            W.o.col_scale ? BF(ov * __half2float(W.o.col_scale[c0])) : BF(ov);
      } else {
        float* pt = a.partials + ((size_t)(m * H + h) * SPB + sp) * (D + 2);
        pt[d] = acc;
        if (d == 0) { pt[D] = gm; pt[D + 1] = l; }
      }
    }
    __syncthreads();
    PROF_ADD(a, 3);
#endif  // CBK_ATTN_TC_TREE
#if CBK_ATTN_TC_REDUCE
    // ---- cross-split reduce by the LAST split block of this kv-head (no attn_reduce phase, no
    // grid.sync): partials -> fence -> per-kv-head arrival counter (a.mcnt, zero between phases).
    if (SPB > 1) {
      __threadfence();
      __syncthreads();
      __shared__ unsigned s_last;
      if (threadIdx.x == 0) s_last = atomicAdd(a.mcnt + (m * a.n_kv + kvh), 1u);
      __syncthreads();
      const bool last = (s_last == (unsigned)(SPB - 1));
      if (last) {
        __threadfence();
        if (threadIdx.x == 0) a.mcnt[m * a.n_kv + kvh] = 0u;
        for (int o = threadIdx.x; o < KG * D; o += CBK_THREADS) {
          const int hg = o / D, d = o - hg * D;
          const int h = kvh * KG + hg;
          const float* pb = a.partials + (size_t)(m * H + h) * SPB * (D + 2);
          float gm = -1e30f, l = 0.f, acc = 0.f;
          for (int q = 0; q < SPB; ++q) {
            const float* pq = pb + (size_t)q * (D + 2);
            const float pm = __ldcg(pq + D), nm = fmaxf(gm, pm);
            const float co = __expf(gm - nm), cn = __expf(pm - nm);
            l = l * co + __ldcg(pq + D + 1) * cn;
            acc = acc * co + __ldcg(pq + d) * cn;
            gm = nm;
          }
          const float ov = rb(acc / l);
          const int c0 = h * D + d;
          a.attn_out[(size_t)c0 * M + m] =
              W.o.col_scale ? BF(ov * __half2float(W.o.col_scale[c0])) : BF(ov);
        }
      }
      __syncthreads();
    }
#endif
  }
}
#endif  // CBK_ATTN_TC

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
  __shared__ float s_fss[CBK_MAXM];        // CBK_FUSE_RESID==2: per-block sum(h^2) of the fused residual
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
  // ---- per-step state in shared memory; `a` itself is never written ----
  __shared__ Step S;
  if (threadIdx.x == 0) {
    for (int m = 0; m < M; ++m) { S.tok[m] = a.tok_v[m]; S.pos[m] = a.pos_v[m]; }
    S.split = a.split;
  }
  __syncthreads();
  // ---- in-kernel greedy generation loop (n_steps == 1: the classic single step) ----
  for (int st = 0; st < a.n_steps; ++st) {
  if (st > 0) {
    if (threadIdx.x == 0) {
      int maxpos = 0;
      for (int m = 0; m < M; ++m) {
        S.tok[m] = a.amax_idx[(size_t)gridDim.x * M + m];   // published by block 0, after grid.sync
        S.pos[m] += 1;
        maxpos = max(maxpos, S.pos[m]);
      }
      S.split = min(a.split_max, max(1, (maxpos + 1 + a.kpb - 1) / a.kpb));   // == runner._split_for
    }
    __syncthreads();
  }
  {  // RoPE cos/sin for this step's positions, once.
    const int HD = a.head_dim / 2;
    for (int t = blockIdx.x * CBK_THREADS + threadIdx.x; t < 2 * M * HD;
         t += gridDim.x * CBK_THREADS) {
      const int j = t % HD, r = t / HD;
      const int m = r % M, lt = r / M;
      const float ang = (float)S.pos[m] * (lt ? a.inv_global[j] : a.inv_local[j]);
      a.rope_cs[t] = rb(cosf(ang));
      a.rope_sn[t] = rb(sinf(ang));
    }
  }
  float rr[M], rrd[M];
  for (int L = 0; L < a.n_layers; ++L) {
    const LayerW& W = a.layers[L];
    // 1. residual tail of the previous layer -> h
    const bool fused_h = CBK_FUSE_RESID && L > 0 && (a.layers[L - 1].post_ff_ln == nullptr);
    if (!fused_h) {
      if (L > 0) block_rms<M>(a.dbuf, hid, a.eps, s_red, rrd);
      phase_h<M>(a, S, L > 0 ? &a.layers[L - 1] : nullptr, L, rrd, s_red, part);
      grid.sync();
    }
    STAMP();
    // 2. input_layernorm + the column-scaled copies of x for q/k/v
    if (fused_h && CBK_FUSE_RESID == 1) block_rms<M>(a.h, hid, a.eps, s_red, rr);   // h from down's epilogue
    else                                rms_from_partials<M>(part, hid, a.eps, s_red, rr);
    write_xprime<M>(a.h, hid, W.in_ln, rr, W.qkv.col_scale ? 3 : 1, W.qkv.col_scale,
                    a.xbuf, a.norm_plus_one);
    grid.sync(); STAMP();
    // 3. fused qkv GEMV
    {
      PROF_T(a);
      __nv_bfloat16* out = a.qkv;
      const int ld = a.nqkv;
      const int hd = hid * M;
      const __nv_bfloat16* xb = CBK_XSTAGE(a.xbuf, (W.qkv.col_scale ? 3 : 1) * hd);
      const bool cs = (W.qkv.col_scale != nullptr);
      gemv_phase<M, 1, CBK_MMA_PF_L2, CBK_MMA_UPW>((float*)s_x,
          W.qkv, (int)W.qkv.K,
          [=] __device__(int r0, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
            const int t = cs ? cbk::qkv_cs_row(r0, nq, nk) : 0;
            *pa = xb + (size_t)t * hd; *pb = *pa;
          },
          [=] __device__(int r, int m, float v) { out[(size_t)m * ld + r] = BF(v); },
          /*pow2_only=*/cs, a.mpart, a.mcnt);
      PROF_ADD(a, 5);
    }
    grid.sync(); STAMP();
    // 4. q/k norm + RoPE + KV append (folded into phase 5 when CBK_FUSEPREP)
#if !CBK_FUSEPREP
    phase_attn_prep<M>(a, S, W, L);
    grid.sync();
#endif
    STAMP();
    // 5. attention
#if CBK_ATTN_TC
    phase_attn_tc<M, KG>(a, S, W, L, s_x);
#else
    phase_attn<M, KG>(a, S, W, L, s_x);
#endif
    grid.sync(); STAMP();
    if (S.split > 1 && !(CBK_ATTN_TC && CBK_ATTN_TC_REDUCE)) {
      phase_attn_reduce<M>(a, S, W, s_x);
      grid.sync();
    }
    STAMP();
    // 6. o_proj (its column scale was folded into the attention output write)
    {
      PROF_T(a);
      __nv_bfloat16* out = a.obuf;
      const __nv_bfloat16* xo = CBK_XSTAGE(a.attn_out, (size_t)nq * M);
      const __nv_bfloat16* hp = a.h;
      __nv_bfloat16* h2p = a.h2;
      const bool fo = CBK_FUSE_RESID && (W.post_attn_ln == nullptr);   // uniform
      float* fss = s_fss;
      if (CBK_FUSE_RESID == 2 && fo) { if (threadIdx.x < M) s_fss[threadIdx.x] = 0.f; __syncthreads(); }
      gemv_phase<M, 2, CBK_MMA_PF_L2, CBK_MMA_UPW_O>((float*)s_x,
          W.o, (int)W.o.K,
          [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
            *pa = xo; *pb = xo;
          },
          [=] __device__(int r, int m, float v) {
            if (fo) {
              const __nv_bfloat16 b = BF(F(hp[(size_t)m * hid + r]) + rb(v));
              h2p[(size_t)m * hid + r] = b;
              if (CBK_FUSE_RESID == 2) atomicAdd(&fss[m], F(b) * F(b));
            } else out[(size_t)m * hid + r] = BF(v);
          }, false, a.mpart, a.mcnt);
      if (CBK_FUSE_RESID == 2 && fo) {
        __syncthreads();
        if (threadIdx.x < M) part[(size_t)blockIdx.x * M + threadIdx.x] = s_fss[threadIdx.x];
      }
      PROF_ADD(a, 6);
    }
    grid.sync(); STAMP();
    // 7. h2 = h + post_attention_layernorm(o)   (fused into 6 for layers without the post-norm)
    const bool fused_h2 = CBK_FUSE_RESID && (W.post_attn_ln == nullptr);
    if (!fused_h2) {
      float rro[M];
      block_rms<M>(a.obuf, hid, a.eps, s_red, rro);
      phase_h2<M>(a, W, rro, s_red, part);
      grid.sync();
    }
    STAMP();
    // 8. pre_feedforward_layernorm + the column-scaled copies of x for gate/up
    if (fused_h2 && CBK_FUSE_RESID == 1) block_rms<M>(a.h2, hid, a.eps, s_red, rr);
    else                                 rms_from_partials<M>(part, hid, a.eps, s_red, rr);
    write_xprime<M>(a.h2, hid, W.pre_ff_ln, rr, W.gateup.col_scale ? 2 : 1,
                    W.gateup.col_scale, a.xbuf, a.norm_plus_one);
    grid.sync(); STAMP();
    // 9. gate/up + GeGLU; down_proj's column scale is folded into the write
    {
      PROF_T(a);
      __nv_bfloat16* out = a.act;
      const int ld = a.inter;
      const __nv_bfloat16* xg = CBK_XSTAGE(a.xbuf, (W.gateup.col_scale ? 2 : 1) * (size_t)hid * M);
      const __nv_bfloat16* xu = xg + (W.gateup.col_scale ? (size_t)hid * M : 0);
      const __half* dcs = W.down.col_scale;
      const int ag = a.act_gelu;
      // a.act is down_proj's activation, read straight from global -> interleaved.
      auto of = [=] __device__(int p, int m, float gv, float uv) {
        const float av = (ag ? rb(gelu_tanh(rb(gv))) : rb(silu_f(rb(gv)))) * rb(uv);
        out[(size_t)p * M + m] = dcs ? BF(rb(av) * __half2float(dcs[p])) : BF(rb(av));
      };
      (void)ld;
      const int RP = (CBK_GATEUP_RP != 0) ? CBK_GATEUP_RP
                                          : pick_R(a.inter, gridDim.x * CBK_WARPS, 2);
      (void)RP;
      bool gu_done = false;
#if (CBK_STREAM & 4) && CBK_BLK1632_ARM == 4
      if constexpr (M == 1)
      if (cbk::layout_is_blk1632((int)W.gateup.layout) && (a.inter & 1) == 0 &&
          (hid & (32 * CBK_STREAM_PF - 1)) == 0) {
        gu_done = true;
        cbk::detail::gemv_phase_stream<4, 2, CBK_STREAM_S, CBK_STREAM_PF, 4>(W.gateup, 2 * a.inter,
            [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
              *pa = xg; *pb = xu;
            },
            [=] __device__(int r0, const float* v) {
              if ((threadIdx.x & 31) == 0) { of(r0 >> 1, 0, v[0], v[1]); of((r0 >> 1) + 1, 0, v[2], v[3]); }
            }, a.mpart, a.mcnt, reinterpret_cast<uint8_t*>(s_x));
      }
#endif
#if (CBK_CHUNK & 4) && (CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6) && !(CBK_BLK1632_MMA && (CBK_MMA_PHASES & 4))
      if constexpr (M == 1)
      if (!gu_done && cbk::layout_is_blk1632((int)W.gateup.layout) && (a.inter & 1) == 0) {
        gu_done = true;
        // pairs of (gate, up) rows: R = 4 rows = 2 pairs per group, NX = 2 (odd rows read x_up)
        gemv_phase_chunk<M, 4, 2, 2>(W.gateup, 2 * a.inter,
            [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
              *pa = xg; *pb = xu;
            },
            [=] __device__(int r0, const float* v) {
              if ((threadIdx.x & 31) == 0) { of(r0 >> 1, 0, v[0], v[1]); of((r0 >> 1) + 1, 0, v[2], v[3]); }
            }, a.mpart, a.mcnt);
      }
#endif
#if CBK_BLK1632_MMA && (CBK_BLK1632_ARM == 4 || CBK_BLK1632_ARM == 6)
      if (!gu_done && (CBK_MMA_PHASES & 4) && M == 1 && cbk::layout_is_blk1632((int)W.gateup.layout) &&
          (hid & 127) == 0 && ((2 * a.inter) & 15) == 0) {
        gu_done = true;
        // interleaved gate/up rows: B column 0 = x_gate, column 1 = x_up; pair (2p, 2p+1) sits
        // in lanes (l, l+1) of the finishing warp
        auto gfin = [=] __device__(int row0, int lane, float v) {
              const float u = __shfl_down_sync(0xffffffffu, v, 1);
              if (lane < 16 && !(lane & 1)) of((row0 + lane) >> 1, 0, v, u);
            };
        auto gxf = [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
              *pa = xg; *pb = xu;
            };
        if (CBK_CHUNK & 4)
          cbk::detail::gemv_phase_mma_chunk<CBK_BLK1632_ARM, 2, CBK_MMA_PF>(W.gateup, 2 * a.inter, gxf, gfin,
                                                                           a.mpart, a.mcnt);
        else
          cbk::detail::gemv_phase_mma<CBK_BLK1632_ARM, 2, CBK_MMA_PF, CBK_MMA_UPW>(W.gateup, 2 * a.inter, gxf, gfin,
                                                                                  a.mpart, a.mcnt);
      }
#endif
      if (!gu_done) {
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
      }
      PROF_ADD(a, 7);
    }
    grid.sync(); STAMP();
    // 10. down_proj (column scale folded into a.act)
    {
      PROF_T(a);
      __nv_bfloat16* out = a.dbuf;
      const __nv_bfloat16* xd = CBK_XSTAGE(a.act, (size_t)a.inter * M);
      const __nv_bfloat16* h2p = a.h2;
      __nv_bfloat16* hp = a.h;
      const bool fd = CBK_FUSE_RESID && (W.post_ff_ln == nullptr);     // uniform
      float* fss = s_fss;
      if (CBK_FUSE_RESID == 2 && fd) { if (threadIdx.x < M) s_fss[threadIdx.x] = 0.f; __syncthreads(); }
      gemv_phase<M, 8, CBK_MMA_PF, CBK_MMA_UPW>((float*)s_x,
          W.down, (int)W.down.K,
          [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
            *pa = xd; *pb = xd;
          },
          [=] __device__(int r, int m, float v) {
            if (fd) {
              const __nv_bfloat16 b = BF(F(h2p[(size_t)m * hid + r]) + rb(v));
              hp[(size_t)m * hid + r] = b;
              if (CBK_FUSE_RESID == 2) atomicAdd(&fss[m], F(b) * F(b));
            } else out[(size_t)m * hid + r] = BF(v);
          }, false, a.mpart, a.mcnt);
      if (CBK_FUSE_RESID == 2 && fd) {
        __syncthreads();
        if (threadIdx.x < M) part[(size_t)blockIdx.x * M + threadIdx.x] = s_fss[threadIdx.x];
      }
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
  const bool fused_tail = CBK_FUSE_RESID && (a.layers[a.n_layers - 1].post_ff_ln == nullptr);
  if (!fused_tail) {
    block_rms<M>(a.dbuf, hid, a.eps, s_red, rrd);
    phase_h<M>(a, S, &a.layers[a.n_layers - 1], a.n_layers, rrd, s_red, part);
    grid.sync();
  }
  STAMP();
  if (fused_tail && CBK_FUSE_RESID == 1) block_rms<M>(a.h, hid, a.eps, s_red, rr);
  else                                   rms_from_partials<M>(part, hid, a.eps, s_red, rr);
  write_xprime<M>(a.h, hid, a.final_norm, rr, 1, nullptr, a.xbuf, a.norm_plus_one);
  grid.sync(); STAMP();
  {
    PROF_T(a);
    float* out = a.logits;
    const int V = a.vocab;
    const __nv_bfloat16* xb = CBK_XSTAGE(a.xbuf, (size_t)hid * M);
    gemv_phase<M, 16, CBK_MMA_PF, CBK_MMA_UPW>((float*)s_x,
        a.lm_head, V,
        [=] __device__(int, const __nv_bfloat16** pa, const __nv_bfloat16** pb) {
          *pa = xb; *pb = xb;
        },
        [=] __device__(int r, int m, float v) { out[(size_t)m * V + r] = v; }, false, a.mpart, a.mcnt);
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
    if (a.out_tokens) a.out_tokens[(size_t)st * M + m] = bi;
  }
  STAMP();
  if (a.n_steps > 1) grid.sync();   // publish this step's argmax to every block before it is consumed
  }  // for st
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

// KG = kv_group (query heads per kv head) is a TEMPLATE parameter of the attention phase,
// so the build instantiates exactly ONE ratio, CBK_KG (runner.py passes the model's:
// Gemma3 = 2, Mistral/Llama-3 = 4, MHA = 1).  Default 2 keeps the Gemma3 build identical.
// A runtime KG that does not match the build is a hard error, never a silent fallback
// (the old KG!=2 -> <1,1> fallback produced NaN logits on Mistral).
#ifndef CBK_KG
#define CBK_KG 2
#endif
static void* pick_kernel(int M, int KG) {
  if (KG != CBK_KG) {
    printf("[cobaltkernel] kv_group %d but the extension was built for CBK_KG=%d "
           "(set COBALT_KG or let runner.py pick it)\n", KG, CBK_KG);
    return nullptr;
  }
  switch (M) {
    case 1: return (void*)megakernel<1, CBK_KG, CBK_MINB>;
#ifndef CBK_M1ONLY
    case 2: return (void*)megakernel<2, CBK_KG, 2>;
    case 4: return (void*)megakernel<4, CBK_KG, 2>;
    case 8: return (void*)megakernel<8, CBK_KG, 1>;
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
