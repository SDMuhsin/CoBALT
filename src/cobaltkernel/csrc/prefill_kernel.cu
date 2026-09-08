// prefill_kernel.cu -- persistent cooperative CUDA kernel for the Gemma3 PREFILL pass.
//
//   ONE cooperative launch processes a whole prompt (M tokens of one sequence), fills the
//   decode KV cache and produces fp32 logits for a selected set of positions.
//
// Spec: docs/KERNELS.md    Oracle: src/cobaltkernel/{ref_gemma3,dequant_ref}.py
// KV contract: pf::kv_off() in prefill_kernel.cuh (shared with the decode megakernel)
// Every linear goes through cbk::gemm_tile (cobalt_gemm.cuh) at the
// recommended prefill configuration MT=128 / WM=2 / BR=256 / KT=128 (64 000 B smem,
// 1 block/SM).  Attention is a smem-tiled fp32 online-softmax phase (see phase_attn).
//
// Phases per layer, separated by grid.sync():
//   1 qkv GEMM (fused q|k|v, col_scale row picked from the weight row tile)
//   2 QK-norm + RoPE + KV-cache write
//   3 attention (causal, sliding-window, GQA)
//   4 o_proj GEMM
//   5 post_attention_layernorm + residual, then pre_feedforward_layernorm  (fused, 1 phase)
//   6 gate/up GEMM with the fused GeGLU epilogue
//   7 down GEMM
//   8 post_feedforward_layernorm + residual, then the NEXT layer's input_layernorm
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>

#include "prefill_kernel.cuh"
#include "cobalt_gemm.cuh"

namespace cg = cooperative_groups;

namespace pf {

// Tail-wave re-cut at BR/2.  Build with -DPF_TAILSPLIT=1 to enable.
#ifndef PF_TAILSPLIT
#define PF_TAILSPLIT 0
#endif

// ---- CoBALT-16:32 arm.  0 = OFF = the DENSE4 control build, bit-for-bit as originally
// shipped (the BLK path is not instantiated at all).
// 4 / 6 = the decoder GEMMs (qkv, o_proj, gateup, down_proj) read a BLK1632_4 / BLK1632_6
// matrix; the tied embedding / lm_head stays DENSE4 in EVERY arm, so it keeps BLK = 0.
#ifndef PF_BLK1632
#define PF_BLK1632 0
#endif
constexpr int PBLK = PF_BLK1632;

// ---- MIXED-LAYOUT arms.  The o_proj-hybrid artifacts
// (`b1632_4_ohyb`, `b1632_6_ohyb4`) keep `o_proj` on canonical global-top-k, i.e. DENSE4,
// while the other six matrices are BLK16_32.  PF_BLK1632_O is the BLK code width of the
// o_proj GEMM ONLY; it defaults to PF_BLK1632, so an unmixed build is unchanged (and when
// PF_BLK1632 == 0 the whole BLK path is still not instantiated).  0 = o_proj is DENSE4.
#ifndef PF_BLK1632_O
#define PF_BLK1632_O PF_BLK1632
#endif
constexpr int PBLKO = PF_BLK1632_O;

// ---- the one tile configuration used by every prefill GEMM (measured best of a
// (MT, WM, S, BR, KT) sweep; reproduce with test_gemm.py --roofline)
constexpr int PMT = 128, PBR = 256, PKT = 128, PWM = 1;   // WM=1 measured best at BR=256
constexpr int PBR2 = PBR;                                 // prefill: one tile for everything
constexpr int PS  = cbk::gt_stages<PMT>::value;
// b = 6 needs a full BYTE per column in the staged weight tile (gt_ws8), so its tile is
// 80 384 B instead of 64 000 -- still under the 101 376 B opt-in limit, still 1 block/SM.
constexpr int PSMEM = cbk::gemm_smem_bytes<PMT, PS, PBR, PKT>(PBLK == 6);

// ---- batched-decode tile (M = 8..32 rows): the measured best "M = 32" configuration,
// (MT, WM, BR, KT) = (32, 1, 128, 256) at minb = 2 -> 38 400 B, 2 blocks/SM.
#ifndef PF_BATCH_MINB
#define PF_BATCH_MINB 2
#endif
// BBR2 (qkv / o_proj / down_proj) is SMALLER than BBR (gateup / lm_head): at M = 32 those
// matrices have too few BR=128 row tiles to fill the 188-block grid (down_proj: 42 items
// for 188 blocks), and halving BR halves the phase time without adding weight traffic.
constexpr int BMT = 32, BBR = 128, BBR2 = 64, BKT = 256, BWM = 1;
constexpr int BS = cbk::gt_stages<BMT>::value;
constexpr int BSMEM = cbk::gemm_smem_bytes<BMT, BS, BBR, BKT>(PBLK == 6);  // 38 400 B (b=6: 54 784)

__device__ __forceinline__ cbk::Mat to_mat(const MatDesc& d) {
  cbk::Mat m;
  m.K = (int)d.K; m.N = (int)d.N; m.G = (int)d.G; m.layout = (int)d.layout;
  m.data = d.data; m.row_off = d.row_off; m.scale = d.scale; m.zero = d.zero;
  return m;
}

// --------------------------------------------------------------------- GEMM phase
// Work items = (weight-row tile, activation-row tile), distributed over blocks.
// `cs_split` != 0 selects the fused-qkv column-scale row from row0 (q|k|v boundaries are
// multiples of BR for every supported shape -- checked on the host).
template <int EPI, int TAIL = 0, int MT = PMT, int BR = PBR, int KT = PKT, int WM = PWM,
          int S = PS, int BLK = 0>
__device__ __forceinline__ void gemm_phase(const MatDesc& d, const __nv_bfloat16* X, int ldX,
                                           int M, void* Y, int ldY, uint8_t* smem,
                                           int cs_pairs, int nq, int nkv) {
  const cbk::Mat w = to_mat(d);
  const int K = w.K;
  const int nrt = ceildiv(K, BR), nmt = ceildiv(M, MT);
  const long long items = (long long)nrt * nmt;
  const int G = gridDim.x;
  // TAIL: the last partial wave (items % G tiles over G blocks) is re-cut at BR/2 so that
  // twice as many blocks share it -- the wave costs ~0.6x instead of 1x.  Same smem
  // (49 664 B <= 64 000) and fewer accumulator registers, so occupancy is unchanged.
  const int rem = (int)(items % G);
  // Only re-cut when the halves fit in ONE sub-wave (2*rem <= G); otherwise the split
  // costs a second sub-wave and loses (measured at M=1024).
  const bool split = TAIL && items > G && rem > 0 && 2 * rem <= G;
  const long long full = split ? items - rem : items;

  auto one = [&](long long it, int row0, int nrows, auto BRT) {
    constexpr int BRV = decltype(BRT)::value;
    const int mt = (int)(it % nmt);
    const __half* cs = d.col_scale;
    if (cs && nq > 0) {           // fused qkv: pick the col_scale row for this tile
      const int r = (row0 < nq) ? 0 : ((row0 < nq + nkv) ? 1 : 2);
      cs = cbk::cs_row_ptr(cs, r, w.N);
    }
    cbk::gemm_tile<MT, EPI, WM, S, BRV, KT, 1, BLK>(w, cs, cs_pairs, row0, nrows, X, ldX, M,
                                                    mt * MT, nullptr, nullptr, Y, ldY, smem);
  };

  for (long long it = blockIdx.x; it < full; it += G) {
    // M-tile INNER: the nmt blocks that share a weight row tile run concurrently, so the
    // weight bytes are fetched into L2 once instead of once per M-tile.
    const int rt = (int)(it / nmt);
    const int row0 = rt * BR;
    one(it, row0, min(BR, K - row0), cbk::detail::ic<BR>{});
  }
  if constexpr (TAIL != 0) {                 // `if constexpr`: never instantiate the
    if (split) {                             // BR/2 tile when the tail path is disabled
    const int ntail = rem;                              // < G/2, uniform across blocks
    for (int j = blockIdx.x; j < 2 * ntail; j += G) {
      const long long it = full + (j >> 1);
      const int rt = (int)(it / nmt);
      const int row0 = rt * BR + (j & 1) * (BR / 2);
      const int nrows = min(BR / 2, K - row0);
      if (nrows > 0) one(it, row0, nrows, cbk::detail::ic<BR / 2>{});
    }
    }
  }
}

// --------------------------------------------------------------- elementwise phases
// One WARP per activation row.  rr = rsqrt(mean(x^2) + eps) in fp32, then
// y = bf16(x * rr * (1 + w))  -- exactly Gemma3RMSNorm (see ref_gemma3.py::rms_norm).
// 8 bf16 (one 16-B vector) <-> 8 floats.  Every hidden size we support is a multiple of
// 256, so a warp covers a row in 16-B strides with no tail.
__device__ __forceinline__ void ld8(const __nv_bfloat16* p, float* o) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(q[i]);
    o[2 * i] = f.x; o[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void st8(__nv_bfloat16* p, const float* v) {
  uint4 o;
  __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = o;
}

__device__ __forceinline__ float warp_rsum(float ss) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  return ss;
}

__device__ __forceinline__ float row_rms(const __nv_bfloat16* x, int N, float eps) {
  const int lane = threadIdx.x & 31;
  float ss = 0.f, v[8];
  for (int t = lane; t < (N >> 3); t += 32) {
    ld8(x + t * 8, v);
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
  }
  return rsqrtf(warp_rsum(ss) / (float)N + eps);
}

__device__ __forceinline__ void phase_norm(const __nv_bfloat16* X, __nv_bfloat16* Y,
                                           const __nv_bfloat16* w, int M, int N, float eps) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int m = blockIdx.x * PF_WARPS + warp; m < M; m += gridDim.x * PF_WARPS) {
    const __nv_bfloat16* x = X + (size_t)m * N;
    const float rr = row_rms(x, N, eps);
    __nv_bfloat16* y = Y + (size_t)m * N;
    float v[8], g[8];
    for (int t = lane; t < (N >> 3); t += 32) {
      ld8(x + t * 8, v); ld8(w + t * 8, g);
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = v[i] * rr * (1.f + g[i]);
      st8(y + t * 8, v);
    }
  }
}

// h += rmsnorm(branch, wb);  then  out = rmsnorm(h, w2).   One warp per row, no barrier.
__device__ __forceinline__ void phase_addnorm(__nv_bfloat16* H, const __nv_bfloat16* Br,
                                              const __nv_bfloat16* wb, const __nv_bfloat16* w2,
                                              __nv_bfloat16* Y, int M, int N, float eps) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int NV = N >> 3;
  for (int m = blockIdx.x * PF_WARPS + warp; m < M; m += gridDim.x * PF_WARPS) {
    const __nv_bfloat16* b = Br + (size_t)m * N;
    __nv_bfloat16* h = H + (size_t)m * N;
    const float rr = row_rms(b, N, eps);
    float v[8], g[8], hv[8], ss = 0.f;
    for (int t = lane; t < NV; t += 32) {
      ld8(b + t * 8, v); ld8(wb + t * 8, g); ld8(h + t * 8, hv);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        hv[i] += rb(v[i] * rr * (1.f + g[i]));
        hv[i] = rb(hv[i]);                       // the residual add is a bf16 add
        ss += hv[i] * hv[i];
      }
      st8(h + t * 8, hv);
    }
    // the second RMS sum is accumulated during the first pass (one fewer read of h)
    const float rr2 = rsqrtf(warp_rsum(ss) / (float)N + eps);
    __nv_bfloat16* y = Y + (size_t)m * N;
    for (int t = lane; t < NV; t += 32) {
      ld8(h + t * 8, hv); ld8(w2 + t * 8, g);
#pragma unroll
      for (int i = 0; i < 8; ++i) hv[i] = hv[i] * rr2 * (1.f + g[i]);
      st8(y + t * 8, hv);
    }
  }
}

// Embedding lookup (DENSE4/DENSE8 CBK1 row) * bf16 embed_scale, then input_layernorm.
__device__ __forceinline__ void phase_embed(const PArgs& a) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int N = a.hidden;
  const __nv_bfloat16* w0 = a.layers[0].in_ln;
  for (int m = blockIdx.x * PF_WARPS + warp; m < a.M; m += gridDim.x * PF_WARPS) {
    const int tok = a.tokens[m];
    __nv_bfloat16* h = a.h + (size_t)m * N;
    for (int j = lane; j < N; j += 32)
      h[j] = PF_BF(rb(dense_elem(a.embed, tok, j)) * a.embed_scale);
    __syncwarp();
    const float rr = row_rms(h, N, a.eps);
    __nv_bfloat16* y = a.xn + (size_t)m * N;
    for (int j = lane; j < N; j += 32)
      y[j] = PF_BF(PF_F(h[j]) * rr * (1.f + PF_F(w0[j])));
  }
}

// cos/sin tables, fp32 values already rounded to bf16 (see ref_gemma3.py::_inv_freq).
__device__ __forceinline__ void phase_rope_tables(const PArgs& a) {
  const int HD = a.head_dim / 2;
  const long long n = (long long)a.M * HD;
  for (long long i = blockIdx.x * PF_THREADS + threadIdx.x; i < 2 * n;
       i += (long long)gridDim.x * PF_THREADS) {
    const int t = (int)(i / n);
    const int m = (int)((i % n) / HD), j = (int)(i % HD);
    const float inv = t ? a.inv_global[j] : a.inv_local[j];
    const float ang = (float)row_pos(a, m) * inv;
    a.rope_cs[i] = rb(cosf(ang));
    a.rope_sn[i] = rb(sinf(ang));
  }
}

// ------------------------------------------------ QK-norm + RoPE + KV-cache write
// One warp per (token, head).  Lane l owns head dims {l, l+32, ...}; the NeoX half-split
// partner (d +/- D/2) is index i +/- DPL/2 in the SAME lane (D/2 is a multiple of 32).
__device__ __forceinline__ void phase_rope(const PArgs& a, const PLayerW& W, int L) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, DPL = D / 32, HD = D / 2;
  const int nh = a.n_heads + a.n_kv;
  const long long items = (long long)a.M * nh;
  const int roff = W.is_sliding ? 0 : (a.M * HD);
  for (long long it = blockIdx.x * PF_WARPS + warp; it < items;
       it += (long long)gridDim.x * PF_WARPS) {
    const int m = (int)(it / nh), id = (int)(it % nh);
    const bool isq = id < a.n_heads;
    const int hh = isq ? id : (id - a.n_heads);
    const __nv_bfloat16* src = a.qkvb + (size_t)m * a.nqkv +
                               (isq ? hh * D : (a.nq_dim + hh * D));
    const __nv_bfloat16* nw = isq ? W.q_norm : W.k_norm;
    const float* cs = a.rope_cs + roff + (size_t)m * HD;
    const float* sn = a.rope_sn + roff + (size_t)m * HD;

    float x[PF_MAXDPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < PF_MAXDPL; ++i)
      if (i < DPL) { x[i] = PF_F(src[lane + 32 * i]); ss += x[i] * x[i]; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float rr = rsqrtf(ss / (float)D + a.eps);
#pragma unroll
    for (int i = 0; i < PF_MAXDPL; ++i)
      if (i < DPL) x[i] = rb(x[i] * rr * (1.f + PF_F(nw[lane + 32 * i])));
    float y[PF_MAXDPL];
#pragma unroll
    for (int i = 0; i < PF_MAXDPL; ++i)
      if (i < DPL) {
        const int d = lane + 32 * i;
        const int j = (d < HD) ? d : (d - HD);
        const int ip = (d < HD) ? (i + DPL / 2) : (i - DPL / 2);
        const float sgn = (d < HD) ? -1.f : 1.f;
        y[i] = rb(rb(x[i] * cs[j]) + rb(sgn * x[ip] * sn[j]));
      }
    if (isq) {
      __nv_bfloat16* q = a.qb + (size_t)m * a.nq_dim + hh * D;
#pragma unroll
      for (int i = 0; i < PF_MAXDPL; ++i) if (i < DPL) q[lane + 32 * i] = PF_BF(y[i]);
    } else {
      const size_t base =
          kv_off(L, a.kv_b, row_seq(a, m), a.n_kv, hh, a.max_ctx, row_pos(a, m), D);
      const __nv_bfloat16* vs = src + a.nkv_dim;
#pragma unroll
      for (int i = 0; i < PF_MAXDPL; ++i)
        if (i < DPL) { a.kcache[base + lane + 32 * i] = PF_BF(y[i]);
                       a.vcache[base + lane + 32 * i] = vs[lane + 32 * i]; }
    }
  }
}

// ------------------------------------------------------------------ attention
// Work item = (head, block of PF_WARPS*PF_QR = 32 query rows).  K/V tiles are staged in
// shared memory; each warp owns PF_QR query rows and runs a register online softmax.
// Bank layout: the smem row stride is D+2 halves = (D+2)/2 words, and (D+2)/2 % 32 == 1
// for D in {128, 256}, so the 32 lanes' K reads (one key per lane) are conflict free.
template <int KPL>            // keys per lane per tile: 2 for head_dim<=128, else 1
__device__ __forceinline__ void phase_attn(const PArgs& a, const PLayerW& W, int L,
                                           uint8_t* smem) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, DPL = D / 32, DS = D + 2;
  constexpr int ktile = 32 * KPL;                   // 64 keys (D=128) / 32 (D=256)
  const int QPB = PF_WARPS * PF_QR;                 // query rows per block

  __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Vs = Ks + ktile * DS;
  __nv_bfloat16* Qs = Vs + ktile * DS;              // [QPB][D]
  float* Ps = reinterpret_cast<float*>(Qs + QPB * D);   // [PF_WARPS][PF_QR][ktile]

  // staging map, computed ONCE (an integer div/mod per staged element used to cost more
  // than the whole inner loop): thread t owns the 8-half vector `dv0` of row `kl0`, and
  // 256 threads cover PF_THREADS/DV whole rows per pass.
  const int DV = D >> 3;                      // 16-B vectors per row
  const int rpp = PF_THREADS / DV;            // rows staged per pass (16 for D=128)
  const int kl0 = threadIdx.x / DV, dv0 = (threadIdx.x % DV) * 8;

  const int nqb = ceildiv(a.M, QPB);
  const long long items = (long long)a.n_heads * nqb;
  for (long long it = blockIdx.x; it < items; it += gridDim.x) {
    const int h = (int)(it % a.n_heads), qbi = (int)(it / a.n_heads);
    const int kvh = h / a.kv_group;
    const int mb = qbi * QPB;
    const int p_first = a.pos0 + mb;
    const int m_last = min(a.M - 1, mb + QPB - 1);
    const int kmax = a.pos0 + m_last + 1;
    const int kmin = W.is_sliding ? max(0, p_first - a.sliding_window + 1) : 0;
    const size_t kvbase = kv_off(L, a.kv_b, a.seq, a.n_kv, kvh, a.max_ctx, 0, D);

    __syncthreads();
    for (int r = kl0; r < QPB; r += rpp) {
      const int m = mb + r;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (m < a.M)
        v = *reinterpret_cast<const uint4*>(a.qb + (size_t)m * a.nq_dim + h * D + dv0);
      *reinterpret_cast<uint4*>(Qs + r * D + dv0) = v;
    }

    float acc[PF_QR][PF_MAXDPL], mx[PF_QR], ls[PF_QR];
#pragma unroll
    for (int r = 0; r < PF_QR; ++r) {
      mx[r] = -1e30f; ls[r] = 0.f;
#pragma unroll
      for (int i = 0; i < PF_MAXDPL; ++i) acc[r][i] = 0.f;
    }

    for (int kt0 = kmin; kt0 < kmax; kt0 += ktile) {
      __syncthreads();
      for (int kl = kl0; kl < ktile; kl += rpp) {
        const int kk = kt0 + kl;
        uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = kv;
        if (kk < kmax) {
          kv = *reinterpret_cast<const uint4*>(a.kcache + kvbase + (size_t)kk * D + dv0);
          vv = *reinterpret_cast<const uint4*>(a.vcache + kvbase + (size_t)kk * D + dv0);
        }
        // Ks/Vs rows are DS = D+2 halves apart, i.e. 4-B (not 16-B) aligned: store 4 words.
        uint32_t* kd = reinterpret_cast<uint32_t*>(Ks + kl * DS + dv0);
        uint32_t* vd = reinterpret_cast<uint32_t*>(Vs + kl * DS + dv0);
        kd[0] = kv.x; kd[1] = kv.y; kd[2] = kv.z; kd[3] = kv.w;
        vd[0] = vv.x; vd[1] = vv.y; vd[2] = vv.z; vd[3] = vv.w;
      }
      __syncthreads();

      // ---- scores: lane owns keys kt0 + lane + 32*c
      float s[PF_QR][KPL];
#pragma unroll
      for (int r = 0; r < PF_QR; ++r)
#pragma unroll
        for (int c = 0; c < KPL; ++c) s[r][c] = 0.f;
      // 8 head dims per step: the query rows come in as one LDS.128 each (Qs rows are
      // 16-B aligned), the keys as 4 LDS.32 (Ks rows are D+2 halves apart).
      for (int d = 0; d < D; d += 8) {
        uint32_t kk4[KPL][4];
#pragma unroll
        for (int c = 0; c < KPL; ++c) {
          const uint32_t* kp = reinterpret_cast<const uint32_t*>(&Ks[(lane + 32 * c) * DS + d]);
#pragma unroll
          for (int i = 0; i < 4; ++i) kk4[c][i] = kp[i];
        }
#pragma unroll
        for (int r = 0; r < PF_QR; ++r) {
          const uint4 qv = *reinterpret_cast<const uint4*>(&Qs[(warp * PF_QR + r) * D + d]);
          const uint32_t* qw = reinterpret_cast<const uint32_t*>(&qv);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 qf = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&qw[i]));
#pragma unroll
            for (int c = 0; c < KPL; ++c) {
              const float2 f = __bfloat1622float2(
                  *reinterpret_cast<const __nv_bfloat162*>(&kk4[c][i]));
              s[r][c] = fmaf(qf.x, f.x, fmaf(qf.y, f.y, s[r][c]));
            }
          }
        }
      }

      float corr[PF_QR];
#pragma unroll
      for (int r = 0; r < PF_QR; ++r) {
        const int m = mb + warp * PF_QR + r;
        const int pm = a.pos0 + m;
        const int lo = W.is_sliding ? (pm - a.sliding_window) : -1;   // kv > pm - window
        float lm = -1e30f;
#pragma unroll
        for (int c = 0; c < KPL; ++c) {
          const int kk = kt0 + lane + 32 * c;
          const bool ok = (m < a.M) && (kk < kmax) && (kk <= pm) && (kk > lo);
          s[r][c] = ok ? s[r][c] * a.attn_scale : -1e30f;
          lm = fmaxf(lm, s[r][c]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, off));
        const float nm = fmaxf(mx[r], lm);
        corr[r] = (mx[r] <= -1e29f) ? 0.f : __expf(mx[r] - nm);
        float ps = 0.f;
#pragma unroll
        for (int c = 0; c < KPL; ++c) {
          const float p = (s[r][c] <= -1e29f) ? 0.f : __expf(s[r][c] - nm);
          Ps[(warp * PF_QR + r) * ktile + lane + 32 * c] = p;
          ps += p;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
        ls[r] = ls[r] * corr[r] + ps;
        mx[r] = nm;
#pragma unroll
        for (int i = 0; i < PF_MAXDPL; ++i) acc[r][i] *= corr[r];
      }
      __syncwarp();

      // ---- P @ V.  V loads are hoisted out of the query-row loop.
      const int nk = min(ktile, kmax - kt0);
      for (int k = 0; k < nk; ++k) {
        float vv[PF_MAXDPL];
#pragma unroll
        for (int i = 0; i < PF_MAXDPL; ++i)
          if (i < DPL) vv[i] = PF_F(Vs[k * DS + lane + 32 * i]);
#pragma unroll
        for (int r = 0; r < PF_QR; ++r) {
          const float p = Ps[(warp * PF_QR + r) * ktile + k];
#pragma unroll
          for (int i = 0; i < PF_MAXDPL; ++i)
            if (i < DPL) acc[r][i] = fmaf(p, vv[i], acc[r][i]);
        }
      }
      __syncwarp();
    }

#pragma unroll
    for (int r = 0; r < PF_QR; ++r) {
      const int m = mb + warp * PF_QR + r;
      if (m >= a.M || ls[r] <= 0.f) continue;
      const float inv = 1.f / ls[r];
      __nv_bfloat16* o = a.ao + (size_t)m * a.nq_dim + h * D;
#pragma unroll
      for (int i = 0; i < PF_MAXDPL; ++i)
        if (i < DPL) o[lane + 32 * i] = PF_BF(acc[r][i] * inv);
    }
  }
  __syncthreads();
}

// -------------------------------------------- batched decode attention (one step)
// Work item = (sequence m, kv head kvh) -> ONE BLOCK; the block's 8 warps split that
// sequence's key range and combine through shared memory (no cross-block reduce).  The
// block serves ALL kv_group query heads of that kv head, so every K/V byte is read once
// per (sequence, kv head) instead of once per query head.
// Lane l owns the CONTIGUOUS head dims [l*DPL, (l+1)*DPL), so a lane's K/V/Q read is one
// 8-B (D=128) or 16-B (D=256) vector and a warp's read is one 256-B / 512-B line.
__device__ __forceinline__ void phase_attn_batch(const PArgs& a, const PLayerW& W, int L,
                                                 uint8_t* smem) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int D = a.head_dim, DPL = D / 32, GQ = a.kv_group;
  float* sacc = reinterpret_cast<float*>(smem);          // [PF_WARPS][D]
  float* smx = sacc + PF_WARPS * D;                      // [PF_WARPS]
  float* ssum = smx + PF_WARPS;                          // [PF_WARPS]

  auto ldvec = [&](const __nv_bfloat16* p, float* o) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(p + lane * DPL);
#pragma unroll
    for (int j = 0; j < PF_MAXDPL / 2; ++j)
      if (2 * j < DPL) {
        const uint32_t v = w[j];
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
        o[2 * j] = f.x; o[2 * j + 1] = f.y;
      }
  };

  const long long items = (long long)a.M * a.n_kv;
  for (long long it = blockIdx.x; it < items; it += gridDim.x) {
    const int m = (int)(it / a.n_kv), kvh = (int)(it % a.n_kv);
    const int pos = a.positions[m];
    const int lo = W.is_sliding ? max(0, pos - a.sliding_window + 1) : 0;
    const int S = pos + 1 - lo;
    const int per = ceildiv(S, PF_WARPS);
    const int w0 = lo + warp * per, w1 = min(pos + 1, w0 + per);
    const size_t kvbase = kv_off(L, a.kv_b, m, a.n_kv, kvh, a.max_ctx, 0, D);

    for (int g0 = 0; g0 < GQ; g0 += PF_MAXGQ) {
      const int ng = min(PF_MAXGQ, GQ - g0);
      float q[PF_MAXGQ][PF_MAXDPL], acc[PF_MAXGQ][PF_MAXDPL], mx[PF_MAXGQ], ls[PF_MAXGQ];
#pragma unroll
      for (int g = 0; g < PF_MAXGQ; ++g) {
        mx[g] = -1e30f; ls[g] = 0.f;
#pragma unroll
        for (int i = 0; i < PF_MAXDPL; ++i) { q[g][i] = 0.f; acc[g][i] = 0.f; }
        if (g < ng)
          ldvec(a.qb + (size_t)m * a.nq_dim + (size_t)(kvh * GQ + g0 + g) * D, q[g]);
      }
      // One key per iteration.  A 2-key unroll (to interleave the two shfl reduction
      // chains) was measured: attention 27.5 -> 26.0 ms at M=32, but it pushes the
      // kernel's spills 128 -> 332 B and the GEMM phases pay more than attention gains
      // (486 -> 464 output tok/s).  Reverted.
      for (int kk = w0; kk < w1; ++kk) {
        float kv[PF_MAXDPL];
        ldvec(a.kcache + kvbase + (size_t)kk * D, kv);
        float sc[PF_MAXGQ];
#pragma unroll
        for (int g = 0; g < PF_MAXGQ; ++g) {
          float t = 0.f;
#pragma unroll
          for (int i = 0; i < PF_MAXDPL; ++i) if (i < DPL) t = fmaf(q[g][i], kv[i], t);
          sc[g] = t;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
          for (int g = 0; g < PF_MAXGQ; ++g) sc[g] += __shfl_xor_sync(0xffffffffu, sc[g], off);
        float vv[PF_MAXDPL];
        ldvec(a.vcache + kvbase + (size_t)kk * D, vv);
#pragma unroll
        for (int g = 0; g < PF_MAXGQ; ++g) {
          const float sv = sc[g] * a.attn_scale;
          const float nm = fmaxf(mx[g], sv);
          const float corr = (mx[g] <= -1e29f) ? 0.f : __expf(mx[g] - nm);
          const float pr = __expf(sv - nm);
          ls[g] = ls[g] * corr + pr;
          mx[g] = nm;
#pragma unroll
          for (int i = 0; i < PF_MAXDPL; ++i)
            if (i < DPL) acc[g][i] = fmaf(pr, vv[i], acc[g][i] * corr);
        }
      }
      // ---- combine the 8 warps' partials, one query head at a time
#pragma unroll
      for (int g = 0; g < PF_MAXGQ; ++g) {
        if (g >= ng) break;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < PF_MAXDPL; ++i)
          if (i < DPL) sacc[warp * D + lane * DPL + i] = acc[g][i];
        if (lane == 0) { smx[warp] = mx[g]; ssum[warp] = ls[g]; }
        __syncthreads();
        if (warp == 0) {
          float gm = -1e30f;
#pragma unroll
          for (int w = 0; w < PF_WARPS; ++w) gm = fmaxf(gm, smx[w]);
          float l = 0.f, aa[PF_MAXDPL];
#pragma unroll
          for (int i = 0; i < PF_MAXDPL; ++i) aa[i] = 0.f;
#pragma unroll
          for (int w = 0; w < PF_WARPS; ++w) {
            const float c = (smx[w] <= -1e29f) ? 0.f : __expf(smx[w] - gm);
            l += ssum[w] * c;
#pragma unroll
            for (int i = 0; i < PF_MAXDPL; ++i)
              if (i < DPL) aa[i] = fmaf(sacc[w * D + lane * DPL + i], c, aa[i]);
          }
          const float inv = (l > 0.f) ? (1.f / l) : 0.f;
          __nv_bfloat16* o = a.ao + (size_t)m * a.nq_dim + (size_t)(kvh * GQ + g0 + g) * D;
#pragma unroll
          for (int i = 0; i < PF_MAXDPL; ++i)
            if (i < DPL) o[lane * DPL + i] = PF_BF(aa[i] * inv);
        }
      }
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------ the kernel
// The whole model, shared by both entry points.  BATCH=0: M rows are consecutive tokens
// of one sequence (prefill).  BATCH=1: M rows are M different sequences, one step each.
// The tile shape is a template parameter because the two entry points carry different
// __launch_bounds__ (1 vs 2 blocks/SM) and therefore different register budgets.
template <int BATCH, int MT, int BR, int BR2, int KT, int WM, int S, int TAIL, int BLK = 0,
          int BLKO = BLK>
__device__ __forceinline__ void run_model(const PArgs& a, uint8_t* smem) {
  cg::grid_group grid = cg::this_grid();
  int ts = 0;
#define PF_STAMP()                                                     \
  do { if (a.timings && blockIdx.x == 0 && threadIdx.x == 0)           \
         a.timings[ts] = clock64();                                    \
       ++ts; } while (0)

  PF_STAMP();
  phase_rope_tables(a);
  phase_embed(a);
  grid.sync();
  PF_STAMP();

  for (int L = 0; L < a.n_layers; ++L) {
    const PLayerW& W = a.layers[L];
    // 1. fused qkv
    gemm_phase<cbk::EPI_BF16, TAIL, MT, BR2, KT, WM, S, BLK>(W.qkv, a.xn, a.hidden, a.M, a.qkvb, a.nqkv, smem, 0,
                              a.nq_dim, a.nkv_dim);
    grid.sync(); PF_STAMP();
    // 2. QK-norm + RoPE + KV write
    phase_rope(a, W, L);
    grid.sync(); PF_STAMP();
    // 3. attention
    if (BATCH) {
      phase_attn_batch(a, W, L, smem);
    } else if (a.head_dim <= 128) {
      phase_attn<2>(a, W, L, smem);
    } else {
      phase_attn<1>(a, W, L, smem);
    }
    grid.sync(); PF_STAMP();
    // 4. o_proj
    gemm_phase<cbk::EPI_BF16, 0, MT, BR2, KT, WM, S, BLKO>(W.o, a.ao, a.nq_dim, a.M, a.ob, a.hidden, smem, 0, 0, 0);
    grid.sync(); PF_STAMP();
    // 5. residual + post_attention_layernorm, then pre_feedforward_layernorm
    phase_addnorm(a.h, a.ob, W.post_attn_ln, W.pre_ff_ln, a.xn, a.M, a.hidden, a.eps);
    grid.sync(); PF_STAMP();
    // 6. gate/up + GeGLU
    gemm_phase<cbk::EPI_GEGLU_BF16, 0, MT, BR, KT, WM, S, BLK>(W.gateup, a.xn, a.hidden, a.M, a.act, a.inter, smem, 1,
                                    0, 0);
    grid.sync(); PF_STAMP();
    // 7. down
    gemm_phase<cbk::EPI_BF16, 0, MT, BR2, KT, WM, S, BLK>(W.down, a.act, a.inter, a.M, a.db, a.hidden, smem, 0, 0, 0);
    grid.sync(); PF_STAMP();
    // 8. residual + post_feedforward_layernorm, then the next input_layernorm
    //    (the final model norm for the last layer)
    phase_addnorm(a.h, a.db, W.post_ff_ln,
                  (L + 1 < a.n_layers) ? a.layers[L + 1].in_ln : a.final_norm,
                  a.xn, a.M, a.hidden, a.eps);
    grid.sync(); PF_STAMP();
  }

  // ---- gather the requested rows of the (already normalised) final hidden state
  {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * PF_WARPS + warp; r < a.R; r += gridDim.x * PF_WARPS) {
      const int m = a.logit_rows[r];
      const __nv_bfloat16* s = a.xn + (size_t)m * a.hidden;
      __nv_bfloat16* d = a.hg + (size_t)r * a.hidden;
      for (int j = lane; j < a.hidden; j += 32) d[j] = s[j];
    }
  }
  grid.sync(); PF_STAMP();
  gemm_phase<cbk::EPI_F32, 0, MT, BR, KT, WM, S>(a.embed, a.hg, a.hidden, a.R, a.logits, a.vocab, smem, 0, 0, 0);
  grid.sync(); PF_STAMP();
#undef PF_STAMP
}


// ---- entry points.  Two separate __global__ functions so that the batched step can ask
// for 2 blocks/SM (small tile) without forcing the prefill
// path -- which needs all 255 registers for the 128x256 tile -- to fit the same budget.
__global__ __launch_bounds__(PF_THREADS, 1) void prefill_kernel(PArgs a) {
  extern __shared__ uint8_t smem[];
  run_model<0, PMT, PBR, PBR2, PKT, PWM, PS, PF_TAILSPLIT, PBLK, PBLKO>(a, smem);
}

__global__ __launch_bounds__(PF_THREADS, PF_BATCH_MINB) void batch_kernel(PArgs a) {
  extern __shared__ uint8_t smem[];
  run_model<1, BMT, BBR, BBR2, BKT, BWM, BS, 0, PBLK, PBLKO>(a, smem);
}

// ------------------------------------------------------------------ host launcher
int pf_smem_bytes(int head_dim) {
  const int DS = head_dim + 2, ktile = 8192 / head_dim;
  const int att = 2 * ktile * DS * 2 + PF_WARPS * PF_QR * head_dim * 2
                + PF_WARPS * PF_QR * ktile * 4;
  return att > PSMEM ? att : PSMEM;
}

// Batched-step smem: the small gemm tile, or the attention combine buffer, whichever is
// bigger.  [PF_WARPS][head_dim] floats + 2*PF_WARPS.
int pf_smem_bytes_batch(int head_dim) {
  const int att = (PF_WARPS * head_dim + 2 * PF_WARPS) * 4;
  return att > BSMEM ? att : BSMEM;
}

int pf_plan_batch(int smem_bytes, int* blocks_out) {
  void* k = (void*)batch_kernel;
  if (smem_bytes > 48 * 1024)
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) !=
        cudaSuccess)
      return -1;
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, PF_THREADS, smem_bytes) !=
      cudaSuccess)
    return -2;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  *blocks_out = nb * sms;
  return nb;
}

int pf_launch_batch(PArgs a, int blocks, int smem_bytes, cudaStream_t stream) {
  void* k = (void*)batch_kernel;
  void* args[] = {(void*)&a};
  cudaError_t e = cudaLaunchCooperativeKernel(k, dim3(blocks), dim3(PF_THREADS), args,
                                              smem_bytes, stream);
  if (e != cudaSuccess) {
    printf("[cobaltkernel-prefill] batch launch failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

int pf_plan(int smem_bytes, int* blocks_out) {
  void* k = (void*)prefill_kernel;
  if (smem_bytes > 48 * 1024)
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) !=
        cudaSuccess)
      return -1;
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, PF_THREADS, smem_bytes) !=
      cudaSuccess)
    return -2;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  *blocks_out = nb * sms;
  return nb;
}

int pf_launch(PArgs a, int blocks, int smem_bytes, cudaStream_t stream) {
  void* k = (void*)prefill_kernel;
  void* args[] = {(void*)&a};
  cudaError_t e = cudaLaunchCooperativeKernel(k, dim3(blocks), dim3(PF_THREADS), args,
                                              smem_bytes, stream);
  if (e != cudaSuccess) {
    printf("[cobaltkernel-prefill] launch failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace pf
