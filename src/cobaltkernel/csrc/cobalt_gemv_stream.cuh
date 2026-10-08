// cobalt_gemv_stream.cuh -- the BLK16_32 row decoder fed by a cp.async ring (M == 1 decode).
//
// The shipped walk (gemv_blk1632_multi) loads a granule batch into registers, decodes it, then loads
// the next batch: a warp alternates [memory latency][issue-bound decode], and the two never overlap
// inside one warp.  Measured on the 2g slice: gate|up streams at 2.05 ms with the decode deleted
// (LOADONLY) and decodes in 2.18 ms with the weight bytes deleted (NOLD), but takes 2.59 ms with
// both -- the sum, not the max.  Here the weight bytes of iteration k+S-1 are copied global->shared
// by cp.async (no registers, asynchronous) while iteration k is decoded from shared memory, so the
// stream runs S-1 iterations ahead of the decoder at all times and the phase can approach
// max(memory, issue).
//
// Work is a contiguous equal share per warp of the (row-group, granule) items (same accounting as
// gemv_phase_chunk): a warp's run may start and end mid-row; a group split between warps is summed
// through f32 atomics and finished by its last-arriving warp.  Every lane owns one granule per
// iteration (32 items per iteration, PF of them per lane), decodes R rows of it, and accumulates into
// acc (the iteration's first group) or acc2 (the next one) -- at most one group boundary falls inside
// an iteration because NT >= 32*PF.
#pragma once
#include "cobalt_gemv.cuh"
#include "megakernel.cuh"

#ifndef CBK_STREAM
#define CBK_STREAM 0        // bitmask of phases on the cp.async ring (1 qkv 2 o 4 gate|up 8 down 16 lm_head)
#endif
#ifndef CBK_STREAM_S
#define CBK_STREAM_S 3      // ring depth (iterations in flight = S-1)
#endif
#ifndef CBK_STREAM_PF
#define CBK_STREAM_PF 1     // granules per lane per iteration
#endif

namespace cbk {
namespace detail {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async_4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// bytes of ring per warp: S slots x PF x R x 32 lanes x (4 mask + 8 nibble)
template <int R, int S, int PF>
struct StreamRing {
  static constexpr int MASK_BYTES = S * PF * R * 32 * 4;
  static constexpr int NIB_BYTES = S * PF * R * 32 * 8;
  static constexpr int BYTES = MASK_BYTES + NIB_BYTES;
  __device__ static __forceinline__ uint32_t* mask(uint8_t* ring, int slot, int s, int i, int lane) {
    return reinterpret_cast<uint32_t*>(ring) + ((slot * PF + s) * R + i) * 32 + lane;
  }
  __device__ static __forceinline__ uint2* nib(uint8_t* ring, int slot, int s, int i, int lane) {
    return reinterpret_cast<uint2*>(ring + MASK_BYTES) + ((slot * PF + s) * R + i) * 32 + lane;
  }
};

// fin(r0, v): called by ALL lanes with the complete sums v[R] of rows r0 .. r0+R-1.
template <int R, int NX, int S, int PF, int BITS, class XF, class Fin>
__device__ __forceinline__ void gemv_phase_stream(const MatDesc& d, int K, XF xf, Fin fin,
                                                  float* __restrict__ mpart, unsigned* __restrict__ mcnt,
                                                  uint8_t* __restrict__ ring_all) {
  static_assert(CBK_BLK1632_ZF && CBK_BLK1632_BIAS, "the stream decoder assumes the shipped ZF + BIAS form");
  static_assert(BITS == 4, "4-bit planes only");
  using Ring = StreamRing<R, S, PF>;
  constexpr int IT = 32 * PF;                     // items per iteration
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  uint8_t* ring = ring_all + (size_t)warp * Ring::BYTES;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const size_t noff = (size_t)(N >> 3);
  const int NT = N >> 5;
  const int ngrp = K / R;
  const long items = (long)ngrp * NT;
  const long C = (items + GW - 1) / GW;
  const long beg = (long)gw * C;
  const long end = min(items, beg + C);
#if CBK_BLK1632_LUT == 2 || CBK_BLK1632_LUT == 4
  const uint32_t* __restrict__ lut = prmt_smem();
#else
  const uint32_t* __restrict__ lut = prmt_lut;
#endif
  if (beg >= end) return;
  const int nit = (int)((end - beg + IT - 1) / IT);

  // ---- producer: the weight bytes of iteration k into slot k % S
  auto issue = [&](int k) {
    if (k < nit) {
      const int slot = k % S;
#pragma unroll
      for (int s = 0; s < PF; ++s) {
        const long item = beg + (long)k * IT + lane + 32 * s;
        if (item < end) {
          const int g = (int)(item / NT), ts = (int)(item - (long)g * NT);
          const uint8_t* rb = d.data + (size_t)(g * R) * rs;
#pragma unroll
          for (int i = 0; i < R; ++i) {
            cp_async_4(Ring::mask(ring, slot, s, i, lane), rb + (size_t)i * rs + ((size_t)ts << 2));
            cp_async_8(Ring::nib(ring, slot, s, i, lane), rb + (size_t)i * rs + noff + ((size_t)ts << 3));
          }
        }
      }
    }
    cp_async_commit();                            // always: one group per iteration keeps the count uniform
  };
#pragma unroll
  for (int k = 0; k < S - 1; ++k) issue(k);

  float acc[R], acc2[R];
#pragma unroll
  for (int i = 0; i < R; ++i) { acc[i] = 0.f; acc2[i] = 0.f; }
  // scale / zero of this lane's items, prefetched one iteration ahead
  __half scn[PF][R]; uint8_t zrn[PF][R];
  auto load_sz = [&](int k) {
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      const long item = beg + (long)k * IT + lane + 32 * s;
      const bool ok = (k < nit) && (item < end);
      const int g = ok ? (int)(item / NT) : 0;
      const int ts = ok ? (int)(item - (long)g * NT) : 0;
#pragma unroll
      for (int i = 0; i < R; ++i) {
        scn[s][i] = ok ? d.scale[(size_t)(g * R + i) * G + (ts >> 2)] : __float2half(0.f);
        zrn[s][i] = ok ? d.zero[(size_t)(g * R + i) * G + (ts >> 2)] : (uint8_t)0;
      }
    }
  };
  load_sz(0);

  auto finish = [&](int g, float* v) {
    // warp-reduce, then emit (full group) or accumulate + last-arriver (split group)
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
      for (int dd = 16; dd > 0; dd >>= 1) v[i] += __shfl_xor_sync(0xffffffffu, v[i], dd);
    const long gb = (long)g * NT, ge = gb + NT - 1;
    const int r0 = g * R;
    if (beg <= gb && ge < end) {
      fin(r0, v);
    } else {
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
  };

  for (int k = 0; k < nit; ++k) {
    issue(k + S - 1);
    cp_async_wait<S - 1>();                       // iteration k's copies have landed (this lane's)
    __half sc[PF][R]; uint8_t zr[PF][R];
#pragma unroll
    for (int s = 0; s < PF; ++s)
#pragma unroll
      for (int i = 0; i < R; ++i) { sc[s][i] = scn[s][i]; zr[s][i] = zrn[s][i]; }
    load_sz(k + 1);
    const long pos0 = beg + (long)k * IT;
    const int g0 = (int)(pos0 / NT);
    const int slot = k % S;
    const __nv_bfloat16 *xa = nullptr, *xb = nullptr;
    xf(g0 * R, &xa, &xb);                         // the same x' serves every group of the phase
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      const long item = pos0 + lane + 32 * s;
      const bool ok = item < end;
      const int g = ok ? (int)(item / NT) : g0;
      const int ts = ok ? (int)(item - (long)g * NT) : 0;
      uint32_t mw[R]; uint64_t buf[R]; uint32_t zbv[R];
#pragma unroll
      for (int i = 0; i < R; ++i) {
        mw[i] = *Ring::mask(ring, slot, s, i, lane);
        const uint2 nv = *Ring::nib(ring, slot, s, i, lane);
        buf[i] = (((uint64_t)nv.y) << 32) | (uint64_t)nv.x;
        zbv[i] = (uint32_t)zr[s][i] * 0x01010101u;
      }
      float qs[R][1], xs[R][1], xsa[1] = {0.f}, xsb[1] = {0.f};
#pragma unroll
      for (int i = 0; i < R; ++i) { qs[i][0] = 0.f; xs[i][0] = 0.f; }
      const int col = ts << 5;
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint4 va[1], vb[1];
        va[0] = *reinterpret_cast<const uint4*>(xa + (size_t)(col + (q4 << 3)));
        if (NX == 2) vb[0] = *reinterpret_cast<const uint4*>(xb + (size_t)(col + (q4 << 3)));
        const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(va);
        const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(vb);
        float fa[8], fb[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          fa[j] = __bfloat162float(ha[j]); xsa[0] += fa[j];
          if (NX == 2) { fb[j] = __bfloat162float(hb[j]); xsb[0] += fb[j]; }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
          if (NX == 2 && (i & 1))
            blk1632_q4<1, BITS, true, true>(mw[i], buf[i], 0u, q4, fa, fb, qs[i], xs[i], lut, zbv[i]);
          else
            blk1632_q4<1, BITS, false, true>(mw[i], buf[i], 0u, q4, fa, fb, qs[i], xs[i], lut, zbv[i]);
        }
      }
      const bool first = (g == g0);
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const float sf = __half2float(sc[s][i]);
        const float zf = (float)zr[s][i] + 128.f;
        const float xsum = (NX == 2 && (i & 1)) ? xsb[0] : xsa[0];
        const float val = ok ? sf * (qs[i][0] - zf * xsum) : 0.f;
        acc[i] += first ? val : 0.f;
        acc2[i] += first ? 0.f : val;
      }
    }
    // ---- group bookkeeping for this iteration
    const long last = min(pos0 + IT - 1, end - 1);
    const int gl = (int)(last / NT);
    if (gl > g0) {                                // a boundary inside: g0 is complete for this warp
      finish(g0, acc);
#pragma unroll
      for (int i = 0; i < R; ++i) { acc[i] = acc2[i]; acc2[i] = 0.f; }
    }
    if (last == (long)(gl + 1) * NT - 1 || last == end - 1) {
      finish(gl, acc);
#pragma unroll
      for (int i = 0; i < R; ++i) acc[i] = 0.f;
    }
  }
  cp_async_wait<0>();
}

}  // namespace detail
}  // namespace cbk
