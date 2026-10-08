// cobalt_gemv_mma.cuh -- the BLK16_32 decode on TENSOR CORES (M == 1 decode).
//
// The shipped decoder (gemv_blk1632_multi) spends ~230 SASS per 32-column row-granule: the
// 16:32 expansion to 8 code bytes per bitmap byte is ~22 integer ops, then every one of the 8
// positions costs a biased PRMT + an FFMA, and every granule converts 16 bf16 activations to
// f32.  Here the SAME expansion feeds mma.m16n8k16 (bf16 x bf16 -> f32):
//   * code bytes -> bf16 pairs by ONE PRMT per pair (0x43 high byte: bf16 0x43cc == 128 + cc
//     exactly, so the 128 bias of CBK_BLK1632_BIAS carries over unchanged),
//   * the activation is loaded as bf16 pairs straight into the B fragment (no conversion),
//   * the 32 MACs per row-granule are done by the tensor core,
//   * sum(x') per group, needed to cancel the bias and the ZF zero-fill, comes from a second
//     mma with a constant all-ones A fragment, so no prep phase changes.
// Fragment mapping (m16n8k16, lane = 4g + t): the warp owns a 16-row tile; lane (g, t) decodes
// rows g and g+8 of block 4j+t of every 128-column group j.  The k index of the mma is a
// permutation of the real columns, which is free as long as A and B agree: k-slots (2t, 2t+1)
// <-> positions 4*tau + {0,1} and (2t+8, 2t+9) <-> 4*tau + {2,3} of block 4j+t, with
// tau = 2*q4 + half the mma tile index (8 tiles per group).  B column n == 0 carries x_a and
// n == 1 carries x_b (gate | up: row parity == g parity), the other six columns are ignored.
// Work units are (tile, column slice); slices of one tile are summed through f32 atomics and
// the last-arriving warp runs the epilogue, so small-K phases no longer pay a whole wave for a
// few leftover rows.
#pragma once
#include "cobalt_gemv.cuh"
#include "megakernel.cuh"   // MatDesc

#ifndef CBK_BLK1632_MMA
#define CBK_BLK1632_MMA 0
#endif
#ifndef CBK_MMA_NOLD
#define CBK_MMA_NOLD 0    // DIAGNOSTIC: constants instead of weight loads (WRONG RESULT)
#endif
#ifndef CBK_MMA_COAL
#define CBK_MMA_COAL 0    // 1: coalesced LDG.128 -> per-warp smem stage -> LDS by the fragment lanes
#endif
#ifndef CBK_MMA_ACC2
#define CBK_MMA_ACC2 0    // 1: two independent mma accumulator pairs per group (even/odd q4) for ILP
#endif
#ifndef CBK_MMA_UPW
#define CBK_MMA_UPW 4     // target (tile x slice) work units per warp when sizing the column slices
#endif
#ifndef CBK_MMA_UPW_O
#define CBK_MMA_UPW_O CBK_MMA_UPW   // o_proj's own target (K = hidden: fewest tiles)
#endif
#ifndef CBK_MMA_PF
#define CBK_MMA_PF 4      // 128-column groups of weight loads in flight per lane (streaming phases)
#endif
#ifndef CBK_MMA_PF_L2
#define CBK_MMA_PF_L2 1   // same, for the phases that run L2-resident (qkv, o_proj)
#endif
#ifndef CBK_MMA_SZPRE
#define CBK_MMA_SZPRE 1   // scale / zero prefetched with the weights (see gemv_blk1632_mma_tile)
#endif
#ifndef CBK_MMA_PHASES
#define CBK_MMA_PHASES 3  // bitmask: 1 qkv, 2 o_proj, 4 gate|up, 8 down, 16 lm_head
#endif

namespace cbk {
namespace detail {

__device__ __forceinline__ void mma_bf16_16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The 8 code bytes of bitmap byte q4 of one 16:32 block (the shipped expansion of blk1632_q4,
// ZF form: pruned slots carry the group's zero code so one sum(x') cancels them).
template <int BITS>
__device__ __forceinline__ void blk1632_expand(uint32_t m32, uint64_t buf, uint32_t h, int q4,
                                               const uint32_t* __restrict__ lut, uint32_t zb,
                                               uint32_t& e0, uint32_t& e1) {
  const uint32_t b8 = (m32 >> (q4 << 3)) & 0xFFu;
  const int o = (q4 == 0) ? 0 : __popc(m32 & ((1u << (q4 << 3)) - 1u));
  const uint32_t cwd = (uint32_t)(buf >> ((o & 15) << 2));
  uint32_t lo = cwd & 0x0F0F0F0Fu;
  uint32_t hi = (cwd >> 4) & 0x0F0F0F0Fu;
  if (BITS == 6) {
    const int ia = o >> 1, od = o & 1;
    const uint32_t wa = od ? (h >> 16) : h;
    const uint32_t wb = od ? h : (h >> 16);
    const uint32_t pe = (wa >> (ia << 1)) & 0xFFu;
    const uint32_t po = (wb >> ((ia + od) << 1)) & 0xFFu;
    uint32_t ve = pe | (pe << 12); ve = (ve | (ve << 6)) & 0x03030303u;
    uint32_t vo = po | (po << 12); vo = (vo | (vo << 6)) & 0x03030303u;
    lo |= ve << 4;
    hi |= vo << 4;
  }
#if CBK_BLK1632_LUT == 4
  const uint4 lte = reinterpret_cast<const uint4*>(lut)[b8];
  const uint32_t sel0 = lte.x, sel1 = lte.y, mk0 = lte.z, mk1 = lte.w;
#else
  const uint32_t sp0 = (((b8 & 0xFu) * 0x00204081u) & 0x01010101u);
  const uint32_t sp1 = ((((b8 >> 4) & 0xFu) * 0x00204081u) & 0x01010101u);
  const uint32_t mk0 = sp0 * 0xFFu;
  const uint32_t mk1 = sp1 * 0xFFu;
  const uint32_t sel0 = lut[b8 << 1];
  const uint32_t sel1 = lut[(b8 << 1) | 1];
#endif
  e0 = (__byte_perm(lo, hi, sel0) & mk0) | (zb & ~mk0);
  e1 = (__byte_perm(lo, hi, sel1) & mk1) | (zb & ~mk1);
}

// One warp: rows [tile*16, tile*16+16) over groups [j0, j1).  On return lane (g, t == 0) holds
// the f32 dot of row tile*16+g in acc_a and of row tile*16+g+8 in acc_b (other lanes: junk).
template <int BITS, int NX, int PF>
__device__ __forceinline__ void gemv_blk1632_mma_tile(
    const uint8_t* __restrict__ data, size_t rs,
    const __half* __restrict__ scale, const uint8_t* __restrict__ zero, int G, int N,
    const __nv_bfloat16* __restrict__ xa, const __nv_bfloat16* __restrict__ xb,
    int tile, int j0, int j1, const uint32_t* __restrict__ lut, float& acc_a, float& acc_b) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int ra = tile * 16 + g, rb = ra + 8;
  const uint8_t* pa = data + (size_t)ra * rs;
  const uint8_t* pb = data + (size_t)rb * rs;
  const size_t noff = (size_t)(N >> 3), hoff = (size_t)N * 3 >> 3;
  const __nv_bfloat16* xs = (NX == 2 && g == 1) ? xb : xa;    // B column n == g
  const uint32_t ONES[4] = {0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u};
  acc_a = 0.f; acc_b = 0.f;
  // PF groups of weight loads are issued together, then computed (24 B per lane per
  // group in flight: at PF 4 this matches the shipped loop's R*PF = 8 granules)
  for (int jc = j0; jc < j1; jc += PF) {
    uint32_t mwa[PF], mwb[PF], hwa[PF], hwb[PF];
    uint2 nwa[PF], nwb[PF];
#if CBK_MMA_SZPRE
    // per-group scale / zero of both rows issued WITH the weight loads: the zero feeds the ZF expansion
    // and the scale the epilogue, so loading them inside the compute stage put an L2 round trip on
    // every group's dependency chain
    __half spa[PF], spb[PF]; uint8_t zpa[PF], zpb[PF];
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int j = min(jc + s, j1 - 1);
      spa[s] = scale[(size_t)ra * G + j]; spb[s] = scale[(size_t)rb * G + j];
      zpa[s] = zero[(size_t)ra * G + j];  zpb[s] = zero[(size_t)rb * G + j];
    }
#endif
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int blk = 4 * (jc + s) + t;
      const bool ok = (jc + s) < j1;
#if CBK_MMA_NOLD
      mwa[s] = 0x0F0F0F0Fu ^ (uint32_t)blk; mwb[s] = 0xF0F0F0F0u ^ (uint32_t)blk;
      nwa[s] = make_uint2(0x12345678u, 0x9ABCDEF0u ^ (uint32_t)blk); nwb[s] = make_uint2(0x0F1E2D3Cu, (uint32_t)blk);
      (void)ok;
#else
      mwa[s] = ok ? __ldcg(reinterpret_cast<const uint32_t*>(pa) + blk) : 0u;
      mwb[s] = ok ? __ldcg(reinterpret_cast<const uint32_t*>(pb) + blk) : 0u;
      nwa[s] = ok ? __ldcg(reinterpret_cast<const uint2*>(pa + noff + ((size_t)blk << 3))) : make_uint2(0u, 0u);
      nwb[s] = ok ? __ldcg(reinterpret_cast<const uint2*>(pb + noff + ((size_t)blk << 3))) : make_uint2(0u, 0u);
#endif
      hwa[s] = 0u; hwb[s] = 0u;
      if (BITS == 6) {
        hwa[s] = ok ? __ldcg(reinterpret_cast<const uint32_t*>(pa + hoff) + blk) : 0u;
        hwb[s] = ok ? __ldcg(reinterpret_cast<const uint32_t*>(pb + hoff) + blk) : 0u;
      }
    }
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int j = jc + s;
      if (j >= j1) break;
      const int blk = 4 * j + t;
      const uint64_t bufa = (((uint64_t)nwa[s].y) << 32) | (uint64_t)nwa[s].x;
      const uint64_t bufb = (((uint64_t)nwb[s].y) << 32) | (uint64_t)nwb[s].x;
#if CBK_MMA_SZPRE
      const uint32_t za = zpa[s], zb_ = zpb[s];
      const float sa = __half2float(spa[s]), sb = __half2float(spb[s]);
#else
      const uint32_t za = zero[(size_t)ra * G + j], zb_ = zero[(size_t)rb * G + j];
      const float sa = __half2float(scale[(size_t)ra * G + j]);
      const float sb = __half2float(scale[(size_t)rb * G + j]);
#endif
      const uint32_t zba = za * 0x01010101u, zbb = zb_ * 0x01010101u;
      const uint4* xp = reinterpret_cast<const uint4*>(xs + (size_t)blk * 32);
      float C[4] = {0.f, 0.f, 0.f, 0.f}, X[4] = {0.f, 0.f, 0.f, 0.f};
#if CBK_MMA_ACC2
      float C2[4] = {0.f, 0.f, 0.f, 0.f}, X2[4] = {0.f, 0.f, 0.f, 0.f};   // odd q4 -> second chain
#endif
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const uint4 xv = xp[q4];                      // 8 bf16: positions 8*q4 .. 8*q4+7
        uint32_t ea0, ea1, eb0, eb1, A[4], B[2];
#if CBK_MMA_ACC2
        float* Cq = (q4 & 1) ? C2 : C; float* Xq = (q4 & 1) ? X2 : X;
#else
        float* Cq = C; float* Xq = X;
#endif
        blk1632_expand<BITS>(mwa[s], bufa, hwa[s], q4, lut, zba, ea0, ea1);
        blk1632_expand<BITS>(mwb[s], bufb, hwb[s], q4, lut, zbb, eb0, eb1);
        // tile 2*q4: positions 8*q4 + {0,1,2,3}
        A[0] = __byte_perm(ea0, 0x43434343u, 0x4140u);
        A[1] = __byte_perm(eb0, 0x43434343u, 0x4140u);
        A[2] = __byte_perm(ea0, 0x43434343u, 0x4342u);
        A[3] = __byte_perm(eb0, 0x43434343u, 0x4342u);
        B[0] = xv.x; B[1] = xv.y;
        mma_bf16_16816(Cq, A, B);
        mma_bf16_16816(Xq, ONES, B);
        // tile 2*q4+1: positions 8*q4 + {4,5,6,7}
        A[0] = __byte_perm(ea1, 0x43434343u, 0x4140u);
        A[1] = __byte_perm(eb1, 0x43434343u, 0x4140u);
        A[2] = __byte_perm(ea1, 0x43434343u, 0x4342u);
        A[3] = __byte_perm(eb1, 0x43434343u, 0x4342u);
        B[0] = xv.z; B[1] = xv.w;
        mma_bf16_16816(Cq, A, B);
        mma_bf16_16816(Xq, ONES, B);
      }
#if CBK_MMA_ACC2
#pragma unroll
      for (int k = 0; k < 4; ++k) { C[k] += C2[k]; X[k] += X2[k]; }
#endif
      // C fragment: c0 (row g, n 2t) c1 (row g, n 2t+1) c2 (row g+8, n 2t) c3 (row g+8, n 2t+1);
      // the real columns live in lanes t == 0: n = 0 (x_a) or n = 1 (x_b, odd rows of gate|up).
      const bool odd = (NX == 2) && (g & 1);
      const float va = odd ? C[1] : C[0], xa_ = odd ? X[1] : X[0];
      const float vb = odd ? C[3] : C[2], xb_ = odd ? X[3] : X[2];
      acc_a = fmaf(sa, va - (128.f + (float)za) * xa_, acc_a);
      acc_b = fmaf(sb, vb - (128.f + (float)zb_) * xb_, acc_b);
    }
  }
}

#if CBK_MMA_COAL
// Coalesced variant: a 4-group chunk (512 columns) of the 16-row tile is staged per WARP in
// shared memory with LDG.128 that touch 4 rows x 128 B (nibbles) / 8 rows x 64 B (masks) per
// instruction instead of 8 rows x 16 B, then the fragment lanes read their block with LDS.
// Row pads keep the LDS conflict-free: masks 20 words/row (bank 4g+t), nibbles 40 words/row
// (bank 8g+2t per half-warp).
#define CBK_MMA_CH 4
#define CBK_MMA_MSKROW 80     // bytes per staged mask row  (16 words + 4 pad)
#define CBK_MMA_NIBROW 160    // bytes per staged nibble row (32 words + 8 pad)
#define CBK_MMA_STAGE (16 * (CBK_MMA_MSKROW + CBK_MMA_NIBROW))   // 3840 B per warp
template <int BITS, int NX>
__device__ __forceinline__ void gemv_blk1632_mma_tile_coal(
    const uint8_t* __restrict__ data, size_t rs,
    const __half* __restrict__ scale, const uint8_t* __restrict__ zero, int G, int N,
    const __nv_bfloat16* __restrict__ xa, const __nv_bfloat16* __restrict__ xb,
    int tile, int j0, int j1, const uint32_t* __restrict__ lut, float& acc_a, float& acc_b) {
  static_assert(BITS == 4, "CBK_MMA_COAL stages the 4-bit planes only");
  __shared__ __align__(16) uint8_t s_mma_stage[CBK_WARPS][CBK_MMA_STAGE];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  uint8_t* s_msk = s_mma_stage[warp];
  uint8_t* s_nib = s_msk + 16 * CBK_MMA_MSKROW;
  const int ra = tile * 16 + g, rb = ra + 8;
  const uint8_t* ptile = data + (size_t)tile * 16 * rs;
  const size_t noff = (size_t)(N >> 3);
  const __nv_bfloat16* xs = (NX == 2 && g == 1) ? xb : xa;    // B column n == g
  const uint32_t ONES[4] = {0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u};
  acc_a = 0.f; acc_b = 0.f;
  for (int jc = j0; jc < j1; jc += CBK_MMA_CH) {
    // ---- stage: nibbles 16 rows x 128 B (4 x LDG.128 per lane), masks 16 rows x 64 B (2 x)
    {
      uint4 nv[4], mv[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = 4 * i + (lane >> 3), c = lane & 7;          // chunk c = blocks 2c, 2c+1
        const bool ok = (jc + (c >> 1)) < j1;
        nv[i] = ok ? __ldcg(reinterpret_cast<const uint4*>(ptile + (size_t)r * rs + noff + ((size_t)(4 * jc) << 3) + (c << 4)))
                   : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = 8 * i + (lane >> 2), c = lane & 3;          // chunk c = group jc + c
        const bool ok = (jc + c) < j1;
        mv[i] = ok ? __ldcg(reinterpret_cast<const uint4*>(ptile + (size_t)r * rs + ((size_t)(4 * jc) << 2) + (c << 4)))
                   : make_uint4(0u, 0u, 0u, 0u);
      }
      __syncwarp();                                               // previous chunk fully consumed
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = 4 * i + (lane >> 3), c = lane & 7;
        *reinterpret_cast<uint4*>(s_nib + r * CBK_MMA_NIBROW + (c << 4)) = nv[i];
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = 8 * i + (lane >> 2), c = lane & 3;
        *reinterpret_cast<uint4*>(s_msk + r * CBK_MMA_MSKROW + (c << 4)) = mv[i];
      }
      __syncwarp();
    }
    // per-row scale / zero of the chunk's 4 groups: lane t fetches group jc+t, shuffled per group
    const int jt = min(jc + t, j1 - 1);
    const float sva = __half2float(scale[(size_t)ra * G + jt]), svb = __half2float(scale[(size_t)rb * G + jt]);
    const int zva = zero[(size_t)ra * G + jt], zvb = zero[(size_t)rb * G + jt];
#pragma unroll
    for (int q = 0; q < CBK_MMA_CH; ++q) {
      const int j = jc + q;
      if (j >= j1) break;
      const int lb = 4 * q + t, blk = 4 * j + t;                  // block within the chunk / row
      const int src = (lane & ~3) | q;
      const float sa = __shfl_sync(0xffffffffu, sva, src), sb = __shfl_sync(0xffffffffu, svb, src);
      const uint32_t za = (uint32_t)__shfl_sync(0xffffffffu, zva, src);
      const uint32_t zb_ = (uint32_t)__shfl_sync(0xffffffffu, zvb, src);
      const uint32_t cma = *reinterpret_cast<const uint32_t*>(s_msk + g * CBK_MMA_MSKROW + (lb << 2));
      const uint32_t cmb = *reinterpret_cast<const uint32_t*>(s_msk + (g + 8) * CBK_MMA_MSKROW + (lb << 2));
      const uint2 nwa = *reinterpret_cast<const uint2*>(s_nib + g * CBK_MMA_NIBROW + (lb << 3));
      const uint2 nwb = *reinterpret_cast<const uint2*>(s_nib + (g + 8) * CBK_MMA_NIBROW + (lb << 3));
      const uint64_t bufa = (((uint64_t)nwa.y) << 32) | (uint64_t)nwa.x;
      const uint64_t bufb = (((uint64_t)nwb.y) << 32) | (uint64_t)nwb.x;
      const uint32_t zba = za * 0x01010101u, zbb = zb_ * 0x01010101u;
      const uint4* xp = reinterpret_cast<const uint4*>(xs + (size_t)blk * 32);
      float C[4] = {0.f, 0.f, 0.f, 0.f}, X[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const uint4 xv = xp[q4];
        uint32_t ea0, ea1, eb0, eb1, A[4], B[2];
        blk1632_expand<BITS>(cma, bufa, 0u, q4, lut, zba, ea0, ea1);
        blk1632_expand<BITS>(cmb, bufb, 0u, q4, lut, zbb, eb0, eb1);
        A[0] = __byte_perm(ea0, 0x43434343u, 0x4140u);
        A[1] = __byte_perm(eb0, 0x43434343u, 0x4140u);
        A[2] = __byte_perm(ea0, 0x43434343u, 0x4342u);
        A[3] = __byte_perm(eb0, 0x43434343u, 0x4342u);
        B[0] = xv.x; B[1] = xv.y;
        mma_bf16_16816(C, A, B);
        mma_bf16_16816(X, ONES, B);
        A[0] = __byte_perm(ea1, 0x43434343u, 0x4140u);
        A[1] = __byte_perm(eb1, 0x43434343u, 0x4140u);
        A[2] = __byte_perm(ea1, 0x43434343u, 0x4342u);
        A[3] = __byte_perm(eb1, 0x43434343u, 0x4342u);
        B[0] = xv.z; B[1] = xv.w;
        mma_bf16_16816(C, A, B);
        mma_bf16_16816(X, ONES, B);
      }
      const bool odd = (NX == 2) && (g & 1);
      const float va = odd ? C[1] : C[0], xa_ = odd ? X[1] : X[0];
      const float vb = odd ? C[3] : C[2], xb_ = odd ? X[3] : X[2];
      acc_a = fmaf(sa, va - (128.f + (float)za) * xa_, acc_a);
      acc_b = fmaf(sb, vb - (128.f + (float)zb_) * xb_, acc_b);
    }
  }
}
#endif  // CBK_MMA_COAL

// Phase driver.  fin(row0, lane, v) is called by ALL 32 lanes of the finishing warp with lane l < 16
// holding the f32 dot of row row0 + l (lanes >= 16: junk) -- so a pair epilogue may shuffle.
template <int BITS, int NX, int PF, int UPW, class XF, class Fin>
__device__ __forceinline__ void gemv_phase_mma(const MatDesc& d, int K, XF xf, Fin fin,
                                               float* __restrict__ mpart, unsigned* __restrict__ mcnt) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int NT = K >> 4, NG = N >> 7;
  // groups per unit: halve until there are >= UPW units per warp, min 1 group
  int gpu = NG;
  while (gpu > 1 && NT * ((NG + gpu - 1) / gpu) < UPW * GW) gpu >>= 1;
  const int S = (NG + gpu - 1) / gpu;
  const int nunits = NT * S;
#if CBK_BLK1632_LUT == 2 || CBK_BLK1632_LUT == 4
  const uint32_t* __restrict__ lut = prmt_smem();
#else
  const uint32_t* __restrict__ lut = prmt_lut;
#endif
  for (int u = gw; u < nunits; u += GW) {
    const int tile = u / S, sl = u - tile * S;
    const int j0 = sl * gpu, j1 = min(NG, j0 + gpu);
    const __nv_bfloat16 *xa = nullptr, *xb = nullptr;
    xf(tile * 16, &xa, &xb);
    float acc_a, acc_b;
#if CBK_MMA_COAL
    gemv_blk1632_mma_tile_coal<BITS, NX>(d.data, rs, d.scale, d.zero, G, N, xa, xb, tile, j0, j1, lut,
                                         acc_a, acc_b);
#else
    gemv_blk1632_mma_tile<BITS, NX, PF>(d.data, rs, d.scale, d.zero, G, N, xa, xb, tile, j0, j1, lut,
                                        acc_a, acc_b);
#endif
    // rows 0..15 of the tile -> lanes 0..15 (row g sits in lane 4g, row g+8 in lane 4(g) too)
    const float va = __shfl_sync(0xffffffffu, acc_a, (lane & 7) << 2);
    const float vb = __shfl_sync(0xffffffffu, acc_b, (lane & 7) << 2);
    float v = (lane < 8) ? va : vb;
    if (S == 1) {
      fin(tile * 16, lane, v);
    } else {
      if (lane < 16) atomicAdd(mpart + (size_t)tile * 16 + lane, v);
      __threadfence();
      unsigned old = 0u;
      if (lane == 0) old = atomicAdd(mcnt + tile, 1u);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old == (unsigned)(S - 1)) {        // last slice of this tile: reduce + epilogue, reset
        __threadfence();
        if (lane < 16) v = atomicExch(mpart + (size_t)tile * 16 + lane, 0.f);
        if (lane == 0) mcnt[tile] = 0u;
        fin(tile * 16, lane, v);
      }
    }
  }
}


// Contiguous equal-work chunking of the tile walk (CBK_CHUNK): the (tile, group) items are handed
// out as one contiguous run per warp, C = ceil(items / warps), so every warp streams the same
// bytes (+-1 group) instead of ceil(units / warps) whole units.  A tile split between warps is
// summed through the same f32 atomics; the number of contributors of tile T is known from the
// arithmetic (first and last warp of its item range), so no extra traffic for whole tiles.
template <int BITS, int NX, int PF, class XF, class Fin>
__device__ __forceinline__ void gemv_phase_mma_chunk(const MatDesc& d, int K, XF xf, Fin fin,
                                                     float* __restrict__ mpart, unsigned* __restrict__ mcnt) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * CBK_WARPS + warp, GW = gridDim.x * CBK_WARPS;
  const int N = (int)d.N, G = (int)d.G;
  const size_t rs = cbk::dense_row_stride((int)d.layout, N);
  const int NG = N >> 7;
  const long items = (long)(K >> 4) * NG;
  const long C = (items + GW - 1) / GW;
  long pos = (long)gw * C;
  const long end = min(items, pos + C);
#if CBK_BLK1632_LUT == 2 || CBK_BLK1632_LUT == 4
  const uint32_t* __restrict__ lut = prmt_smem();
#else
  const uint32_t* __restrict__ lut = prmt_lut;
#endif
  while (pos < end) {
    const int tile = (int)(pos / NG);
    const int j0 = (int)(pos - (long)tile * NG);
    const int j1 = (int)min((long)NG, j0 + (end - pos));
    const __nv_bfloat16 *xa = nullptr, *xb = nullptr;
    xf(tile * 16, &xa, &xb);
    float acc_a, acc_b;
    gemv_blk1632_mma_tile<BITS, NX, PF>(d.data, rs, d.scale, d.zero, G, N, xa, xb, tile, j0, j1, lut,
                                        acc_a, acc_b);
    const float va = __shfl_sync(0xffffffffu, acc_a, (lane & 7) << 2);
    const float vb = __shfl_sync(0xffffffffu, acc_b, (lane & 7) << 2);
    float v = (lane < 8) ? va : vb;
    if (j0 == 0 && j1 == NG) {
      fin(tile * 16, lane, v);
    } else {
      const long tb = (long)tile * NG, te = tb + NG - 1;
      const unsigned contrib = (unsigned)(te / C - tb / C + 1);
      if (lane < 16) atomicAdd(mpart + (size_t)tile * 16 + lane, v);
      __threadfence();
      unsigned old = 0u;
      if (lane == 0) old = atomicAdd(mcnt + tile, 1u);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old == contrib - 1u) {
        __threadfence();
        if (lane < 16) v = atomicExch(mpart + (size_t)tile * 16 + lane, 0.f);
        if (lane == 0) mcnt[tile] = 0u;
        fin(tile * 16, lane, v);
      }
    }
    pos += (j1 - j0);
  }
}

}  // namespace detail
}  // namespace cbk
