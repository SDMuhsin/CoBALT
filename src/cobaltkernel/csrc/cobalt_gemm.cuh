// cobalt_gemm.cuh -- device-side DENSE4/DENSE8 tile GEMM on bf16 mma.sync (m16n8k16),
// usable INSIDE the persistent megakernel.  See docs/KERNELS.md.
//
//   Y[m][k] = sum_j W_hat[k][j] * X[m][j],   W_hat = (code - zero) * scale * col_scale[j]
//
// One 256-thread block computes a tile of MT activation rows x BR weight rows, streaming
// the input axis N in KT-column chunks.  The column scale is folded into the WEIGHT
// fragment (not into X), so a fused gate/up matrix with two different col_scale vectors
// still needs only ONE staged activation tile.
//
// KT (the stage depth along N) is a template parameter: KT=256 gives one full 128-B cache
// line of every weight row per stage, 4x fewer __syncthreads per byte and >=4 independent
// 16-B loads in flight per thread.  DENSE4 dequant uses the bf16 magic-number route
// (0x4300|nibble == 128+nibble exactly) so a nibble pair becomes a bf16x2 with
// SHF+LOP+LOP3, and scale/zero/col_scale are applied with three bf16x2 ops.
//
// PF (the prefetch depth) is a template parameter too: the global loads for stage s+PF are
// issued right after stage s is committed to smem, so a tile's loads have PF compute
// phases to land instead of one.  PF=1 is the round-2 behaviour.  Costs PF*(WPT+XPT+1)
// uint4 registers.
#pragma once
#include "cobalt_format.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace cbk {

enum GemmEpi : int {
  EPI_F32        = 0,  // Y is float*        , Y[m*ldY + row]
  EPI_BF16       = 1,  // Y is __nv_bfloat16*, Y[m*ldY + row]
  EPI_GEGLU_BF16 = 2,  // fused gateup: rows 2i/2i+1 -> Y[m*ldY + i] = gelu(gate)*up, bf16
  EPI_SWIGLU_BF16 = 3  // same pairing, silu(gate)*up (llama / mistral)
};

// ---------------------------------------------------------------- tile geometry
constexpr int GT_BR   = 64;   // default weight rows per block tile (template arg BR)
constexpr int GT_KT   = 64;   // default input columns per pipeline stage (template arg KT)
constexpr int GT_GC   = 4;    // default quantization groups whose scale/zero are cached
// One stage spans max(1, KT/GROUP) groups; cache just enough of them (saves smem at big BR).
template <int KT> struct gt_gc { static constexpr int value = (KT >= 2 * GROUP) ? 2 : 4; };

// smem row strides, in *bytes* for W and in *halves* for X.  Both carry 16 B of padding so
// that the 8 weight rows / 16 activation rows a warp touches land in distinct bank groups.
template <int KT> struct gt_xs  { static constexpr int value = KT + 8; };        // halves
template <int KT> struct gt_ws  { static constexpr int value = KT / 2 + 16; };   // bytes, DENSE4
template <int KT> struct gt_ws8 { static constexpr int value = KT + 16; };       // bytes, DENSE8

// Default pipeline depth per M-tile, chosen so the tile always fits in 48 KB of smem.
template <int MT> struct gt_stages { static constexpr int value = (MT <= 32) ? 4 : (MT == 64 ? 3 : 2); };

// smem bytes needed by gemm_tile<MT,...,KT>.  Independent of PF: the pipeline lives in
// REGISTERS, so smem stays single buffered and the S argument is source compatibility only.
template <int MT, int S = gt_stages<MT>::value, int BR = GT_BR, int KT = GT_KT>
__host__ __device__ constexpr int gemm_smem_bytes(int dense8 = 0) {
  return MT * gt_xs<KT>::value * 2                                        // X tile
       + BR * (dense8 ? gt_ws8<KT>::value : gt_ws<KT>::value)             // packed W tile
       + KT * 4                                                          // col_scale[2][KT/2] bf16x2
       + BR * gt_gc<KT>::value * 8;                                      // (zero,scale) bf16x2 cache
}

namespace detail {

template <int V> struct ic { static constexpr int value = V; };   // compile-time buffer index

// Load the four m16n8k16 A fragments of one 16x16 row-major bf16 smem tile in ONE
// instruction (replaces 4 scalar LDS.32 per fragment).  Lane l supplies the address of
// row (l&15), column block ((l>>4)*8); the resulting register order is exactly the A
// operand order the mma wants.  Each row address must be 16-B aligned.
__device__ __forceinline__ void ldmatrix_x4(uint32_t* d, const void* p) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(s));
}

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// (a & b) | c  in one LOP3.
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

// One packed byte (two 4-bit codes) -> bf16x2 (128+lo, 128+hi), EXACT.
// bf16 0x4300 == 128.0 and its 7 mantissa bits hold integers 0..127 exactly.
__device__ __forceinline__ uint32_t nib2_to_bf16x2(uint32_t y) {
  return lop3_and_or(y | (y << 12), 0x000F000Fu, 0x43004300u);
}

__device__ __forceinline__ uint32_t bf2_sub(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&a),
                             *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bf2_mul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),
                             *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// ------------------------------------------------------ BLK16_32 -> dense expansion
// (docs/FORMAT.md sec.3.2 -- the 16:32 block layout)
//
// The tile GEMM consumes a FIXED-STRIDE dense weight plane out of shared memory.  A
// BLK16_32 row is three planes ([mask N/8][nibble N/4][hi2 N/8]) with a 16-of-32
// fixed-cardinality mask, so a 32-column block's codes sit at a KNOWN offset and the
// block can be expanded to 32 dense code slots with NO prefix popcount over the row and
// NO sliding shift chain -- which is exactly what makes it tileable where SPARSE4* was
// not.
//
// The expansion runs ONCE PER (weight row, stage) and its result is reused by all MT
// activation rows of the tile, so its ALU cost is amortised MT-fold against the mma --
// the opposite of the decode GEMV, where it sits in the per-granule dependency chain.
//
// PRUNED POSITIONS: the expansion writes the group's integer `zero` into every pruned
// slot, so the dequant `(128+code) - (128+zero)` is EXACTLY 0.0 in bf16 (both operands
// are the same bf16 value; 128..255 has ulp 1) and the mma adds 0.0*x.  This is exact in
// a way DENSE4's GEMV accumulator is not (FORMAT.md sec.13.3): there the cancellation is
// deferred to `qs - zero*xs` in fp32, here it happens inside the B fragment.
//
// CBK_GEMM_BLK_LUT: 0 = arithmetic selector (no table, default), 1 = the 512-entry
// __byte_perm selector table shared with the GEMV decoder (csrc/prmt_lut.inc).
#ifndef CBK_GEMM_BLK_LUT
#define CBK_GEMM_BLK_LUT 0
#endif
#if CBK_GEMM_BLK_LUT == 1
__device__ static const uint32_t gt_prmt_lut[512] = {
#include "prmt_lut.inc"
};
#endif

// Arithmetic __byte_perm selectors (`prmt_sel_arith` in cobalt_gemv.cuh):
// for bitmap byte b8 the selector nibble of output slot j is s(r_j) with
// r_j = popc(b8 & ((1<<j)-1)) and s(k) = (k>>1) | ((k&1)<<2).  `sp` is the 0x00204081
// bit->byte spread of the relevant nibble of b8, which the pruned-slot mask already needs.
__device__ __forceinline__ uint32_t gt_sel_arith(uint32_t sp, uint32_t o4) {
  const uint32_t P = ((sp << 8) * 0x01010101u) + o4 * 0x01010101u;   // byte j = r_j
  const uint32_t sb = ((P >> 1) & 0x03030303u) | ((P & 0x01010101u) << 2);
  const uint32_t mm = (sb & 0x000F000Fu) | ((sb >> 4) & 0x00F000F0u);
  return (mm | (mm >> 8)) & 0xFFFFu;
}

// Two 8-bit codes -> bf16x2 (128+c0, 128+c1), EXACT for c <= 255.  ONE prmt: the result
// bytes are [c0, 0x43, c1, 0x43], i.e. the bf16 pair (0x4300|c0, 0x4300|c1).  This is the
// byte-plane analogue of nib2_to_bf16x2 and is one instruction CHEAPER than it.
__device__ __forceinline__ uint32_t byte2_to_bf16x2(uint32_t y) {
  return __byte_perm(y, 0x43434343u, 0x4140u);
}

// Expand ONE 32-column BLK16_32 block into dense code slots.
//   m32 : the block's 32-bit mask (popcount == 16 by construction)
//   nib : the block's 8 nibble-plane bytes (16 survivor low-nibbles, column order)
//   h   : the block's hi2 word (BITS == 6 only; parity-split, FORMAT.md sec.13.2)
//   zb  : the group's integer zero, splatted into all 4 bytes (pruned-slot fill)
//   out : BITS==6 -> 8 words = 32 code BYTES;  BITS==4 -> 4 words = 32 code NIBBLES
//         (DENSE4 order: byte j holds column 2j in the low nibble)
template <int BITS>
__device__ __forceinline__ void blk_expand32(uint32_t m32, uint2 nib, uint32_t h,
                                             uint32_t zb, uint32_t* out) {
  const uint64_t buf = ((uint64_t)nib.y << 32) | (uint64_t)nib.x;
#pragma unroll
  for (int q4 = 0; q4 < 4; ++q4) {
    const uint32_t b8 = (m32 >> (q4 << 3)) & 0xFFu;
    // survivors preceding this bitmap byte INSIDE the block: one POPC, no chain
    const int o = q4 ? __popc(m32 & ((1u << (q4 << 3)) - 1u)) : 0;
    const uint32_t cwd = (uint32_t)(buf >> ((o & 15) << 2));   // 8 survivor nibbles
    uint32_t lo = cwd & 0x0F0F0F0Fu;          // bytes = survivors o, o+2, o+4, o+6
    uint32_t hi = (cwd >> 4) & 0x0F0F0F0Fu;   // bytes = survivors o+1, o+3, o+5, o+7
    if (BITS == 6) {
      const int ia = o >> 1, od = o & 1;
      const uint32_t wa = od ? (h >> 16) : h;
      const uint32_t wb = od ? h : (h >> 16);
      const uint32_t pe = (wa >> (ia << 1)) & 0xFFu;
      const uint32_t po = (wb >> ((ia + od) << 1)) & 0xFFu;
      // carry-free OR spread (a spread-MULTIPLY corrupts the next byte, FORMAT.md 13.8)
      uint32_t ve = pe | (pe << 12); ve = (ve | (ve << 6)) & 0x03030303u;
      uint32_t vo = po | (po << 12); vo = (vo | (vo << 6)) & 0x03030303u;
      lo |= ve << 4;
      hi |= vo << 4;
    }
    const uint32_t sp0 = ((b8 & 0xFu) * 0x00204081u) & 0x01010101u;
    const uint32_t sp1 = (((b8 >> 4) & 0xFu) * 0x00204081u) & 0x01010101u;
    const uint32_t mk0 = sp0 * 0xFFu;      // byte mask of the KEPT slots 0..3
    const uint32_t mk1 = sp1 * 0xFFu;      //                        slots 4..7
#if CBK_GEMM_BLK_LUT == 1
    const uint32_t sel0 = gt_prmt_lut[b8 << 1], sel1 = gt_prmt_lut[(b8 << 1) | 1];
#else
    const uint32_t sel0 = gt_sel_arith(sp0, 0u);
    const uint32_t sel1 = gt_sel_arith(sp1, (uint32_t)__popc((int)(b8 & 0xFu)));
#endif
    const uint32_t e0 = (__byte_perm(lo, hi, sel0) & mk0) | (zb & ~mk0);
    const uint32_t e1 = (__byte_perm(lo, hi, sel1) & mk1) | (zb & ~mk1);
    if (BITS == 6) {
      out[2 * q4 + 0] = e0;
      out[2 * q4 + 1] = e1;
    } else {                     // compact 8 code BYTES (all < 16) into 8 nibbles
      const uint32_t t0 = (e0 | (e0 >> 4)) & 0x00FF00FFu;
      const uint32_t t1 = (e1 | (e1 >> 4)) & 0x00FF00FFu;
      out[q4] = ((t0 | (t0 >> 8)) & 0xFFFFu) | (((t1 | (t1 >> 8)) & 0xFFFFu) << 16);
    }
  }
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

}  // namespace detail

// ---------------------------------------------------------------- gemm_tile
// w        : CBK1 matrix, layout DENSE4 or DENSE8 (fixed row stride).  N % (PF*KT) need not
//            divide, but N % KT == 0 is REQUIRED.
//            DENSE8 is supported for KT == 64 only (register budget of the prefetch).
// col_scale: fp16 [N] (cs_pairs=0) or [2][N] (cs_pairs=1: row parity picks the vector,
//            i.e. the fused gateup convention row 2i = gate, 2i+1 = up).  May be null.
// row0     : first weight row of this tile (multiple of BR recommended, MUST be even
//            for EPI_GEGLU_BF16).   nrows: rows to compute, <= BR, masked.
// X        : GLOBAL row-major activations, X[m*ldX + j], bf16, NOT column-scaled.
// norm_w/rrms: optional Gemma3 RMSNorm fused into the staging
//            (x' = x * rrms[m] * (1 + norm_w[j])); pass nullptr to skip (fast path).
// m0       : first activation row of this tile;  M: total rows.
// Y, ldY   : output, see GemmEpi.
// smem     : gemm_smem_bytes<MT,S,BR,KT>(dense8) bytes, 16-B aligned.
// BLK      : 0 = DENSE4/DENSE8 as before (bit-for-bit unchanged codegen); 4 or 6 = the
//            CoBALT-16:32 BLK1632_4 / BLK1632_6 layout, expanded to a dense smem plane
//            at staging time (see blk_expand32).  BLK=4 produces the DENSE4 nibble plane
//            and reuses DENSE4's inner loop verbatim; BLK=6 produces a byte plane
//            (gt_ws8 stride) read with a one-prmt magic-number dequant.
template <int MT, int EPI, int WM = 1, int S = gt_stages<MT>::value, int BR = GT_BR,
          int KT = GT_KT, int PF = 1, int BLK = 0>
__device__ void gemm_tile(const Mat& w, const __half* col_scale, int cs_pairs, int row0,
                          int nrows, const __nv_bfloat16* X, int ldX, int M, int m0,
                          const __nv_bfloat16* norm_w, const float* rrms, void* Y, int ldY,
                          uint8_t* smem) {
  static_assert(MT == 16 || MT == 32 || MT == 64 || MT == 128, "MT in {16,32,64,128}");
  static_assert(KT == 64 || KT == 128 || KT == 256, "KT in {64,128,256}");
  static_assert(PF >= 1 && PF <= 3, "PF in {1,2,3}");
  static_assert(BLK == 0 || BLK == 4 || BLK == 6, "BLK in {0,4,6}");
  constexpr bool ISBLK  = (BLK != 0);
  constexpr bool BYTEPL = (BLK == 6);           // 6-bit codes need a full byte per column
  static_assert(!ISBLK || PF == 1, "BLK16_32 staging is written for PF == 1");
  constexpr int MSUB = MT / 16;
  constexpr int WN   = 8 / WM;                // warp grid cols
  constexpr int MPW  = (MSUB >= WM) ? MSUB / WM : 1;  // M-subtiles per warp
  constexpr int NPW  = (BR / 8) / WN;         // weight-row subtiles per warp
  constexpr int NG   = (KT > GROUP) ? (KT / GROUP) : 1;   // groups spanned by one stage
  constexpr int GC   = gt_gc<KT>::value;                  // groups held in the smem cache
  constexpr int XS   = gt_xs<KT>::value;
  static_assert(MSUB % WM == 0, "WM must divide MT/16");
  static_assert((BR / 8) % WN == 0, "warp grid must divide BR/8");

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int grp  = lane >> 2;      // 0..7
  const int tig  = lane & 3;       // 0..3
  const int wm   = warp / WN;
  const int wn   = warp % WN;
  const int lmrow = lane & 15;          // ldmatrix: row this lane addresses
  const int lmcol = (lane >> 4) * 8;    // ldmatrix: column block this lane addresses

  const bool d8 = ISBLK ? false : (w.layout == LAYOUT_DENSE8);
  constexpr int WSB = BYTEPL ? gt_ws8<KT>::value : gt_ws<KT>::value;   // BLK smem stride
  static_assert(WSB % 16 == 0, "BLK smem row stride must stay 16-B aligned (uint4 stores)");
  const int  WS = ISBLK ? WSB : (d8 ? gt_ws8<KT>::value : gt_ws<KT>::value);
  const int  N  = w.N;
  const size_t wstride = ISBLK ? (BLK == 4 ? (size_t)(N * 3 / 8) : (size_t)(N >> 1))
                               : (d8 ? (size_t)N : (size_t)(N >> 1));
  // BLK16_32 staging work item = ONE (weight row, 32-column block) pair.
  constexpr int NBLKS  = KT / 32;                  // blocks per row per stage
  constexpr int BITEMS = ISBLK ? (BR * NBLKS) : 1;
  constexpr int BPT    = (BITEMS + 255) / 256;     // items per thread

  // ---- smem carve-up (the PF-deep pipeline lives in REGISTERS)
  // WPT is sized for DENSE8 at KT=64 (4 chunks/row) and for DENSE4 otherwise (KT/32).
  constexpr int WCH = (KT == 64) ? 4 : (KT / 32);
  constexpr int WPT = (BR * WCH > 256) ? (BR * WCH / 256) : 1;   // uint4 per thread, weights
  constexpr int XPT = (MT * KT / 8 > 256) ? (MT * KT / 8 / 256) : 1;  // uint4/thread, acts
  constexpr int CPT = 2 * KT / 8;                     // threads that carry col_scale
  __nv_bfloat16* Xs = reinterpret_cast<__nv_bfloat16*>(smem);          // [MT][XS]
  uint8_t*       Ws = smem + MT * XS * 2;                              // [BR][WS]
  uint32_t*      Cs = reinterpret_cast<uint32_t*>(Ws + BR * WS);       // [2][KT/2] bf16x2
  uint32_t*      SZ = Cs + KT;                                         // [BR][GC][2] bf16x2

  const int mrows = min(MT, M - m0);
  const int wchunks = (d8 ? KT : KT / 2) / 16;    // 16-B chunks of one weight row per stage

  float acc[MPW][NPW][4];
#pragma unroll
  for (int i = 0; i < MPW; ++i)
#pragma unroll
    for (int j = 0; j < NPW; ++j)
#pragma unroll
      for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.f;

  const int nstage = N / KT;
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  uint4 wreg[ISBLK ? 1 : PF][ISBLK ? 1 : WPT], xreg[PF][XPT], creg[PF];
  // BLK16_32 raw planes, one set per staged (row, block) item.
  uint32_t mreg[ISBLK ? PF : 1][ISBLK ? BPT : 1];      // mask word
  uint2    nreg[ISBLK ? PF : 1][ISBLK ? BPT : 1];      // 16 survivor nibbles
  uint32_t hreg[ISBLK ? PF : 1][ISBLK ? BPT : 1];      // hi2 word (b = 6 only)
  uint32_t zreg[ISBLK ? PF : 1][ISBLK ? BPT : 1];      // group zero, byte-splatted

  // Plain vectorised global loads kept in registers across the compute of the previous PF
  // stages.  MEASURED on sm_120/MIG: this streams ~2.6x faster than a cp.async pipeline.
  // The buffer index B is a compile-time constant so the arrays stay in registers.
  auto prefetch = [&](auto BT, int j0) {
    constexpr int B = decltype(BT)::value;
    if constexpr (ISBLK) {
      // Three separately-contiguous planes; every load is naturally aligned and the
      // NBLKS threads of one row cover a contiguous run in each plane.
      const size_t nbase = (size_t)(N >> 3);            // nibble plane offset
      const size_t hbase = (size_t)(3 * N / 8);         // hi2 plane offset (b = 6)
#pragma unroll
      for (int u = 0; u < BPT; ++u) {
        const int i = tid + u * 256;
        mreg[B][u] = 0u; nreg[B][u] = make_uint2(0u, 0u); zreg[B][u] = 0u;
        if (BYTEPL) hreg[B][u] = 0u;
        if (i < BITEMS) {
          const int r = i / NBLKS, bq = i % NBLKS;
          const int gr = row0 + r;
          if (r < nrows && gr < w.K) {
            const uint8_t* rp = w.data + (size_t)gr * wstride;
            mreg[B][u] = *reinterpret_cast<const uint32_t*>(rp + (j0 >> 3) + 4 * bq);
            nreg[B][u] = *reinterpret_cast<const uint2*>(rp + nbase + (j0 >> 2) + 8 * bq);
            if (BYTEPL)
              hreg[B][u] = *reinterpret_cast<const uint32_t*>(rp + hbase + (j0 >> 3) + 4 * bq);
            zreg[B][u] = 0x01010101u * (uint32_t)w.zero[(size_t)gr * w.G +
                                                        (j0 + 32 * bq) / GROUP];
          }
        }
      }
    } else {
#pragma unroll
    for (int u = 0; u < WPT; ++u) {
      const int i = tid + u * 256;
      wreg[B][u] = zero4;
      if (i < BR * wchunks) {
        const int r = i / wchunks, c = i % wchunks;
        const int gr = row0 + r;
        if (r < nrows && gr < w.K)
          wreg[B][u] = *reinterpret_cast<const uint4*>(
              w.data + (size_t)gr * wstride + (d8 ? j0 : (j0 >> 1)) + c * 16);
      }
    }
    }
    if (norm_w == nullptr) {
#pragma unroll
      for (int u = 0; u < XPT; ++u) {
        const int i = tid + u * 256;
        xreg[B][u] = zero4;
        if (i < MT * (KT / 8)) {
          const int m = i / (KT / 8), c = i % (KT / 8);
          if (m < mrows)
            xreg[B][u] = *reinterpret_cast<const uint4*>(X + (size_t)(m0 + m) * ldX + j0 + c * 8);
        }
      }
    }
    if (tid < CPT) {                 // col_scale: 2*KT halves
      const int p = tid / (KT / 8), c = (tid % (KT / 8)) * 8;
      const __half* cp = col_scale ? (col_scale + (cs_pairs ? (size_t)p * N : 0)) : nullptr;
      creg[B] = cp ? *reinterpret_cast<const uint4*>(cp + j0 + c) : zero4;
    }
  };

  auto commit = [&](auto BT, int j0) {
    constexpr int B = decltype(BT)::value;
    if constexpr (ISBLK) {
#pragma unroll
      for (int u = 0; u < BPT; ++u) {
        const int i = tid + u * 256;
        if (i < BITEMS) {
          const int r = i / NBLKS, bq = i % NBLKS;
          uint32_t ex[BYTEPL ? 8 : 4];
          detail::blk_expand32<BLK>(mreg[B][u], nreg[B][u],
                                    BYTEPL ? hreg[B][u] : 0u, zreg[B][u], ex);
          if (BYTEPL) {
            uint4* dst = reinterpret_cast<uint4*>(Ws + (size_t)r * WSB + bq * 32);
            dst[0] = make_uint4(ex[0], ex[1], ex[2], ex[3]);
            dst[1] = make_uint4(ex[4], ex[5], ex[6], ex[7]);
          } else {
            *reinterpret_cast<uint4*>(Ws + (size_t)r * WSB + bq * 16) =
                make_uint4(ex[0], ex[1], ex[2], ex[3]);
          }
        }
      }
    } else {
#pragma unroll
    for (int u = 0; u < WPT; ++u) {
      const int i = tid + u * 256;
      if (i < BR * wchunks) {
        const int r = i / wchunks, c = i % wchunks;
        *reinterpret_cast<uint4*>(Ws + (size_t)r * WS + c * 16) = wreg[B][u];
      }
    }
    }
    if (norm_w == nullptr) {
#pragma unroll
      for (int u = 0; u < XPT; ++u) {
        const int i = tid + u * 256;
        if (i < MT * (KT / 8)) {
          const int m = i / (KT / 8), c = i % (KT / 8);
          *reinterpret_cast<uint4*>(Xs + (size_t)m * XS + c * 8) = xreg[B][u];
        }
      }
    } else {                                   // fused RMSNorm: slower scalar staging
      for (int i = tid; i < MT * KT; i += blockDim.x) {
        const int m = i / KT, c = i % KT;
        float v = 0.f;
        if (m < mrows)
          v = __bfloat162float(X[(size_t)(m0 + m) * ldX + j0 + c]) * rrms[m0 + m] *
              (1.f + __bfloat162float(norm_w[j0 + c]));
        Xs[(size_t)m * XS + c] = __float2bfloat16(v);
      }
    }
    if (tid < CPT) {
      const int p = tid / (KT / 8), c = (tid % (KT / 8)) * 8;
      const __half* h = reinterpret_cast<const __half*>(&creg[B]);
      uint32_t* dst = Cs + p * (KT / 2) + c / 2;
#pragma unroll
      for (int t = 0; t < 4; ++t)
        dst[t] = col_scale ? detail::pack_bf16x2(__half2float(h[2 * t]),
                                                 __half2float(h[2 * t + 1]))
                           : 0x3F803F80u;   // bf16x2(1,1)
    }
  };

  // (zero, scale) live in a [BR rows][GC groups] cache refreshed every GC*GROUP columns.
  // Stored pre-packed as bf16x2 so the inner loop is two smem words per weight row.
  auto load_sz = [&](int g0) {
    for (int i = tid; i < BR * GC; i += blockDim.x) {
      const int r = i / GC, c = i % GC;
      const int gr = row0 + r, g = g0 + c;
      const bool ok = (r < nrows) && (gr < w.K) && (g < w.G);
      const float sf = ok ? __half2float(w.scale[(size_t)gr * w.G + g]) : 0.f;
      const float zf = ok ? (float)w.zero[(size_t)gr * w.G + g] : 0.f;
      // magic offset: nib2_to_bf16x2 yields 128+code, so DENSE4 subtracts (128+zero);
      // DENSE8 keeps the raw zero (both are integers <= 256, exact in bf16).
      const float zo = d8 ? zf : (128.f + zf);
      SZ[2 * i + 0] = detail::pack_bf16x2(zo, zo);
      SZ[2 * i + 1] = detail::pack_bf16x2(sf, sf);
    }
  };

  int szblk = -1;

  // One pipeline stage: commit buffer B (loaded PF stages ago), refill it with stage s+PF,
  // then run the mma over the smem tile.
  auto stage = [&](auto BT, int s) {
    const int j0 = s * KT;
    __syncthreads();                     // everyone finished reading the previous stage
    commit(BT, j0);
    const int g = j0 / GROUP;
    if ((g / GC) != szblk) { szblk = g / GC; load_sz(szblk * GC); }
    __syncthreads();
    if (s + PF < nstage) prefetch(BT, j0 + PF * KT);   // issue: overlaps the compute below
    const int gsel = g - szblk * GC;
    const __nv_bfloat16* xb = Xs;
    const uint8_t*       wb = Ws;

    // Per-warp weight rows are FIXED for the whole stage, so hoist their (zero, scale)
    // and (for cs_pairs) their col_scale parity out of the k loop.
    uint32_t zsr[NPW][NG], scr[NPW][NG];
#pragma unroll
    for (int j = 0; j < NPW; ++j) {
      const int rl = (wn * NPW + j) * 8 + grp;
#pragma unroll
      for (int gg = 0; gg < NG; ++gg) {
        const uint32_t* p = SZ + (size_t)(rl * GC + gsel + gg) * 2;
        zsr[j][gg] = p[0];
        scr[j][gg] = p[1];
      }
    }
    // rl = base + grp with base a multiple of 8 => the parity of (row0+rl) is the same for
    // every k-step and every j, so the gate/up vector selection is a per-lane constant.
    const uint32_t* csb = Cs + (cs_pairs ? ((size_t)((row0 + grp) & 1) * (KT / 2)) : 0);

    // ---- compute this stage (KT/16 k-steps)
#ifndef CBK_GEMM_NOCOMPUTE
#pragma unroll
    for (int ks = 0; ks < KT / 16; ++ks) {
      const int kb = ks * 16;
      constexpr int KPG = (KT > GROUP) ? (GROUP / 16) : (KT / 16);  // k-steps per group
      const int gg = (NG > 1) ? (ks / KPG) : 0;
      uint32_t a[MPW][4];
#pragma unroll
      for (int i = 0; i < MPW; ++i) {
        const int mbase = (wm * MPW + i) * 16;
#ifdef CBK_GEMM_NO_LDMATRIX
        const __nv_bfloat16* p0 = xb + (size_t)(mbase + grp) * XS + kb + 2 * tig;
        const __nv_bfloat16* p1 = xb + (size_t)(mbase + grp + 8) * XS + kb + 2 * tig;
        a[i][0] = *reinterpret_cast<const uint32_t*>(p0);
        a[i][1] = *reinterpret_cast<const uint32_t*>(p1);
        a[i][2] = *reinterpret_cast<const uint32_t*>(p0 + 8);
        a[i][3] = *reinterpret_cast<const uint32_t*>(p1 + 8);
#else
        detail::ldmatrix_x4(a[i], xb + (size_t)(mbase + lmrow) * XS + kb + lmcol);
#endif
      }
      const uint32_t cl = csb[kb / 2 + tig];        // bf16x2 for columns kb+2tig, +1
      const uint32_t ch = csb[kb / 2 + tig + 4];    // bf16x2 for columns kb+2tig+8, +9
#pragma unroll
      for (int j = 0; j < NPW; ++j) {
        const int rl = (wn * NPW + j) * 8 + grp;
        uint32_t b[2];
        if constexpr (BYTEPL) {
          // BLK1632_6: one code BYTE per column, magic-number dequant in ONE prmt.
          const uint8_t* rp = wb + (size_t)rl * WSB + kb;
          const uint32_t v0 = *reinterpret_cast<const uint16_t*>(rp + 2 * tig);
          const uint32_t v1 = *reinterpret_cast<const uint16_t*>(rp + 2 * tig + 8);
          b[0] = detail::bf2_mul(
              detail::bf2_mul(detail::bf2_sub(detail::byte2_to_bf16x2(v0), zsr[j][gg]),
                              scr[j][gg]), cl);
          b[1] = detail::bf2_mul(
              detail::bf2_mul(detail::bf2_sub(detail::byte2_to_bf16x2(v1), zsr[j][gg]),
                              scr[j][gg]), ch);
        } else if (d8) {
          const float sf = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scr[j][gg]));
          const float zf = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&zsr[j][gg]));
          const __nv_bfloat162 c0 = *reinterpret_cast<const __nv_bfloat162*>(&cl);
          const __nv_bfloat162 c1 = *reinterpret_cast<const __nv_bfloat162*>(&ch);
          const uint8_t* rp = wb + (size_t)rl * WS + kb;
          const uint16_t v0 = *reinterpret_cast<const uint16_t*>(rp + 2 * tig);
          const uint16_t v1 = *reinterpret_cast<const uint16_t*>(rp + 2 * tig + 8);
          b[0] = detail::pack_bf16x2(((float)(v0 & 0xFF) - zf) * sf * __bfloat162float(c0.x),
                                     ((float)(v0 >> 8) - zf) * sf * __bfloat162float(c0.y));
          b[1] = detail::pack_bf16x2(((float)(v1 & 0xFF) - zf) * sf * __bfloat162float(c1.x),
                                     ((float)(v1 >> 8) - zf) * sf * __bfloat162float(c1.y));
        } else {
          const uint8_t* rp = wb + (size_t)rl * WS + (kb >> 1);
          const uint32_t y0 = rp[tig], y1 = rp[tig + 4];
          b[0] = detail::bf2_mul(
              detail::bf2_mul(detail::bf2_sub(detail::nib2_to_bf16x2(y0), zsr[j][gg]),
                              scr[j][gg]), cl);
          b[1] = detail::bf2_mul(
              detail::bf2_mul(detail::bf2_sub(detail::nib2_to_bf16x2(y1), zsr[j][gg]),
                              scr[j][gg]), ch);
        }
#pragma unroll
        for (int i = 0; i < MPW; ++i) detail::mma16816(acc[i][j], a[i], b);
      }
    }
#endif
  };

  // ---- prime the PF-deep pipeline, then run it
  prefetch(detail::ic<0>{}, 0);
  if (PF > 1 && 1 < nstage) prefetch(detail::ic<(PF > 1) ? 1 : 0>{}, KT);
  if (PF > 2 && 2 < nstage) prefetch(detail::ic<(PF > 2) ? 2 : 0>{}, 2 * KT);
  for (int s = 0; s < nstage; s += PF) {
    stage(detail::ic<0>{}, s);
    if (PF > 1 && s + 1 < nstage) stage(detail::ic<(PF > 1) ? 1 : 0>{}, s + 1);
    if (PF > 2 && s + 2 < nstage) stage(detail::ic<(PF > 2) ? 2 : 0>{}, s + 2);
  }

  // ---------------- epilogue
#pragma unroll
  for (int i = 0; i < MPW; ++i) {
    const int mbase = (wm * MPW + i) * 16;
#pragma unroll
    for (int j = 0; j < NPW; ++j) {
      const int nbase = (wn * NPW + j) * 8;
#pragma unroll
      for (int h = 0; h < 2; ++h) {          // h=0 -> rows grp, h=1 -> rows grp+8
        const int ml = mbase + grp + 8 * h;
        if (ml >= mrows) continue;
        const int m = m0 + ml;
        const float v0 = acc[i][j][2 * h + 0];   // weight-row nbase + 2*tig
        const float v1 = acc[i][j][2 * h + 1];   // weight-row nbase + 2*tig + 1
        const int rl0 = nbase + 2 * tig;
        if (EPI == EPI_GEGLU_BF16 || EPI == EPI_SWIGLU_BF16) {
          if (rl0 + 1 < nrows) {
            const int idx = (row0 + rl0) >> 1;
            const float g = (EPI == EPI_GEGLU_BF16) ? detail::gelu_tanh(v0) : detail::silu_f(v0);
            reinterpret_cast<__nv_bfloat16*>(Y)[(size_t)m * ldY + idx] =
                __float2bfloat16(g * v1);
          }
        } else if (EPI == EPI_BF16) {
          if (rl0 + 0 < nrows)
            reinterpret_cast<__nv_bfloat16*>(Y)[(size_t)m * ldY + row0 + rl0] =
                __float2bfloat16(v0);
          if (rl0 + 1 < nrows)
            reinterpret_cast<__nv_bfloat16*>(Y)[(size_t)m * ldY + row0 + rl0 + 1] =
                __float2bfloat16(v1);
        } else {
          if (rl0 + 0 < nrows)
            reinterpret_cast<float*>(Y)[(size_t)m * ldY + row0 + rl0] = v0;
          if (rl0 + 1 < nrows)
            reinterpret_cast<float*>(Y)[(size_t)m * ldY + row0 + rl0 + 1] = v1;
        }
      }
    }
  }
}

}  // namespace cbk
