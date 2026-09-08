// cobalt_gemv.cuh -- warp-cooperative dequant-GEMV over a CBK1 packed matrix.
//
// Contract (kernel B codes against this):
//   template<int M> __device__ void cbk::gemv_rows(const Mat& m, int row,
//                                                  const __nv_bfloat16* x, float out[M],
//                                                  uint8_t* scratch);
//   __host__ __device__ constexpr int cbk::scratch_bytes(int N);
//   __device__ void cbk::dequant_row(const Mat& m, int row, __nv_bfloat16* out);
//
// * Called by a FULL warp (32 active lanes). Result is valid in ALL lanes.
// * `x` holds M activation vectors of length N, laid out x[m*N + j], bf16, ALREADY
//   multiplied by the matrix's column scale c[j]. Usually in shared memory.
// * `out[M]` is ACCUMULATED into? No -- it is OVERWRITTEN (assigned).
// * M in {1,2,4,8}. fp32 FMA path.
#pragma once
#include "cobalt_format.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace cbk {

// scratch is currently unused by every layout; a non-zero value is returned so callers
// never request a 0-byte dynamic-smem region (see ENV.md gotcha #5).
__host__ __device__ constexpr int scratch_bytes(int /*N*/) { return 16; }

namespace detail {

constexpr uint32_t FULL = 0xffffffffu;

// Activation addressing.  Default (contract) layout is x[m*N + j].  Building the
// extension with -DCBK_X_INTERLEAVED switches to x[j*M + m], which removes the shared-
// memory bank conflict between the M loads of one column (they become contiguous and
// vectorisable).  Measured: identical at M=1, ~4-7x faster at M=4/8.  See FORMAT.md sec.8.
template<int M>
__device__ __forceinline__ float xat(const __nv_bfloat16* x, int m, int N, int col) {
#ifdef CBK_X_INTERLEAVED
  return __bfloat162float(x[(size_t)col * M + m]);
#else
  return __bfloat162float(x[(size_t)m * N + col]);
#endif
}

template<int M>
__device__ __forceinline__ void warp_reduce(float* a) {
  #pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    #pragma unroll
    for (int m = 0; m < M; ++m) a[m] += __shfl_xor_sync(FULL, a[m], d);
  }
}


// ---------------------------------------------------------------- EXPAND-TO-DENSE
// PRMT selector table for the expand-to-dense sparse decoder.  Indexed [2*b] / [2*b+1]
// for a bitmap byte b; each uint32 holds four 4-bit __byte_perm selectors.
//   kept  position p -> selector = s(r), r = popcount(b & ((1<<p)-1)),
//                       s(k) = (k>>1) | ((k&1)<<2)   (byte index inside {lo,hi})
//   pruned position p -> selector = 8.  MEASURED on sm_120: CUDA's __byte_perm ignores
//   bit 3 of a selector nibble (no PTX prmt sign-replicate mode), so the pruned slots are
//   zeroed instead by ANDing with a byte mask built arithmetically from the bitmap byte
//   (bit i -> byte i, via the 0x00204081 spread-multiply) -- no second table needed.
//
// CBK_BLK1632_LUT selects HOW the selector is obtained inside
// the BLK16_32 multi-row decoder, which is the *latency-bound* path in the decode
// megakernel: the table costs TWO dependent loads per
// bitmap byte inside the per-granule dependency chain.
//   0 = __device__ global const table (the shipped path, L1-resident)
//   1 = __constant__ table            (NB: the index is lane-DIVERGENT, so the constant
//                                      cache serialises per distinct address)
//   2 = __shared__ copy staged once per block at megakernel entry
//   3 = ARITHMETIC, no table at all   (see prmt_sel_arith below)
#ifndef CBK_BLK1632_LUT
#define CBK_BLK1632_LUT 0
#endif
#ifndef CBK_BLK1632_NOXB
#define CBK_BLK1632_NOXB 0
#endif
// CBK_BLK1632_ZF: the DENSE4-style zero-point trick for the MULTI-ROW
// BLK16_32 decoder.  The shipped path keeps pruned positions exactly 0 by accumulating a
// per-ROW MASKED sum of x (`xs[m] = fmaf(km, xv, xs[m])`, plus the `km` extraction) --
// 8*M FMAs and ~16 ALU ops per bitmap byte PER ROW.  With ZF the pruned slots are filled
// with the group's ZERO code instead, so their contribution cancels in the final
// `qs - zf*xs` exactly as it does for a DENSE4 pruned weight, and `xs` becomes a PLAIN
// sum(x) shared by all R rows -- structurally identical to gemv_dense4_multi.  Cost: the
// cancellation is now floating-point rather than structural, i.e. the SAME numerics class
// the shipped DENSE4 arm already has.  Requires a verify_kernel.py re-gate.
#ifndef CBK_BLK1632_ZF
#define CBK_BLK1632_ZF 0
#endif
// CBK_BLK1632_HALFJ -- DIAGNOSTIC ONLY, NUMERICALLY WRONG.
// An open question is whether COMPRESSING x onto the 16 survivor lanes (16 FMAs) beats
// EXPANDING the codes to 32 dense slots (32 FMAs, ~40 % of them on pruned lanes).  Any
// compression scheme must pay for the compaction on top; this macro measures the
// CEILING first -- it keeps every load, every selector and every e0/e1 ALU op, and only
// halves the number of (byte-extract + cvt + FMA) triples in the inner loop.  Whatever
// this buys is a STRICT UPPER BOUND on that alternative.  Never enable in a record build.
#ifndef CBK_BLK1632_HALFJ
#define CBK_BLK1632_HALFJ 0
#endif
// CBK_BLK1632_NOOVH -- DIAGNOSTIC ONLY, NUMERICALLY WRONG.  The complement
// of HALFJ: keeps all 8 (extract, cvt, FMA) triples per bitmap byte but deletes the
// per-granule SELECTOR/MASK overhead (the two spreads, the two byte masks, the two
// selector lookups, the two __byte_perm's and the ZF blend) by using the raw nibble
// planes as if they were already expanded.  Together HALFJ and NOOVH decompose the
// granule's instruction cost into its two halves.  Never enable in a record build.
#ifndef CBK_BLK1632_NOOVH
#define CBK_BLK1632_NOOVH 0
#endif
// CBK_BLK1632_BIAS: magic-bias byte->float.  See the patch_bias docstring.
// Needs CBK_BLK1632_ZF (the bias is cancelled through the plain sum(x)).
#ifndef CBK_BLK1632_BIAS
#define CBK_BLK1632_BIAS 0
#endif
// CBK_BLK1632_NOLD -- DIAGNOSTIC ONLY, NUMERICALLY WRONG.  BLK16_32 issues
// TWO weight loads per row per granule (a uint32 from the mask plane and a uint2 from the
// nibble plane, 12 B) where DENSE4 issues ONE uint4 (16 B).  1 = drop the mask load,
// 2 = drop the nibble load, 3 = drop both.  The ALU is untouched, so the delta prices the
// per-granule LOAD-INSTRUCTION count (not bytes) in the latency-bound decode kernel.
#ifndef CBK_BLK1632_NOLD
#define CBK_BLK1632_NOLD 0
#endif
// CBK_BLK1632_MSCHED: where the MASK-plane load sits.  MEASURED (2g, 27B):
// deleting the mask load is worth +32 % and deleting the nibble load +22 %, i.e. what is
// left of the DENSE4 gap lives in the WEIGHT-LOAD path, not in the decoder's arithmetic
// (the whole (extract, cvt, FMA) inner loop is capped at +2.4 %).  The
// mask word also sits at the HEAD of the per-granule dependency chain
// (mask -> popc -> funnel shift -> byte_perm -> FMA), so its latency is paid twice.
//   0 = shipped: mask and nibble loads interleaved per (granule, row)
//   1 = MFIRST : every mask load of the iteration issued BEFORE any nibble load (free)
//   2 = MPF    : the NEXT iteration's masks are issued before the current compute
//                (one extra PF*R registers of lookahead)
#ifndef CBK_BLK1632_MSCHED
#define CBK_BLK1632_MSCHED 0
#endif
#if CBK_BLK1632_BIAS && !CBK_BLK1632_ZF
#error "CBK_BLK1632_BIAS requires CBK_BLK1632_ZF"
#endif
#if CBK_BLK1632_ZF
#define ZBV(i) zbv[i]
#else
#define ZBV(i) 0u
#endif

#if CBK_BLK1632_LUT == 1
__constant__ uint32_t prmt_lut[512] = {
#include "prmt_lut.inc"
};
#else
__device__ static const uint32_t prmt_lut[512] = {
#include "prmt_lut.inc"
};
#endif

#if CBK_BLK1632_LUT == 2
// One shared copy per BLOCK.  The array lives inside a __noinline__ accessor so that
// exactly ONE allocation exists no matter how many times the decoder is inlined; the
// megakernel stages it once at entry (prmt_smem_stage) where every thread is present.
// If nvcc were to inline it anyway the staged and the read copies would differ and the
// 120-case unit test would fail loudly -- it does not.
__device__ __noinline__ uint32_t* prmt_smem() { __shared__ uint32_t s[512]; return s; }
__device__ __forceinline__ void prmt_smem_stage() {
  uint32_t* s = prmt_smem();
  for (int i = threadIdx.x; i < 512; i += blockDim.x) s[i] = prmt_lut[i];
}
#else
__device__ __forceinline__ void prmt_smem_stage() {}
#endif

// ---- CBK_BLK1632_LUT == 3: the selector computed, no memory op at all.
// For bitmap byte b8 and output position j, the selector nibble is s(r_j) with
// r_j = popc(b8 & ((1<<j)-1)) and s(k) = (k>>1) | ((k&1)<<2).  Both halves follow from
// the SAME 0x00204081 bit->byte spread the pruned-slot mask already builds:
//   s0 = spread(b8 & 0xF)                      byte j = bit j
//   P0 = (s0 << 8) * 0x01010101                byte j = exclusive prefix popcount = r_j
//   P1 = (s1 << 8) * 0x01010101 + popc(b8&0xF) (the high half starts at o4)
// then bytes r -> nibbles s(r) by  (r>>1) | ((r&1)<<2)  and a byte->nibble compaction.
// VERIFIED against the 512-entry table for all 256 bitmap bytes at every KEPT position
// (pruned positions are don't-care -- they are zeroed by mk0/mk1 either way).
__device__ __forceinline__ uint32_t prmt_sel_arith(uint32_t sp) {
  const uint32_t P = (sp << 8) * 0x01010101u;      // byte j = r_j
  const uint32_t sb = ((P >> 1) & 0x03030303u) | ((P & 0x01010101u) << 2);
  const uint32_t mm = (sb & 0x000F000Fu) | ((sb >> 4) & 0x00F000F0u);
  return (mm | (mm >> 8)) & 0xFFFFu;
}
__device__ __forceinline__ uint32_t prmt_sel_arith_hi(uint32_t sp, uint32_t o4) {
  const uint32_t P = ((sp << 8) * 0x01010101u) + o4 * 0x01010101u;
  const uint32_t sb = ((P >> 1) & 0x03030303u) | ((P & 0x01010101u) << 2);
  const uint32_t mm = (sb & 0x000F000Fu) | ((sb >> 4) & 0x00F000F0u);
  return (mm | (mm >> 8)) & 0xFFFFu;
}

// ---------------------------------------------------------------- SPARSE4 / SPARSE4X
// Window = 1024 columns = 8 groups; lane l owns the 32 columns [c0+32l, c0+32l+32),
// i.e. exactly one 32-bit bitmap word.  4 lanes per group.
template<int M, bool HAS_GOFF>
__device__ __forceinline__ void gemv_sparse(const Mat& mt, int row,
                                            const __nv_bfloat16* __restrict__ x, float* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N, G = mt.G;
  const uint8_t*  rb   = mt.data + mt.row_off[row];
  const uint32_t* bmp  = reinterpret_cast<const uint32_t*>(rb);
  const int       hdr  = sparse_row_header(HAS_GOFF ? LAYOUT_SPARSE4X : LAYOUT_SPARSE4, N, G);
  const uint16_t* goff = HAS_GOFF ? reinterpret_cast<const uint16_t*>(rb + (N >> 3)) : nullptr;
  const uint32_t* codes = reinterpret_cast<const uint32_t*>(rb + hdr);
  const __half*   sc = mt.scale + (size_t)row * G;
  const uint8_t*  zp = mt.zero  + (size_t)row * G;
  const int nwords = N >> 5;

  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  int run_bytes = 0;                       // codes byte offset of the current window
  for (int c0 = 0; c0 < N; c0 += 1024) {
    const int wi = (c0 >> 5) + lane;
    uint32_t bits = (wi < nwords) ? bmp[wi] : 0u;
    const int p = __popc(bits);
    const int j = lane >> 2;                       // group index inside the window
    const int gg = (c0 >> 7) + j;                  // global group index
    int gbyte, e;

    if (HAS_GOFF) {
      // explicit per-group byte offsets: only a 4-lane segmented scan is needed
      int s = p;
      int t1 = __shfl_up_sync(FULL, s, 1); if ((lane & 3) >= 1) s += t1;
      int t2 = __shfl_up_sync(FULL, s, 2); if ((lane & 3) >= 2) s += t2;
      e = s - p;
      gbyte = (gg < G) ? (int)goff[gg] : 0;
    } else {
      // derive the group byte offsets from a 32-lane warp scan of the popcounts
      int s = p;
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) { int t = __shfl_up_sync(FULL, s, d); if (lane >= d) s += t; }
      int sm4 = __shfl_up_sync(FULL, s, 4); if (lane < 4) sm4 = 0;
      // lanes l == 3 (mod 4) hold their group's survivor count in (s - sm4)
      const uint32_t oddm = __ballot_sync(FULL, ((lane & 3) == 3) && (((s - sm4) & 1) != 0));
      // NB: __shfl_sync must be executed by every lane of the mask -- never inside a ternary.
      const int esh = __shfl_sync(FULL, s, (j == 0) ? 0 : ((j << 2) - 1));
      const int egrp = (j == 0) ? 0 : esh;         // group's nibble prefix in the window
      e = (s - p) - egrp;                          // this lane's nibble prefix in its group
      // sum_{i<j} ceil(cnt_i/2) = (e + #odd groups before j) / 2   (always even)
      gbyte = run_bytes + ((egrp + __popc(oddm & ((1u << (j << 2)) - 1u))) >> 1);
      const int stot = __shfl_sync(FULL, s, 31);
      run_bytes += (stot + __popc(oddm)) >> 1;
    }

    if (bits) {
      int idx = (gbyte << 1) + e;                  // this lane's first nibble index
      // Measured: a straight scalar walk of the nibble stream beats both a 6-word
      // register preload and a software-pipelined 64-bit shift register (the refill load
      // almost always hits L1, and the extra live registers cost occupancy).  See
      // FORMAT.md sec. 7 for the three variants and their numbers.
      uint32_t cwv = codes[idx >> 3];
      const float sf = __half2float(sc[gg]);
      const float zf = (float)zp[gg];
      const int base = c0 + (lane << 5);
      float qs[M], xs[M];
      #pragma unroll
      for (int m = 0; m < M; ++m) { qs[m] = 0.f; xs[m] = 0.f; }
      while (bits) {
        const int t = __ffs(bits) - 1; bits &= bits - 1;
        const float qv = (float)((cwv >> ((idx & 7) << 2)) & 0xFu);
        ++idx;
        if (bits && ((idx & 7) == 0)) cwv = codes[idx >> 3];
        const int col = base + t;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          const float xv = xat<M>(x, m, N, col);
          qs[m] = fmaf(qv, xv, qs[m]);
          xs[m] += xv;
        }
      }
      #pragma unroll
      for (int m = 0; m < M; ++m) acc[m] = fmaf(sf, qs[m] - zf * xs[m], acc[m]);
    }
  }
  warp_reduce<M>(acc);
  #pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}


// ---------------------------------------------------------------- SPARSE4E
// Expand-to-dense sparse decoder.  Never gathers x: for each bitmap BYTE it expands the
// (up to 8) survivor nibbles to 8 dense byte slots with two __byte_perm, then runs the
// DENSE4 inner loop with sequential (vectorisable) activation reads.
//   pruned slot -> expanded byte 0, and the zero-point is applied only over KEPT slots
//   via a masked xs accumulator, so sum_kept (code-z)*x is exact.
// Container = SPARSE4X (goff table makes the per-lane start a single uint16 read).
template<int M>
__device__ __forceinline__ void gemv_sparse_expand(const Mat& mt, int row,
                                                   const __nv_bfloat16* __restrict__ x,
                                                   float* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N, G = mt.G;
  const uint8_t*  rb    = mt.data + mt.row_off[row];
  const uint32_t* bmp   = reinterpret_cast<const uint32_t*>(rb);
  const uint16_t* goff  = reinterpret_cast<const uint16_t*>(rb + (N >> 3));
  const uint32_t* codes = reinterpret_cast<const uint32_t*>(
      rb + sparse_row_header(LAYOUT_SPARSE4X, N, G));
  const __half*   sc = mt.scale + (size_t)row * G;
  const uint8_t*  zp = mt.zero  + (size_t)row * G;
  const int nwords = N >> 5;

  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (int c0 = 0; c0 < N; c0 += 1024) {
    const int wi = (c0 >> 5) + lane;
    const uint32_t bits = (wi < nwords) ? bmp[wi] : 0u;
    const int p = __popc(bits);
    // 4-lane segmented scan -> this lane's nibble prefix inside its group
    int sc4 = p;
    { int t1 = __shfl_up_sync(FULL, sc4, 1); if ((lane & 3) >= 1) sc4 += t1;
      int t2 = __shfl_up_sync(FULL, sc4, 2); if ((lane & 3) >= 2) sc4 += t2; }
    const int e = sc4 - p;
    if (!bits) continue;
    const int gg = (c0 >> 7) + (lane >> 2);
    const int idx = ((int)goff[gg] << 1) + e;

    const uint32_t* cw = codes + (idx >> 3);
    int wk = 1;
    uint64_t buf = ((((uint64_t)cw[1]) << 32) | (uint64_t)cw[0]) >> ((idx & 7) << 2);
    int have = 16 - (idx & 7);
    const float sf = __half2float(sc[gg]);
    const float zf = (float)zp[gg];
    const int base = c0 + (lane << 5);
    float qs[M], xs[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) { qs[m] = 0.f; xs[m] = 0.f; }

    #pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      if (have < 8) { buf |= ((uint64_t)cw[++wk]) << (have << 2); have += 8; }
      const uint32_t b8 = (bits >> (q4 << 3)) & 0xFFu;
      const uint32_t cwd = (uint32_t)buf;               // next 8 survivor nibbles
      const uint32_t lo = cwd & 0x0F0F0F0Fu;            // bytes = nibbles 0,2,4,6
      const uint32_t hi = (cwd >> 4) & 0x0F0F0F0Fu;     // bytes = nibbles 1,3,5,7
      // bit i of a nibble -> byte i, then 0x01 -> 0xFF : zeroes the pruned slots
      const uint32_t mk0 = ((((b8 & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu);
      const uint32_t mk1 = (((((b8 >> 4) & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu);
#ifdef CBK_LUT_FAKE
      const uint32_t sel0 = 0x3210u, sel1 = 0x7654u;   // DIAGNOSTIC ONLY: wrong results
#else
      const uint32_t sel0 = prmt_lut[b8 << 1], sel1 = prmt_lut[(b8 << 1) | 1];
#endif
      const uint32_t e0 = __byte_perm(lo, hi, sel0) & mk0;        // slots 0..3
      const uint32_t e1 = __byte_perm(lo, hi, sel1) & mk1;        // slots 4..7
      const int pc = __popc(b8);
      buf >>= (pc << 2); have -= pc;
      #pragma unroll
      for (int t = 0; t < 8; ++t) {
        const uint32_t ew = (t < 4) ? e0 : e1;
        const float qv = (float)((ew >> ((t & 3) << 3)) & 0xFFu);
        const float km = ((b8 >> t) & 1u) ? 1.f : 0.f;
        const int col = base + (q4 << 3) + t;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          const float xv = xat<M>(x, m, N, col);
          qs[m] = fmaf(qv, xv, qs[m]);
          xs[m] = fmaf(km, xv, xs[m]);
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = fmaf(sf, qs[m] - zf * xs[m], acc[m]);
  }
  warp_reduce<M>(acc);
  #pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}

// ---------------------------------------------------------------- BLK16_32 (sec.13)
// Fixed-cardinality 16-of-32 mask: EVERY aligned block of 32 columns has EXACTLY 16
// survivors, so its codes sit at a FIXED byte offset in the row.  Consequences vs
// SPARSE4E: no goff table, no warp prefix scan, NO loop-carried 64-bit shift chain.
// Inside a block the four bitmap bytes are decoded with four INDEPENDENT funnel shifts
// (`o` = popcount of the preceding bitmap bytes, computed directly from the mask word),
// then expanded to dense byte slots with SPARSE4E's two __byte_perm + arithmetic AND
// mask, and consumed by the DENSE4 inner loop over SEQUENTIAL activations (no gather,
// no shared-memory staging required).
//
//   BITS = 4 : row = [mask N/8][nibble N/4]                 (3N/8 B/row, 3.0 bpw codes)
//   BITS = 6 : row = [mask N/8][nibble N/4][hi2 N/8]        (N/2 B/row, 4.0 bpw codes)
//   BPG      : blocks decoded per lane per granule (1 -> 32 cols, 2 -> 64, 4 -> 128)
//
// hi2 plane (BITS=6): one uint32 per block; bits [0,16) hold the high-2-bit halves of
// the survivors with EVEN within-block index (0,2,..,14), bits [16,32) the ODD ones.
// That parity split is what makes the pool extraction two shifts: the __byte_perm pool
// wants {c_o, c_{o+2}, c_{o+4}, c_{o+6}} in one operand and {c_{o+1}, ...} in the other,
// which are exactly the two parity words at a contiguous offset.
//
// Pruned positions carry NO code; the zero point is applied only over KEPT slots via the
// masked `xs` accumulator, so a pruned column contributes EXACTLY 0.0f regardless of
// whether `zero` is on the quantization grid (FORMAT.md sec.11 trap does not apply).
template<int M, int BITS, int BPG>
__device__ __forceinline__ void gemv_blk1632(const Mat& mt, int row,
                                             const __nv_bfloat16* __restrict__ x,
                                             float* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N;
  const size_t stride = (BITS == 6) ? (size_t)(N >> 1) : ((size_t)N * 3 >> 3);
  const uint8_t* rb = mt.data + (size_t)row * stride;
  const uint32_t* mp = reinterpret_cast<const uint32_t*>(rb);
  const uint8_t*  np = rb + (N >> 3);
  const uint32_t* hp = reinterpret_cast<const uint32_t*>(rb + ((size_t)N * 3 >> 3));
  const __half*  sc = mt.scale + (size_t)row * mt.G;
  const uint8_t* zp = mt.zero  + (size_t)row * mt.G;
  const int NB = N >> 5;                      // 32-column blocks in the row

  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (int t0 = lane * BPG; t0 < NB; t0 += 32 * BPG) {
    uint32_t mw[BPG]; uint64_t nw[BPG]; uint32_t hw[BPG];
    if (BPG == 1) {
      mw[0] = mp[t0];
      nw[0] = *reinterpret_cast<const uint64_t*>(np + ((size_t)t0 << 3));
      if (BITS == 6) hw[0] = hp[t0];
    } else if (BPG == 2) {
      const uint2 mv = *reinterpret_cast<const uint2*>(mp + t0);
      mw[0] = mv.x; mw[1] = mv.y;
      const uint4 nv = *reinterpret_cast<const uint4*>(np + ((size_t)t0 << 3));
      nw[0] = (((uint64_t)nv.y) << 32) | (uint64_t)nv.x;
      nw[1] = (((uint64_t)nv.w) << 32) | (uint64_t)nv.z;
      if (BITS == 6) { const uint2 hv = *reinterpret_cast<const uint2*>(hp + t0);
                       hw[0] = hv.x; hw[1] = hv.y; }
    } else {   // BPG == 4: one 128-column quantization GROUP per lane, all loads 16-B
      const uint4 mv = *reinterpret_cast<const uint4*>(mp + t0);
      mw[0] = mv.x; mw[1] = mv.y; mw[2] = mv.z; mw[3] = mv.w;
      const uint4 n0 = *reinterpret_cast<const uint4*>(np + ((size_t)t0 << 3));
      const uint4 n1 = *reinterpret_cast<const uint4*>(np + ((size_t)t0 << 3) + 16);
      nw[0] = (((uint64_t)n0.y) << 32) | (uint64_t)n0.x;
      nw[1] = (((uint64_t)n0.w) << 32) | (uint64_t)n0.z;
      nw[2] = (((uint64_t)n1.y) << 32) | (uint64_t)n1.x;
      nw[3] = (((uint64_t)n1.w) << 32) | (uint64_t)n1.z;
      if (BITS == 6) { const uint4 hv = *reinterpret_cast<const uint4*>(hp + t0);
                       hw[0] = hv.x; hw[1] = hv.y; hw[2] = hv.z; hw[3] = hv.w; }
    }
    #pragma unroll
    for (int u = 0; u < BPG; ++u) {
      const int t = t0 + u;
      const uint32_t m32 = mw[u];
      const uint64_t buf = nw[u];
      const int gg = t >> 2;                   // GROUP=128 = 4 blocks
      const float sf = __half2float(sc[gg]);
      const uint32_t zi8 = (uint32_t)zp[gg];
      const float zf = (float)zi8;
      const int base = t << 5;
#ifdef CBK_BLK1632_ZFILL
      // DIAGNOSTIC (see FORMAT.md sec.13): fill pruned slots with `zero` instead of 0 and
      // run DENSE4's inner loop verbatim (plain sum(x), no masked xs).  Faster, but the
      // pruned cancellation then happens only in the final `qs - zf*xs` subtraction, so it
      // is NOT bit-exact for large activations -- exactly the DENSE4 behaviour.
      const uint32_t zb = zi8 * 0x01010101u;
#endif
      float qs[M], xs[M];
      #pragma unroll
      for (int m = 0; m < M; ++m) { qs[m] = 0.f; xs[m] = 0.f; }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t b8 = (m32 >> (i << 3)) & 0xFFu;
        // survivors preceding this bitmap byte INSIDE the block -- independent per i
        const int o = (i == 0) ? 0 : __popc(m32 & ((1u << (i << 3)) - 1u));
        const uint32_t cwd = (uint32_t)(buf >> ((o & 15) << 2));   // 8 survivor nibbles
        uint32_t lo = cwd & 0x0F0F0F0Fu;          // bytes = survivors o, o+2, o+4, o+6
        uint32_t hi = (cwd >> 4) & 0x0F0F0F0Fu;   // bytes = survivors o+1, o+3, o+5, o+7
        if (BITS == 6) {
          const uint32_t h = hw[u];
          const int ia = o >> 1, od = o & 1;
          const uint32_t wa = od ? (h >> 16) : h;   // parity word holding survivor o
          const uint32_t wb = od ? h : (h >> 16);   // parity word holding survivor o+1
          const uint32_t pe = (wa >> (ia << 1)) & 0xFFu;
          const uint32_t po = (wb >> ((ia + od) << 1)) & 0xFFu;
          // 2-bit field j of a byte -> byte j.  MEASURED: the spread-MULTIPLY used for
          // the 1-bit case (0x00204081) is WRONG here -- the shifted copies overlap
          // (spacing 2 vs shift 6) and the adds carry into the next byte.  An OR-spread
          // is carry-free: v = pe | pe<<6 | pe<<12 | pe<<18, built in two steps.
          uint32_t ve = pe | (pe << 12); ve = (ve | (ve << 6)) & 0x03030303u;
          uint32_t vo = po | (po << 12); vo = (vo | (vo << 6)) & 0x03030303u;
          lo |= ve << 4;
          hi |= vo << 4;
        }
        const uint32_t mk0 = ((((b8 & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu);
        const uint32_t mk1 = (((((b8 >> 4) & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu);
#ifdef CBK_BLK1632_ZFILL
        const uint32_t e0 = (__byte_perm(lo, hi, prmt_lut[b8 << 1]) & mk0) | (zb & ~mk0);
        const uint32_t e1 = (__byte_perm(lo, hi, prmt_lut[(b8 << 1) | 1]) & mk1) | (zb & ~mk1);
#else
        const uint32_t e0 = __byte_perm(lo, hi, prmt_lut[b8 << 1]) & mk0;
        const uint32_t e1 = __byte_perm(lo, hi, prmt_lut[(b8 << 1) | 1]) & mk1;
#endif
        #pragma unroll
        for (int tt = 0; tt < 8; ++tt) {
          const uint32_t ew = (tt < 4) ? e0 : e1;
          const float qv = (float)((ew >> ((tt & 3) << 3)) & 0xFFu);
#ifndef CBK_BLK1632_ZFILL
          const float km = ((b8 >> tt) & 1u) ? 1.f : 0.f;
#endif
          const int col = base + (i << 3) + tt;
          #pragma unroll
          for (int m = 0; m < M; ++m) {
            const float xv = xat<M>(x, m, N, col);
            qs[m] = fmaf(qv, xv, qs[m]);
#ifdef CBK_BLK1632_ZFILL
            xs[m] += xv;
#else
            xs[m] = fmaf(km, xv, xs[m]);
#endif
          }
        }
      }
      #pragma unroll
      for (int m = 0; m < M; ++m) acc[m] = fmaf(sf, qs[m] - zf * xs[m], acc[m]);
    }
  }
  warp_reduce<M>(acc);
  #pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}

// ---------------------------------------------------------------- DENSE4
// Window = 1024 columns; lane l owns 32 columns = 16 bytes = one uint4 load.
template<int M>
__device__ __forceinline__ void gemv_dense4(const Mat& mt, int row,
                                            const __nv_bfloat16* __restrict__ x, float* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N;
  const uint8_t* rb = mt.data + (size_t)row * (size_t)(N >> 1);
  const __half*  sc = mt.scale + (size_t)row * mt.G;
  const uint8_t* zp = mt.zero  + (size_t)row * mt.G;

  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (int c0 = 0; c0 < N; c0 += 1024) {
    const int col = c0 + (lane << 5);
    if (col >= N) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(rb + (col >> 1));
    const int gg = col >> 7;
    const float sf = __half2float(sc[gg]);
    const float zf = (float)zp[gg];
    float qs[M], xs[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) { qs[m] = 0.f; xs[m] = 0.f; }
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t cw = (w == 0) ? v.x : (w == 1) ? v.y : (w == 2) ? v.z : v.w;
      #pragma unroll
      for (int t = 0; t < 8; ++t) {
        const float qv = (float)((cw >> (t << 2)) & 0xF);
        const int c = col + (w << 3) + t;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          const float xv = xat<M>(x, m, N, c);
          qs[m] = fmaf(qv, xv, qs[m]);
          xs[m] += xv;
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = fmaf(sf, qs[m] - zf * xs[m], acc[m]);
  }
  warp_reduce<M>(acc);
  #pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}

// ---------------------------------------------------------------- DENSE4, MULTI-ROW
// Deeper-MLP variant of gemv_dense4, added for the decode megakernel.
// Differences from gemv_dense4:
//   * R CONSECUTIVE rows are accumulated in ONE pass over the columns and PF column
//     granules are fetched per loop body, so a warp has PF*R independent 16-B weight
//     loads in flight instead of one;
//   * the activation vector is read as uint4 (8 bf16) rather than 32 scalar loads per
//     granule -- 4*M vector loads per 32 columns instead of 32*M scalar ones, and the
//     sum(x) term of the zero-point correction is computed ONCE for all R rows;
//   * `x` lives in GLOBAL memory (L1-resident); weights use __ldcg so the streaming
//     weight traffic does not evict x from L1;
//   * NX == 2 selects a SECOND activation vector for ODD rows (the fused gate/up
//     matrix: row 2p = gate uses xa, row 2p+1 = up uses xb).
// The per-lane column ORDER is identical to gemv_dense4 (granule t = lane, lane+32, ...)
// and scale/zero are applied per 32-column granule exactly as there, so a full-row call
// is bit-identical to a full-row gemv_dense4 call.
template<int M, int R, int NX, int PF>
__device__ __forceinline__ void gemv_dense4_multi(
    const uint8_t* __restrict__ data, size_t rstride,
    const __half* __restrict__ scale, const uint8_t* __restrict__ zero, int G, int N,
    const __nv_bfloat16* __restrict__ xa, const __nv_bfloat16* __restrict__ xb,
    float acc[R][M]) {
  const int lane = threadIdx.x & 31;
  const int NT = N >> 5;                       // 32-column granules
  #pragma unroll
  for (int i = 0; i < R; ++i)
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[i][m] = 0.f;

  for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
    uint4 w[PF][R];
    #pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int ts = t0 + s * 32;
      #pragma unroll
      for (int i = 0; i < R; ++i)
        w[s][i] = (ts < NT) ? __ldcg(reinterpret_cast<const uint4*>(
                                  data + (size_t)i * rstride + ((size_t)ts << 4)))
                            : make_uint4(0u, 0u, 0u, 0u);
    }
    #pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int ts = t0 + s * 32;
      if (ts >= NT) break;
      const int col = ts << 5, gg = ts >> 2;
      float qs[R][M], xsa[M], xsb[M];
      #pragma unroll
      for (int i = 0; i < R; ++i)
        #pragma unroll
        for (int m = 0; m < M; ++m) qs[i][m] = 0.f;
      #pragma unroll
      for (int m = 0; m < M; ++m) { xsa[m] = 0.f; xsb[m] = 0.f; }
      #pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint4 va[M], vb[M];
        #pragma unroll
        for (int m = 0; m < M; ++m)
          va[m] = *reinterpret_cast<const uint4*>(xa + (size_t)(col + (q4 << 3)) * M + m * 8);
        if (NX == 2) {
          #pragma unroll
          for (int m = 0; m < M; ++m)
            vb[m] = *reinterpret_cast<const uint4*>(xb + (size_t)(col + (q4 << 3)) * M + m * 8);
        }
        const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(va);
        const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(vb);
        float fa[8][M], fb[8][M];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
          #pragma unroll
          for (int m = 0; m < M; ++m) {
            fa[j][m] = __bfloat162float(ha[j * M + m]);
            xsa[m] += fa[j][m];
            if (NX == 2) { fb[j][m] = __bfloat162float(hb[j * M + m]); xsb[m] += fb[j][m]; }
          }
        #pragma unroll
        for (int i = 0; i < R; ++i) {
          const uint32_t cw = (q4 == 0) ? w[s][i].x : (q4 == 1) ? w[s][i].y
                            : (q4 == 2) ? w[s][i].z : w[s][i].w;
          #pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float qv = (float)((cw >> (j << 2)) & 0xF);
            #pragma unroll
            for (int m = 0; m < M; ++m)
              qs[i][m] = fmaf(qv, (NX == 2 && (i & 1)) ? fb[j][m] : fa[j][m], qs[i][m]);
          }
        }
      }
      #pragma unroll
      for (int i = 0; i < R; ++i) {
        const float sf = __half2float(scale[(size_t)i * G + gg]);
        const float zf = (float)zero[(size_t)i * G + gg];
        #pragma unroll
        for (int m = 0; m < M; ++m)
          acc[i][m] = fmaf(sf, qs[i][m] - zf * ((NX == 2 && (i & 1)) ? xsb[m] : xsa[m]),
                           acc[i][m]);
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < R; ++i) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      #pragma unroll
      for (int m = 0; m < M; ++m) acc[i][m] += __shfl_xor_sync(FULL, acc[i][m], d);
  }
}

// ------------------------------------------------------ BLK16_32, MULTI-ROW (sec.13)
// The BLK16_32 analogue of gemv_dense4_multi, for the decode megakernel.  Same
// contract, same work decomposition, same activation addressing (interleaved
// x[col*M + m], read as uint4), same __ldcg weight loads, same PF granule prefetch,
// same NX==2 fused gate/up selection.  A 32-column granule IS one 16:32 block, so the
// granule <-> group mapping (gg = ts>>2) is identical to DENSE4's.
//
//   BITS = 4 : row = [mask N/8][nibble N/4]              stride 3N/8
//   BITS = 6 : row = [mask N/8][nibble N/4][hi2 N/8]     stride N/2
//
// The zero-point term uses the MASKED sum of x (only kept columns), which is per-ROW
// here -- unlike DENSE4, where one sum(x) serves all R rows.  That is what keeps the
// pruned positions exactly 0.0f (FORMAT.md sec.13.3).
//
// FLATTAIL: the warp-tail fix.  With NT = N/32 granules handed out lane-cyclically a
// warp pays ceil(NT/32) rounds for NT/32 rounds of work -- 6 vs 5.25 at N=5376, i.e.
// 12.5 % of a LATENCY-bound kernel (DENSE4 has the same tail but is bandwidth-bound and
// does not pay for it).  With FLATTAIL the main loop covers only the floor(NT/32)*32
// granules that fill every lane, and the remaining R*(NT mod 32) (row, granule) items
// are flattened into one work list handed out lane-cyclically, so at R=4 / NT=168 the
// warp does exactly 21 full rounds of 32 items and the tail vanishes.  The flattened
// items do NOT share their x loads across rows, so only the tail pays that; the main
// loop keeps the R-row x sharing.
// USEB picks the SECOND activation vector (NX==2, odd = "up" rows).  It is a template
// parameter, not a runtime pointer select: a runtime `cond ? fb : fa` takes the address
// of both staging arrays and the compiler then keeps them in LOCAL memory -- MEASURED,
// that alone cost gateup NX=2 at b=4 95.6 % -> 73.6 % of ceiling.
template<int M, int BITS, bool USEB, bool ZF = false>
__device__ __forceinline__ void blk1632_q4(uint32_t m32, uint64_t buf, uint32_t h, int q4,
                                           const float (&fa)[8 * M],
                                           const float (&fb)[8 * M],
                                           float* qs, float* xs,
                                           const uint32_t* __restrict__ lut,
                                           uint32_t zb = 0u) {
  const uint32_t b8 = (m32 >> (q4 << 3)) & 0xFFu;
  // survivors preceding this bitmap byte INSIDE the block -- independent per q4
  const int o = (q4 == 0) ? 0 : __popc(m32 & ((1u << (q4 << 3)) - 1u));
  const uint32_t cwd = (uint32_t)(buf >> ((o & 15) << 2));   // 8 survivor nibbles
  uint32_t lo = cwd & 0x0F0F0F0Fu;          // bytes = survivors o, o+2, o+4, o+6
  uint32_t hi = (cwd >> 4) & 0x0F0F0F0Fu;   // bytes = survivors o+1, o+3, o+5, o+7
  if (BITS == 6) {
    const int ia = o >> 1, od = o & 1;
    const uint32_t wa = od ? (h >> 16) : h;
    const uint32_t wb = od ? h : (h >> 16);
    const uint32_t pe = (wa >> (ia << 1)) & 0xFFu;
    const uint32_t po = (wb >> ((ia + od) << 1)) & 0xFFu;
    // carry-free OR spread (a spread-MULTIPLY corrupts the next byte -- FORMAT.md 13.8)
    uint32_t ve = pe | (pe << 12); ve = (ve | (ve << 6)) & 0x03030303u;
    uint32_t vo = po | (po << 12); vo = (vo | (vo << 6)) & 0x03030303u;
    lo |= ve << 4;
    hi |= vo << 4;
  }
  // bit i of the nibble -> byte i (the 0x00204081 spread); *0xFF turns it into a byte
  // mask that zeroes the pruned slots.  CBK_BLK1632_LUT==3 reuses the SAME two spreads
  // to build the byte_perm selectors arithmetically, so the table disappears from the
  // per-granule dependency chain entirely.
#if CBK_BLK1632_NOOVH
  // DIAGNOSTIC: no selector, no mask, no blend -- the nibble planes stand in for the
  // expanded codes.  WRONG RESULT; measures what the per-granule overhead costs.
  const uint32_t e0 = lo, e1 = hi;
  (void)lut;
#else
  const uint32_t sp0 = (((b8 & 0xFu) * 0x00204081u) & 0x01010101u);
  const uint32_t sp1 = ((((b8 >> 4) & 0xFu) * 0x00204081u) & 0x01010101u);
  const uint32_t mk0 = sp0 * 0xFFu;
  const uint32_t mk1 = sp1 * 0xFFu;
#if CBK_BLK1632_LUT == 3
  const uint32_t sel0 = prmt_sel_arith(sp0);
  const uint32_t sel1 = prmt_sel_arith_hi(sp1, (uint32_t)__popc((int)(b8 & 0xFu)));
#else
  const uint32_t sel0 = lut[b8 << 1];
  const uint32_t sel1 = lut[(b8 << 1) | 1];
#endif
  // ZF: pruned slots carry the group's zero code (zb = zero * 0x01010101) instead of 0,
  // so `qs - zf*sum(x)` cancels them and no per-row masked x-sum is needed.
  const uint32_t e0 = ZF ? ((__byte_perm(lo, hi, sel0) & mk0) | (zb & ~mk0))
                         :  (__byte_perm(lo, hi, sel0) & mk0);
  const uint32_t e1 = ZF ? ((__byte_perm(lo, hi, sel1) & mk1) | (zb & ~mk1))
                         :  (__byte_perm(lo, hi, sel1) & mk1);
#endif
#if CBK_BLK1632_HALFJ
  // DIAGNOSTIC: half the positions.  e1 is XORed into e0 so that sel1 / mk1 / sp1 and
  // the second __byte_perm stay live (the granule's ALU cost is unchanged); only the
  // 8 -> 4 (extract, cvt, FMA) triples per bitmap byte are removed.  WRONG RESULT.
  const uint32_t ehj = e0 ^ e1;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t ew = ehj;
#if CBK_BLK1632_BIAS
    const float qv = __uint_as_float(
        __byte_perm(ew, 0x43000000u, 0x7044u | (uint32_t)((j & 3) << 8)));
#else
    const float qv = (float)((ew >> ((j & 3) << 3)) & 0xFFu);
#endif
#else
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t ew = (j < 4) ? e0 : e1;
#if CBK_BLK1632_BIAS
    // 0x43 <c> 00 00 == 128.0f * (1 + c/128) == 128 + c, exactly.  One prmt, no cvt.
    const float qv = __uint_as_float(
        __byte_perm(ew, 0x43000000u, 0x7044u | (uint32_t)((j & 3) << 8)));
#else
    const float qv = (float)((ew >> ((j & 3) << 3)) & 0xFFu);
#endif
#endif
    #pragma unroll
    for (int m = 0; m < M; ++m) {
      const float xv = USEB ? fb[j * M + m] : fa[j * M + m];
      qs[m] = fmaf(qv, xv, qs[m]);
    }
    if (!ZF) {
      const float km = ((b8 >> j) & 1u) ? 1.f : 0.f;
      #pragma unroll
      for (int m = 0; m < M; ++m)
        xs[m] = fmaf(km, USEB ? fb[j * M + m] : fa[j * M + m], xs[m]);
    }
  }
}

template<int M, int R, int NX, int PF, int BITS, bool FLATTAIL>
__device__ __forceinline__ void gemv_blk1632_multi(
    const uint8_t* __restrict__ data, size_t rstride,
    const __half* __restrict__ scale, const uint8_t* __restrict__ zero, int G, int N,
    const __nv_bfloat16* __restrict__ xa, const __nv_bfloat16* __restrict__ xb,
    float acc[R][M]) {
  const int lane = threadIdx.x & 31;
#if CBK_BLK1632_LUT == 2
  const uint32_t* __restrict__ lut = prmt_smem();   // staged once per block at entry
#else
  const uint32_t* __restrict__ lut = prmt_lut;      // unused when LUT==3
#endif
  const int NT = N >> 5;                        // 32-column granules == 16:32 blocks
  const size_t noff = (size_t)(N >> 3);         // nibble plane base
  const size_t hoff = (size_t)N * 3 >> 3;       // hi2 plane base (BITS==6)
  const int NTAIL = FLATTAIL ? (NT & 31) : 0;
  const int NMAIN = NT - NTAIL;
  #pragma unroll
  for (int i = 0; i < R; ++i)
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[i][m] = 0.f;

#if CBK_BLK1632_NOLD & 1
#define CBK_MLOAD(rb, ts, ok) (0x0F0F0F0Fu)           /* DIAGNOSTIC: no mask load */
#else
#define CBK_MLOAD(rb, ts, ok) \
  ((ok) ? __ldcg(reinterpret_cast<const uint32_t*>(rb) + (ts)) : 0u)
#endif
#if CBK_BLK1632_MSCHED == 2
  uint32_t mnx[PF][R];
  #pragma unroll
  for (int s = 0; s < PF; ++s) {
    const int ts = lane + s * 32;
    #pragma unroll
    for (int i = 0; i < R; ++i)
      mnx[s][i] = CBK_MLOAD(data + (size_t)i * rstride, ts, ts < NMAIN);
  }
#endif
  for (int t0 = lane; t0 < NMAIN; t0 += 32 * PF) {
    uint32_t mw[PF][R]; uint2 nw[PF][R]; uint32_t hw[PF][R];
#if CBK_BLK1632_MSCHED == 2
    // the masks for THIS iteration were issued one iteration ago; issue the NEXT ones now
    #pragma unroll
    for (int s = 0; s < PF; ++s)
      #pragma unroll
      for (int i = 0; i < R; ++i) mw[s][i] = mnx[s][i];
    #pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int tn = t0 + 32 * PF + s * 32;
      #pragma unroll
      for (int i = 0; i < R; ++i)
        mnx[s][i] = CBK_MLOAD(data + (size_t)i * rstride, tn, tn < NMAIN);
    }
#elif CBK_BLK1632_MSCHED == 1
    // every mask load of this iteration issued before any nibble load
    #pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int ts = t0 + s * 32;
      #pragma unroll
      for (int i = 0; i < R; ++i)
        mw[s][i] = CBK_MLOAD(data + (size_t)i * rstride, ts, ts < NMAIN);
    }
#endif
    #pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int ts = t0 + s * 32;
      const bool ok = (ts < NMAIN);
      #pragma unroll
      for (int i = 0; i < R; ++i) {
        const uint8_t* rb = data + (size_t)i * rstride;
#if CBK_BLK1632_MSCHED == 0
        mw[s][i] = CBK_MLOAD(rb, ts, ok);
#endif
#if CBK_BLK1632_NOLD & 2
        nw[s][i] = make_uint2(0x12345678u, 0x9ABCDEF0u);  // DIAGNOSTIC: no nibble load
#else
        nw[s][i] = ok ? __ldcg(reinterpret_cast<const uint2*>(rb + noff + ((size_t)ts << 3)))
                      : make_uint2(0u, 0u);
#endif
        if (BITS == 6)
          hw[s][i] = ok ? __ldcg(reinterpret_cast<const uint32_t*>(rb + hoff) + ts) : 0u;
      }
    }
    #pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int ts = t0 + s * 32;
      if (ts >= NMAIN) break;
      const int col = ts << 5, gg = ts >> 2;
      float qs[R][M], xs[R][M];
#if CBK_BLK1632_ZF
      // ZF: one PLAIN sum(x) per activation stream, shared by all R rows (DENSE4's shape)
      float xsa[M], xsb[M];
      uint32_t zbv[R];
      #pragma unroll
      for (int m = 0; m < M; ++m) { xsa[m] = 0.f; xsb[m] = 0.f; }
      #pragma unroll
      for (int i = 0; i < R; ++i)
        zbv[i] = (uint32_t)zero[(size_t)i * G + gg] * 0x01010101u;
#endif
      #pragma unroll
      for (int i = 0; i < R; ++i)
        #pragma unroll
        for (int m = 0; m < M; ++m) { qs[i][m] = 0.f; xs[i][m] = 0.f; }
      #pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {          // 8 columns == one bitmap byte
        uint4 va[M], vb[M];
        #pragma unroll
        for (int m = 0; m < M; ++m)
          va[m] = *reinterpret_cast<const uint4*>(xa + (size_t)(col + (q4 << 3)) * M + m * 8);
        // CBK_BLK1632_NOXB: DIAGNOSTIC ONLY -- drop the second activation stream (odd
        // "up" rows then read x_gate).  NUMERICALLY WRONG; it exists to put a measured
        // CEILING on what folding the gate/up column scales could ever buy,
        // before anyone pays for a format change.  Never enable in a record build.
#if !CBK_BLK1632_NOXB
        if (NX == 2) {
          #pragma unroll
          for (int m = 0; m < M; ++m)
            vb[m] = *reinterpret_cast<const uint4*>(xb + (size_t)(col + (q4 << 3)) * M + m * 8);
        }
#endif
        const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(va);
        const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(vb);
        float fa[8 * M], fb[8 * M];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
          #pragma unroll
          for (int m = 0; m < M; ++m) {
            fa[j * M + m] = __bfloat162float(ha[j * M + m]);
#if CBK_BLK1632_ZF
            xsa[m] += fa[j * M + m];
#endif
#if !CBK_BLK1632_NOXB
            if (NX == 2) {
              fb[j * M + m] = __bfloat162float(hb[j * M + m]);
#if CBK_BLK1632_ZF
              xsb[m] += fb[j * M + m];
#endif
            }
#endif
          }
        #pragma unroll
        for (int i = 0; i < R; ++i) {
          const uint64_t buf = (((uint64_t)nw[s][i].y) << 32) | (uint64_t)nw[s][i].x;
          if (!CBK_BLK1632_NOXB && NX == 2 && (i & 1))
            blk1632_q4<M, BITS, true, (bool)CBK_BLK1632_ZF>(
                mw[s][i], buf, (BITS == 6) ? hw[s][i] : 0u, q4,
                fa, fb, qs[i], xs[i], lut, ZBV(i));
          else
            blk1632_q4<M, BITS, false, (bool)CBK_BLK1632_ZF>(
                mw[s][i], buf, (BITS == 6) ? hw[s][i] : 0u, q4,
                fa, fb, qs[i], xs[i], lut, ZBV(i));
        }
      }
      #pragma unroll
      for (int i = 0; i < R; ++i) {
        const float sf = __half2float(scale[(size_t)i * G + gg]);
        const float zf = (float)zero[(size_t)i * G + gg];
        #pragma unroll
        for (int m = 0; m < M; ++m)
#if CBK_BLK1632_ZF
          acc[i][m] = fmaf(sf, qs[i][m] - (zf + (float)(128 * CBK_BLK1632_BIAS))
                                             * ((!CBK_BLK1632_NOXB && NX == 2 && (i & 1))
                                                ? xsb[m] : xsa[m]), acc[i][m]);
#else
          acc[i][m] = fmaf(sf, qs[i][m] - zf * xs[i][m], acc[i][m]);
#endif
      }
    }
  }

#undef CBK_MLOAD
  if (FLATTAIL && NTAIL) {
    // flattened (row, granule) work list: item q -> row i, granule NMAIN + (q - i*NTAIL)
    for (int q = lane; q < R * NTAIL; q += 32) {
      int i = 0;
      #pragma unroll
      for (int ii = 1; ii < R; ++ii) i += (q >= ii * NTAIL);
      const int ts = NMAIN + (q - i * NTAIL);
      const int col = ts << 5, gg = ts >> 2;
      const uint8_t* rb = data + (size_t)i * rstride;
      const uint32_t m32 = __ldcg(reinterpret_cast<const uint32_t*>(rb) + ts);
      const uint2 nv = __ldcg(reinterpret_cast<const uint2*>(rb + noff + ((size_t)ts << 3)));
      const uint32_t hv = (BITS == 6)
          ? __ldcg(reinterpret_cast<const uint32_t*>(rb + hoff) + ts) : 0u;
      const uint64_t buf = (((uint64_t)nv.y) << 32) | (uint64_t)nv.x;
      const __nv_bfloat16* xp = (NX == 2 && (i & 1)) ? xb : xa;
      float qs[M], xs[M];
      #pragma unroll
      for (int m = 0; m < M; ++m) { qs[m] = 0.f; xs[m] = 0.f; }
#if CBK_BLK1632_ZF
      const uint32_t zbt = (uint32_t)zero[(size_t)i * G + gg] * 0x01010101u;
#endif
      #pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint4 va[M];
        #pragma unroll
        for (int m = 0; m < M; ++m)
          va[m] = *reinterpret_cast<const uint4*>(xp + (size_t)(col + (q4 << 3)) * M + m * 8);
        const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(va);
        float fa[8 * M];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
          #pragma unroll
          for (int m = 0; m < M; ++m) {
            fa[j * M + m] = __bfloat162float(ha[j * M + m]);
#if CBK_BLK1632_ZF
            xs[m] += fa[j * M + m];
#endif
          }
#if CBK_BLK1632_ZF
        blk1632_q4<M, BITS, false, true>(m32, buf, hv, q4, fa, fa, qs, xs, lut, zbt);
#else
        blk1632_q4<M, BITS, false>(m32, buf, hv, q4, fa, fa, qs, xs, lut);
#endif
      }
      const float sf = __half2float(scale[(size_t)i * G + gg]);
      const float zf = (float)zero[(size_t)i * G + gg];
      #pragma unroll
      for (int ii = 0; ii < R; ++ii)
        if (ii == i)
          #pragma unroll
          for (int m = 0; m < M; ++m)
            acc[ii][m] = fmaf(sf, qs[m] - (zf + (float)(128 * CBK_BLK1632_BIAS)) * xs[m],
                              acc[ii][m]);
    }
  }

  #pragma unroll
  for (int i = 0; i < R; ++i) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      #pragma unroll
      for (int m = 0; m < M; ++m) acc[i][m] += __shfl_xor_sync(FULL, acc[i][m], d);
  }
}

// ---------------------------------------------------------------- DENSE8
// Window = 512 columns; lane l owns 16 columns = 16 bytes = one uint4 load.
template<int M>
__device__ __forceinline__ void gemv_dense8(const Mat& mt, int row,
                                            const __nv_bfloat16* __restrict__ x, float* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N;
  const uint8_t* rb = mt.data + (size_t)row * (size_t)N;
  const __half*  sc = mt.scale + (size_t)row * mt.G;
  const uint8_t* zp = mt.zero  + (size_t)row * mt.G;

  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (int c0 = 0; c0 < N; c0 += 512) {
    const int col = c0 + (lane << 4);
    if (col >= N) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(rb + col);
    const int gg = col >> 7;
    const float sf = __half2float(sc[gg]);
    const float zf = (float)zp[gg];
    float qs[M], xs[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) { qs[m] = 0.f; xs[m] = 0.f; }
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t cw = (w == 0) ? v.x : (w == 1) ? v.y : (w == 2) ? v.z : v.w;
      #pragma unroll
      for (int t = 0; t < 4; ++t) {
        const float qv = (float)((cw >> (t << 3)) & 0xFF);
        const int c = col + (w << 2) + t;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          const float xv = xat<M>(x, m, N, c);
          qs[m] = fmaf(qv, xv, qs[m]);
          xs[m] += xv;
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = fmaf(sf, qs[m] - zf * xs[m], acc[m]);
  }
  warp_reduce<M>(acc);
  #pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}

// ---------------------------------------------------------------- BF16 passthrough
template<int M>
__device__ __forceinline__ void gemv_bf16(const Mat& mt, int row,
                                          const __nv_bfloat16* __restrict__ x, float* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N;
  const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(mt.data) + (size_t)row * N;
  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int c = lane; c < N; c += 32) {
    const float wv = __bfloat162float(rb[c]);
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = fmaf(wv, xat<M>(x, m, N, c), acc[m]);
  }
  warp_reduce<M>(acc);
  #pragma unroll
  for (int m = 0; m < M; ++m) out[m] = acc[m];
}

} // namespace detail

// ---------------------------------------------------------------- public API
template<int M>
__device__ __forceinline__ void gemv_rows(const Mat& m, int row,
                                          const __nv_bfloat16* x, float out[M],
                                          uint8_t* /*scratch*/) {
  switch (m.layout) {
    case LAYOUT_SPARSE4:  detail::gemv_sparse<M, false>(m, row, x, out); break;
    case LAYOUT_SPARSE4X: detail::gemv_sparse<M, true >(m, row, x, out); break;
    case LAYOUT_SPARSE4E: detail::gemv_sparse_expand<M>(m, row, x, out);  break;
    case LAYOUT_DENSE4:   detail::gemv_dense4<M>(m, row, x, out);        break;
#ifndef CBK_BLK1632_BPG
#define CBK_BLK1632_BPG 1
#endif
    case LAYOUT_BLK1632_4: detail::gemv_blk1632<M, 4, CBK_BLK1632_BPG>(m, row, x, out); break;
    case LAYOUT_BLK1632_6: detail::gemv_blk1632<M, 6, CBK_BLK1632_BPG>(m, row, x, out); break;
    case LAYOUT_DENSE8:   detail::gemv_dense8<M>(m, row, x, out);        break;
    default:              detail::gemv_bf16<M>(m, row, x, out);          break;
  }
}

// ---------------------------------------------------------------- dequant_row
// Warp-cooperative: writes N bf16 values (pruned positions = 0). Used for the
// embedding lookup and for debugging.
__device__ __forceinline__ void dequant_row(const Mat& mt, int row, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  const int N = mt.N, G = mt.G;
  const __half*  sc = mt.scale + (size_t)row * G;
  const uint8_t* zp = mt.zero  + (size_t)row * G;

  if (mt.layout == LAYOUT_BF16) {
    const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(mt.data) + (size_t)row * N;
    for (int c = lane; c < N; c += 32) out[c] = rb[c];
    return;
  }
  if (mt.layout == LAYOUT_DENSE4 || mt.layout == LAYOUT_DENSE8) {
    const bool d8 = (mt.layout == LAYOUT_DENSE8);
    const uint8_t* rb = mt.data + (size_t)row * dense_row_stride(mt.layout, N);
    for (int c = lane; c < N; c += 32) {
      const int gg = c >> 7;
      const float sf = __half2float(sc[gg]);
      const float zf = (float)zp[gg];
      const uint32_t code = d8 ? (uint32_t)rb[c] : (uint32_t)((rb[c >> 1] >> ((c & 1) << 2)) & 0xF);
      out[c] = __float2bfloat16(((float)code - zf) * sf);
    }
    return;
  }
  if (layout_is_blk1632(mt.layout)) {
    const int bits = blk1632_bits(mt.layout);
    const uint8_t* rb = mt.data + (size_t)row * dense_row_stride(mt.layout, N);
    const uint32_t* mp = reinterpret_cast<const uint32_t*>(rb);
    const uint8_t*  np = rb + (N >> 3);
    const uint32_t* hp = reinterpret_cast<const uint32_t*>(rb + ((size_t)N * 3 >> 3));
    for (int t = lane; t < (N >> 5); t += 32) {           // one 32-column block per lane
      const uint32_t m32 = mp[t];
      const uint64_t buf = *reinterpret_cast<const uint64_t*>(np + ((size_t)t << 3));
      const uint32_t h = (bits == 6) ? hp[t] : 0u;
      const int gg = t >> 2;
      const float sf = __half2float(sc[gg]);
      const float zf = (float)zp[gg];
      int r = 0;                                          // survivor rank inside the block
      for (int j = 0; j < 32; ++j) {
        const int col = (t << 5) + j;
        if (!((m32 >> j) & 1u)) { out[col] = __float2bfloat16(0.f); continue; }
        uint32_t q = (uint32_t)((buf >> (r << 2)) & 0xFull);
        if (bits == 6) {
          const uint32_t w = (r & 1) ? (h >> 16) : h;
          q |= ((w >> ((r >> 1) << 1)) & 3u) << 4;
        }
        out[col] = __float2bfloat16(((float)q - zf) * sf);
        ++r;
      }
    }
    return;
  }
  // SPARSE4 / SPARSE4X
  for (int c = lane; c < N; c += 32) out[c] = __float2bfloat16(0.f);
  __syncwarp();
  const bool hasg = layout_has_goff(mt.layout);
  const uint8_t*  rb   = mt.data + mt.row_off[row];
  const uint32_t* bmp  = reinterpret_cast<const uint32_t*>(rb);
  const int       hdr  = sparse_row_header(mt.layout, N, G);
  const uint16_t* goff = hasg ? reinterpret_cast<const uint16_t*>(rb + (N >> 3)) : nullptr;
  const uint32_t* codes = reinterpret_cast<const uint32_t*>(rb + hdr);
  const int nwords = N >> 5;
  int run_bytes = 0;
  for (int c0 = 0; c0 < N; c0 += 1024) {
    const int wi = (c0 >> 5) + lane;
    uint32_t bits = (wi < nwords) ? bmp[wi] : 0u;
    const int p = __popc(bits);
    int s = p;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int t = __shfl_up_sync(detail::FULL, s, d); if (lane >= d) s += t; }
    int sm4 = __shfl_up_sync(detail::FULL, s, 4); if (lane < 4) sm4 = 0;
    const uint32_t oddm = __ballot_sync(detail::FULL, ((lane & 3) == 3) && (((s - sm4) & 1) != 0));
    const int j = lane >> 2;
    const int esh = __shfl_sync(detail::FULL, s, (j == 0) ? 0 : ((j << 2) - 1));
    const int e = (j == 0) ? 0 : esh;
    const int gg = (c0 >> 7) + j;
    int gbyte;
    if (hasg) gbyte = (gg < G) ? (int)goff[gg] : 0;
    else      gbyte = run_bytes + ((e + __popc(oddm & ((1u << (j << 2)) - 1u))) >> 1);
    if (!hasg) {
      const int stot = __shfl_sync(detail::FULL, s, 31);
      run_bytes += (stot + __popc(oddm)) >> 1;
    }
    if (bits) {
      int idx = (gbyte << 1) + (s - p - e);
      uint32_t cw = codes[idx >> 3];
      const float sf = __half2float(sc[gg]);
      const float zf = (float)zp[gg];
      const int base = c0 + (lane << 5);
      while (bits) {
        const int t = __ffs(bits) - 1; bits &= bits - 1;
        const float qv = (float)((cw >> ((idx & 7) << 2)) & 0xF);
        out[base + t] = __float2bfloat16((qv - zf) * sf);
        ++idx;
        if (bits && ((idx & 7) == 0)) cw = codes[idx >> 3];
      }
    }
  }
}

} // namespace cbk
