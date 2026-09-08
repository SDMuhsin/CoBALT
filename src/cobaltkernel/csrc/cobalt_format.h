// cobalt_format.h -- CBK1 packed-CoBALT container: host + device structs.
// See docs/FORMAT.md for the normative description.
#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#else
struct __half;             // opaque on the host side
struct __nv_bfloat16;
#endif

namespace cbk {

// ---------------------------------------------------------------- constants
constexpr int GROUP      = 128;   // quantization group along the input axis N
constexpr int ROW_ALIGN  = 16;    // every SPARSE row region starts 16-B aligned
constexpr int ARR_ALIGN  = 256;   // every array inside a .bin blob is 256-B aligned
constexpr int BLOB_TAIL  = 64;    // padding at the end of a data blob (over-read slack)

enum Layout : int {
  LAYOUT_SPARSE4  = 0,   // 1-bit bitmap + survivor-only 4-bit codes (warp prefix scan)
  LAYOUT_DENSE4   = 1,   // 4-bit codes for every position (pruned positions carry `zero`)
  LAYOUT_DENSE8   = 2,   // 8-bit codes for every position
  LAYOUT_SPARSE4X = 3,   // = SPARSE4 + explicit uint16 per-group code byte offsets
  LAYOUT_BF16     = 4,   // raw bf16 passthrough (embedding ablation); scale/zero unused
  LAYOUT_SPARSE4E = 5,   // SPARSE4X container, EXPAND-TO-DENSE decoder (see FORMAT.md sec.10)
  LAYOUT_BLK1632_4 = 6,  // CoBALT-16:32 fixed-cardinality block mask, 4-bit survivors (sec.13)
  LAYOUT_BLK1632_6 = 7   // CoBALT-16:32 fixed-cardinality block mask, 6-bit survivors (sec.13)
};

// --------------------------------------------------------------- BLK16_32 (sec.13)
// Fixed-cardinality mask: EXACTLY 16 survivors in every aligned block of 32 input
// columns.  Per row the bytes are three CONTIGUOUS PLANES (row stride is therefore
// fixed and `row_off` is nullptr, exactly like DENSE4):
//     [0,     N/8)             mask plane   : uint32 per 32-column block, LSB-first,
//                                             byte-identical to the RAW/SPARSE4 bitmap
//     [N/8,  3N/8)             nibble plane : low 4 bits of the 16 survivor codes of a
//                                             block (8 B/block), survivors in COLUMN order
//     [3N/8,  N/2)   b=6 only  hi2 plane    : high 2 bits of the same 16 codes (4 B/block)
// A pruned position stores NO code at all; the decoder contributes exactly 0 for it via
// the masked-`xs` accumulator, so `zero` never has to be on the quantization grid.
__host__ __device__ inline bool layout_is_blk1632(int L) {
  return L == LAYOUT_BLK1632_4 || L == LAYOUT_BLK1632_6;
}
__host__ __device__ inline int blk1632_bits(int L) { return L == LAYOUT_BLK1632_6 ? 6 : 4; }

__host__ __device__ inline bool layout_is_sparse(int L) {
  return L == LAYOUT_SPARSE4 || L == LAYOUT_SPARSE4X || L == LAYOUT_SPARSE4E;
}
// SPARSE4E is byte-identical to SPARSE4X on disk; only the decoder differs.
__host__ __device__ inline bool layout_has_goff(int L) {
  return L == LAYOUT_SPARSE4X || L == LAYOUT_SPARSE4E;
}

// bytes per row for the fixed-stride layouts (0 for the sparse ones)
__host__ __device__ inline size_t dense_row_stride(int L, int N) {
  switch (L) {
    case LAYOUT_DENSE4: return (size_t)N >> 1;
    case LAYOUT_DENSE8: return (size_t)N;
    case LAYOUT_BF16:   return (size_t)N * 2;
    case LAYOUT_BLK1632_4: return (size_t)N * 3 / 8;   // mask N/8 + nibble N/4
    case LAYOUT_BLK1632_6: return (size_t)N / 2;       // + hi2 plane N/8
    default:            return 0;
  }
}

// header bytes (bitmap [+ goff table]) at the front of a SPARSE row region
__host__ __device__ inline int sparse_row_header(int L, int N, int G) {
  int h = N >> 3;                                    // bitmap: N/8 bytes, always 16-B multiple
  if (layout_has_goff(L)) h += (G * 2 + 15) & ~15;   // uint16 goff[G], padded to 16 B
  return h;
}

// ---------------------------------------------------------------- fused matrices
// A fused matrix is the plain DENSE layout of the row-CONCATENATED (or, for `gateup`,
// row-INTERLEAVED) sub-matrices; only the per-matrix column scale differs per row, so the
// caller carries a col_scale of shape [CS_ROWS, N] and picks the row from the weight row.
//   gateup : CS_ROWS = 2, row 2i = gate_i (cs row 0), row 2i+1 = up_i (cs row 1)
//   qkv    : CS_ROWS = 3, rows [0,Kq) = q (cs row 0), [Kq,Kq+Kk) = k (1), rest = v (2)
constexpr int CS_ROWS_GATEUP = 2;
constexpr int CS_ROWS_QKV    = 3;

// col_scale row for a row of the fused `gateup` matrix.
__host__ __device__ inline int gateup_cs_row(int row) { return row & 1; }

// col_scale row for a row of the fused `qkv` matrix (row-concatenated q|k|v).
__host__ __device__ inline int qkv_cs_row(int row, int Kq, int Kk) {
  return (row < Kq) ? 0 : ((row < Kq + Kk) ? 1 : 2);
}

// Base pointer of col_scale row `r` in a [CS_ROWS, N] fp16 column-scale array.
__host__ __device__ inline const __half* cs_row_ptr(const __half* cs, int r, int N) {
  // byte arithmetic: __half is opaque in host-only translation units
  return reinterpret_cast<const __half*>(
      reinterpret_cast<const uint8_t*>(cs) + (size_t)r * N * 2);
}

// ---------------------------------------------------------------- Mat
// One packed matrix: K output rows, N input columns, G = N/GROUP groups per row.
struct Mat {
  int K, N, G, layout;
  const uint8_t*  data;     // base of the row regions
  const uint32_t* row_off;  // K+1 byte offsets into `data` (SPARSE only; nullptr otherwise)
  const __half*   scale;    // K*G, row-major  (scale[r*G+g])
  const uint8_t*  zero;     // K*G, row-major  (integer zero point, [0,2^bits-1])
};

// Sub-matrix view of a fused fixed-stride matrix: rows [k0, k0+K) of `m`.
// Byte-identical to the standalone matrix (the alias offsets manifest.json records).
__host__ __device__ inline Mat sub_rows(const Mat& m, int k0, int K) {
  Mat s = m;
  s.K = K;
  s.data  = m.data  + (size_t)k0 * dense_row_stride(m.layout, m.N);
  s.scale = reinterpret_cast<const __half*>(
      reinterpret_cast<const uint8_t*>(m.scale) + (size_t)k0 * m.G * 2);
  s.zero  = m.zero  + (size_t)k0 * m.G;
  s.row_off = nullptr;                 // fixed-stride layouts only
  return s;
}

// ---------------------------------------------------------------- host helpers
#if !defined(__CUDA_ARCH__)
// Build a Mat from a loaded blob base pointer + the byte offsets recorded in manifest.json.
inline Mat make_mat(int K, int N, int layout, const void* blob,
                    size_t off_data, size_t off_row_off, size_t off_scale, size_t off_zero) {
  Mat m;
  m.K = K; m.N = N; m.G = N / GROUP; m.layout = layout;
  const uint8_t* b = reinterpret_cast<const uint8_t*>(blob);
  m.data    = b + off_data;
  m.row_off = layout_is_sparse(layout) ? reinterpret_cast<const uint32_t*>(b + off_row_off) : nullptr;
  m.scale   = reinterpret_cast<const __half*>(b + off_scale);
  m.zero    = b + off_zero;
  return m;
}

// Size of the scale / zero arrays.
inline size_t scale_bytes(int K, int N) { return (size_t)K * (N / GROUP) * 2; }
inline size_t zero_bytes (int K, int N) { return (size_t)K * (N / GROUP) * 1; }
inline size_t row_off_bytes(int K)      { return (size_t)(K + 1) * 4; }
#endif

} // namespace cbk
