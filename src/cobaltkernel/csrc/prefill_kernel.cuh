// prefill_kernel.cuh -- structs + device helpers for the Gemma3 PREFILL megakernel.
//
// This file is INDEPENDENT of megakernel.cu/.cuh on purpose: the device helpers
// below (RMSNorm, RoPE, GeGLU, KV addressing) are COPIES
// under namespace `pf`, so the two kernels can evolve separately.  The one thing that is
// NOT free to diverge is the KV-cache memory layout -- see pf::kv_off() below, which
// is the KV contract the decode megakernel reads back.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace pf {

#define PF_THREADS 256
#define PF_WARPS   (PF_THREADS / 32)
#define PF_MAXDPL  8       // head_dim / 32, head_dim <= 256
#define PF_QR      4       // query rows per warp in the attention phase
#define PF_MAXGQ   2       // query heads per kv head handled at once (batched decode)
#define PF_KPL     2       // max keys per lane per attention tile (ktile/32)
#define PF_MAXPH   16      // timing slots per layer

// One weight matrix (CBK1 packed DENSE4/DENSE8).  Same 9-int64 shape the decode runner
// uses, so the python side can build the table from tensor.data_ptr() values alone.
struct MatDesc {
  const uint8_t*  data;
  const uint32_t* row_off;
  const __half*   scale;
  const uint8_t*  zero;
  const __half*   col_scale;   // [cs_rows][N]
  long long K, N, G, layout;
};

struct PLayerW {
  MatDesc qkv, o, gateup, down;      // qkv = row-concatenated q|k|v (FORMAT.md 2.4b)
  const __nv_bfloat16 *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln, *q_norm, *k_norm;
  long long is_sliding, _pad;
};

struct PArgs {
  const PLayerW* layers;
  MatDesc embed;                     // token embedding
  MatDesc lm_head;                   // == embed when tied (gemma3); separate for llama
  const __nv_bfloat16* final_norm;
  const float* inv_local;            // [head_dim/2]
  const float* inv_global;
  float* rope_cs;                    // [2][M][head_dim/2], bf16-rounded cos
  float* rope_sn;
  __nv_bfloat16* kcache;             // [L][B][kv_heads][max_ctx][head_dim]
  __nv_bfloat16* vcache;
  const int* tokens;                 // [M]
  const int* positions;              // [M] absolute position of each row (batch mode)
  __nv_bfloat16 *h, *xn, *qkvb, *qb, *ao, *ob, *act, *db, *hg;
  float* logits;                     // [R][vocab] fp32
  const int* logit_rows;             // [R] row indices into [0,M)
  long long* timings;
  int hidden, n_layers, n_heads, n_kv, head_dim, inter, vocab;
  int M, R, pos0, kv_b, seq, max_ctx, sliding_window;
  // batch_mode = 0: M rows are consecutive tokens (pos0+m) of ONE sequence `seq`.
  // batch_mode = 1: M rows are M DIFFERENT sequences, row m = sequence m at positions[m].
  int batch_mode;
  float eps, attn_scale, embed_scale;
  int nq_dim, nkv_dim, nqkv, kv_group;
  // model-family switches (arch.py); null norm pointers in PLayerW bypass that norm
  int norm_plus_one;   // 1 gemma (1+w, one rounding)  0 llama (round, then w*x)
  int act_gelu;        // 1 GeGLU  0 SwiGLU
};

// ------------------------------------------------------------------ small helpers
#define PF_F(x)  __bfloat162float(x)
#define PF_BF(x) __float2bfloat16(x)
__device__ __forceinline__ float rb(float v) { return PF_F(PF_BF(v)); }
__device__ __forceinline__ int ceildiv(int a, int b) { return (a + b - 1) / b; }
__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float norm_apply(float v, float rr, float w, int plus_one) {
  return plus_one ? rb(v * rr * (1.f + w)) : rb(rb(v * rr) * w);
}

// Absolute position of activation row m, and the KV-cache sequence slot it belongs to.
__device__ __forceinline__ int row_pos(const PArgs& a, int m) {
  return a.batch_mode ? a.positions[m] : (a.pos0 + m);
}
__device__ __forceinline__ int row_seq(const PArgs& a, int m) {
  return a.batch_mode ? m : a.seq;
}

// KV CONTRACT -- byte-identical to the decode megakernel (megakernel.cu phase_attn):
//   kcache[(((L * B + b) * n_kv + kvh) * max_ctx + t) * head_dim + d]
__device__ __forceinline__ size_t kv_off(int L, int B, int b, int n_kv, int kvh,
                                         int max_ctx, int t, int D) {
  return ((((size_t)L * B + b) * n_kv + kvh) * (size_t)max_ctx + t) * (size_t)D;
}

// Dequantize element j of row r of a DENSE4/DENSE8 CBK1 matrix (no column scale).
__device__ __forceinline__ float dense_elem(const MatDesc& m, int r, int j) {
  const int g = j / 128;
  const float s = __half2float(m.scale[(size_t)r * m.G + g]);
  const float z = (float)m.zero[(size_t)r * m.G + g];
  float q;
  if (m.layout == 2) {  // DENSE8
    q = (float)m.data[(size_t)r * m.N + j];
  } else {              // DENSE4
    const uint8_t byte = m.data[(size_t)r * (m.N >> 1) + (j >> 1)];
    q = (float)((j & 1) ? (byte >> 4) : (byte & 0xF));
  }
  return (q - z) * s;
}

}  // namespace pf
