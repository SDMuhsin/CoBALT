// megakernel.cuh -- structs shared between the CUDA kernel and the host glue.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace cbk {

#define CBK_THREADS 256
#define CBK_WARPS   (CBK_THREADS / 32)
#define CBK_MAXDPL  8      // max head_dim/32 supported (head_dim <= 256)
#define CBK_MAXM    8
#define CBK_CHUNK_MAX 2048 // upper bound on the staged activation chunk (see chunk_len)

// Layout id for an unquantized row-major bf16 matrix (not one of cbk::Layout).
#define CBK_LAYOUT_PLAIN_BF16 (-1)

// One weight matrix: either a plain bf16 [K][N] block or a CBK1 packed cbk::Mat.
// Laid out as 9 int64 so the python loader can build the device table from
// `tensor.data_ptr()` values alone.
struct MatDesc {
  const uint8_t*  data;
  const uint32_t* row_off;    // packed SPARSE only
  const __half*   scale;      // packed only, [K][G]
  const uint8_t*  zero;       // packed only, [K][G]
  const __half*   col_scale;  // packed only, [N]  (gateup: [2][N])
  long long K, N, G, layout;
};

struct LayerW {
  // q/k/v are ONE fused row space (FORMAT.md sec.2.4b): rows [0,Kq) = q, [Kq,Kq+Kk) = k,
  // the rest = v, with col_scale of shape [3, N].  gateup interleaves gate/up rows.
  MatDesc qkv, o, gateup, down;
  const __nv_bfloat16 *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln, *q_norm, *k_norm;
  long long is_sliding, _pad;
};

struct Args {
  const LayerW* layers;
  MatDesc embed;                    // embedding == lm_head (tied)
  const __nv_bfloat16* final_norm;
  const float* inv_local;
  const float* inv_global;
  float* rope_cs;   // [2][M][head_dim/2] bf16-rounded cos, 0=local 1=global
  float* rope_sn;
  __nv_bfloat16* kcache;            // [L][M][kvh][max_ctx][D]
  __nv_bfloat16* vcache;
  const int* tokens;      // unused (kept so the struct stays 8-byte regular)
  const int* positions;   // unused
  // Passed BY VALUE in the kernel parameter block: with these here a decode step is
  // literally ONE cudaLaunchCooperativeKernel and zero memcpys.
  int tok_v[CBK_MAXM];
  int pos_v[CBK_MAXM];
  __nv_bfloat16 *h, *h2, *qkv, *attn_out, *obuf, *act, *dbuf;
  __nv_bfloat16* xbuf;   // [NC][hidden][M] column-scaled activation copies (global)
  float* rsums;          // [blocks][M] per-block partial sum(x^2) for the RMS norms
  float* partials;
  float* logits;
  float* amax_val;
  int*   amax_idx;
  __nv_bfloat16* dbg_h;
  long long* timings;
  int hidden, n_layers, n_heads, n_kv, head_dim, inter, vocab;
  int M, max_ctx, sliding_window, split;
  float eps, attn_scale, embed_scale;
  int nq_dim, nkv_dim, nqkv, kv_group, xcap;
  int prefetch_o;   // 1 = L2-prefetch o_proj during the attention phase
};

}  // namespace cbk
