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
  MatDesc embed;                    // token embedding
  MatDesc lm_head;                  // == embed for tied models (gemma3); separate for llama
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
  // ---- model-family switches (arch.py).  A null norm pointer in LayerW bypasses that
  // norm (plain residual add / no QK-norm); these two pick the arithmetic convention.
  int norm_plus_one;  // 1: y = bf16(x*rr*(1+w)) (gemma)   0: y = bf16(bf16(x*rr)*w) (llama)
  int act_gelu;       // 1: GeGLU (gelu_tanh)                0: SwiGLU (silu)
  // ---- in-kernel greedy generation: ONE cooperative launch produces n_steps tokens.  Step 0 uses
  // tok_v/pos_v; step s>0 feeds the previous step's in-kernel argmax and pos+1, recomputing the
  // attention key-split with the host's formula (kpb, split_max).  n_steps == 1 is the classic
  // one-launch-per-token path, bit-identical to before this field existed.
  int n_steps;
  int kpb, split_max;
  int* out_tokens;    // [n_steps][M] chosen tokens (block 0 writes), or nullptr
  // CBK_XSMEM: dynamic-smem budget (bytes) for staging a GEMV phase's activation x' once per block
  // (0 = never stage; the phase reads x' from global/L1 as shipped).
  int xsmem_bytes;
  // CBK_BLK1632_MMA: f32 tile partials [max K] and per-tile arrival counters [max K / 16] for the
  // (tile, column-slice) work units of the tensor-core GEMV; both are zero between phases.
  float* mpart;
  unsigned* mcnt;
};

}  // namespace cbk
