"""
Test whether changing quantization scales affects the error correlation.

If correlation is controllable: ECAQ is viable
If correlation is fixed: ECAQ may not work
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_with_scale(x, scale, bits=4):
    """Quantize tensor with given scale."""
    levels = 2 ** (bits - 1) - 1
    x_q = (x / scale).round().clamp(-levels, levels) * scale
    return x_q


def compute_attention_error(h, W_Q, W_K, W_V, W_O, scale_w=1.0, scale_kv=1.0,
                            bits_w=3, bits_kv=4, n_heads=16, d_head=128):
    """Compute attention output error with given scales."""
    batch_size, seq_len, _ = h.shape
    n_kv_heads = W_K.shape[0] // d_head

    # Full precision
    Q_fp = (h @ W_Q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = (h @ W_K.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V_fp = (h @ W_V.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

    # Quantized weights with scale
    W_Q_max = W_Q.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    W_K_max = W_K.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    W_V_max = W_V.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)

    s_Q = scale_w * W_Q_max / (2 ** (bits_w - 1) - 1)
    s_K = scale_w * W_K_max / (2 ** (bits_w - 1) - 1)
    s_V = scale_w * W_V_max / (2 ** (bits_w - 1) - 1)

    W_Q_q = quantize_with_scale(W_Q, s_Q, bits_w)
    W_K_q = quantize_with_scale(W_K, s_K, bits_w)
    W_V_q = quantize_with_scale(W_V, s_V, bits_w)

    Q_wq = (h @ W_Q_q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_wq = (h @ W_K_q.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V_wq = (h @ W_V_q.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

    # Quantize KV cache with scale
    K_max = K_fp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    V_max = V_fp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)

    s_K_kv = scale_kv * K_max / (2 ** (bits_kv - 1) - 1)
    s_V_kv = scale_kv * V_max / (2 ** (bits_kv - 1) - 1)

    K_kvq = quantize_with_scale(K_fp, s_K_kv, bits_kv)
    V_kvq = quantize_with_scale(V_fp, s_V_kv, bits_kv)

    # GQA expansion
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        K_fp = K_fp.repeat_interleave(rep, dim=1)
        V_fp = V_fp.repeat_interleave(rep, dim=1)
        K_wq = K_wq.repeat_interleave(rep, dim=1)
        V_wq = V_wq.repeat_interleave(rep, dim=1)
        K_kvq = K_kvq.repeat_interleave(rep, dim=1)
        V_kvq = V_kvq.repeat_interleave(rep, dim=1)

    # Attention computation
    scale_attn = 1.0 / np.sqrt(d_head)

    def attn(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_attn
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    out_fp = attn(Q_fp, K_fp, V_fp)
    out_wq = attn(Q_wq, K_wq, V_wq)
    out_kvq = attn(Q_fp, K_kvq, V_kvq)
    out_both = attn(Q_wq, K_kvq.repeat_interleave(n_heads // n_kv_heads, dim=1) if K_kvq.shape[1] != n_heads else K_kvq,
                    V_kvq.repeat_interleave(n_heads // n_kv_heads, dim=1) if V_kvq.shape[1] != n_heads else V_kvq)

    # Compute errors
    err_w = (out_wq - out_fp).flatten()
    err_kv = (out_kvq - out_fp).flatten()
    err_both = (out_both - out_fp).flatten()

    # Correlation
    corr = torch.corrcoef(torch.stack([err_w, err_kv]))[0, 1].item()

    return {
        'corr': corr,
        'mse_w': (err_w ** 2).mean().item(),
        'mse_kv': (err_kv ** 2).mean().item(),
        'mse_both': (err_both ** 2).mean().item(),
    }


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    text = "The quick brown fox jumps over the lazy dog." * 5
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

    with torch.no_grad():
        embed = model.model.embed_tokens(inputs['input_ids']).float()

    # Test layer 4 (middle layer)
    layer = model.model.layers[4]
    attn = layer.self_attn

    W_Q = attn.q_proj.weight.data.float()
    W_K = attn.k_proj.weight.data.float()
    W_V = attn.v_proj.weight.data.float()
    W_O = attn.o_proj.weight.data.float()

    d_head = attn.head_dim
    n_heads = W_Q.shape[0] // d_head

    print(f"\n{'='*70}")
    print("TESTING SCALE EFFECT ON ERROR CORRELATION")
    print(f"{'='*70}")

    # Test different scale combinations
    scale_w_values = [0.5, 0.8, 1.0, 1.2, 1.5]
    scale_kv_values = [0.5, 0.8, 1.0, 1.2, 1.5]

    print(f"\n{'scale_w':>10} {'scale_kv':>10} {'corr':>12} {'mse_w':>12} {'mse_kv':>12} {'mse_both':>12} {'mult':>10}")
    print("-" * 80)

    results = []
    for scale_w in scale_w_values:
        for scale_kv in scale_kv_values:
            with torch.no_grad():
                r = compute_attention_error(
                    embed, W_Q, W_K, W_V, W_O,
                    scale_w=scale_w, scale_kv=scale_kv,
                    bits_w=3, bits_kv=4, n_heads=n_heads, d_head=d_head
                )
            mult = r['mse_both'] / (r['mse_w'] + r['mse_kv']) if (r['mse_w'] + r['mse_kv']) > 0 else float('inf')
            results.append((scale_w, scale_kv, r['corr'], r['mse_w'], r['mse_kv'], r['mse_both'], mult))
            print(f"{scale_w:>10.1f} {scale_kv:>10.1f} {r['corr']:>12.4f} {r['mse_w']:>12.6f} {r['mse_kv']:>12.6f} {r['mse_both']:>12.6f} {mult:>10.3f}")

    # Analyze results
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    corrs = [r[2] for r in results]
    mults = [r[6] for r in results]

    print(f"Correlation range: {min(corrs):.4f} to {max(corrs):.4f}")
    print(f"Mult factor range: {min(mults):.4f} to {max(mults):.4f}")

    corr_range = max(corrs) - min(corrs)
    mult_range = max(mults) - min(mults)

    if corr_range > 0.1:
        print(f"\n→ Correlation IS controllable via scales (range={corr_range:.4f})")
        print("  ECAQ hypothesis is VIABLE!")
    else:
        print(f"\n→ Correlation NOT controllable via scales (range={corr_range:.4f})")
        print("  ECAQ hypothesis may NOT work")

    # Find optimal scale combination
    best_idx = np.argmin([r[5] for r in results])  # Minimize mse_both
    best = results[best_idx]
    print(f"\nBest configuration:")
    print(f"  scale_w={best[0]:.1f}, scale_kv={best[1]:.1f}")
    print(f"  mse_both={best[5]:.6f}, mult={best[6]:.3f}")

    # Compare to baseline (1.0, 1.0)
    baseline = next(r for r in results if r[0] == 1.0 and r[1] == 1.0)
    improvement = (baseline[5] - best[5]) / baseline[5] * 100
    print(f"\n  Improvement over baseline: {improvement:.1f}%")


if __name__ == "__main__":
    main()
