"""
Phase 4: Micro-validation of ECAQ across all layers.

Test whether joint scale optimization improves attention output error
across all transformer layers, not just one.
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


def compute_layer_error(h, layer, scale_w=1.0, scale_kv=1.0, bits_w=3, bits_kv=4):
    """Compute attention output error for a single layer with given scales."""
    attn = layer.self_attn
    batch_size, seq_len, _ = h.shape

    W_Q = attn.q_proj.weight.data.float()
    W_K = attn.k_proj.weight.data.float()
    W_V = attn.v_proj.weight.data.float()

    d_head = attn.head_dim
    n_heads = W_Q.shape[0] // d_head
    n_kv_heads = W_K.shape[0] // d_head

    # Full precision computation
    Q_fp = (h @ W_Q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = (h @ W_K.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V_fp = (h @ W_V.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

    # Quantize weights with scale
    W_Q_max = W_Q.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    W_K_max = W_K.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    W_V_max = W_V.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)

    s_Q = scale_w * W_Q_max / (2 ** (bits_w - 1) - 1)
    s_K_w = scale_w * W_K_max / (2 ** (bits_w - 1) - 1)
    s_V_w = scale_w * W_V_max / (2 ** (bits_w - 1) - 1)

    W_Q_q = quantize_with_scale(W_Q, s_Q, bits_w)
    W_K_q = quantize_with_scale(W_K, s_K_w, bits_w)
    W_V_q = quantize_with_scale(W_V, s_V_w, bits_w)

    Q_wq = (h @ W_Q_q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_wq = (h @ W_K_q.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V_wq = (h @ W_V_q.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

    # Quantize KV cache with scale
    K_max = K_wq.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    V_max = V_wq.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)

    s_K_kv = scale_kv * K_max / (2 ** (bits_kv - 1) - 1)
    s_V_kv = scale_kv * V_max / (2 ** (bits_kv - 1) - 1)

    K_q = quantize_with_scale(K_wq, s_K_kv, bits_kv)
    V_q = quantize_with_scale(V_wq, s_V_kv, bits_kv)

    # GQA expansion
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        K_fp = K_fp.repeat_interleave(rep, dim=1)
        V_fp = V_fp.repeat_interleave(rep, dim=1)
        K_q = K_q.repeat_interleave(rep, dim=1)
        V_q = V_q.repeat_interleave(rep, dim=1)

    # Attention
    scale_attn = 1.0 / np.sqrt(d_head)

    def attn_fn(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_attn
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    out_fp = attn_fn(Q_fp, K_fp, V_fp)
    out_q = attn_fn(Q_wq, K_q, V_q)

    mse = ((out_q - out_fp) ** 2).mean().item()
    return mse


def search_optimal_scales(h, layer, bits_w=3, bits_kv=4, search_grid=None):
    """Find optimal scale combination for a layer."""
    if search_grid is None:
        search_grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    best_mse = float('inf')
    best_scales = (1.0, 1.0)

    for scale_w in search_grid:
        for scale_kv in search_grid:
            mse = compute_layer_error(h, layer, scale_w, scale_kv, bits_w, bits_kv)
            if mse < best_mse:
                best_mse = mse
                best_scales = (scale_w, scale_kv)

    baseline_mse = compute_layer_error(h, layer, 1.0, 1.0, bits_w, bits_kv)

    return {
        'baseline_mse': baseline_mse,
        'optimal_mse': best_mse,
        'optimal_scale_w': best_scales[0],
        'optimal_scale_kv': best_scales[1],
        'improvement': (baseline_mse - best_mse) / baseline_mse * 100 if baseline_mse > 0 else 0,
    }


def main():
    print("Loading Qwen 1.7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Multiple calibration samples
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago—never mind how long precisely.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
    ]

    with torch.no_grad():
        embeds = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
            embed = model.model.embed_tokens(inputs['input_ids']).float()
            embeds.append(embed)
        h = torch.cat(embeds, dim=0)  # [5, 64, 2048]

    print(f"\nCalibration data shape: {h.shape}")

    print(f"\n{'='*80}")
    print("ECAQ MICRO-VALIDATION ACROSS ALL LAYERS")
    print(f"Config: 3-bit weights, 4-bit KV")
    print(f"{'='*80}")

    results = []
    print(f"\n{'Layer':>6} {'Baseline MSE':>14} {'Optimal MSE':>14} {'scale_w':>10} {'scale_kv':>10} {'Improve %':>12}")
    print("-" * 70)

    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        r = search_optimal_scales(h, layer, bits_w=3, bits_kv=4)
        results.append(r)

        print(f"{layer_idx:>6} {r['baseline_mse']:>14.8f} {r['optimal_mse']:>14.8f} "
              f"{r['optimal_scale_w']:>10.2f} {r['optimal_scale_kv']:>10.2f} {r['improvement']:>12.1f}")

    # Summary statistics
    improvements = [r['improvement'] for r in results]
    avg_improvement = np.mean(improvements)
    positive_improvements = sum(1 for x in improvements if x > 0)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Average improvement: {avg_improvement:.1f}%")
    print(f"Layers with positive improvement: {positive_improvements}/{len(results)}")
    print(f"Max improvement: {max(improvements):.1f}%")
    print(f"Min improvement: {min(improvements):.1f}%")

    # Most common optimal scales
    scale_w_counts = {}
    scale_kv_counts = {}
    for r in results:
        sw = r['optimal_scale_w']
        sk = r['optimal_scale_kv']
        scale_w_counts[sw] = scale_w_counts.get(sw, 0) + 1
        scale_kv_counts[sk] = scale_kv_counts.get(sk, 0) + 1

    print(f"\nOptimal scale_w distribution: {dict(sorted(scale_w_counts.items()))}")
    print(f"Optimal scale_kv distribution: {dict(sorted(scale_kv_counts.items()))}")

    # Verdict
    print(f"\n{'='*80}")
    if avg_improvement > 5:
        print("VERDICT: ECAQ shows consistent improvement across layers!")
        print("Proceed to Phase 5 (Implementation)")
    elif avg_improvement > 0:
        print("VERDICT: ECAQ shows marginal improvement.")
        print("Consider whether the complexity is justified.")
    else:
        print("VERDICT: ECAQ does NOT show consistent improvement.")
        print("Return to Phase 1.")


if __name__ == "__main__":
    main()
