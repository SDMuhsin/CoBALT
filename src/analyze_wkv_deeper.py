"""
Deeper analysis of weight-KV interaction to find potential joint optimization opportunities.

Key questions:
1. Do errors accumulate differently across multiple attention layers?
2. Is there layer-specific variation in error interaction?
3. What's the optimal bit allocation given a memory budget?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_tensor(x, bits=4, per_channel=True, dim=-1):
    """Simple uniform quantization."""
    if per_channel:
        x_max = x.abs().amax(dim=dim, keepdim=True)
    else:
        x_max = x.abs().max()

    x_max = x_max.clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = x_max / levels

    x_q = (x / scale).round().clamp(-levels, levels)
    x_dq = x_q * scale

    return x_dq, scale


def attention_forward(Q, K, V):
    """Compute scaled dot-product attention."""
    d_k = Q.shape[-1]
    scale_factor = 1.0 / np.sqrt(d_k)

    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_factor
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


def analyze_layer(model, layer_idx, x, bits_w=3, bits_kv=4):
    """Analyze error interaction for a single transformer layer."""
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    # Get weight matrices
    W_Q = attn.q_proj.weight.data.float()
    W_K = attn.k_proj.weight.data.float()
    W_V = attn.v_proj.weight.data.float()

    # Dimensions
    d_head = attn.head_dim
    n_heads = W_Q.shape[0] // d_head
    batch_size, seq_len = x.shape[0], x.shape[1]

    # Full precision projections
    Q_fp = (x @ W_Q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = (x @ W_K.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)
    V_fp = (x @ W_V.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)

    # Handle GQA
    n_kv_heads = K_fp.shape[1]
    repeat_factor = n_heads // n_kv_heads if n_kv_heads != n_heads else 1
    if n_kv_heads != n_heads:
        K_fp_expanded = K_fp.repeat_interleave(repeat_factor, dim=1)
        V_fp_expanded = V_fp.repeat_interleave(repeat_factor, dim=1)
    else:
        K_fp_expanded = K_fp
        V_fp_expanded = V_fp

    out_fp, _ = attention_forward(Q_fp, K_fp_expanded, V_fp_expanded)

    # === Scenario 1: Quantize only weights ===
    W_Q_q, _ = quantize_tensor(W_Q, bits=bits_w, per_channel=True, dim=1)
    W_K_q, _ = quantize_tensor(W_K, bits=bits_w, per_channel=True, dim=1)
    W_V_q, _ = quantize_tensor(W_V, bits=bits_w, per_channel=True, dim=1)

    Q_wq = (x @ W_Q_q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_wq = (x @ W_K_q.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)
    V_wq = (x @ W_V_q.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)

    if n_kv_heads != n_heads:
        K_wq_expanded = K_wq.repeat_interleave(repeat_factor, dim=1)
        V_wq_expanded = V_wq.repeat_interleave(repeat_factor, dim=1)
    else:
        K_wq_expanded = K_wq
        V_wq_expanded = V_wq

    out_weight_only, _ = attention_forward(Q_wq, K_wq_expanded, V_wq_expanded)

    # === Scenario 2: Quantize only KV cache (after FP projection) ===
    K_kvq, _ = quantize_tensor(K_fp, bits=bits_kv, per_channel=True, dim=-1)
    V_kvq, _ = quantize_tensor(V_fp, bits=bits_kv, per_channel=True, dim=-1)

    if n_kv_heads != n_heads:
        K_kvq_expanded = K_kvq.repeat_interleave(repeat_factor, dim=1)
        V_kvq_expanded = V_kvq.repeat_interleave(repeat_factor, dim=1)
    else:
        K_kvq_expanded = K_kvq
        V_kvq_expanded = V_kvq

    out_kv_only, _ = attention_forward(Q_fp, K_kvq_expanded, V_kvq_expanded)

    # === Scenario 3: Quantize both (independent) ===
    K_both, _ = quantize_tensor(K_wq, bits=bits_kv, per_channel=True, dim=-1)
    V_both, _ = quantize_tensor(V_wq, bits=bits_kv, per_channel=True, dim=-1)

    if n_kv_heads != n_heads:
        K_both_expanded = K_both.repeat_interleave(repeat_factor, dim=1)
        V_both_expanded = V_both.repeat_interleave(repeat_factor, dim=1)
    else:
        K_both_expanded = K_both
        V_both_expanded = V_both

    out_both, _ = attention_forward(Q_wq, K_both_expanded, V_both_expanded)

    # === Error Analysis ===
    err_weight = ((out_weight_only - out_fp) ** 2).mean().item()
    err_kv = ((out_kv_only - out_fp) ** 2).mean().item()
    err_both = ((out_both - out_fp) ** 2).mean().item()

    err_sum_independent = err_weight + err_kv
    mult_factor = err_both / err_sum_independent if err_sum_independent > 0 else float('inf')

    return {
        'layer': layer_idx,
        'err_weight': err_weight,
        'err_kv': err_kv,
        'err_both': err_both,
        'err_sum': err_sum_independent,
        'mult_factor': mult_factor,
    }


def analyze_multi_layer_accumulation(model, x, num_layers=5, bits_w=3, bits_kv=4):
    """
    Analyze how errors accumulate when passing through multiple layers.
    This simulates the "prefix accumulation" effect in long-context generation.
    """
    print(f"\n{'='*80}")
    print("MULTI-LAYER ERROR ACCUMULATION ANALYSIS")
    print(f"{'='*80}")

    device = x.device
    batch_size, seq_len, d_model = x.shape

    # We'll track hidden states through layers
    h_fp = x.clone()
    h_wq = x.clone()  # Weight-only quantized path
    h_kvq = x.clone()  # KV-only quantized path
    h_both = x.clone()  # Both quantized path

    layer_errors = []

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        W_Q = attn.q_proj.weight.data.float()
        W_K = attn.k_proj.weight.data.float()
        W_V = attn.v_proj.weight.data.float()
        W_O = attn.o_proj.weight.data.float()

        d_head = attn.head_dim
        n_heads = W_Q.shape[0] // d_head
        n_kv_heads = W_K.shape[0] // d_head
        repeat_factor = n_heads // n_kv_heads if n_kv_heads != n_heads else 1

        # Quantized weights
        W_Q_q, _ = quantize_tensor(W_Q, bits=bits_w, per_channel=True, dim=1)
        W_K_q, _ = quantize_tensor(W_K, bits=bits_w, per_channel=True, dim=1)
        W_V_q, _ = quantize_tensor(W_V, bits=bits_w, per_channel=True, dim=1)
        W_O_q, _ = quantize_tensor(W_O, bits=bits_w, per_channel=True, dim=1)

        def process_attention(h, W_Q_use, W_K_use, W_V_use, W_O_use, quant_kv=False, kv_bits=4):
            Q = (h @ W_Q_use.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            K = (h @ W_K_use.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
            V = (h @ W_V_use.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

            if quant_kv:
                K, _ = quantize_tensor(K, bits=kv_bits, per_channel=True, dim=-1)
                V, _ = quantize_tensor(V, bits=kv_bits, per_channel=True, dim=-1)

            if n_kv_heads != n_heads:
                K = K.repeat_interleave(repeat_factor, dim=1)
                V = V.repeat_interleave(repeat_factor, dim=1)

            out, _ = attention_forward(Q, K, V)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            out = out @ W_O_use.T

            return h + out  # Residual connection

        # FP path
        h_fp = process_attention(h_fp, W_Q, W_K, W_V, W_O, quant_kv=False)

        # Weight-only quantized path
        h_wq = process_attention(h_wq, W_Q_q, W_K_q, W_V_q, W_O_q, quant_kv=False)

        # KV-only quantized path (FP weights, quantized KV)
        h_kvq = process_attention(h_kvq, W_Q, W_K, W_V, W_O, quant_kv=True, kv_bits=bits_kv)

        # Both quantized path
        h_both = process_attention(h_both, W_Q_q, W_K_q, W_V_q, W_O_q, quant_kv=True, kv_bits=bits_kv)

        # Measure errors relative to FP
        err_wq = ((h_wq - h_fp) ** 2).mean().item()
        err_kvq = ((h_kvq - h_fp) ** 2).mean().item()
        err_both = ((h_both - h_fp) ** 2).mean().item()

        err_sum = err_wq + err_kvq
        mult = err_both / err_sum if err_sum > 0 else float('inf')

        layer_errors.append({
            'layer': layer_idx,
            'err_wq': err_wq,
            'err_kvq': err_kvq,
            'err_both': err_both,
            'err_sum': err_sum,
            'mult': mult,
        })

        print(f"Layer {layer_idx:2d}: Err_WQ={err_wq:.6f}, Err_KVQ={err_kvq:.6f}, "
              f"Err_Both={err_both:.6f}, Sum={err_sum:.6f}, Mult={mult:.3f}")

    # Final summary
    final_mult = layer_errors[-1]['mult']
    print(f"\n{'='*40}")
    print(f"After {num_layers} layers: Multiplicative factor = {final_mult:.3f}")
    if final_mult > 1.2:
        print("→ SUPER-ADDITIVE: Errors compound through layers!")
    elif final_mult < 0.8:
        print("→ SUB-ADDITIVE: Errors cancel through layers")
    else:
        print("→ NEAR-ADDITIVE: Limited interaction benefit")

    return layer_errors


def analyze_bit_allocation(model, x, memory_budget_ratio=1.0):
    """
    Given a fixed memory budget, what's the optimal bit allocation between weights and KV?

    memory_budget_ratio = 1.0 means baseline (4-bit W + 4-bit KV)
    We explore different allocations that use the same memory.
    """
    print(f"\n{'='*80}")
    print("BIT ALLOCATION ANALYSIS")
    print(f"{'='*80}")

    # Baseline: 4-bit weights, 4-bit KV
    # Memory model (simplified):
    # - Weight memory is fixed at model load
    # - KV cache grows with context length

    # For a fixed context length, we can trade weight bits for KV bits
    # 3-bit W + 5.33-bit KV ≈ 4-bit W + 4-bit KV (in total bits)
    # 2-bit W + 6-bit KV ≈ 4-bit W + 4-bit KV

    configs = [
        (4, 4, "Baseline (4W+4KV)"),
        (3, 4, "3W+4KV"),
        (4, 2, "4W+2KV"),
        (3, 2, "3W+2KV (aggressive)"),
        (2, 4, "2W+4KV"),
    ]

    results = []
    print(f"\n{'Config':<25} {'Total MSE':>15} {'W_MSE':>12} {'KV_MSE':>12}")
    print("-" * 70)

    for bits_w, bits_kv, name in configs:
        # Average over several layers
        errs = []
        for layer_idx in [0, 7, 14, 21, 27]:
            r = analyze_layer(model, layer_idx, x, bits_w=bits_w, bits_kv=bits_kv)
            errs.append(r)

        avg_err_both = np.mean([e['err_both'] for e in errs])
        avg_err_w = np.mean([e['err_weight'] for e in errs])
        avg_err_kv = np.mean([e['err_kv'] for e in errs])

        results.append((name, bits_w, bits_kv, avg_err_both, avg_err_w, avg_err_kv))
        print(f"{name:<25} {avg_err_both:>15.6f} {avg_err_w:>12.6f} {avg_err_kv:>12.6f}")

    # Find optimal given different memory constraints
    print(f"\n{'='*40}")
    print("OPTIMAL ALLOCATION INSIGHTS:")

    # Compare 3W+4KV vs 4W+2KV (similar memory for long context)
    err_3w4kv = next(r[3] for r in results if r[0] == "3W+4KV")
    err_4w2kv = next(r[3] for r in results if r[0] == "4W+2KV")

    print(f"\n3W+4KV vs 4W+2KV (trade weight bits for KV bits):")
    print(f"  3W+4KV error: {err_3w4kv:.6f}")
    print(f"  4W+2KV error: {err_4w2kv:.6f}")
    if err_3w4kv < err_4w2kv:
        print("  → Better to reduce weight bits and preserve KV precision!")
    else:
        print("  → Better to preserve weight precision and reduce KV bits")

    return results


def main():
    print("Loading Qwen 1.7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Generate some input activations
    text = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)

    with torch.no_grad():
        embed = model.model.embed_tokens(inputs['input_ids'])
        x = embed.float()

    print(f"\nInput shape: {x.shape}")

    # Analysis 1: Multi-layer error accumulation
    analyze_multi_layer_accumulation(model, x, num_layers=8, bits_w=3, bits_kv=4)

    # Analysis 2: Bit allocation
    analyze_bit_allocation(model, x)

    print("\n" + "=" * 80)
    print("KEY INSIGHT FOR JOINT OPTIMIZATION")
    print("=" * 80)
    print("""
    Even if per-layer errors are near-additive, there are two potential avenues:

    1. MULTI-LAYER ACCUMULATION: Errors may compound differently through the full
       transformer stack. Weight errors affect all future computations, while KV
       errors only affect the attention of specific tokens.

    2. BIT ALLOCATION: Given a memory budget, the optimal bit split between weights
       and KV cache may not be obvious. Joint optimization can help find this split.

    3. LAYER-SPECIFIC OPTIMIZATION: Different layers may benefit from different
       weight/KV precision ratios based on their role in the model.
    """)


if __name__ == "__main__":
    main()
