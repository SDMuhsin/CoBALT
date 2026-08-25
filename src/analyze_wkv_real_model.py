"""
Analyze weight-KV interaction with REAL model weights from Qwen 1.7B.
This tests if the near-additive error pattern holds with actual LLM weights.
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
    d_model = W_Q.shape[1]
    d_head = attn.head_dim
    n_heads = W_Q.shape[0] // d_head
    batch_size, seq_len = x.shape[0], x.shape[1]

    # Full precision projections
    Q_fp = (x @ W_Q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = (x @ W_K.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)  # May have different n_kv_heads
    V_fp = (x @ W_V.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)

    # Handle GQA (grouped query attention) - repeat K, V if needed
    n_kv_heads = K_fp.shape[1]
    if n_kv_heads != n_heads:
        repeat_factor = n_heads // n_kv_heads
        K_fp = K_fp.repeat_interleave(repeat_factor, dim=1)
        V_fp = V_fp.repeat_interleave(repeat_factor, dim=1)

    out_fp, _ = attention_forward(Q_fp, K_fp, V_fp)

    # === Scenario 1: Quantize only weights ===
    W_Q_q, _ = quantize_tensor(W_Q, bits=bits_w, per_channel=True, dim=1)
    W_K_q, _ = quantize_tensor(W_K, bits=bits_w, per_channel=True, dim=1)
    W_V_q, _ = quantize_tensor(W_V, bits=bits_w, per_channel=True, dim=1)

    Q_wq = (x @ W_Q_q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_wq = (x @ W_K_q.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)
    V_wq = (x @ W_V_q.T).view(batch_size, seq_len, -1, d_head).transpose(1, 2)

    if K_wq.shape[1] != n_heads:
        K_wq = K_wq.repeat_interleave(repeat_factor, dim=1)
        V_wq = V_wq.repeat_interleave(repeat_factor, dim=1)

    out_weight_only, _ = attention_forward(Q_wq, K_wq, V_wq)

    # === Scenario 2: Quantize only KV cache (after FP projection) ===
    K_kvq, _ = quantize_tensor(K_fp, bits=bits_kv, per_channel=True, dim=-1)
    V_kvq, _ = quantize_tensor(V_fp, bits=bits_kv, per_channel=True, dim=-1)

    out_kv_only, _ = attention_forward(Q_fp, K_kvq, V_kvq)

    # === Scenario 3: Quantize both (independent) ===
    K_both, _ = quantize_tensor(K_wq, bits=bits_kv, per_channel=True, dim=-1)
    V_both, _ = quantize_tensor(V_wq, bits=bits_kv, per_channel=True, dim=-1)

    out_both, _ = attention_forward(Q_wq, K_both, V_both)

    # === Error Analysis ===
    err_weight = ((out_weight_only - out_fp) ** 2).mean().item()
    err_kv = ((out_kv_only - out_fp) ** 2).mean().item()
    err_both = ((out_both - out_fp) ** 2).mean().item()

    err_sum_independent = err_weight + err_kv
    err_interaction = err_both - err_sum_independent
    mult_factor = err_both / err_sum_independent if err_sum_independent > 0 else float('inf')

    return {
        'layer': layer_idx,
        'err_weight': err_weight,
        'err_kv': err_kv,
        'err_both': err_both,
        'err_sum': err_sum_independent,
        'err_interaction': err_interaction,
        'mult_factor': mult_factor,
    }


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

    # Get hidden states after embedding
    with torch.no_grad():
        embed = model.model.embed_tokens(inputs['input_ids'])
        x = embed.float()

    print("\n" + "=" * 100)
    print("REAL MODEL WEIGHT-KV ERROR INTERACTION ANALYSIS (Qwen 1.7B)")
    print("=" * 100)

    configs = [
        (3, 4),  # 3-bit weights, 4-bit KV (our target)
        (3, 2),  # 3-bit weights, 2-bit KV
        (4, 4),  # 4-bit weights, 4-bit KV
    ]

    for bits_w, bits_kv in configs:
        print(f"\n--- Config: {bits_w}-bit weights, {bits_kv}-bit KV ---")
        print(f"{'Layer':>6} {'Err_W':>12} {'Err_KV':>12} {'Err_Both':>12} {'Sum':>12} {'Interact':>12} {'Mult':>8}")
        print("-" * 80)

        mult_factors = []
        for layer_idx in [0, 7, 14, 21, 27]:  # Sample layers
            result = analyze_layer(model, layer_idx, x, bits_w=bits_w, bits_kv=bits_kv)
            mult_factors.append(result['mult_factor'])

            print(f"{result['layer']:>6} {result['err_weight']:>12.6f} {result['err_kv']:>12.6f} "
                  f"{result['err_both']:>12.6f} {result['err_sum']:>12.6f} "
                  f"{result['err_interaction']:>12.6f} {result['mult_factor']:>8.3f}")

        avg_mult = np.mean(mult_factors)
        print(f"\n  Average multiplicative factor: {avg_mult:.3f}")
        if avg_mult > 1.1:
            print("  → SUPER-ADDITIVE: Joint optimization has potential!")
        elif avg_mult < 0.9:
            print("  → SUB-ADDITIVE: Errors partially cancel")
        else:
            print("  → NEAR-ADDITIVE: Limited joint optimization benefit")

    print("\n" + "=" * 100)
    print("CONCLUSIONS FOR JOINT WEIGHT-KV OPTIMIZATION")
    print("=" * 100)


if __name__ == "__main__":
    main()
