"""
Phase 1.2 Analysis: Understanding Weight-KV Error Interaction in Attention

This script analyzes how quantization errors in weights and KV cache interact
in the attention computation:
    Attention(Q, K, V) = softmax(Q·K^T / √d) · V

Error sources:
1. Q = x·W_Q where W_Q is quantized → error ΔQ
2. K is quantized in KV cache → error ΔK
3. V is quantized in KV cache → error ΔV

Goal: Derive the error interaction terms and measure their magnitude empirically.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')


def quantize_tensor(x, bits=4, per_channel=True):
    """Simple uniform quantization."""
    if per_channel:
        # Per-channel along last dim
        x_max = x.abs().amax(dim=-1, keepdim=True)
    else:
        x_max = x.abs().max()

    x_max = x_max.clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = x_max / levels

    x_q = (x / scale).round().clamp(-levels, levels)
    x_dq = x_q * scale

    return x_dq, scale


def attention_forward(Q, K, V, scale_factor=None):
    """Compute scaled dot-product attention."""
    d_k = Q.shape[-1]
    if scale_factor is None:
        scale_factor = 1.0 / np.sqrt(d_k)

    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_factor
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


def analyze_error_decomposition(batch_size=1, seq_len=128, d_model=512, n_heads=8, bits_w=3, bits_kv=4):
    """
    Decompose attention output error into:
    1. Weight-only error (Q quantized, K/V full precision)
    2. KV-only error (Q full precision, K/V quantized)
    3. Interaction error (difference between combined error and sum of individual errors)
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d_head = d_model // n_heads

    # Generate random inputs
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Generate random weight matrices (for Q, K, V projections)
    W_Q = torch.randn(d_model, d_model, device=device) * 0.02
    W_K = torch.randn(d_model, d_model, device=device) * 0.02
    W_V = torch.randn(d_model, d_model, device=device) * 0.02

    # Full precision projections
    Q_fp = x @ W_Q
    K_fp = x @ W_K
    V_fp = x @ W_V

    # Reshape for multi-head attention
    Q_fp = Q_fp.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = K_fp.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    V_fp = V_fp.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

    # Full precision attention output (ground truth)
    out_fp, _ = attention_forward(Q_fp, K_fp, V_fp)

    # === Scenario 1: Quantize only weights (W_Q, W_K, W_V) ===
    W_Q_q, _ = quantize_tensor(W_Q, bits=bits_w, per_channel=True)
    W_K_q, _ = quantize_tensor(W_K, bits=bits_w, per_channel=True)
    W_V_q, _ = quantize_tensor(W_V, bits=bits_w, per_channel=True)

    Q_wq = x @ W_Q_q
    K_wq = x @ W_K_q  # K from quantized weights, but stored in FP KV cache
    V_wq = x @ W_V_q

    Q_wq = Q_wq.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_wq = K_wq.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    V_wq = V_wq.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

    out_weight_only, _ = attention_forward(Q_wq, K_wq, V_wq)

    # === Scenario 2: Quantize only KV cache (K, V after projection) ===
    K_kvq, _ = quantize_tensor(K_fp, bits=bits_kv, per_channel=True)
    V_kvq, _ = quantize_tensor(V_fp, bits=bits_kv, per_channel=True)

    out_kv_only, _ = attention_forward(Q_fp, K_kvq, V_kvq)

    # === Scenario 3: Quantize both (independent optimization) ===
    K_both, _ = quantize_tensor(K_wq, bits=bits_kv, per_channel=True)  # K from quant weights, then quant KV
    V_both, _ = quantize_tensor(V_wq, bits=bits_kv, per_channel=True)

    out_both, _ = attention_forward(Q_wq, K_both, V_both)

    # === Error Analysis ===
    err_weight = ((out_weight_only - out_fp) ** 2).mean().item()
    err_kv = ((out_kv_only - out_fp) ** 2).mean().item()
    err_both = ((out_both - out_fp) ** 2).mean().item()

    # If errors were independent and additive:
    # err_both ≈ err_weight + err_kv
    # Interaction term = err_both - (err_weight + err_kv)
    err_sum_independent = err_weight + err_kv
    err_interaction = err_both - err_sum_independent
    interaction_ratio = err_interaction / err_both if err_both > 0 else 0

    return {
        'err_weight_only': err_weight,
        'err_kv_only': err_kv,
        'err_both_independent': err_both,
        'err_sum_if_additive': err_sum_independent,
        'err_interaction': err_interaction,
        'interaction_ratio': interaction_ratio,
        'multiplicative_factor': err_both / err_sum_independent if err_sum_independent > 0 else float('inf'),
    }


def analyze_across_bit_configs():
    """Analyze error interaction across different quantization bit configurations."""
    print("=" * 80)
    print("WEIGHT-KV ERROR INTERACTION ANALYSIS")
    print("=" * 80)

    configs = [
        (4, 4),  # 4-bit weights, 4-bit KV
        (4, 2),  # 4-bit weights, 2-bit KV
        (3, 4),  # 3-bit weights, 4-bit KV
        (3, 2),  # 3-bit weights, 2-bit KV
        (2, 4),  # 2-bit weights, 4-bit KV
        (2, 2),  # 2-bit weights, 2-bit KV
    ]

    print(f"\n{'Bits_W':>8} {'Bits_KV':>8} {'Err_W':>12} {'Err_KV':>12} {'Err_Both':>12} {'Sum':>12} {'Interaction':>12} {'Mult_Factor':>12}")
    print("-" * 100)

    results = []
    for bits_w, bits_kv in configs:
        r = analyze_error_decomposition(
            batch_size=1, seq_len=256, d_model=1024, n_heads=8,
            bits_w=bits_w, bits_kv=bits_kv
        )
        results.append((bits_w, bits_kv, r))

        print(f"{bits_w:>8} {bits_kv:>8} {r['err_weight_only']:>12.6f} {r['err_kv_only']:>12.6f} "
              f"{r['err_both_independent']:>12.6f} {r['err_sum_if_additive']:>12.6f} "
              f"{r['err_interaction']:>12.6f} {r['multiplicative_factor']:>12.2f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
    - Mult_Factor > 1.0: Errors compound MORE than additively (super-additive)
    - Mult_Factor = 1.0: Errors are additive (independent)
    - Mult_Factor < 1.0: Errors partially cancel (sub-additive)

    If Mult_Factor > 1.0, joint optimization has potential to reduce the interaction term.
    """)

    return results


def analyze_error_correlation():
    """Analyze correlation between weight error and KV error per position/head."""
    print("\n" + "=" * 80)
    print("ERROR CORRELATION ANALYSIS")
    print("=" * 80)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, seq_len, d_model, n_heads = 1, 256, 1024, 8
    d_head = d_model // n_heads
    bits_w, bits_kv = 3, 4

    # Generate inputs and weights
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    W_Q = torch.randn(d_model, d_model, device=device) * 0.02
    W_K = torch.randn(d_model, d_model, device=device) * 0.02
    W_V = torch.randn(d_model, d_model, device=device) * 0.02

    # Full precision
    Q_fp = (x @ W_Q).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = (x @ W_K).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    V_fp = (x @ W_V).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    out_fp, attn_fp = attention_forward(Q_fp, K_fp, V_fp)

    # Quantized weights
    W_Q_q, _ = quantize_tensor(W_Q, bits=bits_w)
    W_K_q, _ = quantize_tensor(W_K, bits=bits_w)
    W_V_q, _ = quantize_tensor(W_V, bits=bits_w)

    Q_wq = (x @ W_Q_q).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_wq = (x @ W_K_q).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    V_wq = (x @ W_V_q).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

    # Quantized KV cache
    K_kvq, _ = quantize_tensor(K_wq, bits=bits_kv)
    V_kvq, _ = quantize_tensor(V_wq, bits=bits_kv)

    # Error in Q (from weight quantization)
    Q_err = Q_wq - Q_fp

    # Error in K/V (from weight quant + KV quant)
    K_err = K_kvq - K_fp
    V_err = V_kvq - V_fp

    # Per-head correlation between Q error and K error
    print(f"\nPer-head correlation (Q_err, K_err):")
    correlations = []
    for h in range(n_heads):
        q_e = Q_err[0, h].flatten().cpu().numpy()
        k_e = K_err[0, h].flatten().cpu().numpy()
        corr = np.corrcoef(q_e, k_e)[0, 1]
        correlations.append(corr)
        print(f"  Head {h}: {corr:.4f}")

    print(f"\n  Mean correlation: {np.mean(correlations):.4f}")
    print(f"  Std correlation:  {np.std(correlations):.4f}")

    # Interpretation
    if abs(np.mean(correlations)) > 0.1:
        print("\n  → Errors are CORRELATED: Joint optimization may help by exploiting this structure")
    else:
        print("\n  → Errors are largely UNCORRELATED: Joint optimization benefit may be limited")


def main():
    results = analyze_across_bit_configs()
    analyze_error_correlation()

    print("\n" + "=" * 80)
    print("SUMMARY: KEY FINDINGS FOR JOINT OPTIMIZATION")
    print("=" * 80)

    # Find the configuration with highest multiplicative factor
    max_mult = max(r['multiplicative_factor'] for _, _, r in results)
    min_mult = min(r['multiplicative_factor'] for _, _, r in results)

    print(f"""
    1. Multiplicative factor range: {min_mult:.2f}x - {max_mult:.2f}x
       {'→ SUPER-ADDITIVE errors: Joint optimization has potential!' if max_mult > 1.1 else '→ Near-additive errors: Limited joint optimization benefit'}

    2. The interaction term comes from:
       - Softmax(Q·K^T) is non-linear: errors in Q and K interact non-linearly
       - V error is multiplied by attention weights that are themselves erroneous

    3. Joint optimization opportunity:
       - If we can adjust weight scales to reduce Q error in directions that
         align with K error, we may reduce the multiplicative interaction.
       - This requires calibration data to measure the actual Q-K error alignment.
    """)


if __name__ == "__main__":
    main()
