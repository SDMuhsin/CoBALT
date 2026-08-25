"""
Verify whether ECAQ benefit comes from:
A) Better individual quantization (tighter scale)
B) Error cancellation (cross-term reduction)
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize(x, bits, scale_mult=1.0, dim=-1):
    """Quantize with scale multiplier."""
    x_max = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = scale_mult * x_max / levels
    x_q = (x / scale).round().clamp(-levels, levels) * scale
    return x_q


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    layer = model.model.layers[4].self_attn

    W_Q = layer.q_proj.weight.data.float()
    W_K = layer.k_proj.weight.data.float()
    W_V = layer.v_proj.weight.data.float()

    d_head = layer.head_dim
    n_heads = W_Q.shape[0] // d_head
    n_kv_heads = W_K.shape[0] // d_head

    # Input
    text = "The quick brown fox jumps over the lazy dog. " * 5
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
    with torch.no_grad():
        h = model.model.embed_tokens(inputs['input_ids']).float()

    batch_size, seq_len, _ = h.shape

    print(f"\n{'='*80}")
    print("DECOMPOSING ECAQ BENEFIT")
    print(f"{'='*80}")

    # Full precision
    Q_fp = (h @ W_Q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K_fp = (h @ W_K.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V_fp = (h @ W_V.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

    if n_kv_heads != n_heads:
        K_fp_exp = K_fp.repeat_interleave(n_heads // n_kv_heads, dim=1)
        V_fp_exp = V_fp.repeat_interleave(n_heads // n_kv_heads, dim=1)
    else:
        K_fp_exp = K_fp
        V_fp_exp = V_fp

    def attn(Q, K, V):
        scale = 1.0 / np.sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    out_fp = attn(Q_fp, K_fp_exp, V_fp_exp)

    # Test different configurations
    configs = [
        # (scale_w, scale_kv, description)
        (1.0, 1.0, "Baseline (1.0, 1.0)"),
        (0.5, 1.0, "Better W only (0.5, 1.0)"),
        (1.0, 0.8, "Better KV only (1.0, 0.8)"),
        (0.5, 0.8, "ECAQ joint (0.5, 0.8)"),
    ]

    print(f"\n{'Config':<25} {'Err_W':>12} {'Err_KV':>12} {'Err_Both':>12} {'Sum':>12} {'Mult':>8}")
    print("-" * 85)

    for scale_w, scale_kv, desc in configs:
        # Quantize weights
        W_Q_q = quantize(W_Q, 3, scale_w, dim=1)
        W_K_q = quantize(W_K, 3, scale_w, dim=1)
        W_V_q = quantize(W_V, 3, scale_w, dim=1)

        # Project with quantized weights
        Q_wq = (h @ W_Q_q.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        K_wq = (h @ W_K_q.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
        V_wq = (h @ W_V_q.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

        # Quantize KV cache
        K_q = quantize(K_wq, 4, scale_kv, dim=-1)
        V_q = quantize(V_wq, 4, scale_kv, dim=-1)

        # Also compute weight-only and KV-only paths for decomposition
        K_kvq = quantize(K_fp, 4, scale_kv, dim=-1)
        V_kvq = quantize(V_fp, 4, scale_kv, dim=-1)

        # Expand for GQA
        if n_kv_heads != n_heads:
            K_wq_exp = K_wq.repeat_interleave(n_heads // n_kv_heads, dim=1)
            V_wq_exp = V_wq.repeat_interleave(n_heads // n_kv_heads, dim=1)
            K_q_exp = K_q.repeat_interleave(n_heads // n_kv_heads, dim=1)
            V_q_exp = V_q.repeat_interleave(n_heads // n_kv_heads, dim=1)
            K_kvq_exp = K_kvq.repeat_interleave(n_heads // n_kv_heads, dim=1)
            V_kvq_exp = V_kvq.repeat_interleave(n_heads // n_kv_heads, dim=1)
        else:
            K_wq_exp = K_wq
            V_wq_exp = V_wq
            K_q_exp = K_q
            V_q_exp = V_q
            K_kvq_exp = K_kvq
            V_kvq_exp = V_kvq

        # Compute errors
        out_w = attn(Q_wq, K_wq_exp, V_wq_exp)  # Weight-only
        out_kv = attn(Q_fp, K_kvq_exp, V_kvq_exp)  # KV-only
        out_both = attn(Q_wq, K_q_exp, V_q_exp)  # Both

        err_w = ((out_w - out_fp) ** 2).mean().item()
        err_kv = ((out_kv - out_fp) ** 2).mean().item()
        err_both = ((out_both - out_fp) ** 2).mean().item()
        err_sum = err_w + err_kv
        mult = err_both / err_sum if err_sum > 0 else float('inf')

        print(f"{desc:<25} {err_w:>12.8f} {err_kv:>12.8f} {err_both:>12.8f} {err_sum:>12.8f} {mult:>8.3f}")

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    print("""
    If ECAQ benefit is from BETTER INDIVIDUAL QUANTIZATION:
    - "Better W only" should have lower Err_W than "Baseline"
    - "Better KV only" should have lower Err_KV than "Baseline"
    - "ECAQ joint" should be approximately = Better_W_only + Better_KV_only

    If ECAQ benefit is from ERROR CANCELLATION:
    - "ECAQ joint" mult factor should be significantly < 1.0
    - The cross-term should be more negative than other configs

    Compare the results above to determine the source of improvement!
    """)


if __name__ == "__main__":
    main()
