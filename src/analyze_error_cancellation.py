"""
Investigate WHY weight and KV errors cancel at the logit level.

Hypotheses:
1. Residual connections average out errors (mean-reverting)
2. LayerNorm compresses error magnitudes
3. Softmax attention clips outlier errors
4. The errors have opposite signs in some subspaces
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_weight(w, bits=4):
    """Quantize weight tensor."""
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = w_max / levels
    w_q = (w / scale).round().clamp(-levels, levels) * scale
    return w_q


def analyze_error_direction(bits_w=3, bits_kv=4):
    """Analyze the direction of weight and KV errors through the transformer."""

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")

    # Get embeddings
    with torch.no_grad():
        embed = model.model.embed_tokens(inputs['input_ids']).float()

    print(f"\nAnalyzing error directions through transformer layers...")
    print(f"Config: {bits_w}-bit weights, {bits_kv}-bit KV")
    print("="*70)

    # We'll track hidden states through a few layers
    h_fp = embed.clone()
    h_wq = embed.clone()
    h_kvq = embed.clone()
    h_both = embed.clone()

    correlations = []

    for layer_idx in range(min(8, len(model.model.layers))):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # Get weights
        W_Q = attn.q_proj.weight.data.float()
        W_K = attn.k_proj.weight.data.float()
        W_V = attn.v_proj.weight.data.float()
        W_O = attn.o_proj.weight.data.float()

        # Quantized weights
        W_Q_q = quantize_weight(W_Q, bits_w)
        W_K_q = quantize_weight(W_K, bits_w)
        W_V_q = quantize_weight(W_V, bits_w)
        W_O_q = quantize_weight(W_O, bits_w)

        d_head = attn.head_dim
        n_heads = W_Q.shape[0] // d_head
        n_kv_heads = W_K.shape[0] // d_head
        batch_size, seq_len = h_fp.shape[0], h_fp.shape[1]

        def forward_attn(h, W_Q_use, W_K_use, W_V_use, W_O_use, quant_kv=False, kv_bits=4):
            Q = (h @ W_Q_use.T).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            K = (h @ W_K_use.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)
            V = (h @ W_V_use.T).view(batch_size, seq_len, n_kv_heads, d_head).transpose(1, 2)

            if quant_kv:
                K_max = K.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
                levels = 2 ** (kv_bits - 1) - 1
                K_scale = K_max / levels
                K = (K / K_scale).round().clamp(-levels, levels) * K_scale

                V_max = V.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
                V_scale = V_max / levels
                V = (V / V_scale).round().clamp(-levels, levels) * V_scale

            # GQA expansion
            if n_kv_heads != n_heads:
                K = K.repeat_interleave(n_heads // n_kv_heads, dim=1)
                V = V.repeat_interleave(n_heads // n_kv_heads, dim=1)

            # Attention
            scale = 1.0 / np.sqrt(d_head)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, V)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            out = out @ W_O_use.T

            return h + out  # Residual

        with torch.no_grad():
            # FP path
            h_fp_new = forward_attn(h_fp, W_Q, W_K, W_V, W_O, quant_kv=False)

            # Weight-only path
            h_wq_new = forward_attn(h_wq, W_Q_q, W_K_q, W_V_q, W_O_q, quant_kv=False)

            # KV-only path
            h_kvq_new = forward_attn(h_kvq, W_Q, W_K, W_V, W_O, quant_kv=True, kv_bits=bits_kv)

            # Both path
            h_both_new = forward_attn(h_both, W_Q_q, W_K_q, W_V_q, W_O_q, quant_kv=True, kv_bits=bits_kv)

        # Compute errors
        err_wq = (h_wq_new - h_fp_new).flatten()
        err_kvq = (h_kvq_new - h_fp_new).flatten()
        err_both = (h_both_new - h_fp_new).flatten()

        # Correlation between weight error and KV error
        corr = torch.corrcoef(torch.stack([err_wq, err_kvq]))[0, 1].item()
        correlations.append(corr)

        # Error magnitudes
        err_wq_norm = err_wq.norm().item()
        err_kvq_norm = err_kvq.norm().item()
        err_both_norm = err_both.norm().item()
        err_sum_norm = (err_wq_norm**2 + err_kvq_norm**2)**0.5  # If independent

        # Cross-term: err_both² = err_wq² + err_kvq² + 2*err_wq·err_kvq
        # If sub-additive, the cross-term err_wq·err_kvq must be negative
        cross_term = (err_both_norm**2 - err_wq_norm**2 - err_kvq_norm**2) / 2

        print(f"Layer {layer_idx:2d}: "
              f"||err_W||={err_wq_norm:8.4f}, ||err_KV||={err_kvq_norm:8.4f}, "
              f"||err_both||={err_both_norm:8.4f}, "
              f"corr={corr:+.4f}, cross={cross_term:+.4f}")

        # Update hidden states
        h_fp = h_fp_new
        h_wq = h_wq_new
        h_kvq = h_kvq_new
        h_both = h_both_new

    print(f"\n{'='*70}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Mean |correlation|: {np.mean(np.abs(correlations)):.4f}")

    if np.mean(correlations) < -0.1:
        print("\n→ Errors are NEGATIVELY correlated! This explains sub-additivity.")
        print("   Weight errors and KV errors point in opposite directions.")
    elif np.mean(correlations) > 0.1:
        print("\n→ Errors are POSITIVELY correlated. Sub-additivity must come from elsewhere.")
    else:
        print("\n→ Errors are UNCORRELATED. Sub-additivity likely from nonlinear compression.")

    return correlations


def analyze_layernorm_effect():
    """Analyze how LayerNorm affects error propagation."""
    print("\n" + "="*70)
    print("LAYERNORM EFFECT ANALYSIS")
    print("="*70)

    # Create random error vectors
    torch.manual_seed(42)
    hidden_dim = 2048
    seq_len = 10

    # Simulate hidden state with errors
    h_fp = torch.randn(1, seq_len, hidden_dim)
    err_w = torch.randn(1, seq_len, hidden_dim) * 0.1  # Weight error
    err_kv = torch.randn(1, seq_len, hidden_dim) * 0.1  # KV error

    h_wq = h_fp + err_w
    h_kvq = h_fp + err_kv
    h_both = h_fp + err_w + err_kv

    # Apply LayerNorm
    ln = torch.nn.LayerNorm(hidden_dim)

    h_fp_ln = ln(h_fp)
    h_wq_ln = ln(h_wq)
    h_kvq_ln = ln(h_kvq)
    h_both_ln = ln(h_both)

    # Compute errors after LayerNorm
    err_wq_ln = (h_wq_ln - h_fp_ln).flatten()
    err_kvq_ln = (h_kvq_ln - h_fp_ln).flatten()
    err_both_ln = (h_both_ln - h_fp_ln).flatten()

    print(f"\nBefore LayerNorm:")
    print(f"  ||err_W|| = {err_w.norm():.4f}")
    print(f"  ||err_KV|| = {err_kv.norm():.4f}")
    print(f"  ||err_W + err_KV|| = {(err_w + err_kv).norm():.4f}")
    print(f"  Sum of norms = {err_w.norm() + err_kv.norm():.4f}")

    print(f"\nAfter LayerNorm:")
    print(f"  ||err_W_ln|| = {err_wq_ln.norm():.4f}")
    print(f"  ||err_KV_ln|| = {err_kvq_ln.norm():.4f}")
    print(f"  ||err_both_ln|| = {err_both_ln.norm():.4f}")
    print(f"  Sum of norms = {(err_wq_ln.norm() + err_kvq_ln.norm()):.4f}")

    ln_mult = err_both_ln.norm() / (err_wq_ln.norm() + err_kvq_ln.norm())
    print(f"\nLayerNorm mult factor: {ln_mult:.3f}")

    if ln_mult < 0.9:
        print("→ LayerNorm COMPRESSES combined errors (sub-additive)")
    else:
        print("→ LayerNorm does not explain sub-additivity")


if __name__ == "__main__":
    correlations = analyze_error_direction(bits_w=3, bits_kv=4)
    analyze_layernorm_effect()

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
    The sub-additive behavior (errors cancel) at the logit level has implications:

    1. INDEPENDENT OPTIMIZATION MAY BE SUBOPTIMAL:
       - If errors cancel, adding more errors doesn't hurt as much
       - Joint optimization could EXPLOIT this by deliberate error alignment

    2. POTENTIAL RESEARCH DIRECTION:
       - Can we design quantization schemes where weight and KV errors
         are MAXIMALLY canceling?
       - This would allow more aggressive quantization with less quality loss

    3. THE KEY QUESTION:
       - Is the cancellation due to random correlation (uncontrollable)?
       - Or due to structural properties we can optimize?
    """)
