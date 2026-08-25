"""
Compare ECAQ scale finding with SINQ's Sinkhorn normalization.

Question: Does SINQ already achieve the benefit that ECAQ's scale_w=0.5 provides?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log


def quantize_naive(W, bits=3):
    """Naive max-abs quantization."""
    W_max = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = W_max / levels
    W_q = (W / scale).round().clamp(-levels, levels) * scale
    return W_q


def quantize_with_scale_mult(W, bits=3, scale_mult=1.0):
    """Naive quantization with scale multiplier."""
    W_max = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = scale_mult * W_max / levels
    W_q = (W / scale).round().clamp(-levels, levels) * scale
    return W_q


def quantize_sinq_style(W, bits=3, group_size=128, order=16):
    """SINQ-style quantization with Sinkhorn normalization per tile."""
    H, D = W.shape
    n_groups = D // group_size

    W_q = torch.zeros_like(W)

    for g in range(n_groups):
        # Extract tile
        tile = W[:, g*group_size:(g+1)*group_size]

        # Apply Sinkhorn normalization
        tile_norm, mu1, mu2 = sinkhorn_log(tile, order=order)

        # Quantize normalized tile
        t_max = tile_norm.abs().max()
        levels = 2 ** bits - 1
        scale = t_max / levels if t_max > 1e-5 else 1.0

        tile_q = (tile_norm / scale).round().clamp(0, levels) * scale

        # De-normalize
        tile_dq = tile_q * mu1 * mu2

        W_q[:, g*group_size:(g+1)*group_size] = tile_dq

    return W_q


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

    print(f"\nWeight shape: {W_Q.shape}")

    # Generate input
    text = "The quick brown fox jumps over the lazy dog. " * 5
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
    with torch.no_grad():
        h = model.model.embed_tokens(inputs['input_ids']).float()

    Q_fp = h @ W_Q.T

    print(f"\n{'='*70}")
    print("COMPARING QUANTIZATION METHODS")
    print(f"{'='*70}")

    methods = [
        ("Naive (scale=1.0)", lambda W: quantize_naive(W, 3)),
        ("Naive (scale=0.5)", lambda W: quantize_with_scale_mult(W, 3, 0.5)),
        ("Naive (scale=0.7)", lambda W: quantize_with_scale_mult(W, 3, 0.7)),
        ("SINQ-style (Sinkhorn)", lambda W: quantize_sinq_style(W, 3, 128, 16)),
    ]

    print(f"\n{'Method':<30} {'W_MSE':>15} {'Proj_MSE':>15}")
    print("-" * 65)

    for name, quant_fn in methods:
        W_Q_q = quant_fn(W_Q)
        Q_q = h @ W_Q_q.T

        w_mse = ((W_Q_q - W_Q) ** 2).mean().item()
        proj_mse = ((Q_q - Q_fp) ** 2).mean().item()

        print(f"{name:<30} {w_mse:>15.8f} {proj_mse:>15.8f}")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
    If SINQ-style quantization has similar error to Naive (scale=0.5):
    → SINQ already achieves the optimal scale, ECAQ is redundant

    If SINQ-style has higher error than Naive (scale=0.5):
    → There may be room for scale adjustment on top of SINQ

    If Naive (scale=0.5) is still best:
    → The max-abs scale is fundamentally suboptimal for this model
    → Consider investigating why tighter scaling helps
    """)


if __name__ == "__main__":
    main()
