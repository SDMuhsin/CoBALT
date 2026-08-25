"""
Test ECAQ on perplexity evaluation.

Compare:
1. Baseline: SINQ weights (3-bit) + standard KV quantization (4-bit)
2. ECAQ: SINQ weights with ECAQ scale adjustment + ECAQ-calibrated KV

This tests whether the MSE improvements translate to perplexity improvements.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.ecaq import ECAQConfig, ECAQCalibrator, calibrate_model_ecaq


def quantize_uniform(x: torch.Tensor, bits: int, scale_mult: float = 1.0) -> torch.Tensor:
    """Uniform quantization."""
    x_max = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = scale_mult * x_max / levels
    x_q = (x / scale).round().clamp(-levels, levels) * scale
    return x_q


def quantize_weight(w: torch.Tensor, bits: int, scale_mult: float = 1.0) -> torch.Tensor:
    """Quantize weight matrix."""
    w_max = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = scale_mult * w_max / levels
    w_q = (w / scale).round().clamp(-levels, levels) * scale
    return w_q


class QuantizedAttention:
    """Simulates quantized attention for perplexity evaluation."""

    def __init__(self, attn_module, bits_w=3, bits_kv=4, scale_w=1.0, scale_kv=1.0):
        self.attn = attn_module
        self.bits_w = bits_w
        self.bits_kv = bits_kv
        self.scale_w = scale_w
        self.scale_kv = scale_kv

        # Pre-quantize weights
        self.W_Q = quantize_weight(attn_module.q_proj.weight.data.float(), bits_w, scale_w)
        self.W_K = quantize_weight(attn_module.k_proj.weight.data.float(), bits_w, scale_w)
        self.W_V = quantize_weight(attn_module.v_proj.weight.data.float(), bits_w, scale_w)
        self.W_O = quantize_weight(attn_module.o_proj.weight.data.float(), bits_w, scale_w)

        self.d_head = attn_module.head_dim
        self.n_heads = self.W_Q.shape[0] // self.d_head
        self.n_kv_heads = self.W_K.shape[0] // self.d_head

    def forward(self, h):
        """Forward pass with quantized weights and KV cache."""
        batch_size, seq_len, _ = h.shape

        # Project
        Q = (h @ self.W_Q.T).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = (h @ self.W_K.T).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = (h @ self.W_V.T).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Quantize KV cache
        K = quantize_uniform(K, self.bits_kv, self.scale_kv)
        V = quantize_uniform(V, self.bits_kv, self.scale_kv)

        # GQA expansion
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)

        # Attention
        scale = 1.0 / np.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = out @ self.W_O.T

        return out


def compute_perplexity(
    model,
    tokenizer,
    test_text: str,
    bits_w: int = 3,
    bits_kv: int = 4,
    layer_scales: dict = None,  # {layer_idx: (scale_w, scale_kv)}
    max_length: int = 512,
):
    """
    Compute perplexity with simulated quantization.

    This is a simplified simulation that quantizes attention weights and KV cache
    but uses FP for other components (MLP, LayerNorm, embeddings).
    """
    if layer_scales is None:
        layer_scales = {}

    inputs = tokenizer(test_text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        # Get embeddings
        h = model.model.embed_tokens(input_ids).float()

        # Process through layers
        for layer_idx, layer in enumerate(model.model.layers):
            # Get scales for this layer
            scale_w, scale_kv = layer_scales.get(layer_idx, (1.0, 1.0))

            # Create quantized attention
            q_attn = QuantizedAttention(
                layer.self_attn,
                bits_w=bits_w,
                bits_kv=bits_kv,
                scale_w=scale_w,
                scale_kv=scale_kv
            )

            # Apply RMSNorm before attention (if exists)
            if hasattr(layer, 'input_layernorm'):
                h_normed = layer.input_layernorm(h)
            else:
                h_normed = h

            # Attention with quantization
            attn_out = q_attn.forward(h_normed)

            # Residual
            h = h + attn_out

            # MLP (use original FP for simplicity)
            if hasattr(layer, 'post_attention_layernorm'):
                h_normed = layer.post_attention_layernorm(h)
            else:
                h_normed = h

            # Simple MLP forward (assuming standard architecture)
            if hasattr(layer, 'mlp'):
                mlp_out = layer.mlp(h_normed)
                h = h + mlp_out

        # Final norm
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)

        # LM head
        logits = model.lm_head(h)

        # Compute perplexity
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        ppl = torch.exp(loss).item()

    return ppl


def main():
    print("Loading Qwen 1.7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Calibration texts
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago—never mind how long precisely.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer.",
        "When Gregor Samsa woke up one morning from unsettling dreams, he found himself changed in his bed into a monstrous vermin.",
    ]

    # Test text (longer for more stable perplexity)
    test_text = """
    The study of artificial intelligence has made remarkable progress in recent years.
    Large language models have demonstrated unprecedented capabilities in natural language
    understanding and generation. These models learn patterns from vast amounts of text data
    and can perform a wide variety of tasks including translation, summarization, and
    question answering. The efficiency of these models during inference is a key concern
    for practical deployment, leading to research in model compression techniques such as
    quantization, pruning, and knowledge distillation.
    """ * 3

    print(f"\n{'='*70}")
    print("ECAQ PERPLEXITY EVALUATION")
    print(f"{'='*70}")

    # 1. Baseline (no quantization)
    print("\n1. Computing FP16 baseline perplexity...")
    with torch.no_grad():
        inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs, labels=inputs['input_ids'])
        ppl_fp = torch.exp(outputs.loss).item()
    print(f"   FP16 PPL: {ppl_fp:.2f}")

    # 2. Standard quantization (scale_w=1.0, scale_kv=1.0)
    print("\n2. Computing standard quantized perplexity (3W+4KV)...")
    ppl_standard = compute_perplexity(
        model, tokenizer, test_text,
        bits_w=3, bits_kv=4,
        layer_scales=None,  # All scales = 1.0
        max_length=512
    )
    print(f"   Standard Quant PPL: {ppl_standard:.2f}")

    # 3. ECAQ calibration
    print("\n3. Running ECAQ calibration...")
    config = ECAQConfig(bits_w=3, bits_kv=4, search_resolution=5)
    calibrator = calibrate_model_ecaq(
        model, tokenizer, calibration_texts, config, device='cpu'
    )

    # 4. ECAQ quantization
    print("\n4. Computing ECAQ-optimized perplexity...")
    ppl_ecaq = compute_perplexity(
        model, tokenizer, test_text,
        bits_w=3, bits_kv=4,
        layer_scales=calibrator.layer_scales,
        max_length=512
    )
    print(f"   ECAQ Quant PPL: {ppl_ecaq:.2f}")

    # 5. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"FP16 baseline:       {ppl_fp:.2f}")
    print(f"Standard 3W+4KV:     {ppl_standard:.2f} (degradation: +{ppl_standard - ppl_fp:.2f})")
    print(f"ECAQ 3W+4KV:         {ppl_ecaq:.2f} (degradation: +{ppl_ecaq - ppl_fp:.2f})")

    improvement = (ppl_standard - ppl_ecaq) / (ppl_standard - ppl_fp) * 100 if (ppl_standard - ppl_fp) > 0 else 0
    print(f"\nECAQ reduces quantization degradation by: {improvement:.1f}%")

    if ppl_ecaq < ppl_standard:
        print("\n✓ ECAQ ACHIEVES LOWER PERPLEXITY THAN STANDARD QUANTIZATION!")
    else:
        print("\n✗ ECAQ did not improve perplexity in this test.")


if __name__ == "__main__":
    main()
