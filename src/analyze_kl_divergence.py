"""
Analyze whether weight-KV errors interact super-additively at the LOGIT level.

Hypothesis: Even if attention output MSE is additive, the effect on final logits
may be super-additive due to nonlinearities in the transformer stack.

This would justify joint optimization for logit/perplexity preservation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer


def kl_divergence(p_logits, q_logits, dim=-1):
    """Compute KL(p || q) where p is the reference."""
    p = F.softmax(p_logits, dim=dim)
    log_p = F.log_softmax(p_logits, dim=dim)
    log_q = F.log_softmax(q_logits, dim=dim)
    return (p * (log_p - log_q)).sum(dim=dim).mean()


def quantize_linear_weight(layer, bits=4):
    """Quantize a linear layer's weight in-place and return original."""
    original_weight = layer.weight.data.clone()

    w = layer.weight.data.float()
    w_max = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    levels = 2 ** (bits - 1) - 1
    scale = w_max / levels

    w_q = (w / scale).round().clamp(-levels, levels) * scale
    layer.weight.data = w_q.to(layer.weight.dtype)

    return original_weight


def restore_weight(layer, original_weight):
    """Restore original weight."""
    layer.weight.data = original_weight


class KVQuantHook:
    """Hook to quantize KV cache during attention computation."""

    def __init__(self, bits=4):
        self.bits = bits
        self.enabled = False

    def quantize_tensor(self, x):
        if not self.enabled:
            return x
        x_max = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        levels = 2 ** (self.bits - 1) - 1
        scale = x_max / levels
        x_q = (x / scale).round().clamp(-levels, levels) * scale
        return x_q


def analyze_logit_interaction(bits_w=3, bits_kv=4):
    """Analyze whether weight-KV errors are additive at the logit level."""

    print(f"Loading Qwen 1.7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Test input
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        # 1. Full precision logits (reference)
        logits_fp = model(**inputs).logits[:, -1, :]  # Last token logits

        # 2. Weight-only quantized logits
        # Store original weights
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'proj' in name:
                original_weights[name] = quantize_linear_weight(module, bits=bits_w)

        logits_wq = model(**inputs).logits[:, -1, :]

        # Restore weights
        for name, module in model.named_modules():
            if name in original_weights:
                restore_weight(module, original_weights[name])

        # 3. KV-only quantized logits
        # This is harder without modifying the model internals
        # We'll approximate by quantizing the K and V projection outputs
        # For simplicity, we'll use a different approach: quantize K, V projections
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ('k_proj' in name or 'v_proj' in name):
                original_weights[name] = quantize_linear_weight(module, bits=bits_kv)

        logits_kvq = model(**inputs).logits[:, -1, :]

        # Restore
        for name, module in model.named_modules():
            if name in original_weights:
                restore_weight(module, original_weights[name])

        # 4. Both quantized
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'proj' in name:
                if 'k_proj' in name or 'v_proj' in name:
                    original_weights[name] = quantize_linear_weight(module, bits=bits_kv)
                else:
                    original_weights[name] = quantize_linear_weight(module, bits=bits_w)

        logits_both = model(**inputs).logits[:, -1, :]

        # Restore
        for name, module in model.named_modules():
            if name in original_weights:
                restore_weight(module, original_weights[name])

    # Compute KL divergences
    kl_wq = kl_divergence(logits_fp, logits_wq).item()
    kl_kvq = kl_divergence(logits_fp, logits_kvq).item()
    kl_both = kl_divergence(logits_fp, logits_both).item()

    kl_sum = kl_wq + kl_kvq
    mult_factor = kl_both / kl_sum if kl_sum > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"LOGIT-LEVEL KL DIVERGENCE ANALYSIS")
    print(f"Config: {bits_w}-bit weights, {bits_kv}-bit KV projections")
    print(f"{'='*60}")
    print(f"KL(FP || W-quant only):  {kl_wq:.6f}")
    print(f"KL(FP || KV-quant only): {kl_kvq:.6f}")
    print(f"KL(FP || Both quant):    {kl_both:.6f}")
    print(f"Sum (if additive):       {kl_sum:.6f}")
    print(f"Multiplicative factor:   {mult_factor:.3f}")
    print(f"{'='*60}")

    if mult_factor > 1.2:
        print("→ SUPER-ADDITIVE at logit level! Joint optimization has potential!")
    elif mult_factor < 0.8:
        print("→ SUB-ADDITIVE at logit level")
    else:
        print("→ NEAR-ADDITIVE at logit level")

    # Also check MSE at logit level
    mse_wq = ((logits_wq - logits_fp) ** 2).mean().item()
    mse_kvq = ((logits_kvq - logits_fp) ** 2).mean().item()
    mse_both = ((logits_both - logits_fp) ** 2).mean().item()
    mse_mult = mse_both / (mse_wq + mse_kvq) if (mse_wq + mse_kvq) > 0 else float('inf')

    print(f"\nMSE at logit level:")
    print(f"MSE_W:  {mse_wq:.4f}")
    print(f"MSE_KV: {mse_kvq:.4f}")
    print(f"MSE_Both: {mse_both:.4f}")
    print(f"MSE Mult Factor: {mse_mult:.3f}")

    return mult_factor, mse_mult


def analyze_across_configs():
    """Analyze logit interaction across different bit configurations."""
    configs = [
        (3, 4),
        (3, 2),
        (4, 4),
        (4, 2),
    ]

    results = []
    for bits_w, bits_kv in configs:
        print(f"\n{'#'*60}")
        print(f"# Testing {bits_w}-bit W, {bits_kv}-bit KV")
        print(f"{'#'*60}")
        kl_mult, mse_mult = analyze_logit_interaction(bits_w, bits_kv)
        results.append((bits_w, bits_kv, kl_mult, mse_mult))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Bits_W':>8} {'Bits_KV':>8} {'KL_Mult':>12} {'MSE_Mult':>12}")
    print("-" * 44)
    for bits_w, bits_kv, kl_mult, mse_mult in results:
        print(f"{bits_w:>8} {bits_kv:>8} {kl_mult:>12.3f} {mse_mult:>12.3f}")

    avg_kl_mult = np.mean([r[2] for r in results])
    print(f"\nAverage KL mult factor: {avg_kl_mult:.3f}")

    if avg_kl_mult > 1.2:
        print("\n→ CONCLUSION: Joint optimization IS justified at the logit/KL level!")
    else:
        print("\n→ CONCLUSION: Errors remain near-additive even at logit level")


if __name__ == "__main__":
    analyze_across_configs()
