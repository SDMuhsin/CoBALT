#!/usr/bin/env python3
"""
Investigation: Why does Sink 6-4-4 beat the "optimal" allocation?

Hypotheses to test:
1. Zone size matters: Maybe 10% sink zone is special
2. Bit choice matters: Maybe 6-bit vs 8-bit for sink has different behavior
3. The last bin has 0 importance: discretization artifacts
4. Per-position quantization has different error characteristics than per-tensor
5. The objective function is wrong: errors don't scale as 2^{-2b}

Let's systematically test these.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from tqdm import tqdm


def quantize_tensor(x, num_bits, per_channel=False):
    if per_channel:
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
    else:
        x_min = x.min()
        x_max = x.max()

    range_val = x_max - x_min
    if isinstance(range_val, torch.Tensor):
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
    elif range_val == 0:
        return x

    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def quantize_kv_uniform(kv_cache, num_bits):
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def quantize_kv_zone(kv_cache, zone_fraction, zone_bits, rest_bits):
    """Zone-based quantization: first zone_fraction at zone_bits, rest at rest_bits."""
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()
        seq_len = k.shape[2]
        zone_end = max(1, int(seq_len * zone_fraction))

        # Quantize first zone
        k[:, :, :zone_end, :] = quantize_tensor(k[:, :, :zone_end, :], zone_bits)
        v[:, :, :zone_end, :] = quantize_tensor(v[:, :, :zone_end, :], zone_bits)
        # Quantize rest
        if zone_end < seq_len:
            k[:, :, zone_end:, :] = quantize_tensor(k[:, :, zone_end:, :], rest_bits)
            v[:, :, zone_end:, :] = quantize_tensor(v[:, :, zone_end:, :], rest_bits)

        quantized.update(k, v, layer_idx)
    return quantized


def quantize_kv_with_allocation(kv_cache, allocation):
    """Per-bin allocation."""
    quantized = DynamicCache()
    num_bins = len(allocation)

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()
        seq_len = k.shape[2]

        for pos in range(seq_len):
            rel_pos = pos / seq_len
            bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
            num_bits = allocation[bin_idx]

            k[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], num_bits)
            v[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], num_bits)

        quantized.update(k, v, layer_idx)
    return quantized


def evaluate_config(model, tokenizer, prompts, quant_fn):
    device = next(model.parameters()).device
    mse_list, cos_sim_list = [], []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] < 10:
            continue

        with torch.no_grad():
            fp_outputs = model(**inputs, use_cache=True)
            fp_logits = fp_outputs.logits[:, -1, :]
            fp_kv = fp_outputs.past_key_values

            q_kv = quant_fn(fp_kv)

            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
            q_logits = q_outputs.logits[:, -1, :]

            mse = F.mse_loss(fp_logits, q_logits).item()
            cos_sim = F.cosine_similarity(fp_logits, q_logits, dim=-1).mean().item()

            mse_list.append(mse)
            cos_sim_list.append(cos_sim)

    return np.mean(mse_list), np.mean(cos_sim_list)


def main():
    print("=" * 70)
    print("Investigation: Why Sink 6-4-4 Beats Optimal")
    print("=" * 70)

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    test_prompts = [
        "Deep learning models require large amounts of training data.",
        "The ancient Romans built an extensive network of roads.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The stock market experienced significant volatility during the pandemic.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning has revolutionized data processing.",
    ]

    print("\n" + "=" * 70)
    print("TEST 1: Zone Fraction Sweep")
    print("(Keeping zone=6-bit, rest=4-bit, varying zone fraction)")
    print("=" * 70)

    for zone_frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        avg_bits = zone_frac * 6 + (1 - zone_frac) * 4
        mse, cos = evaluate_config(model, tokenizer, test_prompts,
                                   lambda kv, zf=zone_frac: quantize_kv_zone(kv, zf, 6, 4))
        print(f"  Zone {zone_frac*100:4.0f}%: avg_bits={avg_bits:.2f}, MSE={mse:.2f}, CosSim={cos:.4f}")

    print("\n" + "=" * 70)
    print("TEST 2: Zone Bit-Width Sweep")
    print("(Keeping zone=10%, rest=4-bit, varying zone bits)")
    print("=" * 70)

    for zone_bits in [4, 5, 6, 7, 8]:
        avg_bits = 0.1 * zone_bits + 0.9 * 4
        mse, cos = evaluate_config(model, tokenizer, test_prompts,
                                   lambda kv, zb=zone_bits: quantize_kv_zone(kv, 0.1, zb, 4))
        print(f"  Zone {zone_bits}-bit: avg_bits={avg_bits:.2f}, MSE={mse:.2f}, CosSim={cos:.4f}")

    print("\n" + "=" * 70)
    print("TEST 3: Optimal-like Allocation vs Sink Allocation")
    print("(Both at ~4.2 avg bits)")
    print("=" * 70)

    # Sink 6-4-4 as 20-bin allocation (first 2 bins = 6, rest = 4)
    sink_alloc = [6, 6] + [4] * 18  # 10% at 6, 90% at 4

    # "Optimal" allocation from attention importance
    # First bin has ~100x importance, so should get +3-4 bits
    # But let's also try a variant that mimics sink better
    optimal_like = [8, 5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 3, 5, 3, 3, 3, 3, 3, 3]

    # What if we do 8-bit for first 1 bin, then 4 for rest?
    first_high = [8] + [4] * 19

    configs = [
        ("Sink 6-4-4 (zone)", lambda kv: quantize_kv_zone(kv, 0.1, 6, 4), 4.2),
        ("Sink [6,6,4...] (allocation)", lambda kv: quantize_kv_with_allocation(kv, sink_alloc), np.mean(sink_alloc)),
        ("Optimal [8,5,5...]", lambda kv: quantize_kv_with_allocation(kv, optimal_like), np.mean(optimal_like)),
        ("First-8 [8,4,4...]", lambda kv: quantize_kv_with_allocation(kv, first_high), np.mean(first_high)),
    ]

    for name, quant_fn, avg_bits in configs:
        mse, cos = evaluate_config(model, tokenizer, test_prompts, quant_fn)
        print(f"  {name}: avg={avg_bits:.2f}, MSE={mse:.2f}, CosSim={cos:.4f}")

    print("\n" + "=" * 70)
    print("TEST 4: Quantization Error Scaling")
    print("(Does error actually scale as 2^{-2b}?)")
    print("=" * 70)

    # For each bit-width, measure actual MSE with uniform quantization
    print("\nUniform quantization MSE by bit-width:")
    for bits in [2, 3, 4, 5, 6, 8]:
        mse, _ = evaluate_config(model, tokenizer, test_prompts,
                                 lambda kv, b=bits: quantize_kv_uniform(kv, b))
        theoretical_error = 2**(-2*bits)
        print(f"  {bits}-bit: MSE={mse:.2f}, Theoretical ∝ {theoretical_error:.6f}")

    print("\n" + "=" * 70)
    print("TEST 5: Just First Token High Precision")
    print("=" * 70)

    # What if only the very first token gets high precision?
    def quant_first_only(kv_cache, first_bits, rest_bits):
        quantized = DynamicCache()
        for layer_idx in range(len(kv_cache)):
            k = kv_cache.key_cache[layer_idx].clone()
            v = kv_cache.value_cache[layer_idx].clone()

            # First token only
            k[:, :, 0:1, :] = quantize_tensor(k[:, :, 0:1, :], first_bits)
            v[:, :, 0:1, :] = quantize_tensor(v[:, :, 0:1, :], first_bits)
            # Rest
            if k.shape[2] > 1:
                k[:, :, 1:, :] = quantize_tensor(k[:, :, 1:, :], rest_bits)
                v[:, :, 1:, :] = quantize_tensor(v[:, :, 1:, :], rest_bits)

            quantized.update(k, v, layer_idx)
        return quantized

    for first_bits in [4, 6, 8, 16]:
        # Approximate avg bits (first token is tiny fraction)
        mse, cos = evaluate_config(model, tokenizer, test_prompts,
                                   lambda kv, fb=first_bits: quant_first_only(kv, fb, 4))
        print(f"  First={first_bits}-bit, Rest=4-bit: MSE={mse:.2f}, CosSim={cos:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()
