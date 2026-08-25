#!/usr/bin/env python3
"""
Diagnose why optimal allocation fails.

Hypothesis: Per-position min-max quantization has different characteristics
than zone-based quantization.

When we quantize position-by-position:
- Each position gets its own scale
- This might be better or worse than shared scale

Let's test different quantization granularities.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np


def quantize_tensor_minmax(x, num_bits):
    """Per-tensor min-max quantization."""
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min
    if range_val == 0:
        return x
    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def evaluate_kv_quantization(model, tokenizer, prompts, quant_fn):
    """Proper KV quantization evaluation."""
    device = next(model.parameters()).device
    mse_list = []
    cos_list = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] < 10:
            continue

        with torch.no_grad():
            torch.manual_seed(42)
            outputs = model(**inputs, use_cache=True)
            fp_kv = outputs.past_key_values

            prefix_kv_fp = DynamicCache()
            prefix_kv_q = DynamicCache()

            for layer_idx in range(len(fp_kv)):
                k = fp_kv.key_cache[layer_idx][:, :, :-1, :]
                v = fp_kv.value_cache[layer_idx][:, :, :-1, :]

                prefix_kv_fp.update(k.clone(), v.clone(), layer_idx)
                k_q, v_q = quant_fn(k.clone(), v.clone())
                prefix_kv_q.update(k_q, v_q, layer_idx)

            last_token = inputs.input_ids[:, -1:]

            torch.manual_seed(42)
            out_fp = model(input_ids=last_token, past_key_values=prefix_kv_fp, use_cache=False)
            logits_fp = out_fp.logits[:, -1, :]

            torch.manual_seed(42)
            out_q = model(input_ids=last_token, past_key_values=prefix_kv_q, use_cache=False)
            logits_q = out_q.logits[:, -1, :]

            mse = F.mse_loss(logits_fp, logits_q).item()
            cos = F.cosine_similarity(logits_fp, logits_q, dim=-1).item()
            mse_list.append(mse)
            cos_list.append(cos)

    return np.mean(mse_list), np.mean(cos_list)


def main():
    print("=" * 70)
    print("Diagnose Optimal Allocation Failure")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

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
        "Deep learning models require large amounts of training data and compute.",
        "The ancient Romans built roads that still exist today in many parts of Europe.",
        "Photosynthesis is the process by which green plants make food using sunlight.",
    ]

    # Test 1: Compare zone-based vs per-position for same bit allocation
    print("\n" + "=" * 70)
    print("TEST 1: Zone-Based vs Per-Position Quantization")
    print("(Both using first 10% at 6-bit, rest at 4-bit)")
    print("=" * 70)

    # Zone-based (shared scale within zone)
    def zone_quant(k, v):
        seq_len = k.shape[2]
        zone_end = max(1, int(seq_len * 0.1))

        k_out = k.clone()
        v_out = v.clone()

        k_out[:, :, :zone_end, :] = quantize_tensor_minmax(k[:, :, :zone_end, :], 6)
        v_out[:, :, :zone_end, :] = quantize_tensor_minmax(v[:, :, :zone_end, :], 6)
        k_out[:, :, zone_end:, :] = quantize_tensor_minmax(k[:, :, zone_end:, :], 4)
        v_out[:, :, zone_end:, :] = quantize_tensor_minmax(v[:, :, zone_end:, :], 4)

        return k_out, v_out

    # Per-position (separate scale per position)
    def perpos_quant_6_4(k, v):
        seq_len = k.shape[2]
        zone_end = max(1, int(seq_len * 0.1))

        k_out = k.clone()
        v_out = v.clone()

        for pos in range(seq_len):
            bits = 6 if pos < zone_end else 4
            k_out[:, :, pos, :] = quantize_tensor_minmax(k[:, :, pos, :], bits)
            v_out[:, :, pos, :] = quantize_tensor_minmax(v[:, :, pos, :], bits)

        return k_out, v_out

    mse_zone, cos_zone = evaluate_kv_quantization(model, tokenizer, test_prompts, zone_quant)
    mse_perpos, cos_perpos = evaluate_kv_quantization(model, tokenizer, test_prompts, perpos_quant_6_4)

    print(f"  Zone-based:    MSE = {mse_zone:.4f}, CosSim = {cos_zone:.4f}")
    print(f"  Per-position:  MSE = {mse_perpos:.4f}, CosSim = {cos_perpos:.4f}")

    # Test 2: Test per-position vs per-tensor for uniform quantization
    print("\n" + "=" * 70)
    print("TEST 2: Per-Tensor vs Per-Position for Uniform 4-bit")
    print("=" * 70)

    def uniform_pertensor_4bit(k, v):
        return quantize_tensor_minmax(k, 4), quantize_tensor_minmax(v, 4)

    def uniform_perpos_4bit(k, v):
        seq_len = k.shape[2]
        k_out = k.clone()
        v_out = v.clone()
        for pos in range(seq_len):
            k_out[:, :, pos, :] = quantize_tensor_minmax(k[:, :, pos, :], 4)
            v_out[:, :, pos, :] = quantize_tensor_minmax(v[:, :, pos, :], 4)
        return k_out, v_out

    mse_pt, cos_pt = evaluate_kv_quantization(model, tokenizer, test_prompts, uniform_pertensor_4bit)
    mse_pp, cos_pp = evaluate_kv_quantization(model, tokenizer, test_prompts, uniform_perpos_4bit)

    print(f"  Per-tensor:    MSE = {mse_pt:.4f}, CosSim = {cos_pt:.4f}")
    print(f"  Per-position:  MSE = {mse_pp:.4f}, CosSim = {cos_pp:.4f}")

    # Test 3: What if we use zone-based for optimal allocation?
    print("\n" + "=" * 70)
    print("TEST 3: Zone-Based Implementation of Optimal Allocation")
    print("=" * 70)

    # The optimal allocation at B_avg=4.2 was:
    # [8, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
    # First bin (0-5%): 8-bit
    # Bins 1-9 (5-45%): mostly 5-bit
    # Bins 10-19 (45-100%): mostly 3-bit

    # Let's try: first 5% at 8-bit, next 40% at 5-bit, rest at 3-bit
    def optimal_zone_based(k, v):
        seq_len = k.shape[2]
        zone1_end = max(1, int(seq_len * 0.05))  # 8-bit
        zone2_end = max(zone1_end + 1, int(seq_len * 0.50))  # 5-bit

        k_out = k.clone()
        v_out = v.clone()

        # Zone 1: 8-bit
        k_out[:, :, :zone1_end, :] = quantize_tensor_minmax(k[:, :, :zone1_end, :], 8)
        v_out[:, :, :zone1_end, :] = quantize_tensor_minmax(v[:, :, :zone1_end, :], 8)

        # Zone 2: 5-bit
        if zone2_end > zone1_end:
            k_out[:, :, zone1_end:zone2_end, :] = quantize_tensor_minmax(k[:, :, zone1_end:zone2_end, :], 5)
            v_out[:, :, zone1_end:zone2_end, :] = quantize_tensor_minmax(v[:, :, zone1_end:zone2_end, :], 5)

        # Zone 3: 3-bit
        if seq_len > zone2_end:
            k_out[:, :, zone2_end:, :] = quantize_tensor_minmax(k[:, :, zone2_end:, :], 3)
            v_out[:, :, zone2_end:, :] = quantize_tensor_minmax(v[:, :, zone2_end:, :], 3)

        return k_out, v_out

    mse_opt_zone, cos_opt_zone = evaluate_kv_quantization(model, tokenizer, test_prompts, optimal_zone_based)

    print(f"  Optimal (zone-based): MSE = {mse_opt_zone:.4f}, CosSim = {cos_opt_zone:.4f}")
    print(f"  (8-bit for 0-5%, 5-bit for 5-50%, 3-bit for 50-100%)")

    # Also compare with Sink 6-4-4
    print(f"\n  For reference:")
    print(f"  Sink 6-4-4 (zone): MSE = {mse_zone:.4f}, CosSim = {cos_zone:.4f}")

    # Test 4: Just first token high precision
    print("\n" + "=" * 70)
    print("TEST 4: Just First Token High Precision (Zone-Based)")
    print("=" * 70)

    for first_bits in [4, 6, 8]:
        def first_token_high(k, v, fb=first_bits):
            k_out = k.clone()
            v_out = v.clone()

            # First token only
            k_out[:, :, :1, :] = quantize_tensor_minmax(k[:, :, :1, :], fb)
            v_out[:, :, :1, :] = quantize_tensor_minmax(v[:, :, :1, :], fb)
            # Rest at 4-bit
            if k.shape[2] > 1:
                k_out[:, :, 1:, :] = quantize_tensor_minmax(k[:, :, 1:, :], 4)
                v_out[:, :, 1:, :] = quantize_tensor_minmax(v[:, :, 1:, :], 4)

            return k_out, v_out

        mse, cos = evaluate_kv_quantization(model, tokenizer, test_prompts,
                                            lambda k, v, fb=first_bits: first_token_high(k, v, fb))
        print(f"  First={first_bits}-bit, Rest=4-bit: MSE = {mse:.4f}, CosSim = {cos:.4f}")

    # Test 5: Sweep zone sizes for 6-4 configuration
    print("\n" + "=" * 70)
    print("TEST 5: Zone Size Sweep for 6-4 Configuration")
    print("=" * 70)

    for zone_pct in [2, 5, 10, 15, 20]:
        def zone_sweep(k, v, zp=zone_pct):
            seq_len = k.shape[2]
            zone_end = max(1, int(seq_len * zp / 100))

            k_out = k.clone()
            v_out = v.clone()

            k_out[:, :, :zone_end, :] = quantize_tensor_minmax(k[:, :, :zone_end, :], 6)
            v_out[:, :, :zone_end, :] = quantize_tensor_minmax(v[:, :, :zone_end, :], 6)
            if zone_end < seq_len:
                k_out[:, :, zone_end:, :] = quantize_tensor_minmax(k[:, :, zone_end:, :], 4)
                v_out[:, :, zone_end:, :] = quantize_tensor_minmax(v[:, :, zone_end:, :], 4)

            return k_out, v_out

        mse, cos = evaluate_kv_quantization(model, tokenizer, test_prompts,
                                            lambda k, v, zp=zone_pct: zone_sweep(k, v, zp))
        avg_bits = zone_pct/100 * 6 + (1 - zone_pct/100) * 4
        print(f"  Zone {zone_pct:2d}% (avg={avg_bits:.2f}): MSE = {mse:.4f}, CosSim = {cos:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
