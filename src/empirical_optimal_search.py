#!/usr/bin/env python3
"""
Empirical Optimal Bit Allocation Search

Given that the theoretical model (σ² ∝ 2^{-2b}) is WRONG for this model,
we need to find the optimal allocation empirically.

Approach:
1. Define a search space of zone-based configurations
2. Evaluate each configuration
3. Find Pareto-optimal configurations (quality vs memory)

This is still "optimal" in the sense of finding the best configuration
among the search space, but it's not a closed-form theoretical optimum.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from itertools import product
from tqdm import tqdm


def quantize_tensor(x, num_bits):
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min
    if range_val == 0:
        return x
    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def evaluate_zone_config(model, tokenizer, prompts, zone_fractions, zone_bits):
    """
    Evaluate a multi-zone configuration.

    Args:
        zone_fractions: list of zone end fractions (e.g., [0.1, 0.5] for 3 zones)
        zone_bits: list of bits for each zone (e.g., [6, 5, 4] for 3 zones)

    Returns: MSE, CosSim, avg_bits
    """
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
                k = fp_kv.key_cache[layer_idx][:, :, :-1, :].clone()
                v = fp_kv.value_cache[layer_idx][:, :, :-1, :].clone()
                seq_len = k.shape[2]

                prefix_kv_fp.update(k.clone(), v.clone(), layer_idx)

                # Apply zone-based quantization
                k_q = k.clone()
                v_q = v.clone()

                prev_end = 0
                for i, (frac, bits) in enumerate(zip(zone_fractions + [1.0], zone_bits)):
                    zone_end = int(seq_len * frac)
                    if zone_end > prev_end:
                        k_q[:, :, prev_end:zone_end, :] = quantize_tensor(k[:, :, prev_end:zone_end, :], bits)
                        v_q[:, :, prev_end:zone_end, :] = quantize_tensor(v[:, :, prev_end:zone_end, :], bits)
                    prev_end = zone_end

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

    # Compute average bits
    zone_fractions_full = zone_fractions + [1.0]
    avg_bits = 0.0
    prev_frac = 0.0
    for frac, bits in zip(zone_fractions_full, zone_bits):
        zone_width = frac - prev_frac
        avg_bits += zone_width * bits
        prev_frac = frac

    return np.mean(mse_list), np.mean(cos_list), avg_bits


def main():
    print("=" * 70)
    print("Empirical Optimal Bit Allocation Search")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)
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
        "Deep learning models require large amounts of training data.",
        "The ancient Romans built roads that still exist today.",
        "Photosynthesis is the process by which plants make food.",
        "The stock market experienced volatility during the pandemic.",
        "Quantum computers promise to revolutionize cryptography.",
        "Natural language processing has advanced with transformers.",
    ]

    # Phase 1: Two-zone search (similar to Sink X-Y-Y)
    print("\n" + "=" * 70)
    print("PHASE 1: Two-Zone Search")
    print("=" * 70)

    results = []

    # Search space
    sink_fractions = [0.05, 0.10, 0.15, 0.20]
    sink_bits_options = [4, 5, 6, 8]
    rest_bits_options = [3, 4, 5]

    print("\nSearching...")
    for sink_frac in tqdm(sink_fractions, desc="Sink fraction"):
        for sink_bits in sink_bits_options:
            for rest_bits in rest_bits_options:
                if sink_bits <= rest_bits:
                    continue  # Skip if sink has same or fewer bits

                mse, cos, avg_bits = evaluate_zone_config(
                    model, tokenizer, test_prompts,
                    zone_fractions=[sink_frac],
                    zone_bits=[sink_bits, rest_bits]
                )

                results.append({
                    "config": f"Sink {sink_frac*100:.0f}%-{sink_bits}b-{rest_bits}b",
                    "sink_frac": sink_frac,
                    "sink_bits": sink_bits,
                    "rest_bits": rest_bits,
                    "avg_bits": avg_bits,
                    "mse": mse,
                    "cos_sim": cos,
                })

    # Sort by MSE
    results.sort(key=lambda x: x["mse"])

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS (by MSE)")
    print("=" * 70)

    print(f"\n{'Config':<25} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 65)
    for r in results[:10]:
        print(f"{r['config']:<25} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Phase 2: Three-zone search
    print("\n" + "=" * 70)
    print("PHASE 2: Three-Zone Search")
    print("=" * 70)

    three_zone_results = []

    # Best two-zone was around 15% sink at 5/6-bit with rest at 4-bit
    # Let's try adding a middle zone
    zone1_fracs = [0.10, 0.15, 0.20]
    zone2_fracs = [0.40, 0.50, 0.60]
    zone1_bits_options = [5, 6]
    zone2_bits_options = [4, 5]
    zone3_bits_options = [3, 4]

    print("\nSearching...")
    for z1_frac in tqdm(zone1_fracs, desc="Zone 1 frac"):
        for z2_frac in zone2_fracs:
            if z2_frac <= z1_frac:
                continue
            for z1_bits in zone1_bits_options:
                for z2_bits in zone2_bits_options:
                    for z3_bits in zone3_bits_options:
                        if z1_bits < z2_bits or z2_bits < z3_bits:
                            continue  # Monotonic decrease in bits

                        mse, cos, avg_bits = evaluate_zone_config(
                            model, tokenizer, test_prompts,
                            zone_fractions=[z1_frac, z2_frac],
                            zone_bits=[z1_bits, z2_bits, z3_bits]
                        )

                        three_zone_results.append({
                            "config": f"Z:{z1_frac*100:.0f}%-{z2_frac*100:.0f}% B:{z1_bits}-{z2_bits}-{z3_bits}",
                            "avg_bits": avg_bits,
                            "mse": mse,
                            "cos_sim": cos,
                        })

    three_zone_results.sort(key=lambda x: x["mse"])

    print("\n" + "=" * 70)
    print("TOP 10 THREE-ZONE CONFIGURATIONS")
    print("=" * 70)

    print(f"\n{'Config':<40} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 80)
    for r in three_zone_results[:10]:
        print(f"{r['config']:<40} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Phase 3: Find Pareto-optimal configurations
    print("\n" + "=" * 70)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("=" * 70)

    all_results = results + three_zone_results

    # A configuration is Pareto-optimal if no other config has:
    # - Lower MSE AND same or fewer bits
    # - Same or lower MSE AND fewer bits
    pareto_optimal = []
    for r in all_results:
        is_dominated = False
        for other in all_results:
            if other is r:
                continue
            # Check if 'other' dominates 'r'
            if other["mse"] <= r["mse"] and other["avg_bits"] < r["avg_bits"]:
                is_dominated = True
                break
            if other["mse"] < r["mse"] and other["avg_bits"] <= r["avg_bits"]:
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimal.append(r)

    pareto_optimal.sort(key=lambda x: x["avg_bits"])

    print(f"\n{'Config':<40} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 80)
    for r in pareto_optimal:
        print(f"{r['config']:<40} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Compare with uniform baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH UNIFORM BASELINES")
    print("=" * 70)

    uniform_results = []
    for bits in [3, 4, 5, 6]:
        mse, cos, avg_bits = evaluate_zone_config(
            model, tokenizer, test_prompts,
            zone_fractions=[],
            zone_bits=[bits]
        )
        uniform_results.append({
            "config": f"Uniform {bits}-bit",
            "avg_bits": avg_bits,
            "mse": mse,
            "cos_sim": cos,
        })
        print(f"  Uniform {bits}-bit: MSE = {mse:.4f}, CosSim = {cos:.4f}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find best config at ~4.2 bits
    target_bits = 4.2
    best_at_target = min(
        [r for r in all_results if abs(r["avg_bits"] - target_bits) < 0.3],
        key=lambda x: x["mse"]
    )

    print(f"\nBest configuration at ~4.2 bits:")
    print(f"  {best_at_target['config']}")
    print(f"  Avg bits: {best_at_target['avg_bits']:.2f}")
    print(f"  MSE: {best_at_target['mse']:.4f}")
    print(f"  CosSim: {best_at_target['cos_sim']:.4f}")

    # Compare with original Sink 6-4-4
    sink_644 = next(r for r in results if "10%-6b-4b" in r["config"])
    print(f"\nOriginal Sink 6-4-4:")
    print(f"  MSE: {sink_644['mse']:.4f}")
    print(f"  CosSim: {sink_644['cos_sim']:.4f}")

    if best_at_target["mse"] < sink_644["mse"]:
        improvement = (sink_644["mse"] - best_at_target["mse"]) / sink_644["mse"] * 100
        print(f"\n  -> Best config is {improvement:.1f}% BETTER than Sink 6-4-4!")
    else:
        print(f"\n  -> Sink 6-4-4 is the best at this bit budget!")

    print("\n" + "=" * 70)
    print("Search Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
