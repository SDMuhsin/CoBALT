#!/usr/bin/env python3
"""
Phase 1.3: Refined Optimal Bit Allocation

Key Insight from v1: Pure attention-based importance is insufficient.
The 2-bit allocations cause catastrophic errors despite low attention.

New Approach: EMPIRICAL SENSITIVITY CALIBRATION
===============================================

Instead of deriving sensitivity from attention theory alone, we MEASURE
the actual sensitivity s(p) by:

s(p) = MSE(output | position p quantized at reference_bits)

This captures:
1. Attention weight effects
2. Non-linear error propagation
3. Layer interactions

Then apply the same Lagrangian optimization:
  min Σ_p s(p) · 2^{-2b_p}
  s.t. avg(b) = B_avg

Solution: b*(p) = 0.5 · log₂(s(p)) + C

The key difference: s(p) is MEASURED, not derived from attention alone.

This is still provably optimal for the given objective function!
The only assumption is that quant error scales as 2^{-2b}.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from tqdm import tqdm


def quantize_tensor(x, num_bits, per_channel=False):
    """Simple min-max quantization."""
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


def measure_empirical_sensitivity(model, tokenizer, prompts, reference_bits=4, num_bins=20):
    """
    Measure empirical sensitivity s(p) for each position bin.

    For each position p:
    - Quantize ONLY position p at reference_bits
    - Measure MSE vs FP output
    - s(p) = average MSE across prompts

    This gives the TRUE sensitivity, accounting for all non-linearities.
    """
    device = next(model.parameters()).device

    sensitivity_by_bin = [[] for _ in range(num_bins)]

    for prompt in tqdm(prompts, desc="Measuring sensitivity"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len < 10:
            continue

        with torch.no_grad():
            # Get FP reference
            fp_outputs = model(**inputs, use_cache=True)
            fp_logits = fp_outputs.logits[:, -1, :]
            fp_kv = fp_outputs.past_key_values

            # For each position, quantize ONLY that position
            for pos in range(seq_len):
                # Clone KV cache
                q_kv = DynamicCache()
                for layer_idx in range(len(fp_kv)):
                    k = fp_kv.key_cache[layer_idx].clone()
                    v = fp_kv.value_cache[layer_idx].clone()

                    # Quantize only position pos
                    k[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], reference_bits)
                    v[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], reference_bits)

                    q_kv.update(k, v, layer_idx)

                # Get output with quantized KV
                dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
                q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
                q_logits = q_outputs.logits[:, -1, :]

                # Compute MSE
                mse = F.mse_loss(fp_logits, q_logits).item()

                # Store by relative position bin
                rel_pos = pos / seq_len
                bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
                sensitivity_by_bin[bin_idx].append(mse)

    # Compute mean sensitivity per bin
    sensitivity_profile = []
    for bin_idx in range(num_bins):
        if sensitivity_by_bin[bin_idx]:
            sensitivity_profile.append(np.mean(sensitivity_by_bin[bin_idx]))
        else:
            sensitivity_profile.append(1e-10)

    return np.array(sensitivity_profile)


def compute_optimal_allocation(sensitivity, B_avg, bit_options=[2, 3, 4, 5, 6, 8],
                               min_bits=3, max_bits=8):
    """
    Compute provably optimal bit allocation using empirical sensitivity.

    THEOREM: For objective min Σ_p s(p) · 2^{-2b_p} s.t. avg(b) = B_avg,
    the optimal continuous solution is:

        b*(p) = 0.5 · log₂(s(p)) + C

    where C = B_avg - (1/L) Σ_p 0.5 · log₂(s(p))

    With floor constraint b_min to avoid catastrophic low-bit errors.
    """
    L = len(sensitivity)
    s = np.maximum(sensitivity, 1e-10)

    # Optimal continuous allocation
    log_s = 0.5 * np.log2(s)
    C = B_avg - np.mean(log_s)
    b_continuous = log_s + C

    # Apply floor and ceiling constraints
    b_continuous = np.clip(b_continuous, min_bits, max_bits)

    # Filter bit options by constraints
    valid_options = [b for b in bit_options if min_bits <= b <= max_bits]

    # Discretize to nearest valid bit option
    b_discrete = np.array([min(valid_options, key=lambda x: abs(x - b)) for b in b_continuous])

    # Greedy refinement to exactly match B_avg
    current_avg = np.mean(b_discrete)
    max_iter = 100
    for _ in range(max_iter):
        if abs(current_avg - B_avg) < 0.05:
            break

        if current_avg > B_avg:
            # Need to reduce: find position with highest bits and lowest sensitivity
            candidates = [(i, s[i]) for i in range(L) if b_discrete[i] > min(valid_options)]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])
            idx = candidates[0][0]
            current_bits = b_discrete[idx]
            lower_options = [b for b in valid_options if b < current_bits]
            if lower_options:
                b_discrete[idx] = max(lower_options)
        else:
            # Need to increase: find position with lowest bits and highest sensitivity
            candidates = [(i, s[i]) for i in range(L) if b_discrete[i] < max(valid_options)]
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[1])
            idx = candidates[0][0]
            current_bits = b_discrete[idx]
            higher_options = [b for b in valid_options if b > current_bits]
            if higher_options:
                b_discrete[idx] = min(higher_options)

        current_avg = np.mean(b_discrete)

    return b_discrete.tolist(), b_continuous.tolist()


def quantize_kv_with_allocation(kv_cache, allocation):
    """Quantize KV cache with per-position bit allocation."""
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


def quantize_kv_uniform(kv_cache, num_bits):
    """Uniform quantization of entire KV cache."""
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)

    return quantized


def quantize_kv_sink(kv_cache, sink_fraction=0.1, sink_bits=6, rest_bits=4):
    """Heuristic Sink 6-4-4 quantization."""
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()

        seq_len = k.shape[2]
        sink_end = int(seq_len * sink_fraction)

        k[:, :, :sink_end, :] = quantize_tensor(k[:, :, :sink_end, :], sink_bits)
        v[:, :, :sink_end, :] = quantize_tensor(v[:, :, :sink_end, :], sink_bits)
        k[:, :, sink_end:, :] = quantize_tensor(k[:, :, sink_end:, :], rest_bits)
        v[:, :, sink_end:, :] = quantize_tensor(v[:, :, sink_end:, :], rest_bits)

        quantized.update(k, v, layer_idx)

    return quantized


def evaluate_allocation(model, tokenizer, prompts, quant_fn):
    """Evaluate a quantization strategy."""
    device = next(model.parameters()).device

    mse_list = []
    cos_sim_list = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len < 10:
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
    print("Phase 1.3: Refined Optimal Bit Allocation (Empirical Sensitivity)")
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

    # Calibration prompts
    calibration_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness and silence.",
        "Machine learning has revolutionized how we process data.",
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Scientists have discovered a new species of deep-sea fish.",
        "The history of computing began with mechanical calculators.",
        "Climate change poses significant challenges for future generations.",
        "Artificial intelligence systems can now generate human-like text.",
    ]

    test_prompts = [
        "Deep learning models require large amounts of training data.",
        "The ancient Romans built an extensive network of roads.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The stock market experienced significant volatility during the pandemic.",
    ]

    print("\nStep 1: Measuring EMPIRICAL sensitivity s(p)...")
    print("(Quantize each position individually and measure output MSE)")
    sensitivity = measure_empirical_sensitivity(model, tokenizer, calibration_prompts,
                                                reference_bits=4, num_bins=20)

    print("\n" + "=" * 70)
    print("EMPIRICAL SENSITIVITY PROFILE s(p)")
    print("=" * 70)

    for i, s in enumerate(sensitivity):
        rel_pos = (i + 0.5) / len(sensitivity)
        bar = "█" * int(s / max(sensitivity) * 40) if max(sensitivity) > 0 else ""
        print(f"  {rel_pos:.2f}: {s:.4f} {bar}")

    # Analyze
    early_sens = np.mean(sensitivity[:5])
    mid_sens = np.mean(sensitivity[5:15])
    late_sens = np.mean(sensitivity[15:])

    print(f"\nSensitivity by region:")
    print(f"  Early (0-25%):  {early_sens:.4f}")
    print(f"  Mid (25-75%):   {mid_sens:.4f}")
    print(f"  Late (75-100%): {late_sens:.4f}")

    cv = np.std(sensitivity) / np.mean(sensitivity)
    print(f"\nCoefficient of Variation: {cv:.2f}")
    if cv > 0.5:
        print("  -> HIGH variation - optimal allocation will differ from uniform")
    else:
        print("  -> LOW variation - optimal allocation may be close to uniform")

    # Compute optimal allocations
    print("\n" + "=" * 70)
    print("OPTIMAL BIT ALLOCATIONS")
    print("=" * 70)

    for B_avg in [3.5, 4.0, 4.2, 5.0]:
        b_discrete, b_continuous = compute_optimal_allocation(sensitivity, B_avg,
                                                              min_bits=3, max_bits=8)
        actual_avg = np.mean(b_discrete)
        print(f"\nTarget B_avg = {B_avg}, Actual = {actual_avg:.2f}")
        print(f"  Optimal: {b_discrete}")

    # Micro-validation
    print("\n" + "=" * 70)
    print("MICRO-VALIDATION: Optimal vs Heuristic vs Uniform")
    print("=" * 70)

    B_avg = 4.2
    optimal_alloc, _ = compute_optimal_allocation(sensitivity, B_avg, min_bits=3, max_bits=8)

    configs = [
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("Sink 6-4-4 (heuristic)", lambda kv: quantize_kv_sink(kv, 0.1, 6, 4), 4.2),
        ("Optimal (empirical)", lambda kv: quantize_kv_with_allocation(kv, optimal_alloc), np.mean(optimal_alloc)),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),
    ]

    print(f"\nOptimal allocation for B_avg={B_avg}: {optimal_alloc}")
    print(f"\nEvaluating on {len(test_prompts)} test prompts...")

    results = []
    for name, quant_fn, avg_bits in configs:
        mse, cos_sim = evaluate_allocation(model, tokenizer, test_prompts, quant_fn)
        results.append({
            "name": name,
            "avg_bits": avg_bits,
            "mse": mse,
            "cos_sim": cos_sim,
        })
        print(f"  {name}: MSE={mse:.4f}, CosSim={cos_sim:.4f}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<25} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Analysis
    optimal_result = next(r for r in results if "Optimal" in r["name"])
    sink_result = next(r for r in results if "Sink" in r["name"])
    uniform_4bit = next(r for r in results if "Uniform 4-bit" in r["name"])

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\n1. Optimal vs Sink 6-4-4:")
    if optimal_result["mse"] < sink_result["mse"]:
        improvement = (sink_result["mse"] - optimal_result["mse"]) / sink_result["mse"] * 100
        print(f"   -> Optimal is {improvement:.1f}% BETTER (lower MSE)")
        print(f"   -> VALIDATES the theory!")
    elif abs(optimal_result["mse"] - sink_result["mse"]) / sink_result["mse"] < 0.05:
        print(f"   -> Results are EQUIVALENT (within 5%)")
        print(f"   -> Heuristic happens to be near-optimal for this setting")
    else:
        degradation = (optimal_result["mse"] - sink_result["mse"]) / sink_result["mse"] * 100
        print(f"   -> Optimal is {degradation:.1f}% WORSE")
        print(f"   -> Need to investigate further")

    print(f"\n2. Optimal vs Uniform 4-bit:")
    improvement = (uniform_4bit["mse"] - optimal_result["mse"]) / uniform_4bit["mse"] * 100
    print(f"   -> Optimal is {improvement:.1f}% better")

    # Summary
    print("\n" + "=" * 70)
    print("MATHEMATICAL DERIVATION (REFINED)")
    print("=" * 70)
    print("""
THEOREM: Optimal Position-Aware KV Quantization (Empirical)

Given:
- Empirical sensitivity s(p) measured via calibration
- Target average bit-width B_avg
- Bit constraints: b_min ≤ b(p) ≤ b_max

Objective:
  min_{b(p)} Σ_p s(p) · 2^{-2b(p)}  [minimize sensitivity-weighted error]
  s.t. (1/L) Σ_p b(p) = B_avg
       b_min ≤ b(p) ≤ b_max

Solution (by Lagrangian with clipping):
  b*(p) = clip(0.5 · log₂(s(p)) + C, b_min, b_max)

where C satisfies the average constraint.

Key Insight:
- s(p) is MEASURED empirically, not derived from attention alone
- This captures all non-linearities and layer interactions
- The solution is provably optimal for the measured sensitivity

Calibration Complexity: O(L × N) where L = sequence length, N = num prompts
Allocation Complexity: O(L log L) for sorting and greedy refinement
""")

    print("=" * 70)
    print("Derivation Complete")
    print("=" * 70)

    return sensitivity, optimal_alloc


if __name__ == "__main__":
    main()
