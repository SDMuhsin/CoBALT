#!/usr/bin/env python3
"""
Phase 1.2: Calibration for Per-Position Sensitivity Estimation

Goal: Measure s(p) = sensitivity of output to quantization at position p

Method:
1. For each position p, quantize ONLY position p in KV cache
2. Measure the output error (MSE) compared to FP
3. s(p) = MSE when position p is quantized

This gives us the "marginal" sensitivity of each position.
The sensitivity profile s(p) will inform the optimal bit allocation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def quantize_tensor(x, num_bits):
    """Simple min-max quantization."""
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return x
    scale = (x_max - x_min) / (2**num_bits - 1)
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def quantize_single_position(kv_cache, pos, num_bits):
    """Quantize only a single position in the KV cache."""
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()

        # Quantize only position pos
        k[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], num_bits)
        v[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], num_bits)

        quantized.update(k, v, layer_idx)

    return quantized


def measure_position_sensitivity(model, tokenizer, prompts, num_bits=4):
    """
    Measure per-position sensitivity across multiple prompts.

    For each position p:
    - Quantize ONLY position p with num_bits
    - Measure output MSE vs FP
    - s(p) = average MSE across prompts

    Returns:
    - sensitivity_by_relative_pos: dict mapping relative position (0-1) to sensitivity
    """
    device = next(model.parameters()).device

    # Collect sensitivity measurements
    all_sensitivities = {}  # position -> list of MSE values

    for prompt in tqdm(prompts, desc="Calibrating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len < 10:
            continue

        with torch.no_grad():
            # Get FP reference
            fp_outputs = model(**inputs, use_cache=True)
            fp_logits = fp_outputs.logits[:, -1, :]
            fp_kv = fp_outputs.past_key_values

            # Measure sensitivity for each position
            for pos in range(seq_len):
                # Quantize only this position
                q_kv = quantize_single_position(fp_kv, pos, num_bits)

                # Get output with quantized KV
                dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
                q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
                q_logits = q_outputs.logits[:, -1, :]

                # Compute MSE
                mse = F.mse_loss(fp_logits, q_logits).item()

                # Store by relative position
                rel_pos = pos / seq_len
                if rel_pos not in all_sensitivities:
                    all_sensitivities[rel_pos] = []
                all_sensitivities[rel_pos].append(mse)

    # Average sensitivities by relative position (binned)
    num_bins = 20
    binned_sensitivity = [[] for _ in range(num_bins)]

    for rel_pos, mse_list in all_sensitivities.items():
        bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
        binned_sensitivity[bin_idx].extend(mse_list)

    # Compute mean sensitivity per bin
    sensitivity_profile = []
    for bin_idx in range(num_bins):
        if binned_sensitivity[bin_idx]:
            mean_sens = np.mean(binned_sensitivity[bin_idx])
        else:
            mean_sens = 0.0
        sensitivity_profile.append(mean_sens)

    return sensitivity_profile


def compute_optimal_allocation(sensitivity_profile, B_avg, bit_options=[2, 3, 4, 5, 6, 8]):
    """
    Compute optimal bit allocation using Lagrangian relaxation.

    Problem:
        min Σ_p s(p) * σ²(b(p))
        s.t. (1/L) Σ_p b(p) = B_avg

    where σ²(b) = 1/(2^{2b}) (proportional to quantization variance)

    Lagrangian:
        L = Σ_p s(p) * 2^{-2b(p)} + λ * ((1/L) Σ_p b(p) - B_avg)

    For continuous relaxation, taking derivative w.r.t. b(p):
        ∂L/∂b(p) = -2*ln(2) * s(p) * 2^{-2b(p)} + λ/L = 0
        => 2^{-2b(p)} = λ / (2*L*ln(2)*s(p))
        => -2b(p) = log2(λ / (2*L*ln(2)*s(p)))
        => b(p) = -0.5 * log2(λ / (2*L*ln(2)*s(p)))
        => b(p) = 0.5 * log2(2*L*ln(2)*s(p)) - 0.5*log2(λ)

    This shows: b*(p) ∝ 0.5 * log2(s(p)) + constant

    We find λ such that the average bit constraint is satisfied.
    """
    L = len(sensitivity_profile)
    s = np.array(sensitivity_profile)

    # Handle zero sensitivities
    s = np.maximum(s, 1e-10)

    # Continuous solution: b(p) = 0.5 * log2(s(p)) + C
    # where C is chosen to satisfy the average constraint

    log_s = 0.5 * np.log2(s)

    # Find C such that mean(0.5*log2(s) + C) = B_avg
    # => C = B_avg - mean(0.5*log2(s))
    C = B_avg - np.mean(log_s)

    b_continuous = log_s + C

    # Discretize to nearest bit option
    b_discrete = []
    for b in b_continuous:
        nearest = min(bit_options, key=lambda x: abs(x - b))
        b_discrete.append(nearest)

    b_discrete = np.array(b_discrete)

    # Adjust to exactly match B_avg constraint (greedy refinement)
    current_avg = np.mean(b_discrete)
    while abs(current_avg - B_avg) > 0.05:
        if current_avg > B_avg:
            # Need to reduce: find position with highest bits and lowest sensitivity
            candidates = [(i, s[i]) for i in range(L) if b_discrete[i] > min(bit_options)]
            if not candidates:
                break
            # Reduce bit at position with lowest sensitivity
            candidates.sort(key=lambda x: x[1])
            idx = candidates[0][0]
            current_bits = b_discrete[idx]
            lower_options = [b for b in bit_options if b < current_bits]
            if lower_options:
                b_discrete[idx] = max(lower_options)
        else:
            # Need to increase: find position with lowest bits and highest sensitivity
            candidates = [(i, s[i]) for i in range(L) if b_discrete[i] < max(bit_options)]
            if not candidates:
                break
            # Increase bit at position with highest sensitivity
            candidates.sort(key=lambda x: -x[1])
            idx = candidates[0][0]
            current_bits = b_discrete[idx]
            higher_options = [b for b in bit_options if b > current_bits]
            if higher_options:
                b_discrete[idx] = min(higher_options)

        current_avg = np.mean(b_discrete)

    return b_discrete.tolist(), b_continuous.tolist()


def main():
    print("=" * 70)
    print("Phase 1.2: Per-Position Sensitivity Calibration")
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
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness and silence.",
        "Machine learning has revolutionized how we process data.",
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Scientists have discovered a new species of deep-sea fish.",
        "The history of computing began with mechanical calculators.",
        "Climate change poses significant challenges for future generations.",
        "Artificial intelligence systems can now generate human-like text.",
    ]

    print(f"\nCalibrating with {len(prompts)} prompts...")

    # Measure sensitivity profile
    sensitivity_profile = measure_position_sensitivity(model, tokenizer, prompts, num_bits=4)

    print("\n" + "=" * 70)
    print("SENSITIVITY PROFILE (by relative position)")
    print("=" * 70)

    for i, s in enumerate(sensitivity_profile):
        rel_pos = (i + 0.5) / len(sensitivity_profile)
        bar = "█" * int(s / max(sensitivity_profile) * 40)
        print(f"  {rel_pos:.2f}: {s:.4f} {bar}")

    # Compute optimal allocation for different B_avg
    print("\n" + "=" * 70)
    print("OPTIMAL BIT ALLOCATIONS")
    print("=" * 70)

    for B_avg in [3.0, 4.0, 5.0]:
        b_discrete, b_continuous = compute_optimal_allocation(sensitivity_profile, B_avg)
        actual_avg = np.mean(b_discrete)
        print(f"\nTarget B_avg = {B_avg}, Actual = {actual_avg:.2f}")
        print(f"  Allocation: {b_discrete}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)

    # Check if sensitivity varies significantly by position
    early_sens = np.mean(sensitivity_profile[:5])
    mid_sens = np.mean(sensitivity_profile[5:15])
    late_sens = np.mean(sensitivity_profile[15:])

    print(f"\n1. Sensitivity by position range:")
    print(f"   Early (0-25%):  {early_sens:.4f}")
    print(f"   Mid (25-75%):   {mid_sens:.4f}")
    print(f"   Late (75-100%): {late_sens:.4f}")

    if early_sens > 1.5 * mid_sens:
        print("   -> Early positions are MORE sensitive (supports sink token hypothesis)")
    elif late_sens > 1.5 * mid_sens:
        print("   -> Late positions are MORE sensitive")
    else:
        print("   -> Sensitivity is relatively UNIFORM")

    # Check variation
    variation = np.std(sensitivity_profile) / np.mean(sensitivity_profile)
    print(f"\n2. Sensitivity variation (CV): {variation:.2f}")
    if variation > 0.5:
        print("   -> HIGH variation - optimal allocation will differ significantly from uniform")
    else:
        print("   -> LOW variation - optimal allocation may be close to uniform")

    print("\n" + "=" * 70)
    print("Calibration Complete")
    print("=" * 70)

    return sensitivity_profile


if __name__ == "__main__":
    main()
