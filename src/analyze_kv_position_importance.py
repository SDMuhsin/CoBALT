#!/usr/bin/env python3
"""
Phase 1.2 Analysis: KV Cache Position Importance

Analyzes how different KV cache positions contribute to attention output.
Hypothesis: Early positions are accessed in every attention computation,
so their quantization errors propagate more.

Key questions:
1. What is the attention weight distribution across positions?
2. How does quantizing different position ranges affect output?
3. Do early positions (sink tokens) receive disproportionate attention?
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict

def analyze_attention_patterns(model, tokenizer, prompts, max_new_tokens=50):
    """
    Analyze attention patterns to understand position importance.
    Returns attention statistics per position.
    """
    device = next(model.parameters()).device

    position_attention_sums = defaultdict(list)  # position -> list of attention weights received

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        # Generate with attention output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        # Analyze attention patterns from the attentions
        # Each attention tensor: (batch, num_heads, seq_len, seq_len)
        for step_idx, step_attentions in enumerate(outputs.attentions):
            # step_attentions is tuple of (layer_attention for each layer)
            for layer_idx, layer_attn in enumerate(step_attentions):
                # layer_attn: (batch, num_heads, query_len, key_len)
                # For autoregressive, query_len=1 after first step
                attn_weights = layer_attn[0].mean(dim=0)  # Average over heads: (query_len, key_len)

                # Sum attention each position receives (from the last query position)
                if attn_weights.dim() == 2:
                    last_query_attn = attn_weights[-1]  # Attention from last query to all keys
                else:
                    last_query_attn = attn_weights

                for pos_idx, attn_weight in enumerate(last_query_attn.cpu().numpy()):
                    position_attention_sums[pos_idx].append(float(attn_weight))

    return position_attention_sums


def analyze_kv_quantization_sensitivity(model, tokenizer, prompts, num_bits=4):
    """
    Analyze how quantizing KV cache at different positions affects output.

    Method: For each position range, quantize only that range and measure output error.
    """
    device = next(model.parameters()).device

    results = []

    for prompt in prompts[:3]:  # Use subset for efficiency
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        # Get FP reference output with KV cache
        with torch.no_grad():
            fp_outputs = model(**inputs, use_cache=True, output_hidden_states=True)
            fp_logits = fp_outputs.logits
            fp_kv_cache = fp_outputs.past_key_values

        # Analyze position ranges
        ranges = [
            (0, seq_len // 4, "early_25%"),
            (seq_len // 4, seq_len // 2, "mid_early_25%"),
            (seq_len // 2, 3 * seq_len // 4, "mid_late_25%"),
            (3 * seq_len // 4, seq_len, "late_25%"),
        ]

        for start, end, name in ranges:
            # Quantize only this range of KV cache
            quantized_kv = quantize_kv_range(fp_kv_cache, start, end, num_bits)

            # Forward pass with quantized KV
            with torch.no_grad():
                # We need to do a forward pass using the past_key_values
                # For analysis, we'll just measure the KV error directly
                kv_error = compute_kv_error(fp_kv_cache, quantized_kv, start, end)

            results.append({
                "prompt_len": seq_len,
                "range": name,
                "start": start,
                "end": end,
                "kv_mse": kv_error
            })

    return results


def quantize_kv_range(kv_cache, start, end, num_bits):
    """Quantize only a specific position range of the KV cache."""
    from transformers.cache_utils import DynamicCache

    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]

        k_q = k.clone()
        v_q = v.clone()

        # Quantize only positions [start:end]
        k_q[:, :, start:end, :] = quantize_tensor(k[:, :, start:end, :], num_bits)
        v_q[:, :, start:end, :] = quantize_tensor(v[:, :, start:end, :], num_bits)

        quantized.update(k_q, v_q, layer_idx)

    return quantized


def quantize_tensor(x, num_bits):
    """Simple min-max quantization."""
    if x.numel() == 0:
        return x

    x_min = x.min()
    x_max = x.max()

    if x_max == x_min:
        return x

    scale = (x_max - x_min) / (2**num_bits - 1)
    x_q = torch.round((x - x_min) / scale) * scale + x_min

    return x_q


def compute_kv_error(fp_kv, q_kv, start, end):
    """Compute MSE between FP and quantized KV in the specified range."""
    total_mse = 0.0
    count = 0

    for layer_idx in range(len(fp_kv)):
        k_fp = fp_kv.key_cache[layer_idx]
        v_fp = fp_kv.value_cache[layer_idx]
        k_q = q_kv.key_cache[layer_idx]
        v_q = q_kv.value_cache[layer_idx]

        k_mse = F.mse_loss(k_fp[:, :, start:end, :], k_q[:, :, start:end, :]).item()
        v_mse = F.mse_loss(v_fp[:, :, start:end, :], v_q[:, :, start:end, :]).item()
        total_mse += k_mse + v_mse
        count += 2

    return total_mse / count if count > 0 else 0.0


def analyze_position_sensitivity_via_perturbation(model, tokenizer, prompt, max_positions=100):
    """
    Perturb KV cache at each position and measure output sensitivity.
    This directly measures how much each position matters for the output.
    """
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    # Get FP reference with DynamicCache
    with torch.no_grad():
        fp_outputs = model(**inputs, use_cache=True)
        fp_logits = fp_outputs.logits[:, -1, :]  # Last token logits
        fp_kv = fp_outputs.past_key_values

    # Perturb each position and measure output change
    position_sensitivities = []

    positions_to_test = min(seq_len, max_positions)

    for pos in range(positions_to_test):
        # Add noise to position pos in KV cache
        perturbed_kv = perturb_kv_position(fp_kv, pos, noise_scale=0.1)

        # Forward with perturbed KV (using a dummy forward to get next token prediction)
        # We need to do this properly - extend the sequence by one token
        with torch.no_grad():
            # Create a dummy next token input
            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            perturbed_outputs = model(
                input_ids=dummy_input,
                past_key_values=perturbed_kv,
                use_cache=False
            )
            perturbed_logits = perturbed_outputs.logits[:, -1, :]

        # Measure KL divergence between FP and perturbed outputs
        fp_probs = F.softmax(fp_logits, dim=-1)
        perturbed_probs = F.softmax(perturbed_logits, dim=-1)

        kl_div = F.kl_div(
            perturbed_probs.log(),
            fp_probs,
            reduction='batchmean'
        ).item()

        position_sensitivities.append({
            "position": pos,
            "relative_position": pos / seq_len,
            "kl_divergence": kl_div
        })

    return position_sensitivities


def perturb_kv_position(kv_cache, pos, noise_scale=0.1):
    """Add noise to a specific position in KV cache."""
    from transformers.cache_utils import DynamicCache

    perturbed = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]

        k_p = k.clone()
        v_p = v.clone()

        # Add noise to position pos
        k_std = k[:, :, pos:pos+1, :].std()
        v_std = v[:, :, pos:pos+1, :].std()

        k_p[:, :, pos:pos+1, :] += torch.randn_like(k[:, :, pos:pos+1, :]) * k_std * noise_scale
        v_p[:, :, pos:pos+1, :] += torch.randn_like(v[:, :, pos:pos+1, :]) * v_std * noise_scale

        perturbed.update(k_p, v_p, layer_idx)

    return perturbed


def main():
    print("=" * 60)
    print("Phase 1.2: KV Cache Position Importance Analysis")
    print("=" * 60)

    # Load model
    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"  # Need eager for attention output
    )
    model.eval()

    # Test prompts of varying lengths
    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "In a galaxy far far away, there lived a wise old wizard who knew the secrets of the universe. " * 3,
        "Machine learning has revolutionized the way we process and understand data. Neural networks can learn complex patterns. " * 4,
    ]

    print("\n" + "=" * 60)
    print("Analysis 1: Position Sensitivity via Perturbation")
    print("=" * 60)

    # Use a medium-length prompt for perturbation analysis
    test_prompt = "The history of artificial intelligence began in the mid-20th century. Early pioneers like Alan Turing and John McCarthy laid the foundations for what would become one of the most transformative technologies of our time. Today, AI systems can understand language, recognize images, and even generate creative content."

    print(f"\nPrompt length: {len(tokenizer.encode(test_prompt))} tokens")
    print("Testing sensitivity of each position...")

    sensitivities = analyze_position_sensitivity_via_perturbation(
        model, tokenizer, test_prompt, max_positions=50
    )

    # Analyze results
    early_sens = [s["kl_divergence"] for s in sensitivities if s["relative_position"] < 0.25]
    mid_sens = [s["kl_divergence"] for s in sensitivities if 0.25 <= s["relative_position"] < 0.75]
    late_sens = [s["kl_divergence"] for s in sensitivities if s["relative_position"] >= 0.75]

    print("\n" + "-" * 40)
    print("Position Sensitivity (KL Divergence from perturbation):")
    print("-" * 40)
    print(f"Early positions (0-25%):  mean={np.mean(early_sens):.6f}, std={np.std(early_sens):.6f}")
    if mid_sens:
        print(f"Mid positions (25-75%):   mean={np.mean(mid_sens):.6f}, std={np.std(mid_sens):.6f}")
    if late_sens:
        print(f"Late positions (75-100%): mean={np.mean(late_sens):.6f}, std={np.std(late_sens):.6f}")

    # Detailed position breakdown
    print("\n" + "-" * 40)
    print("Per-Position Sensitivity:")
    print("-" * 40)
    for s in sensitivities[:20]:  # First 20 positions
        print(f"  Pos {s['position']:3d} ({s['relative_position']:.1%}): KL = {s['kl_divergence']:.6f}")

    print("\n" + "=" * 60)
    print("Analysis 2: KV Quantization Error by Position Range")
    print("=" * 60)

    quant_results = analyze_kv_quantization_sensitivity(model, tokenizer, prompts, num_bits=4)

    # Group by range
    range_errors = defaultdict(list)
    for r in quant_results:
        range_errors[r["range"]].append(r["kv_mse"])

    print("\n" + "-" * 40)
    print("4-bit Quantization MSE by Position Range:")
    print("-" * 40)
    for range_name in ["early_25%", "mid_early_25%", "mid_late_25%", "late_25%"]:
        if range_name in range_errors:
            errors = range_errors[range_name]
            print(f"  {range_name:15s}: mean MSE = {np.mean(errors):.6e}")

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Determine if early positions are more sensitive
    if early_sens and (mid_sens or late_sens):
        all_other = mid_sens + late_sens
        ratio = np.mean(early_sens) / np.mean(all_other) if all_other else float('inf')
        print(f"\n1. Early position sensitivity ratio: {ratio:.2f}x")
        if ratio > 1.5:
            print("   -> Early positions ARE more sensitive (supports Gap 7 hypothesis)")
        elif ratio < 0.67:
            print("   -> Early positions are LESS sensitive (contradicts Gap 7 hypothesis)")
        else:
            print("   -> Position sensitivity is UNIFORM (Gap 7 may have limited benefit)")

    # Check for sink token pattern
    if len(sensitivities) > 3:
        first_few = [s["kl_divergence"] for s in sensitivities[:3]]
        rest = [s["kl_divergence"] for s in sensitivities[3:]]
        sink_ratio = np.mean(first_few) / np.mean(rest) if rest else float('inf')
        print(f"\n2. Sink token (pos 0-2) sensitivity ratio: {sink_ratio:.2f}x")
        if sink_ratio > 2.0:
            print("   -> Strong sink token effect detected")
        else:
            print("   -> No strong sink token effect")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
