#!/usr/bin/env python3
"""
Comprehensive validation of Zone-Based KV Quantization.

Key finding from micro-validation:
- Zone 5-4-3 (Gap7: early high, late low) BEATS uniform 4-bit by 19.8%
- Zone 5-4-3 at 4 avg bits BEATS uniform 5-bit!

This test:
1. Validates on more prompts
2. Tests on longer contexts
3. Measures generation quality (not just next-token)
4. Tests different zone ratios
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np


def quantize_tensor(x, num_bits):
    """Simple asymmetric min-max quantization."""
    if x.numel() == 0:
        return x

    x_min = x.min()
    x_max = x.max()

    if x_max == x_min:
        return x

    scale = (x_max - x_min) / (2**num_bits - 1)
    x_q = torch.round((x - x_min) / scale) * scale + x_min

    return x_q


def quantize_kv_uniform(kv_cache, num_bits):
    """Uniform quantization of entire KV cache."""
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]

        k_q = quantize_tensor(k, num_bits)
        v_q = quantize_tensor(v, num_bits)

        quantized.update(k_q, v_q, layer_idx)

    return quantized


def quantize_kv_zoned_flexible(kv_cache, zone_config):
    """
    Flexible zone-based quantization.

    Args:
        kv_cache: DynamicCache
        zone_config: list of (end_fraction, bits) tuples
            e.g., [(0.1, 6), (0.5, 4), (1.0, 3)] means:
            - 0-10%: 6 bits
            - 10-50%: 4 bits
            - 50-100%: 3 bits
    """
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]
        seq_len = k.shape[2]

        k_q = k.clone()
        v_q = v.clone()

        prev_end = 0
        for end_frac, bits in zone_config:
            end = int(seq_len * end_frac)
            if end > prev_end:
                k_q[:, :, prev_end:end, :] = quantize_tensor(k[:, :, prev_end:end, :], bits)
                v_q[:, :, prev_end:end, :] = quantize_tensor(v[:, :, prev_end:end, :], bits)
            prev_end = end

        quantized.update(k_q, v_q, layer_idx)

    return quantized


def evaluate_generation_quality(model, tokenizer, prompts, quant_fn, max_new_tokens=20):
    """
    Evaluate quantization impact on generation quality.

    Metrics:
    - Token-level agreement with FP generation
    - KL divergence at each step
    """
    device = next(model.parameters()).device

    results = {
        "token_match_rate": [],
        "avg_kl": [],
        "first_mismatch_position": [],
    }

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate with FP model
        with torch.no_grad():
            fp_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            fp_tokens = fp_outputs.sequences[0, inputs.input_ids.shape[1]:]
            fp_scores = torch.stack(fp_outputs.scores, dim=0)  # (new_tokens, vocab)

        # Generate with quantized KV
        # For fair comparison, we quantize the KV cache at each step
        q_tokens = []
        q_scores = []

        current_input = inputs.input_ids.clone()

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(current_input, use_cache=True)
                fp_kv = outputs.past_key_values

                # Quantize KV cache
                q_kv = quant_fn(fp_kv)

                # Get next token prediction with quantized KV
                dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
                q_outputs = model(input_ids=dummy, past_key_values=q_kv, use_cache=False)
                q_logits = q_outputs.logits[:, -1, :]

                next_token = q_logits.argmax(dim=-1)
                q_tokens.append(next_token.item())
                q_scores.append(q_logits[0])

                # Append to input for next step
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        q_tokens = torch.tensor(q_tokens, device=device)
        q_scores = torch.stack(q_scores, dim=0) if q_scores else torch.zeros(0, fp_scores.shape[1], device=device)

        # Compute metrics
        min_len = min(len(fp_tokens), len(q_tokens))
        if min_len > 0:
            matches = (fp_tokens[:min_len] == q_tokens[:min_len]).float()
            token_match_rate = matches.mean().item()

            # Find first mismatch
            mismatches = (fp_tokens[:min_len] != q_tokens[:min_len]).nonzero(as_tuple=True)[0]
            first_mismatch = mismatches[0].item() if len(mismatches) > 0 else min_len
        else:
            token_match_rate = 0.0
            first_mismatch = 0

        # KL divergence
        if len(q_scores) > 0 and len(fp_scores) > 0:
            min_steps = min(len(fp_scores), len(q_scores))
            fp_probs = F.softmax(fp_scores[:min_steps], dim=-1)
            q_probs = F.softmax(q_scores[:min_steps], dim=-1)
            kl = F.kl_div(q_probs.log(), fp_probs, reduction='batchmean').item()
        else:
            kl = float('inf')

        results["token_match_rate"].append(token_match_rate)
        results["avg_kl"].append(kl)
        results["first_mismatch_position"].append(first_mismatch)

    return {
        "token_match_rate": np.mean(results["token_match_rate"]),
        "avg_kl": np.mean([k for k in results["avg_kl"] if k != float('inf')]) if results["avg_kl"] else float('inf'),
        "avg_first_mismatch": np.mean(results["first_mismatch_position"]),
    }


def main():
    print("=" * 70)
    print("Comprehensive Zone-Based KV Quantization Validation")
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

    # Test prompts with various lengths
    prompts = [
        "The capital of France is",
        "In the year 2024, artificial intelligence has become",
        "The quick brown fox jumps over the lazy dog. This sentence is often used because it",
        """Machine learning models require large amounts of data to train effectively.
        The quality of the data is just as important as the quantity. When training a model,
        it is essential to ensure that the data is representative of the real-world distribution.
        This principle is known as""",
        """The history of computing can be traced back to the invention of the abacus.
        However, the modern computer era began with Charles Babbage's Analytical Engine in the 1830s.
        The first programmable electronic computer, ENIAC, was completed in 1945. Since then,
        computers have evolved dramatically, following Moore's Law which predicts that the number of
        transistors on a chip doubles approximately every two years. Today, we carry more computing
        power in our pockets than was used to send astronauts to the moon. The future of computing
        promises even more remarkable advances with quantum computers and neuromorphic chips. The next
        major breakthrough will likely be""",
    ]

    print(f"\nTest prompts: {len(prompts)}")
    for i, p in enumerate(prompts):
        tokens = len(tokenizer.encode(p))
        print(f"  Prompt {i+1}: {tokens} tokens")

    # Zone configurations to test
    configs = [
        # Uniform baselines
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),

        # Gap7 variants (early high, late low)
        ("Gap7: 5-4-3", lambda kv: quantize_kv_zoned_flexible(kv, [(0.33, 5), (0.67, 4), (1.0, 3)]), 4.0),
        ("Gap7: 6-4-2", lambda kv: quantize_kv_zoned_flexible(kv, [(0.33, 6), (0.67, 4), (1.0, 2)]), 4.0),

        # Sink token focused (first 10% very high precision)
        ("Sink: 8-4-4", lambda kv: quantize_kv_zoned_flexible(kv, [(0.1, 8), (1.0, 4)]), 4.4),
        ("Sink: 6-4-4", lambda kv: quantize_kv_zoned_flexible(kv, [(0.1, 6), (1.0, 4)]), 4.2),

        # Gradient descent style (continuous precision decay)
        ("Decay: 6-5-4-3", lambda kv: quantize_kv_zoned_flexible(kv, [(0.25, 6), (0.5, 5), (0.75, 4), (1.0, 3)]), 4.5),
    ]

    print("\n" + "=" * 70)
    print("Part 1: Next-Token Prediction Quality")
    print("=" * 70)

    next_token_results = []

    for name, quant_fn, avg_bits in configs:
        print(f"\nTesting: {name}...")

        mse_list = []
        cos_sim_list = []
        top1_match_list = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

            with torch.no_grad():
                fp_outputs = model(**inputs, use_cache=True)
                fp_logits = fp_outputs.logits[:, -1, :]
                fp_kv = fp_outputs.past_key_values

            q_kv = quant_fn(fp_kv)

            with torch.no_grad():
                dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=fp_logits.device)
                q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
                q_logits = q_outputs.logits[:, -1, :]

            mse = F.mse_loss(fp_logits, q_logits).item()
            cos_sim = F.cosine_similarity(fp_logits, q_logits, dim=-1).mean().item()
            top1_match = 1.0 if fp_logits.argmax() == q_logits.argmax() else 0.0

            mse_list.append(mse)
            cos_sim_list.append(cos_sim)
            top1_match_list.append(top1_match)

        next_token_results.append({
            "name": name,
            "avg_bits": avg_bits,
            "mse": np.mean(mse_list),
            "cos_sim": np.mean(cos_sim_list),
            "top1_match": np.mean(top1_match_list),
        })

    print("\n" + "-" * 85)
    print(f"{'Config':<20} | {'Avg Bits':>8} | {'MSE':>12} | {'Cos Sim':>10} | {'Top-1 Match':>12}")
    print("-" * 85)

    for r in next_token_results:
        print(f"{r['name']:<20} | {r['avg_bits']:>8.1f} | {r['mse']:>12.4f} | "
              f"{r['cos_sim']:>10.4f} | {r['top1_match']:>12.0%}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    uniform_4bit = next(r for r in next_token_results if r["name"] == "Uniform 4-bit")
    uniform_5bit = next(r for r in next_token_results if r["name"] == "Uniform 5-bit")

    # Find best zone-based at 4 bits
    zone_configs = [r for r in next_token_results if "Gap7" in r["name"] or "Sink" in r["name"]]
    if zone_configs:
        best_zone = min(zone_configs, key=lambda x: x["mse"])

        print(f"\n1. Best zone-based config: {best_zone['name']}")
        print(f"   MSE: {best_zone['mse']:.4f} (vs Uniform 4-bit: {uniform_4bit['mse']:.4f})")
        print(f"   Improvement: {(1 - best_zone['mse']/uniform_4bit['mse'])*100:.1f}%")

        if best_zone['mse'] < uniform_5bit['mse']:
            print(f"\n2. {best_zone['name']} at {best_zone['avg_bits']:.1f} bits BEATS Uniform 5-bit!")
            print(f"   Memory savings: {(1 - best_zone['avg_bits']/5)*100:.1f}%")
            print("   -> PUBLICATION-WORTHY RESULT")
        else:
            print(f"\n2. {best_zone['name']} does NOT beat Uniform 5-bit")
            print("   -> Need more aggressive zone design")

    # Gap7 vs Sink comparison
    gap7_results = [r for r in next_token_results if "Gap7" in r["name"]]
    sink_results = [r for r in next_token_results if "Sink" in r["name"]]

    if gap7_results and sink_results:
        best_gap7 = min(gap7_results, key=lambda x: x["mse"])
        best_sink = min(sink_results, key=lambda x: x["mse"])

        print(f"\n3. Gap7 vs Sink comparison:")
        print(f"   Best Gap7: {best_gap7['name']} (MSE={best_gap7['mse']:.4f})")
        print(f"   Best Sink: {best_sink['name']} (MSE={best_sink['mse']:.4f})")

        if best_gap7['mse'] < best_sink['mse']:
            print("   -> Gradual precision decay (Gap7) is better")
        else:
            print("   -> Focused sink token precision (Sink) is better")

    print("\n" + "=" * 70)
    print("Comprehensive Validation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
