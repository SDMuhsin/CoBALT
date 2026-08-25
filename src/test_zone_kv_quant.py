#!/usr/bin/env python3
"""
Phase 4: Micro-Validation - Zone-Based KV Quantization

Testing the simplified approach:
- Zone 1 (early 33%): 3-bit (instructions/context)
- Zone 2 (middle 33%): 4-bit (general context)
- Zone 3 (late 33%): 5-bit (recent context - based on V3 findings)

This tests whether the zone-based approach provides measurable improvement
over uniform 4-bit quantization.
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


def quantize_kv_zoned(kv_cache, zone_bits):
    """
    Zone-based quantization.

    Args:
        kv_cache: DynamicCache
        zone_bits: dict mapping zone -> bits, e.g., {"early": 3, "mid": 4, "late": 5}

    Zones:
        - early: [0, L/3)
        - mid: [L/3, 2L/3)
        - late: [2L/3, L)
    """
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]
        seq_len = k.shape[2]

        k_q = k.clone()
        v_q = v.clone()

        # Define zone boundaries
        zone1_end = seq_len // 3
        zone2_end = 2 * seq_len // 3

        # Quantize each zone with its assigned bits
        # Zone 1: early positions
        k_q[:, :, :zone1_end, :] = quantize_tensor(k[:, :, :zone1_end, :], zone_bits["early"])
        v_q[:, :, :zone1_end, :] = quantize_tensor(v[:, :, :zone1_end, :], zone_bits["early"])

        # Zone 2: middle positions
        k_q[:, :, zone1_end:zone2_end, :] = quantize_tensor(k[:, :, zone1_end:zone2_end, :], zone_bits["mid"])
        v_q[:, :, zone1_end:zone2_end, :] = quantize_tensor(v[:, :, zone1_end:zone2_end, :], zone_bits["mid"])

        # Zone 3: late positions (recent context)
        k_q[:, :, zone2_end:, :] = quantize_tensor(k[:, :, zone2_end:, :], zone_bits["late"])
        v_q[:, :, zone2_end:, :] = quantize_tensor(v[:, :, zone2_end:, :], zone_bits["late"])

        quantized.update(k_q, v_q, layer_idx)

    return quantized


def evaluate_kv_quantization(model, tokenizer, prompts, quant_fn, quant_name):
    """
    Evaluate KV cache quantization quality.

    Returns metrics:
    - MSE between FP and quantized output logits
    - Cosine similarity
    - Top-10 overlap
    - Next token agreement
    """
    device = next(model.parameters()).device

    results = {
        "mse": [],
        "cosine_sim": [],
        "top10_overlap": [],
        "top1_match": [],
    }

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get FP reference
        with torch.no_grad():
            fp_outputs = model(**inputs, use_cache=True)
            fp_logits = fp_outputs.logits[:, -1, :]
            fp_kv = fp_outputs.past_key_values

        # Quantize KV cache
        q_kv = quant_fn(fp_kv)

        # Get output with quantized KV
        with torch.no_grad():
            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
            q_logits = q_outputs.logits[:, -1, :]

        # Compute metrics
        mse = F.mse_loss(fp_logits, q_logits).item()
        cos_sim = F.cosine_similarity(fp_logits, q_logits, dim=-1).mean().item()

        # Top-k analysis
        fp_top10 = set(fp_logits.topk(10, dim=-1).indices[0].tolist())
        q_top10 = set(q_logits.topk(10, dim=-1).indices[0].tolist())
        top10_overlap = len(fp_top10 & q_top10) / 10

        fp_top1 = fp_logits.argmax(dim=-1).item()
        q_top1 = q_logits.argmax(dim=-1).item()
        top1_match = 1.0 if fp_top1 == q_top1 else 0.0

        results["mse"].append(mse)
        results["cosine_sim"].append(cos_sim)
        results["top10_overlap"].append(top10_overlap)
        results["top1_match"].append(top1_match)

    return {
        "name": quant_name,
        "mse": np.mean(results["mse"]),
        "mse_std": np.std(results["mse"]),
        "cosine_sim": np.mean(results["cosine_sim"]),
        "top10_overlap": np.mean(results["top10_overlap"]),
        "top1_match": np.mean(results["top1_match"]),
    }


def compute_avg_bits(zone_bits, seq_len):
    """Compute average bits for zone-based quantization."""
    zone1_end = seq_len // 3
    zone2_end = 2 * seq_len // 3
    zone3_len = seq_len - zone2_end

    total_bits = (
        zone1_end * zone_bits["early"] +
        (zone2_end - zone1_end) * zone_bits["mid"] +
        zone3_len * zone_bits["late"]
    )
    return total_bits / seq_len


def main():
    print("=" * 70)
    print("Zone-Based KV Quantization: Micro-Validation")
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

    # Test prompts with varying lengths
    prompts = [
        # Short context
        "The quick brown fox jumps over the lazy dog. What is the next word?",
        # Medium context
        """Artificial intelligence has made remarkable progress in recent years.
        Large language models can now understand and generate human-like text.
        This has led to applications in many fields including medicine, law, and education.
        The future of AI remains uncertain but promising.""",
        # Long context
        """The history of artificial intelligence began in the mid-20th century when
        Alan Turing proposed the concept of machine intelligence. In 1950, Turing published his
        seminal paper introducing the Turing Test. The field was officially born at the Dartmouth
        Conference in 1956, where John McCarthy coined the term "artificial intelligence." Early
        AI research focused on symbolic reasoning and problem-solving. The 1980s saw the rise of
        expert systems, while the 1990s brought machine learning advances. The deep learning
        revolution began in 2012 with AlexNet, and large language models emerged in the 2020s.
        Today, AI systems can understand language, recognize images, and even generate creative content.
        The impact of AI on society continues to grow, raising both opportunities and challenges.""",
    ]

    print(f"\nTest prompts: {len(prompts)}")
    for i, p in enumerate(prompts):
        tokens = len(tokenizer.encode(p))
        print(f"  Prompt {i+1}: {tokens} tokens")

    # Define quantization configurations to test
    configs = [
        # Baseline: Uniform quantization
        ("Uniform 3-bit", lambda kv: quantize_kv_uniform(kv, 3), 3.0),
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),

        # Zone-based: V3 hypothesis (late positions need more bits)
        ("Zone 3-4-5 (V3)", lambda kv: quantize_kv_zoned(kv, {"early": 3, "mid": 4, "late": 5}), 4.0),
        ("Zone 2-4-6 (Extreme V3)", lambda kv: quantize_kv_zoned(kv, {"early": 2, "mid": 4, "late": 6}), 4.0),

        # Zone-based: Original Gap 7 hypothesis (early positions need more bits)
        ("Zone 5-4-3 (Gap7)", lambda kv: quantize_kv_zoned(kv, {"early": 5, "mid": 4, "late": 3}), 4.0),

        # Zone-based: Sink token focused (very high early, uniform rest)
        ("Zone 8-4-4 (Sink)", lambda kv: quantize_kv_zoned(kv, {"early": 8, "mid": 4, "late": 4}), 5.33),
    ]

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    results = []
    for name, quant_fn, avg_bits in configs:
        print(f"\nTesting: {name} (avg {avg_bits:.1f} bits)...")
        result = evaluate_kv_quantization(model, tokenizer, prompts, quant_fn, name)
        result["avg_bits"] = avg_bits
        results.append(result)

    # Print comparison table
    print("\n" + "-" * 85)
    print(f"{'Config':<25} | {'Avg Bits':>8} | {'MSE':>12} | {'Cos Sim':>10} | {'Top-10':>8} | {'Top-1':>6}")
    print("-" * 85)

    for r in results:
        print(f"{r['name']:<25} | {r['avg_bits']:>8.1f} | {r['mse']:>12.4f} | "
              f"{r['cosine_sim']:>10.4f} | {r['top10_overlap']:>8.1%} | {r['top1_match']:>6.0%}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Compare zone-based to uniform at same avg bits
    uniform_4bit = next(r for r in results if r["name"] == "Uniform 4-bit")
    zone_v3 = next(r for r in results if r["name"] == "Zone 3-4-5 (V3)")
    zone_gap7 = next(r for r in results if r["name"] == "Zone 5-4-3 (Gap7)")

    print(f"\n1. Zone 3-4-5 (V3) vs Uniform 4-bit:")
    print(f"   MSE improvement: {(1 - zone_v3['mse']/uniform_4bit['mse'])*100:.1f}%")
    print(f"   Cos Sim improvement: {(zone_v3['cosine_sim'] - uniform_4bit['cosine_sim'])*100:.2f}%")

    if zone_v3['mse'] < uniform_4bit['mse']:
        print("   -> V3 hypothesis CONFIRMED: Late positions benefit from more bits")
    else:
        print("   -> V3 hypothesis REJECTED: Zone-based does not beat uniform")

    print(f"\n2. Zone 5-4-3 (Gap7) vs Uniform 4-bit:")
    print(f"   MSE improvement: {(1 - zone_gap7['mse']/uniform_4bit['mse'])*100:.1f}%")
    print(f"   Cos Sim improvement: {(zone_gap7['cosine_sim'] - uniform_4bit['cosine_sim'])*100:.2f}%")

    print(f"\n3. V3 vs Gap7 (head-to-head at same avg bits):")
    if zone_v3['mse'] < zone_gap7['mse']:
        print(f"   -> V3 wins: {(1 - zone_v3['mse']/zone_gap7['mse'])*100:.1f}% lower MSE")
        print("   -> Late positions truly need more precision")
    else:
        print(f"   -> Gap7 wins: {(1 - zone_gap7['mse']/zone_v3['mse'])*100:.1f}% lower MSE")
        print("   -> Early positions (sink tokens) truly need more precision")

    # Check if any zone-based beats uniform 5-bit
    uniform_5bit = next(r for r in results if r["name"] == "Uniform 5-bit")
    print(f"\n4. Best zone-based (4 avg bits) vs Uniform 5-bit:")
    best_zone = min([r for r in results if "Zone" in r["name"] and r["avg_bits"] <= 4.0], key=lambda x: x['mse'])
    print(f"   {best_zone['name']}: MSE={best_zone['mse']:.4f}")
    print(f"   Uniform 5-bit: MSE={uniform_5bit['mse']:.4f}")
    if best_zone['mse'] < uniform_5bit['mse']:
        print("   -> Zone-based at 4 bits BEATS uniform 5-bit! (Memory win)")
    else:
        print("   -> Uniform 5-bit still better (simpler approach wins)")

    print("\n" + "=" * 70)
    print("Micro-Validation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
