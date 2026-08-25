#!/usr/bin/env python3
"""
Validate the optimized PAKV (Sink 20%-5b-4b) implementation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')
from sinq.pakv import PAKVQuantizer, PAKVConfig, create_pakv_quantizer


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


def quantize_uniform(kv_cache, num_bits):
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def evaluate_config(model, tokenizer, prompts, quant_fn):
    device = next(model.parameters()).device
    mse_list, cos_list = [], []

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

                prefix_kv_fp.update(k.clone(), v.clone(), layer_idx)

                # Apply quantization
                prefix_fp = DynamicCache()
                prefix_fp.update(k.clone(), v.clone(), 0)
                q_cache = quant_fn(prefix_fp)
                k_q = q_cache.key_cache[0]
                v_q = q_cache.value_cache[0]

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
    print("Validate Optimized PAKV Implementation")
    print("=" * 70)

    torch.manual_seed(42)
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

    # Test configurations
    print("\n" + "=" * 70)
    print("Testing Configurations")
    print("=" * 70)

    # 1. Default PAKV (new optimized config)
    pakv_optimized = create_pakv_quantizer()  # Default: 20%-5b-4b
    print(f"\nOptimized PAKV config:")
    print(f"  sink_fraction: {pakv_optimized.config.sink_fraction}")
    print(f"  sink_bits: {pakv_optimized.config.sink_bits}")
    print(f"  rest_bits: {pakv_optimized.config.rest_bits}")
    print(f"  avg_bits: {pakv_optimized.config.avg_bits}")

    # 2. Old Sink 6-4-4
    pakv_old = create_pakv_quantizer(sink_fraction=0.1, sink_bits=6, rest_bits=4)
    print(f"\nOld PAKV config (Sink 6-4-4):")
    print(f"  avg_bits: {pakv_old.config.avg_bits}")

    # Evaluate
    configs = [
        ("Uniform 4-bit", lambda kv: quantize_uniform(kv, 4), 4.0),
        ("PAKV Old (10%-6b-4b)", pakv_old.quantize_kv_cache, pakv_old.config.avg_bits),
        ("PAKV Optimized (20%-5b-4b)", pakv_optimized.quantize_kv_cache, pakv_optimized.config.avg_bits),
        ("Uniform 5-bit", lambda kv: quantize_uniform(kv, 5), 5.0),
    ]

    results = []
    for name, quant_fn, avg_bits in configs:
        mse, cos = evaluate_config(model, tokenizer, test_prompts, quant_fn)
        results.append({"name": name, "avg_bits": avg_bits, "mse": mse, "cos_sim": cos})
        print(f"\n  {name}: MSE = {mse:.4f}, CosSim = {cos:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<30} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Compare
    optimized = next(r for r in results if "Optimized" in r["name"])
    old = next(r for r in results if "Old" in r["name"])

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if optimized["mse"] < old["mse"]:
        improvement = (old["mse"] - optimized["mse"]) / old["mse"] * 100
        print(f"\n  PASS: Optimized PAKV is {improvement:.1f}% better than old Sink 6-4-4")
    else:
        print(f"\n  FAIL: Optimized PAKV is worse than old config")

    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
