#!/usr/bin/env python3
"""
Test KV cache quantization at higher bit-widths.

The 4-5 bit quantization is clearly too aggressive.
Let's find the minimum bits needed for acceptable PPL.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def quantize_tensor(x, num_bits):
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min
    if range_val == 0:
        return x
    qmax = 2**num_bits - 1
    scale = range_val / qmax
    return torch.round((x - x_min) / scale) * scale + x_min


def quantize_kv_uniform(kv_cache, num_bits):
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def quantize_kv_zone(kv_cache, sink_fraction, sink_bits, rest_bits):
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()
        seq_len = k.shape[2]
        sink_end = max(1, int(seq_len * sink_fraction))

        k[:, :, :sink_end, :] = quantize_tensor(k[:, :, :sink_end, :], sink_bits)
        v[:, :, :sink_end, :] = quantize_tensor(v[:, :, :sink_end, :], sink_bits)
        if sink_end < seq_len:
            k[:, :, sink_end:, :] = quantize_tensor(k[:, :, sink_end:, :], rest_bits)
            v[:, :, sink_end:, :] = quantize_tensor(v[:, :, sink_end:, :], rest_bits)

        quantized.update(k, v, layer_idx)
    return quantized


def evaluate_ppl_with_kv_quant(model, tokenizer, dataset, quant_fn, max_samples=20):
    device = next(model.parameters()).device

    all_text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0][:max_samples])
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0
    chunk_size = 128
    seq_len = input_ids.size(1)

    with torch.no_grad():
        for start in tqdm(range(0, seq_len - chunk_size, chunk_size), desc="Evaluating", leave=False):
            end = start + chunk_size
            chunk = input_ids[:, start:end]

            outputs = model(chunk, use_cache=True)
            kv_cache = outputs.past_key_values
            q_kv = quant_fn(kv_cache)

            if end + chunk_size <= seq_len:
                next_chunk = input_ids[:, end:end+chunk_size]
                next_outputs = model(next_chunk, past_key_values=q_kv, use_cache=False)
                logits = next_outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = next_chunk[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )

                total_nll += loss.item()
                total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return float('inf')

    return np.exp(total_nll / total_tokens)


def main():
    print("=" * 70)
    print("Test KV Cache Quantization at Higher Bit-Widths")
    print("=" * 70)

    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("\nLoading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = [s for s in dataset if len(s["text"].strip()) > 50]

    # Test 1: Uniform quantization at various bit-widths
    print("\n" + "=" * 70)
    print("TEST 1: Uniform KV Quantization")
    print("=" * 70)

    for bits in [4, 5, 6, 7, 8]:
        print(f"\nUniform {bits}-bit...")
        ppl = evaluate_ppl_with_kv_quant(
            model, tokenizer, dataset,
            lambda kv, b=bits: quantize_kv_uniform(kv, b),
            max_samples=15
        )
        print(f"  PPL = {ppl:.2f}")

    # Test 2: Zone-based at higher bits
    print("\n" + "=" * 70)
    print("TEST 2: Zone-Based (PAKV) at Higher Bits")
    print("=" * 70)

    configs = [
        ("20%-8b-6b", 0.2, 8, 6),
        ("20%-8b-7b", 0.2, 8, 7),
        ("10%-8b-6b", 0.1, 8, 6),
        ("10%-8b-7b", 0.1, 8, 7),
        ("20%-7b-6b", 0.2, 7, 6),
    ]

    for name, sink_frac, sink_bits, rest_bits in configs:
        avg_bits = sink_frac * sink_bits + (1 - sink_frac) * rest_bits
        print(f"\nPAKV {name} (avg={avg_bits:.2f} bits)...")
        ppl = evaluate_ppl_with_kv_quant(
            model, tokenizer, dataset,
            lambda kv, sf=sink_frac, sb=sink_bits, rb=rest_bits: quantize_kv_zone(kv, sf, sb, rb),
            max_samples=15
        )
        print(f"  PPL = {ppl:.2f}")

    # FP16 baseline
    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)
    print("\nFP16 (no quant)...")
    ppl = evaluate_ppl_with_kv_quant(
        model, tokenizer, dataset,
        lambda kv: kv,
        max_samples=15
    )
    print(f"  PPL = {ppl:.2f}")


if __name__ == "__main__":
    main()
