#!/usr/bin/env python3
"""
Debug the PPL evaluation - something is fundamentally wrong.

FP16 baseline PPL of 597 is very high for Qwen 1.7B on WikiText-2.
Expected PPL should be around 10-20.

Let's check the standard evaluation method.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def evaluate_ppl_standard(model, tokenizer, dataset, max_length=1024):
    """Standard PPL evaluation without KV cache manipulation."""
    device = next(model.parameters()).device

    # Concatenate all text
    all_text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0])

    # Tokenize
    encodings = tokenizer(all_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    seq_len = input_ids.size(1)

    # Sliding window evaluation
    stride = max_length // 2

    for begin_loc in tqdm(range(0, min(seq_len, 10000), stride), desc="Evaluating"):
        end_loc = min(begin_loc + max_length, seq_len)
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_len = input_chunk.size(1)

        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
            nll = outputs.loss.item()

            if not np.isnan(nll) and not np.isinf(nll):
                nlls.append(nll)

        if end_loc >= seq_len:
            break

    ppl = np.exp(np.mean(nlls))
    return ppl


def evaluate_ppl_with_kv_quant(model, tokenizer, dataset, quant_fn, max_samples=20):
    """
    Evaluate PPL with KV cache quantization.

    Proper method:
    1. Process sequence in chunks
    2. For each chunk, use model.forward with labels to get loss
    3. KV cache is used internally by the model
    """
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device

    # Concatenate text
    all_text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0][:max_samples])
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0

    chunk_size = 128
    seq_len = input_ids.size(1)

    with torch.no_grad():
        for start in tqdm(range(0, seq_len - chunk_size, chunk_size), desc="Evaluating"):
            end = start + chunk_size
            chunk = input_ids[:, start:end]

            # Forward to get KV cache
            outputs = model(chunk, use_cache=True)
            kv_cache = outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(kv_cache)

            # Now predict the NEXT chunk using quantized KV
            if end + chunk_size <= seq_len:
                next_chunk = input_ids[:, end:end+chunk_size]

                # Get logits for next chunk using quantized KV
                next_outputs = model(next_chunk, past_key_values=q_kv, use_cache=False)
                logits = next_outputs.logits

                # Compute cross-entropy loss
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

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    return ppl


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
    from transformers.cache_utils import DynamicCache
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def quantize_kv_zone(kv_cache, sink_fraction, sink_bits, rest_bits):
    from transformers.cache_utils import DynamicCache
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


def main():
    print("=" * 70)
    print("Debug PPL Evaluation")
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

    # Load WikiText-2
    print("\nLoading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = [s for s in dataset if len(s["text"].strip()) > 50]
    print(f"Loaded {len(dataset)} samples")

    # Test 1: Standard PPL (no KV manipulation)
    print("\n" + "=" * 70)
    print("TEST 1: Standard PPL Evaluation (No KV Manipulation)")
    print("=" * 70)

    ppl_standard = evaluate_ppl_standard(model, tokenizer, dataset, max_length=512)
    print(f"\nStandard PPL: {ppl_standard:.2f}")

    # Test 2: PPL with KV quantization (proper method)
    print("\n" + "=" * 70)
    print("TEST 2: PPL with KV Quantization")
    print("=" * 70)

    configs = [
        ("FP16 (no quant)", lambda kv: kv),
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4)),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5)),
        ("PAKV (10%-6b-4b)", lambda kv: quantize_kv_zone(kv, 0.1, 6, 4)),
        ("PAKV (20%-5b-4b)", lambda kv: quantize_kv_zone(kv, 0.2, 5, 4)),
    ]

    for name, quant_fn in configs:
        print(f"\nEvaluating: {name}...")
        ppl = evaluate_ppl_with_kv_quant(model, tokenizer, dataset, quant_fn, max_samples=20)
        print(f"  PPL = {ppl:.2f}")


if __name__ == "__main__":
    main()
