#!/usr/bin/env python3
"""
Phase 6: Evaluate PAKV on WikiText-2 Perplexity (Fixed Version)

Proper evaluation approach:
1. Concatenate WikiText samples to form long sequences
2. Use sliding window evaluation
3. Quantize KV cache at each window
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/workspace/SINQ')
from sinq.pakv import PAKVQuantizer, PAKVConfig


def quantize_tensor(x, num_bits, per_channel=True):
    """Simple min-max quantization."""
    if per_channel:
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
    else:
        x_min = x.min()
        x_max = x.max()

    range_val = x_max - x_min
    range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)

    qmax = 2**num_bits - 1
    scale = range_val / qmax
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


def evaluate_ppl_sliding_window(
    model,
    tokenizer,
    dataset,
    quant_fn,
    window_size=256,
    stride=128,
    max_tokens=5000,
):
    """
    Evaluate perplexity using sliding window approach with KV quantization.

    For each window:
    1. Process first half as context
    2. Quantize KV cache
    3. Evaluate next-token prediction on second half
    """
    device = next(model.parameters()).device

    # Concatenate all text
    all_text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0])

    # Tokenize
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encodings.input_ids[0].to(device)
    total_len = len(input_ids)

    print(f"  Total tokens: {total_len}")

    nlls = []

    # Slide over the sequence
    for start in tqdm(range(0, total_len - window_size, stride), desc="Windows"):
        end = start + window_size

        # Get window
        window = input_ids[start:end].unsqueeze(0)

        with torch.no_grad():
            # Forward pass to get logits and KV cache
            outputs = model(window, use_cache=True)
            logits = outputs.logits  # (1, window_size, vocab)
            fp_kv = outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(fp_kv)

            # Now measure: how well does the quantized KV predict the next tokens?
            # We use the last position's logits vs the next token

            # Get prediction for position beyond the window
            if end < total_len:
                # The logits from the last position predict the next token
                last_logits = logits[0, -1, :]  # (vocab,)
                next_token = input_ids[end].item()

                # Compute NLL
                log_probs = F.log_softmax(last_logits, dim=-1)
                nll = -log_probs[next_token].item()

                # Also get quantized prediction
                # Use quantized KV to predict
                dummy = window[:, -1:]
                q_outputs = model(input_ids=dummy, past_key_values=q_kv, use_cache=False)
                q_logits = q_outputs.logits[0, -1, :]
                q_log_probs = F.log_softmax(q_logits, dim=-1)
                q_nll = -q_log_probs[next_token].item()

                if not np.isinf(q_nll) and not np.isnan(q_nll):
                    nlls.append(q_nll)

    if not nlls:
        return float('inf')

    avg_nll = np.mean(nlls)
    ppl = np.exp(avg_nll)

    return ppl


def evaluate_ppl_per_token(
    model,
    tokenizer,
    dataset,
    quant_fn,
    max_tokens=2000,
    context_size=128,
):
    """
    Token-by-token PPL evaluation with KV cache quantization.

    For each token position:
    1. Use previous tokens as context
    2. Quantize the KV cache
    3. Predict the current token
    4. Compute NLL
    """
    device = next(model.parameters()).device

    # Concatenate text
    all_text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0])

    # Tokenize
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encodings.input_ids[0].to(device)
    total_len = len(input_ids)

    print(f"  Total tokens: {total_len}")

    nlls = []

    # For each position, use context_size previous tokens
    for pos in tqdm(range(context_size, min(total_len, max_tokens) - 1), desc="Tokens"):
        context_start = max(0, pos - context_size)
        context = input_ids[context_start:pos].unsqueeze(0)
        target = input_ids[pos].item()

        with torch.no_grad():
            # Get KV cache for context
            outputs = model(context, use_cache=True)
            fp_kv = outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(fp_kv)

            # Predict next token with quantized KV
            last_token = context[:, -1:]
            q_outputs = model(input_ids=last_token, past_key_values=q_kv, use_cache=False)
            logits = q_outputs.logits[0, -1, :]

            # Compute NLL
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs[target].item()

            if not np.isinf(nll) and not np.isnan(nll) and nll < 100:
                nlls.append(nll)

    if not nlls:
        return float('inf')

    avg_nll = np.mean(nlls)
    ppl = np.exp(avg_nll)

    return ppl


def main():
    print("=" * 70)
    print("Phase 6: PAKV WikiText-2 Perplexity Evaluation (v2)")
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

    # Load WikiText-2
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = [s for s in dataset if len(s["text"].strip()) > 0]
    print(f"Loaded {len(dataset)} non-empty samples")

    # Define quantization configurations
    pakv_config = PAKVConfig(sink_fraction=0.1, sink_bits=6, rest_bits=4)
    pakv_quantizer = PAKVQuantizer(pakv_config)

    configs = [
        ("FP (no quant)", lambda kv: kv, 16.0),
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("PAKV 6-4", pakv_quantizer.quantize_kv_cache, 4.2),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),
    ]

    print("\n" + "=" * 70)
    print("Evaluating Perplexity (per-token evaluation)")
    print("=" * 70)

    results = []

    for name, quant_fn, avg_bits in configs:
        print(f"\nEvaluating: {name} ({avg_bits} bits)...")
        ppl = evaluate_ppl_per_token(
            model, tokenizer, dataset, quant_fn,
            max_tokens=1000,  # Evaluate on first 1000 tokens
            context_size=128,  # Use 128 tokens of context
        )
        results.append({
            "name": name,
            "avg_bits": avg_bits,
            "ppl": ppl,
        })
        print(f"  PPL = {ppl:.2f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<20} | {'Avg Bits':>10} | {'Perplexity':>12}")
    print("-" * 50)
    for r in results:
        ppl_str = f"{r['ppl']:.2f}" if r['ppl'] < float('inf') else "inf"
        print(f"{r['name']:<20} | {r['avg_bits']:>10.1f} | {ppl_str:>12}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    fp_ppl = next(r for r in results if "FP" in r["name"])["ppl"]
    uniform_4bit = next(r for r in results if r["name"] == "Uniform 4-bit")
    pakv = next(r for r in results if r["name"] == "PAKV 6-4")
    uniform_5bit = next(r for r in results if r["name"] == "Uniform 5-bit")

    if all(r['ppl'] < float('inf') for r in results):
        print(f"\n1. PAKV vs Uniform 4-bit:")
        print(f"   PAKV PPL: {pakv['ppl']:.2f}")
        print(f"   Uniform 4-bit PPL: {uniform_4bit['ppl']:.2f}")
        if pakv['ppl'] < uniform_4bit['ppl']:
            improvement = (uniform_4bit['ppl'] - pakv['ppl']) / uniform_4bit['ppl'] * 100
            print(f"   -> PAKV is {improvement:.1f}% BETTER (lower PPL)")
        else:
            degradation = (pakv['ppl'] - uniform_4bit['ppl']) / uniform_4bit['ppl'] * 100
            print(f"   -> PAKV is {degradation:.1f}% WORSE")

        print(f"\n2. PAKV vs Uniform 5-bit:")
        print(f"   PAKV PPL: {pakv['ppl']:.2f} (4.2 bits)")
        print(f"   Uniform 5-bit PPL: {uniform_5bit['ppl']:.2f} (5.0 bits)")
        if pakv['ppl'] <= uniform_5bit['ppl']:
            memory_savings = (5.0 - 4.2) / 5.0 * 100
            print(f"   -> PAKV MATCHES/BEATS Uniform 5-bit with {memory_savings:.0f}% memory savings!")
            print("   -> PUBLICATION-WORTHY RESULT")
        else:
            gap = pakv['ppl'] - uniform_5bit['ppl']
            print(f"   -> PAKV is {gap:.2f} PPL higher than Uniform 5-bit")

        print(f"\n3. Distance from FP baseline:")
        print(f"   FP PPL: {fp_ppl:.2f}")
        print(f"   PAKV gap: +{pakv['ppl'] - fp_ppl:.2f}")
        print(f"   Uniform 4-bit gap: +{uniform_4bit['ppl'] - fp_ppl:.2f}")
    else:
        print("\nSome configurations returned inf PPL. Check evaluation.")

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
