#!/usr/bin/env python3
"""
Validate PAKV configurations with PERPLEXITY (not MSE).

CRITICAL: MSE improvements do NOT necessarily translate to PPL improvements.
This was learned in Cycle 6 with ECAQ - 48% MSE improvement gave 0% PPL improvement.

We must validate on the END METRIC (perplexity) to claim any real improvement.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def quantize_tensor(x, num_bits):
    """Simple min-max quantization."""
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min
    if range_val == 0:
        return x
    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def quantize_kv_uniform(kv_cache, num_bits):
    """Uniform quantization of KV cache."""
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def quantize_kv_zone(kv_cache, sink_fraction, sink_bits, rest_bits):
    """Zone-based quantization (PAKV style)."""
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()
        seq_len = k.shape[2]
        sink_end = max(1, int(seq_len * sink_fraction))

        # Quantize sink zone
        k[:, :, :sink_end, :] = quantize_tensor(k[:, :, :sink_end, :], sink_bits)
        v[:, :, :sink_end, :] = quantize_tensor(v[:, :, :sink_end, :], sink_bits)

        # Quantize rest
        if sink_end < seq_len:
            k[:, :, sink_end:, :] = quantize_tensor(k[:, :, sink_end:, :], rest_bits)
            v[:, :, sink_end:, :] = quantize_tensor(v[:, :, sink_end:, :], rest_bits)

        quantized.update(k, v, layer_idx)
    return quantized


def evaluate_perplexity(model, tokenizer, dataset, quant_fn, max_samples=100, seq_len=256):
    """
    Evaluate perplexity with KV cache quantization.

    Method:
    1. For each sample, encode the full sequence
    2. At each position, quantize the KV cache up to that position
    3. Compute NLL for predicting the next token
    4. Average NLL and compute PPL = exp(avg_nll)
    """
    device = next(model.parameters()).device

    # Concatenate dataset text
    all_text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0][:max_samples])

    # Tokenize
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=seq_len * max_samples)
    input_ids = encodings.input_ids[0].to(device)
    total_len = min(len(input_ids), seq_len * 50)  # Limit total tokens

    nlls = []
    stride = seq_len // 2  # 50% overlap

    for start in tqdm(range(0, total_len - seq_len, stride), desc="Evaluating"):
        end = start + seq_len
        window = input_ids[start:end].unsqueeze(0)

        with torch.no_grad():
            # Forward pass to get KV cache
            outputs = model(window, use_cache=True)
            fp_kv = outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(fp_kv)

            # For each position in second half of window, compute NLL
            # using quantized KV cache
            for pos in range(seq_len // 2, seq_len - 1):
                # Get prefix KV (positions 0 to pos)
                prefix_kv = DynamicCache()
                for layer_idx in range(len(q_kv)):
                    k = q_kv.key_cache[layer_idx][:, :, :pos+1, :]
                    v = q_kv.value_cache[layer_idx][:, :, :pos+1, :]
                    prefix_kv.update(k.clone(), v.clone(), layer_idx)

                # Predict next token
                current_token = window[:, pos:pos+1]
                out = model(input_ids=current_token, past_key_values=prefix_kv, use_cache=False)
                logits = out.logits[0, -1, :]

                # Compute NLL
                target = window[0, pos + 1].item()
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs[target].item()

                if not np.isnan(nll) and not np.isinf(nll) and nll < 100:
                    nlls.append(nll)

    if not nlls:
        return float('inf')

    avg_nll = np.mean(nlls)
    ppl = np.exp(avg_nll)
    return ppl


def evaluate_perplexity_simple(model, tokenizer, dataset, quant_fn, max_samples=50, context_len=128):
    """
    Simpler perplexity evaluation: encode context, quantize KV, predict next token.
    """
    device = next(model.parameters()).device

    nlls = []
    samples_processed = 0

    for sample in tqdm(dataset, desc="Evaluating", total=max_samples):
        if samples_processed >= max_samples:
            break

        text = sample["text"].strip()
        if len(text) < 50:
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_len + 50)
        input_ids = inputs.input_ids[0].to(device)

        if len(input_ids) < context_len + 10:
            continue

        with torch.no_grad():
            # Use first context_len tokens as context
            context = input_ids[:context_len].unsqueeze(0)

            # Get FP KV cache for context
            outputs = model(context, use_cache=True)
            fp_kv = outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(fp_kv)

            # Predict next tokens using quantized KV
            for i in range(min(20, len(input_ids) - context_len - 1)):
                target_pos = context_len + i
                target_token = input_ids[target_pos].item()

                # Get prefix of quantized KV
                prefix_kv = DynamicCache()
                for layer_idx in range(len(q_kv)):
                    k = q_kv.key_cache[layer_idx][:, :, :context_len+i, :]
                    v = q_kv.value_cache[layer_idx][:, :, :context_len+i, :]
                    prefix_kv.update(k.clone(), v.clone(), layer_idx)

                # Forward with last token
                if i == 0:
                    last_token = context[:, -1:]
                else:
                    last_token = input_ids[target_pos-1:target_pos].unsqueeze(0)

                out = model(input_ids=last_token, past_key_values=prefix_kv, use_cache=False)
                logits = out.logits[0, -1, :]

                # NLL
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs[target_token].item()

                if not np.isnan(nll) and not np.isinf(nll) and nll < 100:
                    nlls.append(nll)

        samples_processed += 1

    if not nlls:
        return float('inf')

    avg_nll = np.mean(nlls)
    ppl = np.exp(avg_nll)
    return ppl


def main():
    print("=" * 70)
    print("PAKV Perplexity Validation")
    print("CRITICAL: MSE != PPL. Must validate on end metric.")
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

    # Load WikiText-2
    print("\nLoading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = [s for s in dataset if len(s["text"].strip()) > 50]
    print(f"Loaded {len(dataset)} samples")

    # Configurations to test
    configs = [
        ("FP16 (no quant)", lambda kv: kv, 16.0),
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),
        ("PAKV Old (10%-6b-4b)", lambda kv: quantize_kv_zone(kv, 0.1, 6, 4), 4.2),
        ("PAKV New (20%-5b-4b)", lambda kv: quantize_kv_zone(kv, 0.2, 5, 4), 4.2),
        ("PAKV (15%-5b-4b)", lambda kv: quantize_kv_zone(kv, 0.15, 5, 4), 4.15),
        ("PAKV (20%-6b-4b)", lambda kv: quantize_kv_zone(kv, 0.2, 6, 4), 4.4),
    ]

    print("\n" + "=" * 70)
    print("Evaluating Perplexity on WikiText-2")
    print("=" * 70)

    results = []
    for name, quant_fn, avg_bits in configs:
        print(f"\nEvaluating: {name}...")
        ppl = evaluate_perplexity_simple(model, tokenizer, dataset, quant_fn,
                                         max_samples=30, context_len=128)
        results.append({
            "name": name,
            "avg_bits": avg_bits,
            "ppl": ppl,
        })
        print(f"  PPL = {ppl:.2f}")

    # Results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<25} | {'Avg Bits':>10} | {'Perplexity':>12}")
    print("-" * 55)
    for r in results:
        ppl_str = f"{r['ppl']:.2f}" if r['ppl'] < 1000 else "inf"
        print(f"{r['name']:<25} | {r['avg_bits']:>10.1f} | {ppl_str:>12}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    fp_ppl = next(r for r in results if "FP16" in r["name"])["ppl"]
    uniform4 = next(r for r in results if "Uniform 4" in r["name"])
    uniform5 = next(r for r in results if "Uniform 5" in r["name"])
    pakv_old = next(r for r in results if "Old" in r["name"])
    pakv_new = next(r for r in results if "New" in r["name"])

    print(f"\nFP16 baseline PPL: {fp_ppl:.2f}")

    print(f"\n1. PAKV New vs PAKV Old (same avg bits = 4.2):")
    print(f"   PAKV Old PPL: {pakv_old['ppl']:.2f}")
    print(f"   PAKV New PPL: {pakv_new['ppl']:.2f}")
    if pakv_new['ppl'] < pakv_old['ppl']:
        improvement = (pakv_old['ppl'] - pakv_new['ppl']) / pakv_old['ppl'] * 100
        print(f"   -> PAKV New is {improvement:.1f}% BETTER")
    elif pakv_new['ppl'] > pakv_old['ppl']:
        degradation = (pakv_new['ppl'] - pakv_old['ppl']) / pakv_old['ppl'] * 100
        print(f"   -> PAKV New is {degradation:.1f}% WORSE")
    else:
        print(f"   -> Results are EQUIVALENT")

    print(f"\n2. PAKV vs Uniform at similar bits:")
    print(f"   Uniform 4-bit PPL: {uniform4['ppl']:.2f}")
    print(f"   PAKV Old (4.2 bits) PPL: {pakv_old['ppl']:.2f}")
    print(f"   PAKV New (4.2 bits) PPL: {pakv_new['ppl']:.2f}")
    print(f"   Uniform 5-bit PPL: {uniform5['ppl']:.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if pakv_new['ppl'] < pakv_old['ppl'] and pakv_new['ppl'] < uniform5['ppl']:
        print("\n  SUCCESS: PAKV New shows PPL improvement over both baselines!")
    elif pakv_new['ppl'] >= pakv_old['ppl']:
        print("\n  FAILURE: PAKV New does NOT improve PPL over PAKV Old.")
        print("  MSE improvements did NOT translate to PPL improvements.")
    else:
        print("\n  PARTIAL: PAKV New beats uniform but not PAKV Old.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
