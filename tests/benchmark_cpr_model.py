#!/usr/bin/env python3
"""
Benchmark CPR quantization on a full model.

This script:
1. Loads a pretrained model
2. Replaces linear layers with CPRLinear
3. Evaluates perplexity on WikiText-2
4. Measures inference throughput

Usage:
    python benchmark_cpr_model.py --model meta-llama/Llama-3.2-1B
"""

import argparse
import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from sinq.cprlinear import CPRLinear, cpr_quant_config


def replace_linear_with_cpr(model, high_frac=0.25, high_bits=6, low_bits=5,
                            tile_size=128, skip_patterns=None):
    """
    Replace all nn.Linear layers in model with CPRLinear.

    Args:
        model: Model to quantize
        high_frac: Fraction of columns at high precision
        high_bits: High precision bit width
        low_bits: Low precision bit width
        tile_size: Tile size for quantization
        skip_patterns: List of layer name patterns to skip
    """
    if skip_patterns is None:
        skip_patterns = ['lm_head', 'embed']

    replaced = 0
    skipped = 0

    for name, module in model.named_modules():
        # Check if this is a linear layer
        if not isinstance(module, nn.Linear):
            continue

        # Check if we should skip this layer
        should_skip = any(pat in name for pat in skip_patterns)
        if should_skip:
            skipped += 1
            continue

        # Get parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with CPRLinear
        cpr_layer = CPRLinear.from_linear(
            module,
            high_frac=high_frac,
            high_bits=high_bits,
            low_bits=low_bits,
            tile_size=tile_size,
        )
        setattr(parent, parts[-1], cpr_layer)
        replaced += 1

    return replaced, skipped


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext",
                        dataset_config="wikitext-2-raw-v1", max_samples=None):
    """Evaluate model perplexity on a dataset."""
    print(f"Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="test")

    # Concatenate all text
    text = "\n\n".join(dataset["text"])

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    # Calculate perplexity in chunks
    seq_len = input_ids.size(1)
    max_length = 2048
    stride = max_length // 2

    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc="Evaluating"):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        if target_len <= 0:
            break

        input_chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
            neg_log_likelihood = outputs.loss * target_len

        nlls.append(neg_log_likelihood)
        prev_end = end

        if max_samples and len(nlls) >= max_samples:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end)
    return ppl.item()


def benchmark_throughput(model, tokenizer, batch_size=1, seq_len=128,
                         n_iters=50, warmup=10):
    """Benchmark inference throughput."""
    device = next(model.parameters()).device

    # Generate random input
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking {n_iters} iterations...")
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (batch_size * seq_len * n_iters) / elapsed
    ms_per_token = elapsed * 1000 / (batch_size * seq_len * n_iters)

    return {
        'tokens_per_sec': tokens_per_sec,
        'ms_per_token': ms_per_token,
        'total_time': elapsed,
    }


def count_parameters_and_bits(model):
    """Count total parameters and effective bits."""
    total_params = 0
    total_bits = 0

    for name, module in model.named_modules():
        if isinstance(module, CPRLinear):
            params = module.in_features * module.out_features
            bits = params * module.compute_avg_bits()
            total_params += params
            total_bits += bits
        elif isinstance(module, nn.Linear):
            params = module.weight.numel()
            bits = params * 16  # FP16
            total_params += params
            total_bits += bits

    avg_bits = total_bits / total_params if total_params > 0 else 0
    return total_params, avg_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model to quantize")
    parser.add_argument("--high_frac", type=float, default=0.25,
                        help="Fraction of high-precision columns")
    parser.add_argument("--high_bits", type=int, default=6,
                        help="High precision bit width")
    parser.add_argument("--low_bits", type=int, default=5,
                        help="Low precision bit width")
    parser.add_argument("--tile_size", type=int, default=128,
                        help="Tile size for quantization")
    parser.add_argument("--cache_weights", action="store_true",
                        help="Cache dequantized weights for faster inference")
    parser.add_argument("--skip_ppl", action="store_true",
                        help="Skip perplexity evaluation")
    parser.add_argument("--max_ppl_samples", type=int, default=None,
                        help="Max samples for perplexity evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("CPR-SINQ Model Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"CPR Config: {args.high_frac*100:.0f}% @ {args.high_bits}-bit, "
          f"{(1-args.high_frac)*100:.0f}% @ {args.low_bits}-bit")

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    # Evaluate baseline
    if not args.skip_ppl:
        print("\n--- Baseline (FP16) ---")
        ppl_baseline = evaluate_perplexity(
            model, tokenizer, max_samples=args.max_ppl_samples
        )
        print(f"Baseline Perplexity: {ppl_baseline:.2f}")

    throughput_baseline = benchmark_throughput(model, tokenizer)
    print(f"Baseline Throughput: {throughput_baseline['tokens_per_sec']:.0f} tokens/sec")

    # Quantize with CPR
    print("\n--- Quantizing with CPR ---")
    replaced, skipped = replace_linear_with_cpr(
        model,
        high_frac=args.high_frac,
        high_bits=args.high_bits,
        low_bits=args.low_bits,
        tile_size=args.tile_size,
    )
    print(f"Replaced {replaced} layers, skipped {skipped} layers")

    # Count parameters
    total_params, avg_bits = count_parameters_and_bits(model)
    print(f"Total linear params: {total_params/1e6:.1f}M")
    print(f"Average bits: {avg_bits:.2f}")

    # Cache weights if requested
    if args.cache_weights:
        print("Caching dequantized weights...")
        for module in model.modules():
            if isinstance(module, CPRLinear):
                module.cache_weights()

    # Evaluate CPR
    if not args.skip_ppl:
        print("\n--- CPR Quantized ---")
        ppl_cpr = evaluate_perplexity(
            model, tokenizer, max_samples=args.max_ppl_samples
        )
        print(f"CPR Perplexity: {ppl_cpr:.2f}")
        print(f"Perplexity Increase: {ppl_cpr - ppl_baseline:.2f} ({(ppl_cpr/ppl_baseline - 1)*100:.1f}%)")

    throughput_cpr = benchmark_throughput(model, tokenizer)
    print(f"CPR Throughput: {throughput_cpr['tokens_per_sec']:.0f} tokens/sec")
    print(f"Throughput Ratio: {throughput_cpr['tokens_per_sec']/throughput_baseline['tokens_per_sec']*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Average Bits: {avg_bits:.2f}")
    if not args.skip_ppl:
        print(f"Baseline PPL: {ppl_baseline:.2f}")
        print(f"CPR PPL: {ppl_cpr:.2f}")
    print(f"Baseline Throughput: {throughput_baseline['tokens_per_sec']:.0f} tokens/sec")
    print(f"CPR Throughput: {throughput_cpr['tokens_per_sec']:.0f} tokens/sec")
    print(f"Throughput Retention: {throughput_cpr['tokens_per_sec']/throughput_baseline['tokens_per_sec']*100:.1f}%")


if __name__ == "__main__":
    main()
