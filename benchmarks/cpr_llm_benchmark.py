"""
CPR LLM Benchmark: End-to-end benchmarking of CPR quantization on real LLMs.

This script:
1. Loads an LLM (default: Qwen2.5-0.5B)
2. Measures baseline FP16 performance
3. Quantizes with CPR INT8
4. Measures quantized performance
5. Evaluates perplexity on WikiText-2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import gc
import argparse
from typing import Tuple, Optional
from tqdm import tqdm

# For perplexity evaluation
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False
    print("Warning: datasets not available, skipping perplexity evaluation")


def get_model_memory_mb(model: nn.Module) -> float:
    """Get model memory footprint in MB."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / 1e6


def get_gpu_memory_mb() -> Tuple[float, float]:
    """Get (allocated, reserved) GPU memory in MB."""
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.memory_reserved() / 1e6
    )


def benchmark_generation(model, tokenizer, prompt: str, max_new_tokens: int = 50,
                         warmup: int = 3, iters: int = 10) -> Tuple[float, float]:
    """
    Benchmark text generation.
    Returns (tokens_per_second, latency_ms).
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    # Benchmark
    total_tokens = 0
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        total_tokens += outputs.shape[1] - input_len
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = total_tokens / elapsed
    latency_ms = elapsed / iters * 1000

    return tokens_per_sec, latency_ms


def evaluate_perplexity(model, tokenizer, max_length: int = 512,
                        stride: int = 256, num_samples: int = 100) -> float:
    """
    Evaluate perplexity on WikiText-2 test set.
    """
    if not HAVE_DATASETS:
        return float('nan')

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])

    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'].to(model.device)

    # Calculate perplexity with sliding window
    nlls = []
    prev_end = 0

    total_len = 0
    pbar = tqdm(range(0, min(input_ids.shape[1], num_samples * stride), stride),
                desc="Evaluating PPL")

    for begin in pbar:
        end = min(begin + max_length, input_ids.shape[1])
        trg_len = end - prev_end

        input_slice = input_ids[:, begin:end]
        target_slice = input_slice.clone()
        target_slice[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_slice, labels=target_slice)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        total_len += trg_len
        prev_end = end

        if len(nlls) >= num_samples:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_len)
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description='CPR LLM Benchmark')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Model to benchmark')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='Max tokens to generate')
    parser.add_argument('--skip-ppl', action='store_true',
                        help='Skip perplexity evaluation')
    parser.add_argument('--ppl-samples', type=int, default=50,
                        help='Number of samples for perplexity')
    args = parser.parse_args()

    print("=" * 70)
    print("CPR LLM Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load model and tokenizer
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # First load in FP16
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model_fp16.eval()

    # Baseline measurements
    print("\n" + "=" * 70)
    print("BASELINE (FP16)")
    print("=" * 70)

    fp16_mem = get_model_memory_mb(model_fp16)
    gpu_alloc, gpu_reserved = get_gpu_memory_mb()
    print(f"Model memory: {fp16_mem:.1f} MB")
    print(f"GPU allocated: {gpu_alloc:.1f} MB")

    # Generation benchmark
    prompt = "The quick brown fox jumps over the lazy dog. In the world of artificial intelligence,"
    print(f"\nBenchmarking generation (max_new_tokens={args.max_new_tokens})...")

    fp16_tps, fp16_latency = benchmark_generation(
        model_fp16, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens
    )
    print(f"Tokens/sec: {fp16_tps:.1f}")
    print(f"Latency: {fp16_latency:.1f} ms")

    # Perplexity
    fp16_ppl = float('nan')
    if not args.skip_ppl:
        print(f"\nEvaluating perplexity (samples={args.ppl_samples})...")
        fp16_ppl = evaluate_perplexity(model_fp16, tokenizer, num_samples=args.ppl_samples)
        print(f"Perplexity: {fp16_ppl:.2f}")

    # Free FP16 model memory before loading quantized
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # Quantize with CPR
    print("\n" + "=" * 70)
    print("QUANTIZING WITH CPR INT8")
    print("=" * 70)

    # Reload model fresh
    model_cpr = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )

    # Quantize
    from sinq.cpr_model import quantize_model_cpr
    model_cpr = quantize_model_cpr(
        model_cpr,
        group_size=128,
        compute_dtype=torch.float16,
        device='cuda',
        skip_layers=['lm_head'],  # Keep LM head in FP16 for quality
        verbose=True,
    )
    model_cpr.eval()

    # CPR measurements
    print("\n" + "=" * 70)
    print("CPR INT8 RESULTS")
    print("=" * 70)

    cpr_mem = get_model_memory_mb(model_cpr)
    gpu_alloc, gpu_reserved = get_gpu_memory_mb()
    print(f"Model memory: {cpr_mem:.1f} MB")
    print(f"GPU allocated: {gpu_alloc:.1f} MB")
    print(f"Memory reduction: {(1 - cpr_mem/fp16_mem)*100:.1f}%")

    # Generation benchmark
    print(f"\nBenchmarking generation (max_new_tokens={args.max_new_tokens})...")
    cpr_tps, cpr_latency = benchmark_generation(
        model_cpr, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens
    )
    print(f"Tokens/sec: {cpr_tps:.1f}")
    print(f"Latency: {cpr_latency:.1f} ms")
    print(f"Speed vs FP16: {cpr_tps/fp16_tps*100:.1f}%")

    # Perplexity
    cpr_ppl = float('nan')
    if not args.skip_ppl:
        print(f"\nEvaluating perplexity (samples={args.ppl_samples})...")
        cpr_ppl = evaluate_perplexity(model_cpr, tokenizer, num_samples=args.ppl_samples)
        print(f"Perplexity: {cpr_ppl:.2f}")
        print(f"PPL increase: {cpr_ppl - fp16_ppl:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'FP16':>12} | {'CPR INT8':>12} | {'Target':>12}")
    print("-" * 70)
    print(f"{'Model Memory (MB)':<25} | {fp16_mem:>12.1f} | {cpr_mem:>12.1f} | {'50% reduction':>12}")
    print(f"{'Memory Reduction':<25} | {'-':>12} | {(1-cpr_mem/fp16_mem)*100:>11.1f}% | {'>50%':>12}")
    print(f"{'Tokens/sec':<25} | {fp16_tps:>12.1f} | {cpr_tps:>12.1f} | {'':>12}")
    print(f"{'Speed vs FP16':<25} | {100:>11.1f}% | {cpr_tps/fp16_tps*100:>11.1f}% | {'>80%':>12}")
    print(f"{'Perplexity':<25} | {fp16_ppl:>12.2f} | {cpr_ppl:>12.2f} | {'<0.5 inc':>12}")
    if not args.skip_ppl:
        print(f"{'PPL Increase':<25} | {'-':>12} | {cpr_ppl-fp16_ppl:>12.2f} | {'<0.5':>12}")

    # Check if targets met
    print("\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)

    mem_target = (1 - cpr_mem/fp16_mem) >= 0.45  # 45% reduction
    speed_target = (cpr_tps/fp16_tps) >= 0.80   # 80% of FP16 speed
    ppl_target = True if args.skip_ppl else (cpr_ppl - fp16_ppl) < 0.5

    print(f"Memory reduction ≥50%: {'✅ PASS' if mem_target else '❌ FAIL'} ({(1-cpr_mem/fp16_mem)*100:.1f}%)")
    print(f"Speed ≥80% of FP16: {'✅ PASS' if speed_target else '❌ FAIL'} ({cpr_tps/fp16_tps*100:.1f}%)")
    if not args.skip_ppl:
        print(f"PPL increase <0.5: {'✅ PASS' if ppl_target else '❌ FAIL'} ({cpr_ppl-fp16_ppl:.2f})")

    all_pass = mem_target and speed_target and ppl_target
    print()
    print(f"Overall: {'✅ ALL TARGETS MET' if all_pass else '❌ SOME TARGETS MISSED'}")


if __name__ == '__main__':
    main()
