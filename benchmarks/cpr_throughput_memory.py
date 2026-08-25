"""
CPR Throughput and Memory Benchmark

Compares:
1. FP16 baseline
2. CPR INT8 with fused kernel (all layers)
3. CPR INT8 with hybrid kernel (shape-aware routing)

Measures:
- Throughput (tokens/sec)
- GPU memory during inference
- Peak memory usage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import gc
import argparse
from typing import Tuple, Dict
from tqdm import tqdm


def get_gpu_memory_mb() -> Tuple[float, float]:
    """Get (allocated, peak) GPU memory in MB."""
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.max_memory_allocated() / 1e6
    )


def reset_peak_memory():
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats()


def benchmark_generation(model, tokenizer, prompt: str, max_new_tokens: int = 50,
                         warmup: int = 3, iters: int = 10) -> Dict:
    """
    Benchmark text generation.
    Returns dict with tokens_per_second, latency_ms, peak_memory_mb.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    # Reset peak memory after warmup
    reset_peak_memory()
    gc.collect()
    torch.cuda.empty_cache()

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

    # Get memory stats
    alloc_mb, peak_mb = get_gpu_memory_mb()

    return {
        'tokens_per_sec': tokens_per_sec,
        'latency_ms': latency_ms,
        'allocated_mb': alloc_mb,
        'peak_mb': peak_mb,
    }


def count_layer_types(model) -> Dict[str, int]:
    """Count fused vs dequant layers."""
    from sinq.cpr_model import CPRLinearFused

    fused = 0
    dequant = 0
    fp16 = 0

    for name, module in model.named_modules():
        if isinstance(module, CPRLinearFused):
            if module._use_fused:
                fused += 1
            else:
                dequant += 1
        elif isinstance(module, nn.Linear):
            fp16 += 1

    return {'fused': fused, 'dequant': dequant, 'fp16': fp16}


def main():
    parser = argparse.ArgumentParser(description='CPR Throughput and Memory Benchmark')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B',
                        help='Model to benchmark')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='Max tokens to generate')
    parser.add_argument('--iters', type=int, default=5,
                        help='Number of benchmark iterations')
    args = parser.parse_args()

    print("=" * 80)
    print("CPR Throughput and Memory Benchmark")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sinq.cpr_model import quantize_model_cpr, CPRLinearFused

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog. In the world of artificial intelligence,"

    results = {}

    # =========================================================================
    # Test 1: FP16 Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: FP16 BASELINE")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model_fp16.eval()

    # Get model memory
    model_mem_mb = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / 1e6
    model_mem_mb += sum(b.numel() * b.element_size() for b in model_fp16.buffers()) / 1e6

    print(f"Model memory: {model_mem_mb:.1f} MB")

    # Benchmark
    print(f"\nBenchmarking generation (max_new_tokens={args.max_new_tokens})...")
    fp16_results = benchmark_generation(model_fp16, tokenizer, prompt,
                                        max_new_tokens=args.max_new_tokens,
                                        iters=args.iters)

    print(f"Throughput: {fp16_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"Latency: {fp16_results['latency_ms']:.1f} ms")
    print(f"Peak GPU memory: {fp16_results['peak_mb']:.1f} MB")

    results['fp16'] = {
        'model_mb': model_mem_mb,
        **fp16_results
    }

    # Free FP16 model
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: CPR INT8 - All Fused (original behavior)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: CPR INT8 - ALL FUSED KERNEL")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory()

    model_fused = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )

    # Quantize with force_fused=True for all layers
    print("Quantizing with fused kernel for all layers...")
    linear_layers = []
    for name, module in model_fused.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            linear_layers.append((name, module))

    for name, module in tqdm(linear_layers, desc="Quantizing"):
        cpr_layer = CPRLinearFused.from_linear(module, group_size=128, force_fused=True)
        # Replace module
        parts = name.split('.')
        parent = model_fused
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], cpr_layer)
        del module

    gc.collect()
    torch.cuda.empty_cache()
    model_fused.eval()

    # Get model memory
    model_mem_mb = sum(p.numel() * p.element_size() for p in model_fused.parameters()) / 1e6
    model_mem_mb += sum(b.numel() * b.element_size() for b in model_fused.buffers()) / 1e6

    layer_counts = count_layer_types(model_fused)
    print(f"Model memory: {model_mem_mb:.1f} MB")
    print(f"Layer counts: {layer_counts}")

    # Benchmark
    print(f"\nBenchmarking generation (max_new_tokens={args.max_new_tokens})...")
    fused_results = benchmark_generation(model_fused, tokenizer, prompt,
                                         max_new_tokens=args.max_new_tokens,
                                         iters=args.iters)

    print(f"Throughput: {fused_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"Latency: {fused_results['latency_ms']:.1f} ms")
    print(f"Peak GPU memory: {fused_results['peak_mb']:.1f} MB")
    print(f"Speed vs FP16: {fused_results['tokens_per_sec'] / results['fp16']['tokens_per_sec'] * 100:.1f}%")

    results['fused'] = {
        'model_mb': model_mem_mb,
        'layer_counts': layer_counts,
        **fused_results
    }

    # Free fused model
    del model_fused
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 3: CPR INT8 - Hybrid (shape-aware routing)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: CPR INT8 - HYBRID (SHAPE-AWARE ROUTING)")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory()

    model_hybrid = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )

    # Quantize with auto kernel selection (force_fused=None)
    print("Quantizing with auto kernel selection...")
    linear_layers = []
    for name, module in model_hybrid.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            linear_layers.append((name, module))

    for name, module in tqdm(linear_layers, desc="Quantizing"):
        cpr_layer = CPRLinearFused.from_linear(module, group_size=128, force_fused=None)
        # Replace module
        parts = name.split('.')
        parent = model_hybrid
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], cpr_layer)
        del module

    gc.collect()
    torch.cuda.empty_cache()
    model_hybrid.eval()

    # Get model memory
    model_mem_mb = sum(p.numel() * p.element_size() for p in model_hybrid.parameters()) / 1e6
    model_mem_mb += sum(b.numel() * b.element_size() for b in model_hybrid.buffers()) / 1e6

    layer_counts = count_layer_types(model_hybrid)
    print(f"Model memory: {model_mem_mb:.1f} MB")
    print(f"Layer counts: {layer_counts}")

    # Benchmark
    print(f"\nBenchmarking generation (max_new_tokens={args.max_new_tokens})...")
    hybrid_results = benchmark_generation(model_hybrid, tokenizer, prompt,
                                          max_new_tokens=args.max_new_tokens,
                                          iters=args.iters)

    print(f"Throughput: {hybrid_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"Latency: {hybrid_results['latency_ms']:.1f} ms")
    print(f"Peak GPU memory: {hybrid_results['peak_mb']:.1f} MB")
    print(f"Speed vs FP16: {hybrid_results['tokens_per_sec'] / results['fp16']['tokens_per_sec'] * 100:.1f}%")

    results['hybrid'] = {
        'model_mb': model_mem_mb,
        'layer_counts': layer_counts,
        **hybrid_results
    }

    # Free hybrid model
    del model_hybrid
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fp16 = results['fp16']
    fused = results['fused']
    hybrid = results['hybrid']

    print(f"\n{'Metric':<25} | {'FP16':>12} | {'CPR Fused':>12} | {'CPR Hybrid':>12}")
    print("-" * 80)
    print(f"{'Model Memory (MB)':<25} | {fp16['model_mb']:>12.1f} | {fused['model_mb']:>12.1f} | {hybrid['model_mb']:>12.1f}")
    print(f"{'Peak GPU Memory (MB)':<25} | {fp16['peak_mb']:>12.1f} | {fused['peak_mb']:>12.1f} | {hybrid['peak_mb']:>12.1f}")
    print(f"{'Throughput (tok/s)':<25} | {fp16['tokens_per_sec']:>12.1f} | {fused['tokens_per_sec']:>12.1f} | {hybrid['tokens_per_sec']:>12.1f}")
    print(f"{'Speed vs FP16':<25} | {'100.0%':>12} | {fused['tokens_per_sec']/fp16['tokens_per_sec']*100:>11.1f}% | {hybrid['tokens_per_sec']/fp16['tokens_per_sec']*100:>11.1f}%")
    print(f"{'Memory vs FP16':<25} | {'100.0%':>12} | {fused['peak_mb']/fp16['peak_mb']*100:>11.1f}% | {hybrid['peak_mb']/fp16['peak_mb']*100:>11.1f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Memory savings
    fused_mem_savings = (1 - fused['peak_mb'] / fp16['peak_mb']) * 100
    hybrid_mem_savings = (1 - hybrid['peak_mb'] / fp16['peak_mb']) * 100

    print(f"\nMemory Savings:")
    print(f"  CPR Fused:  {fused_mem_savings:.1f}% reduction in peak GPU memory")
    print(f"  CPR Hybrid: {hybrid_mem_savings:.1f}% reduction in peak GPU memory")

    # Speed comparison
    fused_speed = fused['tokens_per_sec'] / fp16['tokens_per_sec'] * 100
    hybrid_speed = hybrid['tokens_per_sec'] / fp16['tokens_per_sec'] * 100

    print(f"\nThroughput:")
    print(f"  CPR Fused:  {fused_speed:.1f}% of FP16 speed")
    print(f"  CPR Hybrid: {hybrid_speed:.1f}% of FP16 speed")

    # Recommendation
    print(f"\nRecommendation:")
    if hybrid_speed > fused_speed:
        print(f"  Use Hybrid mode - {hybrid_speed - fused_speed:.1f}% faster than all-fused")
    else:
        print(f"  Use Fused mode - {fused_speed - hybrid_speed:.1f}% faster than hybrid")

    # Layer breakdown for hybrid
    print(f"\nHybrid Layer Routing:")
    print(f"  Fused kernel: {hybrid['layer_counts']['fused']} layers (favorable shapes)")
    print(f"  Dequant+cuBLAS: {hybrid['layer_counts']['dequant']} layers (unfavorable shapes)")
    print(f"  FP16 (unquantized): {hybrid['layer_counts']['fp16']} layers")


if __name__ == '__main__':
    main()
