"""
True CPR (Mixed 6-bit/5-bit) LLM Benchmark

Compares:
1. FP16 baseline
2. Uniform INT8 (previous CPRLinearFused)
3. True CPR Mixed (6-bit/5-bit with column reordering)

This benchmark tests the actual CPR quantization scheme with:
- Column sensitivity analysis
- Column reordering
- Mixed precision (25% 6-bit, 75% 5-bit = 5.25 avg bits)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import gc
import argparse
from typing import Dict, Tuple
from tqdm import tqdm


def get_gpu_memory() -> Tuple[float, float]:
    """Return (allocated_mb, peak_mb)."""
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.max_memory_allocated() / 1e6
    )


def benchmark_forward(model, tokenizer, prompt: str,
                      max_new_tokens: int = 50, warmup: int = 3, iters: int = 5) -> Dict:
    """Benchmark forward pass and generation."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    input_len = inputs['input_ids'].shape[1]

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

    alloc_mb, peak_mb = get_gpu_memory()

    return {
        'tokens_per_sec': total_tokens / elapsed,
        'latency_ms': elapsed / iters * 1000,
        'alloc_mb': alloc_mb,
        'peak_mb': peak_mb,
    }


def quantize_layer_int8(module: nn.Linear, force_fused: bool = True):
    """Quantize a linear layer to INT8 (uniform quantization)."""
    from sinq.cpr_model import CPRLinearFused

    cpr = CPRLinearFused(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        group_size=128,
        compute_dtype=torch.float16,
        device=module.weight.device,
        force_fused=force_fused,
    )
    cpr.quantize_weights(module.weight.data)
    if module.bias is not None:
        cpr.bias.data.copy_(module.bias.data.to(torch.float16))
    return cpr


def quantize_layer_mixed(module: nn.Linear, high_frac: float = 0.25):
    """Quantize a linear layer with true CPR (mixed 6-bit/5-bit)."""
    from sinq.triton_cpr_mixed_kernel import CPRLinearMixedFused

    cpr = CPRLinearMixedFused.from_linear(
        module,
        high_frac=high_frac,
        high_bits=6,
        low_bits=5,
        group_size=128,
        compute_dtype=torch.float16,
    )
    return cpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--max-new-tokens', type=int, default=30)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--skip-int8', action='store_true', help='Skip INT8 test')
    args = parser.parse_args()

    print("=" * 80)
    print("TRUE CPR Mixed-Precision LLM Benchmark")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog. In the world of AI,"

    results = {}

    # =========================================================================
    # FP16 Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("FP16 BASELINE")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model_fp16.eval()

    fp16_model_mem = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / 1e6
    fp16_model_mem += sum(b.numel() * b.element_size() for b in model_fp16.buffers()) / 1e6

    print(f"Model memory: {fp16_model_mem:.1f} MB")
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    print(f"\nBenchmarking generation...")
    fp16_results = benchmark_forward(model_fp16, tokenizer, prompt,
                                     max_new_tokens=args.max_new_tokens, iters=args.iters)

    print(f"Throughput: {fp16_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"GPU peak: {fp16_results['peak_mb']:.1f} MB")

    results['fp16'] = {'model_mb': fp16_model_mem, **fp16_results}

    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # INT8 Uniform (if not skipped)
    # =========================================================================
    if not args.skip_int8:
        print("\n" + "=" * 80)
        print("INT8 UNIFORM (8-bit all columns)")
        print("=" * 80)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Load to CPU first
        print("Loading model to CPU...")
        model_cpu = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map='cpu',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        print("Quantizing to INT8 and moving to GPU...")
        model_cpu.model.embed_tokens = model_cpu.model.embed_tokens.to('cuda')

        for i, layer in enumerate(tqdm(model_cpu.model.layers, desc="Quantizing INT8")):
            layer = layer.to('cuda')
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                old_module = getattr(layer.self_attn, name)
                new_module = quantize_layer_int8(old_module, force_fused=True)
                setattr(layer.self_attn, name, new_module)
                del old_module

            for name in ['gate_proj', 'up_proj', 'down_proj']:
                old_module = getattr(layer.mlp, name)
                new_module = quantize_layer_int8(old_module, force_fused=True)
                setattr(layer.mlp, name, new_module)
                del old_module

            model_cpu.model.layers[i] = layer
            gc.collect()
            torch.cuda.empty_cache()

        model_cpu.model.norm = model_cpu.model.norm.to('cuda')
        model_cpu.lm_head = model_cpu.lm_head.to('cuda')

        gc.collect()
        torch.cuda.empty_cache()

        model_int8 = model_cpu
        model_int8.eval()

        int8_model_mem = sum(p.numel() * p.element_size() for p in model_int8.parameters()) / 1e6
        int8_model_mem += sum(b.numel() * b.element_size() for b in model_int8.buffers()) / 1e6

        print(f"\nModel memory: {int8_model_mem:.1f} MB")
        print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"Memory reduction: {(1 - int8_model_mem / fp16_model_mem) * 100:.1f}%")

        print(f"\nBenchmarking generation...")
        int8_results = benchmark_forward(model_int8, tokenizer, prompt,
                                         max_new_tokens=args.max_new_tokens, iters=args.iters)

        print(f"Throughput: {int8_results['tokens_per_sec']:.1f} tokens/sec")
        print(f"GPU peak: {int8_results['peak_mb']:.1f} MB")
        print(f"Speed vs FP16: {int8_results['tokens_per_sec'] / results['fp16']['tokens_per_sec'] * 100:.1f}%")

        results['int8'] = {'model_mb': int8_model_mem, **int8_results}

        del model_int8
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # TRUE CPR Mixed (6-bit/5-bit)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRUE CPR MIXED (6-bit/5-bit, 5.25 avg bits)")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load to CPU first
    print("Loading model to CPU...")
    model_cpu = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Quantizing with TRUE CPR (6-bit/5-bit) and moving to GPU...")
    model_cpu.model.embed_tokens = model_cpu.model.embed_tokens.to('cuda')

    for i, layer in enumerate(tqdm(model_cpu.model.layers, desc="Quantizing CPR")):
        layer = layer.to('cuda')

        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            old_module = getattr(layer.self_attn, name)
            new_module = quantize_layer_mixed(old_module, high_frac=0.25)
            setattr(layer.self_attn, name, new_module)
            del old_module

        for name in ['gate_proj', 'up_proj', 'down_proj']:
            old_module = getattr(layer.mlp, name)
            new_module = quantize_layer_mixed(old_module, high_frac=0.25)
            setattr(layer.mlp, name, new_module)
            del old_module

        model_cpu.model.layers[i] = layer
        gc.collect()
        torch.cuda.empty_cache()

    model_cpu.model.norm = model_cpu.model.norm.to('cuda')
    model_cpu.lm_head = model_cpu.lm_head.to('cuda')

    gc.collect()
    torch.cuda.empty_cache()

    model_cpr = model_cpu
    model_cpr.eval()

    cpr_model_mem = sum(p.numel() * p.element_size() for p in model_cpr.parameters()) / 1e6
    cpr_model_mem += sum(b.numel() * b.element_size() for b in model_cpr.buffers()) / 1e6

    # Calculate theoretical packed memory
    from sinq.triton_cpr_mixed_kernel import CPRLinearMixedFused
    cpr_packed_bytes = 0
    for m in model_cpr.modules():
        if isinstance(m, CPRLinearMixedFused):
            cpr_packed_bytes += m.memory_footprint_packed()
        elif isinstance(m, nn.Linear):
            cpr_packed_bytes += m.weight.numel() * 2  # FP16
            if m.bias is not None:
                cpr_packed_bytes += m.bias.numel() * 2
        elif isinstance(m, nn.Embedding):
            cpr_packed_bytes += m.weight.numel() * 2
    cpr_packed_mem = cpr_packed_bytes / 1e6

    print(f"\nModel memory (INT8 storage): {cpr_model_mem:.1f} MB")
    print(f"Theoretical packed (5.25-bit): {cpr_packed_mem:.1f} MB")
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"Memory reduction vs FP16: {(1 - cpr_model_mem / fp16_model_mem) * 100:.1f}%")
    print(f"Theoretical reduction (packed): {(1 - cpr_packed_mem / fp16_model_mem) * 100:.1f}%")

    print(f"\nBenchmarking generation...")
    cpr_results = benchmark_forward(model_cpr, tokenizer, prompt,
                                    max_new_tokens=args.max_new_tokens, iters=args.iters)

    print(f"Throughput: {cpr_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"GPU peak: {cpr_results['peak_mb']:.1f} MB")
    print(f"Speed vs FP16: {cpr_results['tokens_per_sec'] / results['fp16']['tokens_per_sec'] * 100:.1f}%")

    results['cpr'] = {
        'model_mb': cpr_model_mem,
        'packed_mb': cpr_packed_mem,
        **cpr_results
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fp16 = results['fp16']
    cpr = results['cpr']

    if not args.skip_int8:
        int8 = results['int8']
        print(f"\n{'Metric':<30} | {'FP16':>12} | {'INT8 (8-bit)':>12} | {'CPR (5.25b)':>12}")
        print("-" * 75)
        print(f"{'Model Memory (MB)':<30} | {fp16['model_mb']:>12.1f} | {int8['model_mb']:>12.1f} | {cpr['model_mb']:>12.1f}")
        print(f"{'Packed Memory (MB)':<30} | {fp16['model_mb']:>12.1f} | {int8['model_mb']:>12.1f} | {cpr['packed_mb']:>12.1f}")
        print(f"{'Peak GPU (MB)':<30} | {fp16['peak_mb']:>12.1f} | {int8['peak_mb']:>12.1f} | {cpr['peak_mb']:>12.1f}")
        print(f"{'Throughput (tok/s)':<30} | {fp16['tokens_per_sec']:>12.1f} | {int8['tokens_per_sec']:>12.1f} | {cpr['tokens_per_sec']:>12.1f}")
        print(f"{'Speed vs FP16':<30} | {'100.0%':>12} | {int8['tokens_per_sec']/fp16['tokens_per_sec']*100:>11.1f}% | {cpr['tokens_per_sec']/fp16['tokens_per_sec']*100:>11.1f}%")
        print(f"{'Memory vs FP16':<30} | {'100.0%':>12} | {int8['model_mb']/fp16['model_mb']*100:>11.1f}% | {cpr['model_mb']/fp16['model_mb']*100:>11.1f}%")
        print(f"{'Packed Memory vs FP16':<30} | {'100.0%':>12} | {int8['model_mb']/fp16['model_mb']*100:>11.1f}% | {cpr['packed_mb']/fp16['model_mb']*100:>11.1f}%")
    else:
        print(f"\n{'Metric':<30} | {'FP16':>12} | {'CPR (5.25b)':>12}")
        print("-" * 60)
        print(f"{'Model Memory (MB)':<30} | {fp16['model_mb']:>12.1f} | {cpr['model_mb']:>12.1f}")
        print(f"{'Packed Memory (MB)':<30} | {fp16['model_mb']:>12.1f} | {cpr['packed_mb']:>12.1f}")
        print(f"{'Peak GPU (MB)':<30} | {fp16['peak_mb']:>12.1f} | {cpr['peak_mb']:>12.1f}")
        print(f"{'Throughput (tok/s)':<30} | {fp16['tokens_per_sec']:>12.1f} | {cpr['tokens_per_sec']:>12.1f}")
        print(f"{'Speed vs FP16':<30} | {'100.0%':>12} | {cpr['tokens_per_sec']/fp16['tokens_per_sec']*100:>11.1f}%")
        print(f"{'Memory vs FP16':<30} | {'100.0%':>12} | {cpr['model_mb']/fp16['model_mb']*100:>11.1f}%")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"\nTRUE CPR Mixed Precision Benefits:")
    print(f"  - Average bits: 5.25 (vs 8 for INT8, vs 16 for FP16)")
    print(f"  - Memory reduction: {(1 - cpr['model_mb']/fp16['model_mb'])*100:.1f}% (current INT8 storage)")
    print(f"  - Theoretical (packed): {(1 - cpr['packed_mb']/fp16['model_mb'])*100:.1f}% reduction")
    print(f"  - Speed: {cpr['tokens_per_sec']/fp16['tokens_per_sec']*100:.1f}% of FP16")

    if not args.skip_int8:
        print(f"\nCPR vs Uniform INT8:")
        print(f"  - Memory: {cpr['packed_mb']/int8['model_mb']*100:.1f}% (packed vs INT8)")
        print(f"  - Speed: {cpr['tokens_per_sec']/int8['tokens_per_sec']*100:.1f}%")


if __name__ == '__main__':
    main()
