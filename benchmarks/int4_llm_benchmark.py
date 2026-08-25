"""
INT4 Packed LLM Benchmark - REAL Memory Measurement

This benchmark measures ACTUAL GPU memory, not theoretical estimates.

Goal: Demonstrate real memory savings with INT4 packed quantization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import gc
import argparse
from tqdm import tqdm


def get_real_gpu_memory():
    """Get REAL GPU memory in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_generation(model, tokenizer, prompt, max_new_tokens=30, warmup=2, iters=3):
    """Benchmark generation speed."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    input_len = inputs['input_ids'].shape[1]
    total_tokens = 0
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        total_tokens += outputs.shape[1] - input_len
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    return {
        'tokens_per_sec': total_tokens / elapsed,
        'peak_memory_mb': peak_mem,
    }


def quantize_model_int4(model):
    """Quantize model to INT4."""
    from sinq.int4_packed_kernel import INT4PackedLinear

    # Move embeddings to GPU first
    model.model.embed_tokens = model.model.embed_tokens.to('cuda')

    quantized_count = 0

    for i, layer in enumerate(tqdm(model.model.layers, desc="Quantizing INT4")):
        layer = layer.to('cuda')

        # Quantize attention
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            old = getattr(layer.self_attn, name)
            new = INT4PackedLinear.from_linear(old, group_size=128)
            setattr(layer.self_attn, name, new)
            del old
            quantized_count += 1

        # Quantize MLP
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            old = getattr(layer.mlp, name)
            new = INT4PackedLinear.from_linear(old, group_size=128)
            setattr(layer.mlp, name, new)
            del old
            quantized_count += 1

        model.model.layers[i] = layer
        gc.collect()
        torch.cuda.empty_cache()

    # Move norm and lm_head to GPU (keep FP16)
    model.model.norm = model.model.norm.to('cuda')
    model.lm_head = model.lm_head.to('cuda')

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Quantized {quantized_count} layers to INT4")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--max-new-tokens', type=int, default=30)
    args = parser.parse_args()

    print("=" * 70)
    print("INT4 Packed LLM Benchmark - REAL Memory Measurement")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog. In the world of AI,"

    # =========================================================================
    # FP16 Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("FP16 BASELINE")
    print("=" * 70)

    clear_gpu()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model_fp16.eval()

    fp16_mem = get_real_gpu_memory()
    print(f"GPU memory after loading: {fp16_mem:.1f} MB")

    fp16_results = benchmark_generation(model_fp16, tokenizer, prompt,
                                        max_new_tokens=args.max_new_tokens)

    print(f"Peak GPU memory during inference: {fp16_results['peak_memory_mb']:.1f} MB")
    print(f"Throughput: {fp16_results['tokens_per_sec']:.1f} tokens/sec")

    # Delete FP16 model
    del model_fp16
    clear_gpu()

    # =========================================================================
    # INT4 Packed
    # =========================================================================
    print("\n" + "=" * 70)
    print("INT4 PACKED (4-bit quantization)")
    print("=" * 70)

    clear_gpu()

    # Load to CPU first
    print("Loading model to CPU...")
    model_cpu = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Quantizing to INT4...")
    model_int4 = quantize_model_int4(model_cpu)
    model_int4.eval()

    int4_mem = get_real_gpu_memory()
    print(f"\nGPU memory after loading: {int4_mem:.1f} MB")
    print(f"Memory vs FP16: {int4_mem / fp16_mem * 100:.1f}%")
    print(f"Memory reduction: {(1 - int4_mem / fp16_mem) * 100:.1f}%")

    int4_results = benchmark_generation(model_int4, tokenizer, prompt,
                                        max_new_tokens=args.max_new_tokens)

    print(f"\nPeak GPU memory during inference: {int4_results['peak_memory_mb']:.1f} MB")
    print(f"Throughput: {int4_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"Speed vs FP16: {int4_results['tokens_per_sec'] / fp16_results['tokens_per_sec'] * 100:.1f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY - REAL MEASUREMENTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} | {'FP16':>15} | {'INT4 Packed':>15}")
    print("-" * 65)
    print(f"{'GPU Memory (MB)':<30} | {fp16_mem:>15.1f} | {int4_mem:>15.1f}")
    print(f"{'Peak Memory (MB)':<30} | {fp16_results['peak_memory_mb']:>15.1f} | {int4_results['peak_memory_mb']:>15.1f}")
    print(f"{'Throughput (tok/s)':<30} | {fp16_results['tokens_per_sec']:>15.1f} | {int4_results['tokens_per_sec']:>15.1f}")
    print(f"{'Memory vs FP16':<30} | {'100.0%':>15} | {int4_mem/fp16_mem*100:>14.1f}%")
    print(f"{'Speed vs FP16':<30} | {'100.0%':>15} | {int4_results['tokens_per_sec']/fp16_results['tokens_per_sec']*100:>14.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    mem_reduction = (1 - int4_mem / fp16_mem) * 100
    speed_ratio = int4_results['tokens_per_sec'] / fp16_results['tokens_per_sec'] * 100

    print(f"\nMemory reduction: {mem_reduction:.1f}%")
    print(f"Speed: {speed_ratio:.1f}% of FP16")

    if mem_reduction >= 50:
        print(f"\n[SUCCESS] Achieved {mem_reduction:.1f}% memory reduction (target: >50%)")
    else:
        print(f"\n[FAIL] Only {mem_reduction:.1f}% memory reduction (target: >50%)")

    if speed_ratio >= 100:
        print(f"[BONUS] Speed is {speed_ratio:.1f}% of FP16 (faster than baseline!)")
    elif speed_ratio >= 50:
        print(f"[ACCEPTABLE] Speed is {speed_ratio:.1f}% of FP16")
    else:
        print(f"[WARNING] Speed is only {speed_ratio:.1f}% of FP16")


if __name__ == '__main__':
    main()
