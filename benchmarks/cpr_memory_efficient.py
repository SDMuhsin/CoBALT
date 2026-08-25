"""
Memory-efficient CPR quantization and benchmark.

This script:
1. Loads model layer-by-layer and quantizes immediately
2. Measures true memory savings during inference
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


def quantize_layer_inplace(module: nn.Linear, force_fused: bool = True):
    """Quantize a linear layer to CPR INT8 in-place to save memory."""
    from sinq.cpr_model import CPRLinearFused

    # Get original weight info
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None
    in_features = module.in_features
    out_features = module.out_features
    device = weight.device
    dtype = weight.dtype

    # Create quantized layer
    cpr = CPRLinearFused(
        in_features=in_features,
        out_features=out_features,
        bias=bias is not None,
        group_size=128,
        compute_dtype=torch.float16,
        device=device,
        force_fused=force_fused,
    )

    # Quantize weights
    cpr.quantize_weights(weight)

    # Copy bias
    if bias is not None:
        cpr.bias.data.copy_(bias.to(torch.float16))

    # Clear original weight to free memory immediately
    module.weight.data = torch.empty(0, device='cpu')
    if module.bias is not None:
        module.bias.data = torch.empty(0, device='cpu')

    return cpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--max-new-tokens', type=int, default=30)
    parser.add_argument('--iters', type=int, default=5)
    args = parser.parse_args()

    print("=" * 80)
    print("CPR Memory-Efficient Benchmark")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sinq.cpr_model import CPRLinearFused

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog. In the world of AI,"

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
    print(f"GPU allocated during inference: {fp16_results['alloc_mb']:.1f} MB")
    print(f"Peak GPU during inference: {fp16_results['peak_mb']:.1f} MB")

    # Delete FP16 model
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # CPR INT8 - Direct Load (no FP16 intermediate)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CPR INT8 (QUANTIZED DIRECTLY)")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load to CPU first, then quantize layer by layer
    print("Loading model to CPU...")
    model_cpu = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cpu',  # Load to CPU first
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Quantizing and moving to GPU layer by layer...")

    # Move embeddings to GPU
    model_cpu.model.embed_tokens = model_cpu.model.embed_tokens.to('cuda')

    # Quantize each layer and move to GPU
    for i, layer in enumerate(tqdm(model_cpu.model.layers, desc="Quantizing")):
        # Move layer to GPU
        layer = layer.to('cuda')

        # Quantize linear layers
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            old_module = getattr(layer.self_attn, name)
            new_module = quantize_layer_inplace(old_module, force_fused=True)
            setattr(layer.self_attn, name, new_module)
            del old_module

        for name in ['gate_proj', 'up_proj', 'down_proj']:
            old_module = getattr(layer.mlp, name)
            new_module = quantize_layer_inplace(old_module, force_fused=True)
            setattr(layer.mlp, name, new_module)
            del old_module

        # Keep layer on GPU
        model_cpu.model.layers[i] = layer

        gc.collect()
        torch.cuda.empty_cache()

    # Move final layers to GPU
    model_cpu.model.norm = model_cpu.model.norm.to('cuda')
    model_cpu.lm_head = model_cpu.lm_head.to('cuda')  # Keep LM head in FP16

    gc.collect()
    torch.cuda.empty_cache()

    model_cpr = model_cpu
    model_cpr.eval()

    cpr_model_mem = sum(p.numel() * p.element_size() for p in model_cpr.parameters()) / 1e6
    cpr_model_mem += sum(b.numel() * b.element_size() for b in model_cpr.buffers()) / 1e6

    print(f"\nModel memory: {cpr_model_mem:.1f} MB")
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"Memory reduction: {(1 - cpr_model_mem / fp16_model_mem) * 100:.1f}%")

    # Count layer types
    fused = sum(1 for m in model_cpr.modules() if isinstance(m, CPRLinearFused) and m._use_fused)
    dequant = sum(1 for m in model_cpr.modules() if isinstance(m, CPRLinearFused) and not m._use_fused)
    fp16_layers = sum(1 for m in model_cpr.modules() if isinstance(m, nn.Linear))
    print(f"Layers: {fused} fused, {dequant} dequant, {fp16_layers} FP16")

    print(f"\nBenchmarking generation...")
    cpr_results = benchmark_forward(model_cpr, tokenizer, prompt,
                                    max_new_tokens=args.max_new_tokens, iters=args.iters)

    print(f"Throughput: {cpr_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"GPU allocated during inference: {cpr_results['alloc_mb']:.1f} MB")
    print(f"Peak GPU during inference: {cpr_results['peak_mb']:.1f} MB")
    print(f"Speed vs FP16: {cpr_results['tokens_per_sec'] / fp16_results['tokens_per_sec'] * 100:.1f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<30} | {'FP16':>12} | {'CPR INT8':>12}")
    print("-" * 60)
    print(f"{'Model Memory (MB)':<30} | {fp16_model_mem:>12.1f} | {cpr_model_mem:>12.1f}")
    print(f"{'GPU Allocated (MB)':<30} | {fp16_results['alloc_mb']:>12.1f} | {cpr_results['alloc_mb']:>12.1f}")
    print(f"{'Peak GPU (MB)':<30} | {fp16_results['peak_mb']:>12.1f} | {cpr_results['peak_mb']:>12.1f}")
    print(f"{'Throughput (tok/s)':<30} | {fp16_results['tokens_per_sec']:>12.1f} | {cpr_results['tokens_per_sec']:>12.1f}")
    print(f"{'Speed vs FP16':<30} | {'100.0%':>12} | {cpr_results['tokens_per_sec']/fp16_results['tokens_per_sec']*100:>11.1f}%")
    print(f"{'Memory vs FP16':<30} | {'100.0%':>12} | {cpr_results['peak_mb']/fp16_results['peak_mb']*100:>11.1f}%")

    mem_savings = (1 - cpr_results['peak_mb'] / fp16_results['peak_mb']) * 100
    speed_pct = cpr_results['tokens_per_sec'] / fp16_results['tokens_per_sec'] * 100

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Memory savings: {mem_savings:.1f}%")
    print(f"Speed: {speed_pct:.1f}% of FP16")

    if mem_savings > 0:
        print(f"\n✅ Memory reduced by {mem_savings:.1f}%")
    else:
        print(f"\n❌ Memory increased by {-mem_savings:.1f}%")

    if speed_pct >= 80:
        print(f"✅ Speed target met ({speed_pct:.1f}% >= 80%)")
    else:
        print(f"❌ Speed target missed ({speed_pct:.1f}% < 80%)")


if __name__ == '__main__':
    main()
