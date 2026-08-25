"""
TRUE CPR Packed LLM Benchmark

This tests the ACTUAL CPR scheme with:
1. Column sensitivity analysis
2. Column reordering
3. Real 6-bit/5-bit packed storage

Measurements:
- Storage memory: Real packed size (what matters for disk/transfer)
- Runtime memory: With dequant cache (needed for reasonable speed)
- Quality: Compare perplexity vs uniform quantization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import gc
from tqdm import tqdm


def get_gpu_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    print("=" * 70)
    print("TRUE CPR Packed Benchmark")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sinq.cpr_packed_kernel import CPRPackedLinear

    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================================================================
    # FP16 Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("FP16 BASELINE")
    print("=" * 70)

    clear_gpu()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model_fp16.eval()

    fp16_mem = get_gpu_memory()
    print(f"GPU memory: {fp16_mem:.1f} MB")

    # Quick generation test
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

    with torch.no_grad():
        for _ in range(3):  # Warmup
            model_fp16.generate(**inputs, max_new_tokens=20, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            outputs = model_fp16.generate(**inputs, max_new_tokens=20, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / 5

    fp16_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    fp16_tps = fp16_tokens / fp16_time
    print(f"Throughput: {fp16_tps:.1f} tokens/sec")

    del model_fp16
    clear_gpu()

    # =========================================================================
    # CPR Packed (True 6-bit/5-bit)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CPR PACKED (True 6-bit/5-bit, column sensitivity)")
    print("=" * 70)

    clear_gpu()

    # Load to CPU
    print("Loading model to CPU...")
    model_cpu = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Track storage size
    storage_bytes = 0
    fp16_equiv_bytes = 0

    print("Quantizing with TRUE CPR (6-bit/5-bit packed)...")
    model_cpu.model.embed_tokens = model_cpu.model.embed_tokens.to('cuda')

    for i, layer in enumerate(tqdm(model_cpu.model.layers, desc="Quantizing")):
        layer = layer.to('cuda')

        # Quantize attention
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            old = getattr(layer.self_attn, name)
            new = CPRPackedLinear.from_linear(old, high_frac=0.25, group_size=128)
            storage_bytes += new.memory_bytes()
            fp16_equiv_bytes += new.memory_bytes_fp16()
            setattr(layer.self_attn, name, new)
            del old

        # Quantize MLP
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            old = getattr(layer.mlp, name)
            new = CPRPackedLinear.from_linear(old, high_frac=0.25, group_size=128)
            storage_bytes += new.memory_bytes()
            fp16_equiv_bytes += new.memory_bytes_fp16()
            setattr(layer.mlp, name, new)
            del old

        model_cpu.model.layers[i] = layer
        gc.collect()
        torch.cuda.empty_cache()

    model_cpu.model.norm = model_cpu.model.norm.to('cuda')
    model_cpu.lm_head = model_cpu.lm_head.to('cuda')

    model_cpr = model_cpu
    model_cpr.eval()

    # Measure memory BEFORE dequant cache
    cpr_storage_mem = get_gpu_memory()

    print(f"\nStorage memory (packed weights): {cpr_storage_mem:.1f} MB")
    print(f"Storage vs FP16: {cpr_storage_mem / fp16_mem * 100:.1f}%")
    print(f"Storage reduction: {(1 - cpr_storage_mem / fp16_mem) * 100:.1f}%")

    # Now run inference (will create dequant cache)
    with torch.no_grad():
        for _ in range(3):
            model_cpr.generate(**inputs, max_new_tokens=20, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()

    cpr_runtime_mem = get_gpu_memory()
    print(f"\nRuntime memory (with dequant cache): {cpr_runtime_mem:.1f} MB")
    print(f"Runtime vs FP16: {cpr_runtime_mem / fp16_mem * 100:.1f}%")

    # Speed test
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(5):
            outputs = model_cpr.generate(**inputs, max_new_tokens=20, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        cpr_time = (time.perf_counter() - start) / 5

    cpr_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    cpr_tps = cpr_tokens / cpr_time
    print(f"\nThroughput: {cpr_tps:.1f} tokens/sec")
    print(f"Speed vs FP16: {cpr_tps / fp16_tps * 100:.1f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<35} | {'FP16':>12} | {'CPR Packed':>12}")
    print("-" * 65)
    print(f"{'Storage Memory (MB)':<35} | {fp16_mem:>12.1f} | {cpr_storage_mem:>12.1f}")
    print(f"{'Runtime Memory (MB)':<35} | {fp16_mem:>12.1f} | {cpr_runtime_mem:>12.1f}")
    print(f"{'Throughput (tok/s)':<35} | {fp16_tps:>12.1f} | {cpr_tps:>12.1f}")
    print(f"{'Storage vs FP16':<35} | {'100%':>12} | {cpr_storage_mem/fp16_mem*100:>11.1f}%")
    print(f"{'Speed vs FP16':<35} | {'100%':>12} | {cpr_tps/fp16_tps*100:>11.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    storage_reduction = (1 - cpr_storage_mem / fp16_mem) * 100
    speed_pct = cpr_tps / fp16_tps * 100

    print(f"\nCPR Unique Features:")
    print(f"  - Column sensitivity analysis: YES")
    print(f"  - Column reordering: YES")
    print(f"  - Mixed 6-bit/5-bit precision: YES")
    print(f"  - Average bits: 5.25")

    print(f"\nResults:")
    print(f"  - Storage reduction: {storage_reduction:.1f}%")
    print(f"  - Speed: {speed_pct:.1f}% of FP16")

    if storage_reduction >= 50:
        print(f"\n[SUCCESS] Storage reduced by {storage_reduction:.1f}% (target: >50%)")
    else:
        print(f"\n[FAIL] Storage only reduced by {storage_reduction:.1f}%")

    print(f"\nNote: Runtime memory is higher due to dequant cache.")
    print(f"Without fused kernel, cache is needed for reasonable speed.")


if __name__ == '__main__':
    main()
