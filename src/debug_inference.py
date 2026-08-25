#!/usr/bin/env python3
"""
Debug: Check for numerical issues in inference.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np


def quantize_tensor(x, num_bits):
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min
    if range_val == 0:
        return x
    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def main():
    print("=" * 70)
    print("Debug: Numerical Stability")
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
    device = next(model.parameters()).device

    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run multiple times with FP KV to check stability
    print("\n" + "=" * 70)
    print("TEST 1: Multiple FP Runs (checking determinism)")
    print("=" * 70)

    torch.manual_seed(42)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        fp_kv = outputs.past_key_values
        fp_logits = outputs.logits[:, -1, :]

        logits_list = []
        for i in range(5):
            dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
            out = model(input_ids=dummy, past_key_values=fp_kv, use_cache=False)
            logits_list.append(out.logits[:, -1, :].clone())

        # Check if all are identical
        for i in range(1, 5):
            diff = (logits_list[0] - logits_list[i]).abs().max().item()
            print(f"  Run 0 vs Run {i}: Max diff = {diff:.8f}")

    # Check for NaN/Inf in KV cache
    print("\n" + "=" * 70)
    print("TEST 2: Check for NaN/Inf in KV cache")
    print("=" * 70)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        fp_kv = outputs.past_key_values

        total_nan = 0
        total_inf = 0
        for layer_idx in range(len(fp_kv)):
            k = fp_kv.key_cache[layer_idx]
            v = fp_kv.value_cache[layer_idx]
            total_nan += k.isnan().sum().item() + v.isnan().sum().item()
            total_inf += k.isinf().sum().item() + v.isinf().sum().item()

        print(f"  Total NaN: {total_nan}")
        print(f"  Total Inf: {total_inf}")

    # Run inference in FP32
    print("\n" + "=" * 70)
    print("TEST 3: Compare FP16 vs FP32 inference")
    print("=" * 70)

    # FP16 model already loaded
    with torch.no_grad():
        outputs_fp16 = model(**inputs, use_cache=True)
        logits_fp16 = outputs_fp16.logits[:, -1, :]
        kv_fp16 = outputs_fp16.past_key_values

    # Load FP32 model
    print("  Loading FP32 model...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model_fp32.eval()

    with torch.no_grad():
        outputs_fp32 = model_fp32(**inputs, use_cache=True)
        logits_fp32 = outputs_fp32.logits[:, -1, :]
        kv_fp32 = outputs_fp32.past_key_values

        # Compare
        logits_diff = (logits_fp16.float() - logits_fp32).abs().mean().item()
        print(f"  FP16 vs FP32 logits diff: {logits_diff:.6f}")

        # Compare KV values
        k_diff = (kv_fp16.key_cache[0].float() - kv_fp32.key_cache[0]).abs().mean().item()
        print(f"  FP16 vs FP32 KV diff: {k_diff:.6f}")

    # Check if quantization affects it differently in FP32
    print("\n" + "=" * 70)
    print("TEST 4: Quantization in FP32")
    print("=" * 70)

    for bits in [4, 6, 8]:
        with torch.no_grad():
            q_kv = DynamicCache()
            for layer_idx in range(len(kv_fp32)):
                k = quantize_tensor(kv_fp32.key_cache[layer_idx].clone(), bits)
                v = quantize_tensor(kv_fp32.value_cache[layer_idx].clone(), bits)
                q_kv.update(k, v, layer_idx)

            dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_out = model_fp32(input_ids=dummy, past_key_values=q_kv, use_cache=False)
            q_logits = q_out.logits[:, -1, :]

            mse = F.mse_loss(logits_fp32, q_logits).item()
            print(f"  {bits}-bit (FP32): MSE = {mse:.4f}")

    # Check stability in FP32
    print("\n" + "=" * 70)
    print("TEST 5: FP32 Inference Stability")
    print("=" * 70)

    with torch.no_grad():
        logits_list = []
        for i in range(3):
            dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
            out = model_fp32(input_ids=dummy, past_key_values=kv_fp32, use_cache=False)
            logits_list.append(out.logits[:, -1, :].clone())

        for i in range(1, 3):
            diff = (logits_list[0] - logits_list[i]).abs().max().item()
            print(f"  Run 0 vs Run {i}: Max diff = {diff:.8f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
