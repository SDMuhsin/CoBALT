#!/usr/bin/env python3
"""
Investigation: Non-Monotonic Error Scaling

Critical Discovery: Uniform 4-bit has LOWER MSE than 8-bit!

This suggests:
1. The KV cache values might have special structure
2. Quantization might be interacting strangely with attention
3. There might be numerical precision issues at higher bits

Let's investigate what's happening.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np


def quantize_tensor(x, num_bits, per_channel=False):
    if per_channel:
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
    else:
        x_min = x.min()
        x_max = x.max()

    range_val = x_max - x_min
    if isinstance(range_val, torch.Tensor):
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
    elif range_val == 0:
        return x

    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def main():
    print("=" * 70)
    print("Investigation: Non-Monotonic Error Scaling")
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

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        fp_logits = outputs.logits[:, -1, :]
        fp_kv = outputs.past_key_values

        # Get raw KV values for analysis
        k0 = fp_kv.key_cache[0]  # First layer key
        v0 = fp_kv.value_cache[0]  # First layer value

        print(f"\nKV Cache Shape: {k0.shape}")
        print(f"K range: [{k0.min().item():.4f}, {k0.max().item():.4f}]")
        print(f"V range: [{v0.min().item():.4f}, {v0.max().item():.4f}]")
        print(f"K dtype: {k0.dtype}")

        # Test 1: Quantization error on raw tensors
        print("\n" + "=" * 70)
        print("TEST 1: Raw Quantization Error on K tensor")
        print("=" * 70)

        for bits in [2, 3, 4, 5, 6, 8]:
            k_q = quantize_tensor(k0, bits)
            raw_mse = F.mse_loss(k0, k_q).item()
            print(f"  {bits}-bit: Quantization MSE = {raw_mse:.6f}")

        # Test 2: Check if the model output is sensitive to tiny perturbations
        print("\n" + "=" * 70)
        print("TEST 2: Sensitivity to Perturbation")
        print("=" * 70)

        # Add small noise and see effect
        for noise_scale in [0.0001, 0.001, 0.01, 0.1]:
            noisy_kv = DynamicCache()
            for layer_idx in range(len(fp_kv)):
                k = fp_kv.key_cache[layer_idx].clone()
                v = fp_kv.value_cache[layer_idx].clone()
                k = k + torch.randn_like(k) * noise_scale
                v = v + torch.randn_like(v) * noise_scale
                noisy_kv.update(k, v, layer_idx)

            dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
            noisy_out = model(input_ids=dummy, past_key_values=noisy_kv, use_cache=False)
            noisy_logits = noisy_out.logits[:, -1, :]
            mse = F.mse_loss(fp_logits, noisy_logits).item()
            print(f"  Noise σ={noise_scale}: Output MSE = {mse:.4f}")

        # Test 3: Check quantization at different layers
        print("\n" + "=" * 70)
        print("TEST 3: Per-Layer Analysis (8-bit quantization)")
        print("=" * 70)

        num_layers = len(fp_kv)
        print(f"  Total layers: {num_layers}")

        for layer_idx in [0, 5, 10, num_layers-1]:
            k = fp_kv.key_cache[layer_idx]
            v = fp_kv.value_cache[layer_idx]

            k_q = quantize_tensor(k, 8)
            v_q = quantize_tensor(v, 8)

            k_mse = F.mse_loss(k, k_q).item()
            v_mse = F.mse_loss(v, v_q).item()

            print(f"  Layer {layer_idx}: K_MSE={k_mse:.8f}, V_MSE={v_mse:.8f}")

        # Test 4: Check if it's the quantize-then-rerun that's the issue
        print("\n" + "=" * 70)
        print("TEST 4: Direct Quantization vs Model Re-inference")
        print("=" * 70)

        for bits in [4, 6, 8]:
            # Quantize KV
            q_kv = DynamicCache()
            for layer_idx in range(len(fp_kv)):
                k = quantize_tensor(fp_kv.key_cache[layer_idx].clone(), bits)
                v = quantize_tensor(fp_kv.value_cache[layer_idx].clone(), bits)
                q_kv.update(k, v, layer_idx)

            # Get output with quantized KV
            dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_out = model(input_ids=dummy, past_key_values=q_kv, use_cache=False)
            q_logits = q_out.logits[:, -1, :]

            # Compare
            mse = F.mse_loss(fp_logits, q_logits).item()
            cos = F.cosine_similarity(fp_logits, q_logits, dim=-1).item()

            print(f"  {bits}-bit: MSE={mse:.4f}, CosSim={cos:.4f}")

        # Test 5: Is the FP reference stable?
        print("\n" + "=" * 70)
        print("TEST 5: FP Reference Stability")
        print("=" * 70)

        # Run FP inference twice with the same KV cache
        dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
        fp_out1 = model(input_ids=dummy, past_key_values=fp_kv, use_cache=False)
        fp_out2 = model(input_ids=dummy, past_key_values=fp_kv, use_cache=False)

        fp_logits1 = fp_out1.logits[:, -1, :]
        fp_logits2 = fp_out2.logits[:, -1, :]

        fp_mse = F.mse_loss(fp_logits1, fp_logits2).item()
        print(f"  FP inference variance: MSE = {fp_mse:.8f}")

        # Test 6: Multiple prompts
        print("\n" + "=" * 70)
        print("TEST 6: Multiple Prompts")
        print("=" * 70)

        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming many industries.",
            "In the beginning, there was nothing but darkness.",
        ]

        for bits in [4, 6, 8]:
            mses = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs, use_cache=True)
                fp_logits = outputs.logits[:, -1, :]
                fp_kv = outputs.past_key_values

                q_kv = DynamicCache()
                for layer_idx in range(len(fp_kv)):
                    k = quantize_tensor(fp_kv.key_cache[layer_idx].clone(), bits)
                    v = quantize_tensor(fp_kv.value_cache[layer_idx].clone(), bits)
                    q_kv.update(k, v, layer_idx)

                dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
                q_out = model(input_ids=dummy, past_key_values=q_kv, use_cache=False)
                q_logits = q_out.logits[:, -1, :]

                mse = F.mse_loss(fp_logits, q_logits).item()
                mses.append(mse)

            print(f"  {bits}-bit: MSEs = {[f'{m:.2f}' for m in mses]}, Mean = {np.mean(mses):.2f}")

        # Test 7: Try per-channel quantization
        print("\n" + "=" * 70)
        print("TEST 7: Per-Channel vs Per-Tensor Quantization")
        print("=" * 70)

        prompt = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, use_cache=True)
        fp_logits = outputs.logits[:, -1, :]
        fp_kv = outputs.past_key_values

        for bits in [4, 8]:
            for per_channel in [False, True]:
                q_kv = DynamicCache()
                for layer_idx in range(len(fp_kv)):
                    k = quantize_tensor(fp_kv.key_cache[layer_idx].clone(), bits, per_channel=per_channel)
                    v = quantize_tensor(fp_kv.value_cache[layer_idx].clone(), bits, per_channel=per_channel)
                    q_kv.update(k, v, layer_idx)

                dummy = torch.tensor([[tokenizer.eos_token_id]], device=device)
                q_out = model(input_ids=dummy, past_key_values=q_kv, use_cache=False)
                q_logits = q_out.logits[:, -1, :]

                mse = F.mse_loss(fp_logits, q_logits).item()
                mode = "per-channel" if per_channel else "per-tensor"
                print(f"  {bits}-bit {mode}: MSE = {mse:.4f}")

        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)


if __name__ == "__main__":
    main()
