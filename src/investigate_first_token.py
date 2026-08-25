#!/usr/bin/env python3
"""
Investigate why 4-bit first token beats 8-bit.

This is counterintuitive and suggests something fundamental about
the quantization error distribution.
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
    print("Investigate First Token Quantization")
    print("=" * 70)

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    model_name = "Qwen/Qwen3-1.7B"
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
        fp_kv = outputs.past_key_values

        # Look at the first token's KV values
        k0_first = fp_kv.key_cache[0][:, :, 0, :]  # First layer, first position
        v0_first = fp_kv.value_cache[0][:, :, 0, :]

        print("\n" + "=" * 70)
        print("FIRST TOKEN KV STATISTICS (Layer 0)")
        print("=" * 70)

        print(f"\nK shape: {k0_first.shape}")
        print(f"K range: [{k0_first.min().item():.2f}, {k0_first.max().item():.2f}]")
        print(f"K mean: {k0_first.mean().item():.4f}")
        print(f"K std: {k0_first.std().item():.4f}")

        print(f"\nV shape: {v0_first.shape}")
        print(f"V range: [{v0_first.min().item():.4f}, {v0_first.max().item():.4f}]")
        print(f"V mean: {v0_first.mean().item():.6f}")
        print(f"V std: {v0_first.std().item():.6f}")

        # Compare quantization error for different bits
        print("\n" + "=" * 70)
        print("QUANTIZATION ERROR (RAW) FOR FIRST TOKEN")
        print("=" * 70)

        for bits in [2, 3, 4, 5, 6, 8]:
            k0_first_full = fp_kv.key_cache[0][:, :, :1, :]
            v0_first_full = fp_kv.value_cache[0][:, :, :1, :]

            k_q = quantize_tensor(k0_first_full, bits)
            v_q = quantize_tensor(v0_first_full, bits)

            k_mse = F.mse_loss(k0_first_full, k_q).item()
            v_mse = F.mse_loss(v0_first_full, v_q).item()

            print(f"  {bits}-bit: K_MSE = {k_mse:.6f}, V_MSE = {v_mse:.8f}")

        # Now check the full model error with different first token precisions
        print("\n" + "=" * 70)
        print("OUTPUT ERROR WITH DIFFERENT FIRST TOKEN PRECISION")
        print("=" * 70)

        # Reference: all 4-bit
        def quant_all_4bit(k, v):
            return quantize_tensor(k, 4), quantize_tensor(v, 4)

        # Test: first token at various bits, rest at 4-bit
        for first_bits in [2, 3, 4, 5, 6, 8]:
            prefix_kv_fp = DynamicCache()
            prefix_kv_q = DynamicCache()

            for layer_idx in range(len(fp_kv)):
                k = fp_kv.key_cache[layer_idx][:, :, :-1, :].clone()
                v = fp_kv.value_cache[layer_idx][:, :, :-1, :].clone()

                prefix_kv_fp.update(k.clone(), v.clone(), layer_idx)

                # Quantize first token at first_bits, rest at 4-bit
                k_q = k.clone()
                v_q = v.clone()

                k_q[:, :, :1, :] = quantize_tensor(k[:, :, :1, :], first_bits)
                v_q[:, :, :1, :] = quantize_tensor(v[:, :, :1, :], first_bits)
                if k.shape[2] > 1:
                    k_q[:, :, 1:, :] = quantize_tensor(k[:, :, 1:, :], 4)
                    v_q[:, :, 1:, :] = quantize_tensor(v[:, :, 1:, :], 4)

                prefix_kv_q.update(k_q, v_q, layer_idx)

            last_token = inputs.input_ids[:, -1:]

            torch.manual_seed(42)
            out_fp = model(input_ids=last_token, past_key_values=prefix_kv_fp, use_cache=False)
            logits_fp = out_fp.logits[:, -1, :]

            torch.manual_seed(42)
            out_q = model(input_ids=last_token, past_key_values=prefix_kv_q, use_cache=False)
            logits_q = out_q.logits[:, -1, :]

            mse = F.mse_loss(logits_fp, logits_q).item()
            cos = F.cosine_similarity(logits_fp, logits_q, dim=-1).item()

            print(f"  First={first_bits}-bit: MSE = {mse:.4f}, CosSim = {cos:.4f}")

        # Interesting! Let's see if the issue is consistency across layers
        print("\n" + "=" * 70)
        print("CHECK: ERROR PER LAYER (First Token at 8-bit vs 4-bit)")
        print("=" * 70)

        # What if the first token has different characteristics in different layers?
        for layer_idx in [0, 5, 10, 20, 27]:
            k_layer = fp_kv.key_cache[layer_idx][:, :, :1, :]
            v_layer = fp_kv.value_cache[layer_idx][:, :, :1, :]

            k_range = k_layer.max().item() - k_layer.min().item()
            v_range = v_layer.max().item() - v_layer.min().item()

            k_4_mse = F.mse_loss(k_layer, quantize_tensor(k_layer, 4)).item()
            k_8_mse = F.mse_loss(k_layer, quantize_tensor(k_layer, 8)).item()
            v_4_mse = F.mse_loss(v_layer, quantize_tensor(v_layer, 4)).item()
            v_8_mse = F.mse_loss(v_layer, quantize_tensor(v_layer, 8)).item()

            print(f"  Layer {layer_idx:2d}: K_range={k_range:8.2f}, K_4b_mse={k_4_mse:.6f}, K_8b_mse={k_8_mse:.6f}")
            print(f"           V_range={v_range:8.4f}, V_4b_mse={v_4_mse:.8f}, V_8b_mse={v_8_mse:.8f}")

        # Check: What's the attention weight for the first token?
        print("\n" + "=" * 70)
        print("ATTENTION TO FIRST TOKEN (Across Layers)")
        print("=" * 70)

        outputs_attn = model(**inputs, output_attentions=True)
        attentions = outputs_attn.attentions

        for layer_idx in [0, 5, 10, 20, 27]:
            attn = attentions[layer_idx]  # (batch, heads, seq, seq)
            # Attention from last position to first position
            attn_to_first = attn[0, :, -1, 0].mean().item()  # avg over heads
            attn_to_first_max = attn[0, :, -1, 0].max().item()
            print(f"  Layer {layer_idx:2d}: Attn to first token (mean={attn_to_first:.4f}, max={attn_to_first_max:.4f})")


if __name__ == "__main__":
    main()
