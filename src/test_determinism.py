#!/usr/bin/env python3
"""
Test: Make inference deterministic.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
import os


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
    print("Test: Determinism")
    print("=" * 70)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    # Try to make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic

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

    # Test: Does the issue come from model forward or cache interaction?
    print("\n" + "=" * 70)
    print("TEST 1: Fresh forward pass each time (no cache reuse)")
    print("=" * 70)

    with torch.no_grad():
        logits_list = []
        for i in range(3):
            torch.manual_seed(42 + i)
            out = model(**inputs, use_cache=False)
            logits_list.append(out.logits[:, -1, :].clone())

        for i in range(1, 3):
            diff = (logits_list[0] - logits_list[i]).abs().max().item()
            print(f"  Run 0 vs Run {i}: Max diff = {diff:.8f}")

    # Test: Same seed
    print("\n" + "=" * 70)
    print("TEST 2: Same seed each run")
    print("=" * 70)

    with torch.no_grad():
        logits_list = []
        for i in range(3):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            out = model(**inputs, use_cache=False)
            logits_list.append(out.logits[:, -1, :].clone())

        for i in range(1, 3):
            diff = (logits_list[0] - logits_list[i]).abs().max().item()
            print(f"  Run 0 vs Run {i}: Max diff = {diff:.8f}")

    # Test: Inference only, no random ops
    print("\n" + "=" * 70)
    print("TEST 3: Check for Qwen3's thinking tokens")
    print("=" * 70)

    # Qwen3 has a "thinking" mode that might add randomness
    # Let's check what's happening with generation config

    print(f"  Model config: {model.config.architectures}")
    print(f"  Is Qwen3 model: {'Qwen3' in str(model.config.architectures)}")

    # Test with do_sample=False explicitly
    print("\n" + "=" * 70)
    print("TEST 4: Generate with do_sample=False")
    print("=" * 70)

    with torch.no_grad():
        outputs = []
        for i in range(3):
            out = model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )
            outputs.append(out[0, -1].item())

        print(f"  Generated tokens: {outputs}")
        print(f"  All same: {len(set(outputs)) == 1}")

    # Test: The issue might be the KV cache usage with dummy token
    print("\n" + "=" * 70)
    print("TEST 5: Direct logits comparison without dummy token")
    print("=" * 70)

    with torch.no_grad():
        logits_list = []
        for i in range(3):
            torch.manual_seed(42)
            out = model(**inputs, use_cache=False)
            # Just get the last token's logits directly
            logits_list.append(out.logits[:, -1, :].clone())

        for i in range(1, 3):
            diff = (logits_list[0] - logits_list[i]).abs().max().item()
            print(f"  Run 0 vs Run {i}: Max diff = {diff:.8f}")

    # Test: Compare with and without KV cache properly
    print("\n" + "=" * 70)
    print("TEST 6: Fresh generation for quantization test")
    print("=" * 70)

    # Instead of reusing KV cache, do full forward pass each time
    # This gives the "true" logits for the prompt

    with torch.no_grad():
        # Get FP reference (just the last token's prediction)
        torch.manual_seed(42)
        fp_out = model(**inputs, use_cache=False)
        fp_logits = fp_out.logits[:, -1, :].clone()

        # For quantization test, we need to:
        # 1. Get KV cache
        # 2. Quantize it
        # 3. Run a fresh forward pass with the same input but using quantized KV

        # But this doesn't make sense because the KV cache IS the computation result
        # What we should do is quantize KV during generation

        # Let's do it properly: run generation and quantize KV at each step
        print("\n  Doing proper quantization test...")

        # Method: Encode prompt, quantize KV, then do continuation
        outputs = model(**inputs, use_cache=True)
        fp_kv = outputs.past_key_values
        fp_final_logits = outputs.logits[:, -1, :]  # Last token prediction WITH fp KV

        # Now quantize KV and get prediction
        for bits in [4, 6, 8]:
            q_kv = DynamicCache()
            for layer_idx in range(len(fp_kv)):
                k = quantize_tensor(fp_kv.key_cache[layer_idx].clone(), bits)
                v = quantize_tensor(fp_kv.value_cache[layer_idx].clone(), bits)
                q_kv.update(k, v, layer_idx)

            # Re-run forward with quantized KV
            # We need to compute logits for the last token position using quantized KV
            # This is equivalent to: given the quantized context, what's the next token?

            # But the problem is: model(input_ids, past_key_values) expects
            # input_ids to be the NEW tokens, not the full sequence

            # So we should:
            # 1. Run full sequence to get last token's prediction with FP KV
            # 2. For quantization test, we can't easily redo the forward pass

            # Actually, the proper test is:
            # - Compare output at EACH position with quantized vs FP KV
            # - Or test generation quality over multiple tokens

            # For now, let's just measure the logits divergence
            # by running the model with the last token and quantized KV

            last_token = inputs.input_ids[:, -1:]
            prefix_kv_fp = DynamicCache()
            prefix_kv_q = DynamicCache()

            # Create prefix KV (all but last position)
            for layer_idx in range(len(fp_kv)):
                k = fp_kv.key_cache[layer_idx][:, :, :-1, :]
                v = fp_kv.value_cache[layer_idx][:, :, :-1, :]
                prefix_kv_fp.update(k.clone(), v.clone(), layer_idx)
                prefix_kv_q.update(quantize_tensor(k.clone(), bits), quantize_tensor(v.clone(), bits), layer_idx)

            # Run with FP prefix
            torch.manual_seed(42)
            out_fp = model(input_ids=last_token, past_key_values=prefix_kv_fp, use_cache=False)
            logits_fp = out_fp.logits[:, -1, :]

            # Run with Q prefix
            torch.manual_seed(42)
            out_q = model(input_ids=last_token, past_key_values=prefix_kv_q, use_cache=False)
            logits_q = out_q.logits[:, -1, :]

            mse = F.mse_loss(logits_fp, logits_q).item()
            cos = F.cosine_similarity(logits_fp, logits_q, dim=-1).item()
            print(f"  {bits}-bit: MSE = {mse:.4f}, CosSim = {cos:.4f}")


if __name__ == "__main__":
    main()
