#!/usr/bin/env python3
"""
Test CPR-SINQ on full model with PPL evaluation.

This is the CRITICAL test - does MSE improvement translate to PPL improvement?

Target: Beat 5-bit SINQ PPL with < 5.25 bits average.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
import gc
sys.path.insert(0, '/workspace/SINQ')

from src.cpr_sinq import quantize_cpr, dequantize_cpr, compute_avg_bits

# Seed
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate_ppl(model, tokenizer, max_samples=15, max_length=256):
    """Evaluate perplexity on WikiText-2."""
    device = next(model.parameters()).device

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100][:max_samples]

    all_text = "\n\n".join(texts)
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=max_length * max_samples)
    input_ids = encodings.input_ids.to(device)

    nlls = []
    stride = max_length // 2

    with torch.no_grad():
        for begin in tqdm(range(0, min(input_ids.size(1), max_length * 10), stride), desc="Evaluating", leave=False):
            end = min(begin + max_length, input_ids.size(1))
            chunk = input_ids[:, begin:end]

            outputs = model(chunk, labels=chunk)
            nll = outputs.loss.item()

            if not np.isnan(nll) and not np.isinf(nll):
                nlls.append(nll)

            if end >= input_ids.size(1):
                break

    return np.exp(np.mean(nlls)) if nlls else float('inf')


class CPRLinear(nn.Module):
    """
    Linear layer with CPR quantized weights.

    Replaces nn.Linear for inference with dequantized weights.
    """
    def __init__(self, original_linear, quant_data, compute_dtype=torch.float16):
        super().__init__()
        self.quant_data = quant_data
        self.bias = original_linear.bias
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.compute_dtype = compute_dtype

        # Pre-dequantize for inference (for now - full implementation would fuse)
        self.register_buffer('weight', dequantize_cpr(quant_data).to(compute_dtype))

    def forward(self, x):
        return nn.functional.linear(x.to(self.compute_dtype), self.weight,
                                    self.bias.to(self.compute_dtype) if self.bias is not None else None)


def quantize_model_cpr(model, high_frac=0.25, high_bits=6, low_bits=5, device='cuda:0'):
    """
    Quantize all linear layers in model using CPR.

    Returns model with quantized weights and total average bits.
    """
    total_params = 0
    total_bits = 0

    layers_to_quantize = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers_to_quantize.append((name, module))

    for name, module in tqdm(layers_to_quantize, desc="Quantizing with CPR"):
        weight = module.weight.data.to(device)

        # Quantize with CPR
        quant_data = quantize_cpr(weight, high_frac, high_bits, low_bits)

        # Track bits
        n_params = weight.numel()
        avg_bits = compute_avg_bits(quant_data)
        total_params += n_params
        total_bits += n_params * avg_bits

        # Replace module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        cpr_linear = CPRLinear(module, quant_data)
        setattr(parent, parts[-1], cpr_linear)

    avg_bits_total = total_bits / total_params
    return model, avg_bits_total


def quantize_model_uniform_sinq(model, nbits=5, device='cuda:0'):
    """
    Quantize all linear layers using uniform SINQ.
    """
    from sinq.patch_model import AutoSINQHFModel
    from sinq.sinqlinear import BaseQuantizeConfig

    quant_config = BaseQuantizeConfig(
        nbits=nbits,
        group_size=64,
        axis=1,
        tiling_mode='1D',
        method='sinq_nogemlite'
    )

    # We need tokenizer for quantization
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=device
    )

    return model


def main():
    print("=" * 70)
    print("CPR-SINQ FULL MODEL EVALUATION")
    print("=" * 70)
    print("\nTarget: Beat 5-bit SINQ PPL with < 5.25 bits average")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FP16 baseline
    print("\n" + "=" * 70)
    print("FP16 Baseline")
    print("=" * 70)

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model_fp16.eval()
    baseline_ppl = evaluate_ppl(model_fp16, tokenizer)
    print(f"FP16 PPL: {baseline_ppl:.2f}")
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # Uniform 5-bit SINQ
    print("\n" + "=" * 70)
    print("Uniform 5-bit SINQ")
    print("=" * 70)

    model_5bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model_5bit = quantize_model_uniform_sinq(model_5bit, nbits=5, device=device)
    model_5bit.eval()
    if not hasattr(model_5bit, 'hf_device_map'):
        model_5bit = model_5bit.to(device)
    ppl_5bit = evaluate_ppl(model_5bit, tokenizer)
    print(f"5-bit SINQ PPL: {ppl_5bit:.2f}")
    del model_5bit
    gc.collect()
    torch.cuda.empty_cache()

    # Uniform 4-bit SINQ (for comparison)
    print("\n" + "=" * 70)
    print("Uniform 4-bit SINQ")
    print("=" * 70)

    model_4bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model_4bit = quantize_model_uniform_sinq(model_4bit, nbits=4, device=device)
    model_4bit.eval()
    if not hasattr(model_4bit, 'hf_device_map'):
        model_4bit = model_4bit.to(device)
    ppl_4bit = evaluate_ppl(model_4bit, tokenizer)
    print(f"4-bit SINQ PPL: {ppl_4bit:.2f}")
    del model_4bit
    gc.collect()
    torch.cuda.empty_cache()

    # CPR-SINQ configurations
    cpr_configs = [
        (0.25, 6, 5, "CPR 25%@6b/75%@5b (5.25 bits)"),
        (0.25, 5, 4, "CPR 25%@5b/75%@4b (4.25 bits)"),
    ]

    results = []

    for high_frac, high_bits, low_bits, name in cpr_configs:
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        model_cpr = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model_cpr, avg_bits = quantize_model_cpr(
            model_cpr, high_frac, high_bits, low_bits, device=device
        )
        model_cpr.eval()
        if not hasattr(model_cpr, 'hf_device_map'):
            model_cpr = model_cpr.to(device)

        ppl_cpr = evaluate_ppl(model_cpr, tokenizer)
        print(f"{name} PPL: {ppl_cpr:.2f} (avg bits: {avg_bits:.3f})")

        results.append({
            'name': name,
            'ppl': ppl_cpr,
            'avg_bits': avg_bits
        })

        del model_cpr
        gc.collect()
        torch.cuda.empty_cache()

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFP16 Baseline:       {baseline_ppl:.2f}")
    print(f"SINQ 4-bit:          {ppl_4bit:.2f} (4.0 bits)")
    print(f"SINQ 5-bit:          {ppl_5bit:.2f} (5.0 bits)")

    for r in results:
        print(f"{r['name']}: {r['ppl']:.2f} ({r['avg_bits']:.3f} bits)")

    # Check if target met
    print("\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)

    # Target: Beat 5-bit SINQ PPL with < 5.25 bits
    for r in results:
        if r['avg_bits'] <= 5.25 and r['ppl'] < ppl_5bit:
            improvement = (ppl_5bit - r['ppl']) / ppl_5bit * 100
            print(f"✓ {r['name']} beats 5-bit SINQ by {improvement:.1f}%!")
        elif r['avg_bits'] <= 5.25:
            print(f"✗ {r['name']} does NOT beat 5-bit SINQ (PPL {r['ppl']:.2f} vs {ppl_5bit:.2f})")

    # Linear interpolation check
    print("\n--- Linear interpolation comparison ---")
    for r in results:
        # What would linear interpolation give at this bit width?
        if r['avg_bits'] >= 5.0:
            # Between 5-bit and 6-bit
            expected = ppl_5bit  # 5-bit is our baseline
            print(f"{r['name']}: Actual {r['ppl']:.2f}, 5-bit baseline {expected:.2f}")
        else:
            # Between 4-bit and 5-bit
            frac = (r['avg_bits'] - 4.0) / (5.0 - 4.0)
            expected = ppl_4bit + frac * (ppl_5bit - ppl_4bit)
            print(f"{r['name']}: Actual {r['ppl']:.2f}, Linear interpolation {expected:.2f}")

            if r['ppl'] < expected:
                print(f"  ✓ Beats linear interpolation!")
            else:
                print(f"  ✗ Does not beat linear interpolation")


if __name__ == "__main__":
    main()
