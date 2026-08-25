#!/usr/bin/env python3
"""
Measure actual quantization sensitivity per layer.

Instead of using gradient importance (which doesn't predict quant sensitivity),
directly measure how much PPL degrades when quantizing each layer in isolation.

This gives us actual quantization sensitivity:
- Sensitive layers: high PPL degradation when quantized
- Tolerant layers: low PPL degradation when quantized
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
import gc
import copy
sys.path.insert(0, '/workspace/SINQ')

# Seed
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate_ppl_fast(model, tokenizer, max_samples=10, max_length=128):
    """Fast PPL evaluation for sensitivity measurement."""
    device = next(model.parameters()).device

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100][:max_samples]

    all_text = "\n\n".join(texts)
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=max_length * max_samples)
    input_ids = encodings.input_ids.to(device)

    nlls = []
    stride = max_length // 2

    with torch.no_grad():
        for begin in range(0, min(input_ids.size(1), max_length * 5), stride):
            end = min(begin + max_length, input_ids.size(1))
            chunk = input_ids[:, begin:end]

            outputs = model(chunk, labels=chunk)
            nll = outputs.loss.item()

            if not np.isnan(nll) and not np.isinf(nll):
                nlls.append(nll)

            if end >= input_ids.size(1):
                break

    return np.exp(np.mean(nlls)) if nlls else float('inf')


def quantize_single_layer(model, layer_name, device='cuda:0'):
    """
    Quantize a single linear layer using simple RTN quantization.
    Returns the quantized weight.
    """
    # Find the layer
    parts = layer_name.split('.')
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    final_part = parts[-1]
    layer = getattr(module, final_part)

    if not isinstance(layer, nn.Linear):
        return None

    # Simple 4-bit RTN quantization
    weight = layer.weight.data.float()
    max_val = weight.abs().max(dim=1, keepdim=True).values
    scale = max_val / 7.5  # 4-bit symmetric
    q = torch.round(weight / scale).clamp(-8, 7)
    weight_quant = q * scale

    # Replace weight
    layer.weight.data = weight_quant.to(layer.weight.dtype)

    return True


def measure_layer_sensitivity(model_name, tokenizer, device='cuda:0'):
    """
    Measure quantization sensitivity of each layer.

    Returns dict: layer_name -> ppl_degradation
    """
    # Get baseline PPL
    print("Computing FP16 baseline PPL...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model_fp16.eval()
    baseline_ppl = evaluate_ppl_fast(model_fp16, tokenizer)
    print(f"Baseline PPL: {baseline_ppl:.2f}")

    # Get all linear layer names
    linear_layers = []
    for name, module in model_fp16.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)

    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nMeasuring sensitivity for {len(linear_layers)} linear layers...")

    sensitivity = {}

    for layer_name in tqdm(linear_layers, desc="Measuring sensitivity"):
        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        # Quantize single layer
        success = quantize_single_layer(model, layer_name, device)

        if success:
            # Measure PPL
            ppl = evaluate_ppl_fast(model, tokenizer)
            degradation = ppl - baseline_ppl
            sensitivity[layer_name] = {
                'ppl': ppl,
                'degradation': degradation,
                'relative': degradation / baseline_ppl
            }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return baseline_ppl, sensitivity


def main():
    print("=" * 70)
    print("Quantization Sensitivity Measurement")
    print("=" * 70)
    print("\nMeasuring actual PPL degradation when quantizing each layer.")
    print("This gives us true quantization sensitivity, not gradient importance.\n")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_ppl, sensitivity = measure_layer_sensitivity(model_name, tokenizer, device)

    # Analyze by layer type
    print("\n" + "=" * 70)
    print("SENSITIVITY BY LAYER TYPE")
    print("=" * 70)

    type_sensitivity = {}
    for name, data in sensitivity.items():
        if 'q_proj' in name:
            layer_type = 'Q'
        elif 'k_proj' in name:
            layer_type = 'K'
        elif 'v_proj' in name:
            layer_type = 'V'
        elif 'o_proj' in name:
            layer_type = 'O'
        elif 'up_proj' in name:
            layer_type = 'up'
        elif 'gate_proj' in name:
            layer_type = 'gate'
        elif 'down_proj' in name:
            layer_type = 'down'
        else:
            layer_type = 'other'

        if layer_type not in type_sensitivity:
            type_sensitivity[layer_type] = []
        type_sensitivity[layer_type].append(data['degradation'])

    print(f"\n{'Layer Type':<12}{'Mean Degradation':<20}{'Max Degradation':<20}")
    print("-" * 55)

    sorted_types = sorted(type_sensitivity.items(),
                          key=lambda x: np.mean(x[1]), reverse=True)

    for layer_type, degradations in sorted_types:
        mean_deg = np.mean(degradations)
        max_deg = np.max(degradations)
        bar = "█" * int(mean_deg * 2)
        print(f"{layer_type:<12}{mean_deg:<20.3f}{max_deg:<20.3f} {bar}")

    # Top 10 most sensitive layers
    print("\n" + "=" * 70)
    print("TOP 10 MOST SENSITIVE LAYERS")
    print("=" * 70)

    sorted_sens = sorted(sensitivity.items(),
                         key=lambda x: x[1]['degradation'], reverse=True)[:10]

    print(f"\n{'Layer':<50}{'PPL':<10}{'Degradation':<15}")
    print("-" * 75)
    for name, data in sorted_sens:
        print(f"{name:<50}{data['ppl']:<10.2f}{data['degradation']:<15.3f}")

    # Compare with gradient importance
    print("\n" + "=" * 70)
    print("COMPARISON: Sensitivity vs Gradient Importance")
    print("=" * 70)

    # From our earlier analysis:
    grad_importance = {
        'V': 1.885, 'O': 1.500, 'down': 1.111, 'up': 1.005,
        'K': 1.001, 'gate': 0.740, 'Q': 0.727
    }

    print(f"\n{'Type':<12}{'Grad Importance':<18}{'Quant Sensitivity':<20}{'Match?':<10}")
    print("-" * 60)

    for layer_type, _ in sorted_types:
        if layer_type in grad_importance:
            grad_imp = grad_importance[layer_type]
            quant_sens = np.mean(type_sensitivity[layer_type])

            # Rank comparison
            grad_rank = sorted(grad_importance.values(), reverse=True).index(grad_imp) + 1
            sens_list = [np.mean(v) for k, v in sorted_types if k in grad_importance]
            sens_rank = sorted(sens_list, reverse=True).index(quant_sens) + 1

            match = "✓" if abs(grad_rank - sens_rank) <= 1 else "✗"
            print(f"{layer_type:<12}{grad_imp:<18.3f}{quant_sens:<20.3f}{match}")


if __name__ == "__main__":
    main()
