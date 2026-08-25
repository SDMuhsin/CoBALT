#!/usr/bin/env python3
"""
Analyze importance of Q, K, V projections vs other linear layers.

The hypothesis is that Q and K may need higher precision because
their errors multiply in Q @ K^T, while V errors are linear.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '/workspace/SINQ')

def compute_layer_importance(model, tokenizer, texts, max_length=128, device='cuda:0'):
    """Compute importance by layer type (Q, K, V, O, up, gate, down)."""
    model = model.to(device)
    model.train()

    # Storage for gradient accumulation by layer type
    type_grad_accum = {}

    for text in tqdm(texts[:20], desc="Computing importance"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)

        if input_ids.shape[1] < 10:
            continue

        model.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2 and param.grad is not None:
                # Categorize by type
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

                grad_norm = param.grad.abs().mean().item()

                if layer_type not in type_grad_accum:
                    type_grad_accum[layer_type] = []
                type_grad_accum[layer_type].append(grad_norm)

    model.eval()

    # Average importance by type
    importance = {}
    for layer_type, grads in type_grad_accum.items():
        importance[layer_type] = np.mean(grads)

    # Normalize by mean
    mean_imp = np.mean(list(importance.values()))
    importance = {k: v / mean_imp for k, v in importance.items()}

    return importance


def main():
    print("=" * 70)
    print("Q/K/V Importance Analysis")
    print("=" * 70)

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    cal_texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100][:30]

    importance = compute_layer_importance(model, tokenizer, cal_texts, device=device)

    print("\nLayer Type Importance (normalized, mean=1.0):")
    print("-" * 40)

    # Sort by importance
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for layer_type, imp in sorted_imp:
        bar = "█" * int(imp * 20)
        print(f"  {layer_type:8s}: {imp:.3f}  {bar}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    qk_avg = (importance.get('Q', 1) + importance.get('K', 1)) / 2
    v_avg = importance.get('V', 1)
    mlp_avg = (importance.get('up', 1) + importance.get('gate', 1) + importance.get('down', 1)) / 3

    print(f"\nQ+K average importance: {qk_avg:.3f}")
    print(f"V importance: {v_avg:.3f}")
    print(f"MLP average importance: {mlp_avg:.3f}")

    if qk_avg > v_avg * 1.2:
        print("\n✓ Q/K are significantly more important than V")
        print("  → Attention-specific quantization may help")
    elif v_avg > qk_avg * 1.2:
        print("\n✓ V is significantly more important than Q/K")
        print("  → Standard uniform treatment may be better")
    else:
        print("\n≈ Q/K/V have similar importance")
        print("  → No clear benefit to attention-specific treatment")


if __name__ == "__main__":
    main()
