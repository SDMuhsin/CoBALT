#!/usr/bin/env python3
"""
Phase 1.2: Measure per-layer importance for QEP-SINQ.

Goal: Compute gradient magnitude of final loss w.r.t. each layer's output
to determine which layers have most downstream impact on PPL.

Key question: Does importance vary significantly across layers?
If uniform → QEP premise is FALSE
If non-uniform → QEP premise is VALID, proceed to Phase 1.3
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json


def get_calibration_texts(num_samples=32):
    """Get calibration text from WikiText-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100]
    return texts[:num_samples]


def compute_layer_importance_gradient(model, tokenizer, texts, max_length=256):
    """
    Compute importance of each layer using gradient magnitude.

    For each transformer block, measure:
    w_l = E[||∂L/∂h_l||]

    where L is cross-entropy loss and h_l is the output of layer l.
    """
    device = next(model.parameters()).device
    model.eval()

    # Get model structure
    if hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    # Find decoder layers
    if hasattr(transformer, 'layers'):
        layers = transformer.layers
    elif hasattr(transformer, 'decoder') and hasattr(transformer.decoder, 'layers'):
        layers = transformer.decoder.layers
    else:
        raise ValueError("Cannot find decoder layers in model")

    num_layers = len(layers)
    print(f"Found {num_layers} transformer layers")

    # Storage for gradients
    layer_grad_norms = {i: [] for i in range(num_layers)}

    # Register hooks to capture layer outputs and their gradients
    layer_outputs = {}
    layer_grads = {}
    handles = []

    def make_forward_hook(layer_idx):
        def hook(module, input, output):
            # output is typically (hidden_states, ...) tuple or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store with requires_grad so we can compute gradient
            layer_outputs[layer_idx] = hidden_states
        return hook

    def make_backward_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_output is gradient w.r.t. module output
            if isinstance(grad_output, tuple):
                grad = grad_output[0]
            else:
                grad = grad_output
            if grad is not None:
                layer_grads[layer_idx] = grad.detach().clone()
        return hook

    # Register hooks
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_forward_hook(i)))
        handles.append(layer.register_full_backward_hook(make_backward_hook(i)))

    try:
        for text in tqdm(texts, desc="Computing gradients"):
            # Clear stored values
            layer_outputs.clear()
            layer_grads.clear()

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length)
            input_ids = inputs.input_ids.to(device)

            if input_ids.shape[1] < 10:
                continue

            # Forward pass
            model.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Backward pass to get gradients
            loss.backward()

            # Collect gradient norms for each layer
            for i in range(num_layers):
                if i in layer_grads:
                    grad = layer_grads[i]
                    # Compute L2 norm per token, then mean
                    # grad shape: (batch, seq_len, hidden_dim)
                    grad_norm = grad.norm(dim=-1).mean().item()
                    layer_grad_norms[i].append(grad_norm)

    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()

    # Compute mean gradient norm per layer
    importance = {}
    for i in range(num_layers):
        if layer_grad_norms[i]:
            importance[i] = np.mean(layer_grad_norms[i])
        else:
            importance[i] = 0.0

    return importance


def compute_layer_importance_perturbation(model, tokenizer, texts, max_length=256,
                                          noise_scale=0.01):
    """
    Alternative: Compute importance via perturbation.

    For each layer, add small noise to output and measure PPL change.
    This doesn't require backward pass.
    """
    device = next(model.parameters()).device
    model.eval()

    # Get model structure
    if hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    if hasattr(transformer, 'layers'):
        layers = transformer.layers
    elif hasattr(transformer, 'decoder') and hasattr(transformer.decoder, 'layers'):
        layers = transformer.decoder.layers
    else:
        raise ValueError("Cannot find decoder layers")

    num_layers = len(layers)

    # Compute baseline loss
    baseline_losses = []
    with torch.no_grad():
        for text in texts[:16]:  # Use fewer samples for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length)
            input_ids = inputs.input_ids.to(device)
            if input_ids.shape[1] < 10:
                continue
            outputs = model(input_ids, labels=input_ids)
            baseline_losses.append(outputs.loss.item())

    baseline_loss = np.mean(baseline_losses)
    print(f"Baseline loss: {baseline_loss:.4f}")

    # For each layer, perturb and measure loss change
    importance = {}

    for layer_idx in tqdm(range(num_layers), desc="Perturbation analysis"):
        # Create hook to add noise
        noise_hook = None

        def make_noise_hook(scale):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    noise = torch.randn_like(hidden) * scale * hidden.std()
                    return (hidden + noise,) + output[1:]
                else:
                    noise = torch.randn_like(output) * scale * output.std()
                    return output + noise
            return hook

        handle = layers[layer_idx].register_forward_hook(make_noise_hook(noise_scale))

        perturbed_losses = []
        with torch.no_grad():
            for text in texts[:16]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=max_length)
                input_ids = inputs.input_ids.to(device)
                if input_ids.shape[1] < 10:
                    continue
                outputs = model(input_ids, labels=input_ids)
                perturbed_losses.append(outputs.loss.item())

        handle.remove()

        perturbed_loss = np.mean(perturbed_losses)
        # Importance = how much loss increases when layer is perturbed
        importance[layer_idx] = perturbed_loss - baseline_loss

    return importance, baseline_loss


def main():
    print("=" * 70)
    print("Phase 1.2: Measure Per-Layer Importance for QEP-SINQ")
    print("=" * 70)

    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\nGetting calibration texts...")
    texts = get_calibration_texts(num_samples=32)
    print(f"Got {len(texts)} calibration samples")

    # Method 1: Gradient-based importance
    print("\n" + "=" * 70)
    print("METHOD 1: Gradient-Based Importance")
    print("=" * 70)

    # Need to enable gradients for this
    for param in model.parameters():
        param.requires_grad_(True)

    grad_importance = compute_layer_importance_gradient(
        model, tokenizer, texts[:16], max_length=256
    )

    print("\nGradient-based importance per layer:")
    for i in sorted(grad_importance.keys()):
        print(f"  Layer {i:2d}: {grad_importance[i]:.6f}")

    # Normalize
    values = list(grad_importance.values())
    if max(values) > 0:
        normalized = {k: v / max(values) for k, v in grad_importance.items()}
    else:
        normalized = grad_importance

    print("\nNormalized importance (max=1.0):")
    for i in sorted(normalized.keys()):
        bar = "█" * int(normalized[i] * 40)
        print(f"  Layer {i:2d}: {normalized[i]:.3f} {bar}")

    # Method 2: Perturbation-based importance
    print("\n" + "=" * 70)
    print("METHOD 2: Perturbation-Based Importance")
    print("=" * 70)

    # Disable gradients for speed
    for param in model.parameters():
        param.requires_grad_(False)

    pert_importance, baseline = compute_layer_importance_perturbation(
        model, tokenizer, texts, max_length=256, noise_scale=0.01
    )

    print("\nPerturbation-based importance (loss increase):")
    for i in sorted(pert_importance.keys()):
        print(f"  Layer {i:2d}: {pert_importance[i]:.6f}")

    # Normalize
    values = list(pert_importance.values())
    if max(values) > 0:
        normalized_pert = {k: v / max(values) for k, v in pert_importance.items()}
    else:
        normalized_pert = pert_importance

    print("\nNormalized perturbation importance (max=1.0):")
    for i in sorted(normalized_pert.keys()):
        bar = "█" * int(max(0, normalized_pert[i]) * 40)
        print(f"  Layer {i:2d}: {normalized_pert[i]:.3f} {bar}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check variance in importance
    grad_values = list(grad_importance.values())
    pert_values = list(pert_importance.values())

    grad_cv = np.std(grad_values) / (np.mean(grad_values) + 1e-10)
    pert_cv = np.std(pert_values) / (np.mean(pert_values) + 1e-10)

    grad_range = max(grad_values) / (min(grad_values) + 1e-10) if min(grad_values) > 0 else float('inf')
    pert_range = max(pert_values) / (min(pert_values) + 1e-10) if min(pert_values) > 0 else float('inf')

    print(f"\nGradient method:")
    print(f"  Mean: {np.mean(grad_values):.6f}")
    print(f"  Std:  {np.std(grad_values):.6f}")
    print(f"  CV (coefficient of variation): {grad_cv:.3f}")
    print(f"  Max/Min ratio: {grad_range:.2f}x")

    print(f"\nPerturbation method:")
    print(f"  Mean: {np.mean(pert_values):.6f}")
    print(f"  Std:  {np.std(pert_values):.6f}")
    print(f"  CV (coefficient of variation): {pert_cv:.3f}")
    print(f"  Max/Min ratio: {pert_range:.2f}x")

    # Decision point
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    # If CV > 0.3 or Max/Min > 2x, importance is non-uniform enough to matter
    significant_variation = (grad_cv > 0.3 or pert_cv > 0.3 or
                            grad_range > 2.0 or pert_range > 2.0)

    if significant_variation:
        print("\n✓ PREMISE VALID: Layer importance varies significantly.")
        print("  QEP approach can potentially improve PPL by weighting layers.")
        print("  Proceed to Phase 1.3 (Ideation)")
    else:
        print("\n✗ PREMISE WEAK: Layer importance is relatively uniform.")
        print("  QEP weighting may not provide significant benefit.")
        print("  Consider alternative approaches or proceed with caution.")

    # Save results
    results = {
        "model": model_name,
        "num_layers": len(grad_importance),
        "gradient_importance": grad_importance,
        "perturbation_importance": pert_importance,
        "baseline_loss": baseline,
        "gradient_cv": grad_cv,
        "perturbation_cv": pert_cv,
        "gradient_range": grad_range,
        "perturbation_range": pert_range,
        "significant_variation": significant_variation,
    }

    with open("results/layer_importance.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/layer_importance.json")

    return results


if __name__ == "__main__":
    main()
