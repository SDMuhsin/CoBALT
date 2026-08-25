"""
QEP-SINQ: Quantization Error Propagation for SINQ

This module implements importance-weighted Sinkhorn iterations for SINQ quantization.
High-importance layers (early layers with high gradient norm) get more iterations,
while low-importance layers get fewer iterations.

Key findings from validation:
- Sinkhorn iterations DO affect PPL (order 1: 35.03, order 32: 31.74)
- Layer importance varies 134x (Layer 0 vs Layer 27)
- Diminishing returns after ~8 iterations

Usage:
    from sinq.qep_sinq import compute_layer_importance, get_qep_sinkhorn_order

    # Compute importance weights (one-time calibration)
    importance = compute_layer_importance(model, tokenizer, calibration_texts)

    # Get iteration count for a specific layer
    order = get_qep_sinkhorn_order(layer_idx, importance, base_order=8, alpha=0.5)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


def compute_layer_importance(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    device: Optional[str] = None
) -> Dict[int, float]:
    """
    Compute importance of each transformer layer using gradient magnitude.

    For each layer l:
        w_l = E[||∂L_CE/∂h_l||]

    where L_CE is cross-entropy loss and h_l is the layer output.

    Args:
        model: The transformer model to analyze
        tokenizer: Tokenizer for encoding texts
        texts: List of calibration texts
        max_length: Maximum sequence length
        device: Device to use (defaults to model's device)

    Returns:
        Dictionary mapping layer_idx to normalized importance weight
    """
    if device is None:
        device = next(model.parameters()).device

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
        raise ValueError("Cannot find decoder layers in model")

    num_layers = len(layers)

    # Storage for gradients
    layer_grad_norms = {i: [] for i in range(num_layers)}

    # Register hooks
    layer_grads = {}
    handles = []

    def make_backward_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad = grad_output[0]
            else:
                grad = grad_output
            if grad is not None:
                layer_grads[layer_idx] = grad.detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_full_backward_hook(make_backward_hook(i)))

    # Enable gradients temporarily
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(True)

    try:
        for text in tqdm(texts, desc="Computing layer importance"):
            layer_grads.clear()

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(device)

            if input_ids.shape[1] < 10:
                continue

            model.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for i in range(num_layers):
                if i in layer_grads:
                    grad = layer_grads[i]
                    grad_norm = grad.norm(dim=-1).mean().item()
                    layer_grad_norms[i].append(grad_norm)

    finally:
        # Cleanup
        for handle in handles:
            handle.remove()
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad.get(name, False))

    # Compute mean importance per layer
    importance = {}
    for i in range(num_layers):
        if layer_grad_norms[i]:
            importance[i] = np.mean(layer_grad_norms[i])
        else:
            importance[i] = 1.0  # Default

    # Normalize so mean = 1.0
    mean_importance = np.mean(list(importance.values()))
    if mean_importance > 0:
        importance = {k: v / mean_importance for k, v in importance.items()}

    return importance


def get_qep_sinkhorn_order(
    layer_idx: int,
    importance: Dict[int, float],
    base_order: int = 8,
    alpha: float = 0.5,
    min_order: int = 2,
    max_order: int = 32
) -> int:
    """
    Get the Sinkhorn iteration count for a layer based on its importance.

    Formula:
        order = clamp(round(base_order * importance^alpha), min_order, max_order)

    Args:
        layer_idx: Index of the layer
        importance: Dictionary of layer importance weights
        base_order: Base number of iterations (default 8)
        alpha: Exponent for importance weighting (default 0.5)
        min_order: Minimum iterations (default 2)
        max_order: Maximum iterations (default 32)

    Returns:
        Number of Sinkhorn iterations for this layer
    """
    w = importance.get(layer_idx, 1.0)

    # Handle edge cases
    if w <= 0:
        return min_order

    # Compute weighted order
    order = base_order * (w ** alpha)
    order = int(round(order))

    # Clamp to bounds
    return max(min_order, min(max_order, order))


def create_qep_quantize_dual_scale_shift(importance: Dict[int, float], **kwargs):
    """
    Create a patched quantize_dual_scale_shift function that uses QEP iteration counts.

    This is a factory function that returns a replacement for quantize_dual_scale_shift
    which looks up the current layer and uses importance-weighted iterations.

    Args:
        importance: Dictionary of layer importance weights
        **kwargs: Arguments for get_qep_sinkhorn_order

    Returns:
        Patched quantize_dual_scale_shift function
    """
    from sinq.sinkhorn import sinkhorn_log

    # Track current layer (will be set by the quantization loop)
    current_layer_idx = [0]

    def set_current_layer(idx):
        current_layer_idx[0] = idx

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        """Patched version with QEP iteration allocation."""
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Get QEP order for current layer
        layer_idx = current_layer_idx[0]
        order = get_qep_sinkhorn_order(layer_idx, importance, **kwargs)

        # Use QEP order instead of hardcoded 16
        matrix_normalized, mu1, mu2 = sinkhorn_log(matrix, order)

        if not ('sinq' in method):
            matrix_normalized = matrix_normalized * mu1 * mu2
            mu1 = torch.ones_like(mu1)
            mu2 = torch.ones_like(mu2)

        if 'awq' in method:
            matrix_normalized = matrix_normalized * awq_scale
            mu1 = mu1 / awq_scale.float()

        # Standard quantization
        max_val = matrix_normalized.amax(dim=1, keepdim=True)
        min_val = matrix_normalized.amin(dim=1, keepdim=True)
        max_int = min_max[1]
        min_int = min_max[0]
        scales = (max_val - min_val).clamp(min=1e-4) / max_int
        zeros = -torch.round(min_val / scales)
        q = torch.clamp(torch.round(matrix_normalized / scales + zeros), min_int, max_int).to(torch.int8)

        scales2 = torch.ones(1, matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
        scales = scales * mu2

        q = q.to(dtype).to(dev)
        s1 = scales.to(dtype)
        s2 = scales2.to(dtype)
        z = zeros.to(dtype).to(dev)

        return q, s1.to(dev), s2.to(dev), z

    return patched_quantize_dual_scale_shift, set_current_layer


# Pre-computed importance weights for Qwen models
# These were measured on Qwen3-1.7B and can be used as defaults
QWEN_1_7B_IMPORTANCE = {
    0: 4.80, 1: 3.76, 2: 3.12, 3: 2.60, 4: 2.33, 5: 1.99, 6: 1.68,
    7: 1.46, 8: 1.20, 9: 0.98, 10: 0.77, 11: 0.65, 12: 0.56, 13: 0.46,
    14: 0.38, 15: 0.29, 16: 0.22, 17: 0.16, 18: 0.13, 19: 0.09,
    20: 0.07, 21: 0.06, 22: 0.05, 23: 0.05, 24: 0.04, 25: 0.04,
    26: 0.04, 27: 0.04
}


def print_qep_allocation(importance: Dict[int, float], base_order: int = 8, alpha: float = 0.5):
    """Print the QEP iteration allocation for each layer."""
    print(f"QEP Iteration Allocation (base={base_order}, alpha={alpha})")
    print("-" * 50)
    print(f"{'Layer':<8}{'Importance':<12}{'Iterations':<12}")
    print("-" * 50)

    total_iters = 0
    for layer_idx in sorted(importance.keys()):
        w = importance[layer_idx]
        order = get_qep_sinkhorn_order(layer_idx, importance, base_order, alpha)
        total_iters += order
        print(f"{layer_idx:<8}{w:<12.3f}{order:<12}")

    uniform_total = len(importance) * base_order
    print("-" * 50)
    print(f"Total iterations: {total_iters} (uniform: {uniform_total})")
    print(f"Speedup: {uniform_total / total_iters:.2f}x")
