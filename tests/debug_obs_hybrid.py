"""
Debug: Why is OBS-SINQ Hybrid failing so catastrophically?

The test showed OBS-SINQ producing PPL in the millions while SCAB gets ~15-200.
This is unexpected - OBS importance should work reasonably well.

Possible issues:
1. OBS importance polarity is wrong (pruning important weights)
2. Compensation is overcorrecting
3. Sinkhorn interaction is incompatible with OBS-pruned weights
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_hessian_inverse,
    compute_importance_scores,
    create_sparsity_mask
)


def debug_single_layer():
    """Debug importance scoring on a single layer."""
    print("="*70)
    print("DEBUGGING OBS vs INVERSE-μ IMPORTANCE")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"

    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Get activations
    activations_cache = {}

    def capture_hook(name):
        def hook(module, input, output):
            activations_cache[name] = input[0].detach()
        return hook

    layer = model.model.layers[0]
    handle = layer.self_attn.q_proj.register_forward_hook(capture_hook('q_proj'))

    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        _ = model(tokens.input_ids.to(device))
    handle.remove()

    W = layer.self_attn.q_proj.weight.data.clone().float()
    X = activations_cache['q_proj'].float()
    K, N = W.shape

    print(f"\nWeight shape: {K} x {N}")
    print(f"Activation shape: {X.shape}")

    # 1. Compute Sinkhorn factors
    print("\n" + "-"*50)
    print("SINKHORN FACTORS")
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    print(f"μ₁ (column): min={mu1.min():.4f}, max={mu1.max():.4f}, mean={mu1.mean():.4f}")
    print(f"μ₂ (row):    min={mu2.min():.4f}, max={mu2.max():.4f}, mean={mu2.mean():.4f}")

    # 2. Compute both importance scores
    print("\n" + "-"*50)
    print("IMPORTANCE SCORES COMPARISON")

    # Inverse-μ importance (SCAB)
    from sinq.sparse_quant import compute_activation_norms
    act_norms = compute_activation_norms(X)
    inv_mu_importance = compute_importance_scores(W, mu1, mu2, act_norms, method='sinq_wanda_inverse')
    print(f"\nInverse-μ importance:")
    print(f"  min={inv_mu_importance.min():.6f}, max={inv_mu_importance.max():.6f}")
    print(f"  mean={inv_mu_importance.mean():.6f}, std={inv_mu_importance.std():.6f}")

    # OBS importance
    X_flat = X.view(-1, X.shape[-1])[:256]
    H_inv = compute_hessian_inverse(X_flat)
    H_inv_diag = H_inv.diag()

    # Standard OBS: error = w² / H_inv_diag
    # Higher error = more important to keep
    # So importance = w² * H_inv_diag (larger = more important)
    obs_importance = (W ** 2) * H_inv_diag.view(1, -1)

    print(f"\nOBS importance (W² * H_inv_diag):")
    print(f"  min={obs_importance.min():.6f}, max={obs_importance.max():.6f}")
    print(f"  mean={obs_importance.mean():.6f}, std={obs_importance.std():.6f}")

    # Check if they agree on which weights are important
    inv_mu_flat = inv_mu_importance.view(-1)
    obs_flat = obs_importance.view(-1)

    # Correlation
    correlation = torch.corrcoef(torch.stack([inv_mu_flat, obs_flat]))[0, 1]
    print(f"\nCorrelation between importance scores: {correlation:.4f}")

    # 3. Compare pruning decisions at 50% sparsity
    print("\n" + "-"*50)
    print("PRUNING DECISIONS AT 50% SPARSITY")

    n_weights = K * N
    n_prune = int(n_weights * 0.5)

    # Inverse-μ mask
    inv_threshold = torch.kthvalue(inv_mu_flat, n_prune).values
    inv_mask = (inv_mu_flat > inv_threshold).view(K, N)

    # OBS mask
    obs_threshold = torch.kthvalue(obs_flat, n_prune).values
    obs_mask = (obs_flat > obs_threshold).view(K, N)

    # Agreement
    agreement = (inv_mask == obs_mask).float().mean()
    print(f"Mask agreement: {agreement*100:.1f}%")

    # 4. Check the actual pruning error for both
    print("\n" + "-"*50)
    print("PRUNING ERROR ANALYSIS (before quantization)")

    # Inverse-μ pruning (no compensation)
    W_inv_pruned = W * inv_mask.float()
    mse_inv = ((W - W_inv_pruned) ** 2).mean()
    print(f"Inverse-μ pruning MSE: {mse_inv:.6f}")

    # OBS pruning (no compensation)
    W_obs_pruned = W * obs_mask.float()
    mse_obs = ((W - W_obs_pruned) ** 2).mean()
    print(f"OBS pruning MSE: {mse_obs:.6f}")

    # 5. Now add OBS compensation to both
    print("\n" + "-"*50)
    print("WITH OBS COMPENSATION")

    def apply_obs_compensation(W, mask, H_inv, H_inv_diag):
        W_comp = W.clone()
        pruned_mask = (1 - mask.float())
        for i in range(W.shape[0]):
            pruned_weights = W[i] * pruned_mask[i]
            compensation = -H_inv @ (pruned_weights / H_inv_diag)
            W_comp[i] = W[i] * mask[i].float() + compensation * mask[i].float()
        return W_comp

    W_inv_comp = apply_obs_compensation(W, inv_mask, H_inv, H_inv_diag)
    W_obs_comp = apply_obs_compensation(W, obs_mask, H_inv, H_inv_diag)

    mse_inv_comp = ((W - W_inv_comp) ** 2).mean()
    mse_obs_comp = ((W - W_obs_comp) ** 2).mean()
    print(f"Inverse-μ + OBS comp MSE: {mse_inv_comp:.6f}")
    print(f"OBS + OBS comp MSE: {mse_obs_comp:.6f}")

    # 6. Check output error (what matters for PPL)
    print("\n" + "-"*50)
    print("OUTPUT RECONSTRUCTION ERROR (Y = XW^T)")

    Y_original = X_flat @ W.T
    Y_inv = X_flat @ W_inv_comp.T
    Y_obs = X_flat @ W_obs_comp.T

    output_mse_inv = ((Y_original - Y_inv) ** 2).mean()
    output_mse_obs = ((Y_original - Y_obs) ** 2).mean()
    print(f"Inverse-μ output MSE: {output_mse_inv:.6f}")
    print(f"OBS output MSE: {output_mse_obs:.6f}")

    # 7. Now add SINQ quantization to see where things go wrong
    print("\n" + "-"*50)
    print("WITH SINQ QUANTIZATION")

    # Apply Sinkhorn normalization to compensated weights
    W_inv_norm = W_inv_comp / (mu2.view(-1, 1) * mu1.view(1, -1))
    W_obs_norm = W_obs_comp / (mu2.view(-1, 1) * mu1.view(1, -1))

    # Check normalized weight ranges
    print(f"\nNormalized weight ranges:")
    print(f"  Inverse-μ: min={W_inv_norm.min():.4f}, max={W_inv_norm.max():.4f}")
    print(f"  OBS:       min={W_obs_norm.min():.4f}, max={W_obs_norm.max():.4f}")

    # Quantize both
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]

    q_inv, scales_inv, zeros_inv, _ = quantize_rtn(W_inv_norm, min_max, group_size=group_size)
    q_obs, scales_obs, zeros_obs, _ = quantize_rtn(W_obs_norm, min_max, group_size=group_size)

    print(f"\nQuantized ranges (should be 0-15):")
    print(f"  Inverse-μ: min={q_inv.min()}, max={q_inv.max()}")
    print(f"  OBS:       min={q_obs.min()}, max={q_obs.max()}")

    # Dequantize
    def dequant(q, scales, zeros, mu1, mu2, mask, group_size):
        K, N = q.shape
        n_groups = scales.shape[1]
        q_grouped = q.float().view(K, n_groups, group_size)
        W_deq = (q_grouped - zeros.float()) * scales.float()
        W_deq = W_deq.view(K, N)
        W_deq = W_deq * mu2.view(-1, 1) * mu1.view(1, -1)
        W_deq = W_deq * mask.float()
        return W_deq

    # Apply mask
    n_groups = N // group_size
    W_deq_inv = dequant(q_inv, scales_inv.view(-1, n_groups, 1), zeros_inv.view(-1, n_groups, 1),
                        mu1, mu2, inv_mask, group_size)
    W_deq_obs = dequant(q_obs, scales_obs.view(-1, n_groups, 1), zeros_obs.view(-1, n_groups, 1),
                        mu1, mu2, obs_mask, group_size)

    mse_deq_inv = ((W - W_deq_inv) ** 2).mean()
    mse_deq_obs = ((W - W_deq_obs) ** 2).mean()
    print(f"\nFinal MSE after dequantization:")
    print(f"  Inverse-μ: {mse_deq_inv:.6f}")
    print(f"  OBS:       {mse_deq_obs:.6f}")

    # Output reconstruction
    Y_deq_inv = X_flat @ W_deq_inv.T
    Y_deq_obs = X_flat @ W_deq_obs.T

    output_mse_deq_inv = ((Y_original - Y_deq_inv) ** 2).mean()
    output_mse_deq_obs = ((Y_original - Y_deq_obs) ** 2).mean()
    print(f"\nFinal output MSE:")
    print(f"  Inverse-μ: {output_mse_deq_inv:.6f}")
    print(f"  OBS:       {output_mse_deq_obs:.6f}")

    # 8. Diagnose: Are OBS-pruned weights creating outliers in normalized space?
    print("\n" + "-"*50)
    print("OUTLIER ANALYSIS")

    # Check for outliers in the normalized compensated weights
    inv_outliers = (W_inv_norm.abs() > 10).sum()
    obs_outliers = (W_obs_norm.abs() > 10).sum()
    print(f"Outliers (|W_norm| > 10):")
    print(f"  Inverse-μ: {inv_outliers}")
    print(f"  OBS:       {obs_outliers}")

    # Check scale ranges
    print(f"\nScale ranges:")
    print(f"  Inverse-μ: min={scales_inv.min():.6f}, max={scales_inv.max():.6f}")
    print(f"  OBS:       min={scales_obs.min():.6f}, max={scales_obs.max():.6f}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)


if __name__ == "__main__":
    debug_single_layer()
