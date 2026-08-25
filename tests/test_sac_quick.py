"""
Test Scale-Aware Compensation (SAC).

Key insight: Standard OBS compensation doesn't account for quantization.
The compensation delta might move a weight AWAY from its nearest quantization
level, increasing quantization error.

SAC: Adjust OBS delta to be "quantization-friendly":
1. Compute standard OBS delta
2. Compute where weight would land after delta
3. Snap to nearest quantization level
4. Only apply delta if it reduces overall error

This is different from QAOBS because:
- QAOBS: Iterative refinement AFTER quantization
- SAC: Pre-emptive adjustment BEFORE quantization

This is novel because it unifies compensation with quantization in a single step.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn.functional as F
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_activation_norms,
    create_sparsity_mask,
    compute_hessian_inverse,
)


def quantize_to_levels(w, scale, zero, nbits):
    """Quantize a value and return both quantized int and dequantized float."""
    max_int = 2**nbits - 1
    q_int = torch.clamp(torch.round(w / scale + zero), 0, max_int)
    q_deq = (q_int - zero) * scale
    return q_int, q_deq


def compute_group_scales(W_norm, mask, nbits, group_size):
    """Compute per-group scales for quantization."""
    K, N = W_norm.shape
    device = W_norm.device
    max_int = 2**nbits - 1

    if group_size is None or group_size <= 0 or N <= group_size:
        # Per-row quantization
        W_for_scale = W_norm.clone()
        W_for_scale[mask == 0] = 0  # Don't include pruned weights
        max_val = W_for_scale.amax(dim=1, keepdim=True)
        min_val = W_for_scale.amin(dim=1, keepdim=True)
        scales = (max_val - min_val).clamp(min=1e-8) / max_int
        zeros = -torch.round(min_val / scales)
        return scales, zeros, None

    # Group-wise quantization
    if N % group_size != 0:
        pad_size = group_size - (N % group_size)
        W_norm = F.pad(W_norm, (0, pad_size))
        mask = F.pad(mask, (0, pad_size))
        N_padded = W_norm.shape[1]
    else:
        N_padded = N
        pad_size = 0

    n_groups = N_padded // group_size
    W_grouped = W_norm.view(K, n_groups, group_size)
    mask_grouped = mask.view(K, n_groups, group_size)

    # For each group, compute min/max of non-masked weights
    W_for_max = W_grouped.clone()
    W_for_max[mask_grouped == 0] = -1e10
    max_val = W_for_max.amax(dim=2, keepdim=True)

    W_for_min = W_grouped.clone()
    W_for_min[mask_grouped == 0] = 1e10
    min_val = W_for_min.amin(dim=2, keepdim=True)

    # Handle fully masked groups
    fully_masked = (mask_grouped.sum(dim=2, keepdim=True) == 0)
    max_val = torch.where(fully_masked, torch.ones_like(max_val), max_val)
    min_val = torch.where(fully_masked, torch.zeros_like(min_val), min_val)

    scales = (max_val - min_val).clamp(min=1e-8) / max_int
    zeros = -torch.round(min_val / scales)

    return scales, zeros, (n_groups, group_size, pad_size)


def baseline_with_obs(W, X, sparsity, nbits, group_size=64):
    """Baseline: Standard OBS compensation."""
    K, N = W.shape
    device = W.device

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Importance and mask
    act_norms = compute_activation_norms(X)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    mask = create_sparsity_mask(importance, sparsity)

    # OBS compensation
    X_float = X.float()
    if X_float.dim() == 3:
        X_float = X_float.view(-1, X_float.shape[-1])
    n_samples = min(X_float.shape[0], 256)
    X_sub = X_float[:n_samples]

    H_inv = compute_hessian_inverse(X_sub)
    H_inv_diag = H_inv.diag()

    W_compensated = W.clone()
    n_prune_per_row = int(N * sparsity)
    _, sorted_idx = importance.sort(dim=1)
    prune_idx = sorted_idx[:, :n_prune_per_row]

    for i in range(K):
        J = prune_idx[i]
        w_J = W[i, J]

        H_inv_J = H_inv[:, J]
        H_inv_JJ = H_inv_diag[J]

        valid = H_inv_JJ.abs() > 1e-10
        scale_factor = torch.zeros_like(w_J)
        scale_factor[valid] = w_J[valid] / H_inv_JJ[valid]

        delta = -H_inv_J @ scale_factor
        delta[J] = 0

        W_compensated[i] = W_compensated[i] + delta

    W_compensated = W_compensated * mask

    # Quantize
    scale_factor = mu2.unsqueeze(1) * mu1.unsqueeze(0)
    W_norm_comp = W_compensated / scale_factor
    # Ensure 2D
    if W_norm_comp.dim() != 2:
        W_norm_comp = W_norm_comp.squeeze()
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm_comp, min_max, group_size=group_size)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0) * mask

    return W_deq, mask


def sac_sparse_quant(W, X, sparsity, nbits, group_size=64, snap_threshold=0.5):
    """
    Scale-Aware Compensation (SAC).

    Modify OBS delta to be quantization-friendly.
    """
    K, N = W.shape
    device = W.device

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Importance and mask (same as baseline)
    act_norms = compute_activation_norms(X)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    mask = create_sparsity_mask(importance, sparsity)

    # Precompute quantization scales
    W_sparse_norm = W_norm * mask
    scales, zeros, group_info = compute_group_scales(W_sparse_norm, mask, nbits, group_size)

    # OBS setup
    X_float = X.float()
    if X_float.dim() == 3:
        X_float = X_float.view(-1, X_float.shape[-1])
    n_samples = min(X_float.shape[0], 256)
    X_sub = X_float[:n_samples]

    H_inv = compute_hessian_inverse(X_sub)
    H_inv_diag = H_inv.diag()

    # SAC compensation - process each row
    W_compensated = W.clone()
    n_prune_per_row = int(N * sparsity)
    _, sorted_idx = importance.sort(dim=1)
    prune_idx = sorted_idx[:, :n_prune_per_row]

    max_int = 2**nbits - 1

    for i in range(K):
        J = prune_idx[i]
        w_J = W[i, J]

        H_inv_J = H_inv[:, J]
        H_inv_JJ = H_inv_diag[J]

        valid = H_inv_JJ.abs() > 1e-10
        scale_factor = torch.zeros_like(w_J)
        scale_factor[valid] = w_J[valid] / H_inv_JJ[valid]

        delta = -H_inv_J @ scale_factor
        delta[J] = 0  # Don't change pruned weights

        # SAC modification: Adjust delta to be quantization-friendly
        for j in range(N):
            if mask[i, j] == 0:
                continue  # Skip pruned weights

            w_orig = W[i, j]
            w_obs = W[i, j] + delta[j]

            # Get quantization scale for this position
            if group_info is not None:
                n_groups, gs, pad_size = group_info
                group_idx = j // gs
                if group_idx < n_groups:
                    s = scales[i, group_idx, 0]
                    z = zeros[i, group_idx, 0]
                else:
                    s = scales[i, -1, 0]
                    z = zeros[i, -1, 0]
            else:
                s = scales[i, 0]
                z = zeros[i, 0]

            # Convert to normalized space for quantization
            w_orig_norm = w_orig / (mu2[i] * mu1[j])
            w_obs_norm = w_obs / (mu2[i] * mu1[j])

            # Quantize both
            q_orig = torch.clamp(torch.round(w_orig_norm / s + z), 0, max_int)
            q_obs = torch.clamp(torch.round(w_obs_norm / s + z), 0, max_int)

            # Dequantize
            deq_orig = (q_orig - z) * s
            deq_obs = (q_obs - z) * s

            # Compute errors relative to original
            err_orig = (w_orig_norm - deq_orig).abs()
            err_obs = (w_obs_norm - deq_obs).abs()

            # SAC rule: only apply delta if it reduces quantization error
            # or if the output error reduction outweighs quantization increase
            if err_obs > err_orig * (1 + snap_threshold):
                # Delta increases quantization error too much - reduce it
                # Snap to nearest quantization level instead
                best_q = q_orig  # Start with original quantization
                best_err = (w_orig_norm - deq_orig).abs()

                for q_candidate in [q_orig - 1, q_orig, q_orig + 1]:
                    if q_candidate < 0 or q_candidate > max_int:
                        continue
                    deq_cand = (q_candidate - z) * s
                    err_cand = (w_obs_norm - deq_cand).abs()
                    if err_cand < best_err:
                        best_err = err_cand
                        best_q = q_candidate

                # Adjust delta to land on best quantization level
                target_norm = (best_q - z) * s
                target_orig = target_norm * mu2[i] * mu1[j]
                delta[j] = target_orig - w_orig

        W_compensated[i] = W[i] + delta

    W_compensated = W_compensated * mask

    # Quantize
    scale_factor = mu2.unsqueeze(1) * mu1.unsqueeze(0)
    W_norm_comp = W_compensated / scale_factor
    if W_norm_comp.dim() != 2:
        W_norm_comp = W_norm_comp.squeeze()
    min_max = [0, 2**nbits - 1]
    Q, scales_final, zeros_final, _ = quantize_rtn(W_norm_comp, min_max, group_size=group_size)

    # Dequantize
    if len(scales_final.shape) == 3:
        n_groups = scales_final.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros_final) * scales_final
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros_final) * scales_final

    W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0) * mask

    return W_deq, mask


def test_sac():
    """Test SAC approach."""
    print("="*60)
    print("TEST: Scale-Aware Compensation (SAC)")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 64, 128
    batch = 64
    sparsity = 0.35
    nbits = 3
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    # Target output
    Y_target = X @ W.T

    # Baseline with OBS
    W_base, mask_base = baseline_with_obs(W.float(), X, sparsity, nbits, group_size)
    Y_base = X @ W_base.T
    out_mse_base = ((Y_target - Y_base) ** 2).mean().item()
    wgt_mse_base = ((W - W_base) ** 2).mean().item()

    print(f"\nConfig: {K}x{N}, {sparsity*100:.0f}% sparsity, {nbits}-bit")
    print(f"\nBaseline OBS:")
    print(f"  Weight MSE: {wgt_mse_base:.6f}")
    print(f"  Output MSE: {out_mse_base:.6f}")

    # SAC with different thresholds
    for thresh in [0.0, 0.25, 0.5, 1.0]:
        W_sac, mask_sac = sac_sparse_quant(W.float(), X, sparsity, nbits, group_size, snap_threshold=thresh)
        Y_sac = X @ W_sac.T
        out_mse_sac = ((Y_target - Y_sac) ** 2).mean().item()
        wgt_mse_sac = ((W - W_sac) ** 2).mean().item()

        wgt_imp = (wgt_mse_base - wgt_mse_sac) / wgt_mse_base * 100
        out_imp = (out_mse_base - out_mse_sac) / out_mse_base * 100

        print(f"\nSAC (threshold={thresh}):")
        print(f"  Weight MSE: {wgt_mse_sac:.6f} ({wgt_imp:+.2f}%)")
        print(f"  Output MSE: {out_mse_sac:.6f} ({out_imp:+.2f}%)")


def test_sac_real():
    """Test SAC on real weights."""
    print("\n" + "="*60)
    print("TEST: SAC on Real Qwen-0.5B Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("Loading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        W = model.model.layers[0].mlp.gate_proj.weight.data.float()
        K, N = W.shape
        device = W.device

        print(f"\nLayer 0 gate_proj: {K}x{N}")

        batch = 64
        X = torch.randn(batch, N, device=device, dtype=torch.float32)
        Y_target = X @ W.T

        sparsity = 0.35
        nbits = 3
        group_size = 64

        # Baseline
        W_base, _ = baseline_with_obs(W, X, sparsity, nbits, group_size)
        Y_base = X @ W_base.T
        out_mse_base = ((Y_target - Y_base) ** 2).mean().item()

        print(f"\nBaseline output MSE: {out_mse_base:.6f}")

        # SAC
        W_sac, _ = sac_sparse_quant(W, X, sparsity, nbits, group_size, snap_threshold=0.5)
        Y_sac = X @ W_sac.T
        out_mse_sac = ((Y_target - Y_sac) ** 2).mean().item()

        improvement = (out_mse_base - out_mse_sac) / out_mse_base * 100
        print(f"SAC output MSE:      {out_mse_sac:.6f} ({improvement:+.2f}%)")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_sac()
    test_sac_real()
