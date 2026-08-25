"""
Test Quantization-Aware OBS (QAOBS) hypothesis.

Standard OBS: Compensate for pruning error in continuous weight space
QAOBS: Compensate for the COMBINED pruning + quantization error

Key insight: Standard OBS compensation doesn't account for quantization.
After quantization, the continuous-optimal compensation may no longer be optimal.

Approach:
1. Prune based on importance (standard)
2. Quantize the sparse weights
3. Compute compensation by comparing quantized output to original output
4. Apply compensation to remaining weights and re-quantize

This is an iterative refinement that accounts for quantization.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn.functional as F
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_activation_norms,
    compute_importance_scores,
    create_sparsity_mask,
    compute_hessian_inverse,
)


def quantize_and_dequant(W_norm, nbits, group_size, mu1, mu2, mask):
    """Quantize and dequantize weights."""
    K, N = W_norm.shape
    min_max = [0, 2**nbits - 1]

    Q, scales, zeros, _ = quantize_rtn(W_norm * mask, min_max, group_size=group_size)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return W_deq, Q, scales, zeros


def standard_obs_prune(W, X, sparsity, mu1, mu2, nbits, group_size):
    """
    Standard approach: OBS compensation, then quantize.
    """
    K, N = W.shape
    device = W.device

    # Sinkhorn normalize
    W_norm = W / (mu2 * mu1)

    # Compute importance
    act_norms = torch.norm(X.float(), dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2

    # Create mask
    mask = create_sparsity_mask(importance, sparsity)

    # Standard OBS compensation
    X_float = X.float()
    if X_float.dim() == 3:
        X_float = X_float.view(-1, X_float.shape[-1])
    n_samples = min(X_float.shape[0], 256)
    X_float = X_float[:n_samples]

    H_inv = compute_hessian_inverse(X_float)
    H_inv_diag = H_inv.diag()

    # Compensate each row
    W_compensated = W.clone()
    n_prune_per_row = int(N * sparsity)
    _, sorted_indices = importance.sort(dim=1)
    prune_indices = sorted_indices[:, :n_prune_per_row]

    for i in range(K):
        J = prune_indices[i]
        w_J = W[i, J]

        H_inv_J = H_inv[:, J]
        H_inv_JJ = H_inv_diag[J]

        valid = H_inv_JJ.abs() > 1e-10
        scale = torch.zeros_like(w_J)
        scale[valid] = w_J[valid] / H_inv_JJ[valid]

        delta = -H_inv_J @ scale
        delta[J] = 0

        W_compensated[i] = W_compensated[i] + delta

    # Apply mask
    W_compensated = W_compensated * mask

    # Quantize
    W_norm_comp = W_compensated / (mu2 * mu1)
    W_deq, Q, scales, zeros = quantize_and_dequant(
        W_norm_comp, nbits, group_size, mu1, mu2, mask
    )

    return W_deq, mask


def qaobs_prune(W, X, sparsity, mu1, mu2, nbits, group_size, n_iter=3):
    """
    Quantization-Aware OBS: Iteratively refine compensation accounting for quantization.

    Approach:
    1. Start with standard pruning mask
    2. Quantize
    3. Compute residual error (original output - quantized output)
    4. Apply additional compensation to reduce residual
    5. Re-quantize and repeat
    """
    K, N = W.shape
    device = W.device

    # Sinkhorn normalize
    W_norm = W / (mu2 * mu1)

    # Compute importance
    act_norms = torch.norm(X.float(), dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2

    # Create mask
    mask = create_sparsity_mask(importance, sparsity)

    # Start with the sparse weights (no compensation yet)
    W_current = W * mask

    X_float = X.float()
    if X_float.dim() == 3:
        X_float = X_float.view(-1, X_float.shape[-1])
    n_samples = min(X_float.shape[0], 256)
    X_sub = X_float[:n_samples]

    # Precompute target output
    target_output = X_sub @ W.T  # [n_samples, K]

    H = X_sub.T @ X_sub
    damping = max(0.01 * H.diag().mean().item(), 1e-3)
    H = H + damping * torch.eye(N, device=device, dtype=H.dtype)
    H_inv = torch.linalg.inv(H)

    for it in range(n_iter):
        # Quantize current weights
        W_norm_current = W_current / (mu2 * mu1)
        W_deq, Q, scales, zeros = quantize_and_dequant(
            W_norm_current, nbits, group_size, mu1, mu2, mask
        )

        # Compute quantized output
        quant_output = X_sub @ W_deq.T  # [n_samples, K]

        # Compute residual for each row
        residual = target_output - quant_output  # [n_samples, K]

        # For each row, find correction to remaining weights
        # We want: X @ delta_W = residual
        # Optimal: delta_W = H_inv @ X.T @ residual (per row)

        delta_W = torch.zeros_like(W)
        for i in range(K):
            r_i = residual[:, i]  # [n_samples]
            # delta_w_i = H_inv @ X.T @ r_i
            grad = X_sub.T @ r_i  # [N]
            delta_w = H_inv @ grad
            # Only update non-pruned weights
            delta_W[i] = delta_w * mask[i]

        # Apply correction
        W_current = W_current + delta_W * 0.5  # Damped update

        # Clip to reasonable range
        W_current = W_current.clamp(-W.abs().max() * 2, W.abs().max() * 2)

    # Final quantization
    W_norm_final = W_current / (mu2 * mu1)
    W_deq_final, _, _, _ = quantize_and_dequant(
        W_norm_final, nbits, group_size, mu1, mu2, mask
    )

    return W_deq_final, mask


def test_qaobs_synthetic():
    """Test QAOBS on synthetic data."""
    print("="*60)
    print("TEST: Quantization-Aware OBS (QAOBS)")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 64
    sparsity = 0.35
    nbits = 3
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    # Sinkhorn on original
    _, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    print(f"\nConfig: {K}x{N}, {sparsity*100:.0f}% sparsity, {nbits}-bit")

    # Standard OBS
    W_std, mask_std = standard_obs_prune(W.float(), X, sparsity, mu1, mu2, nbits, group_size)
    mse_std = ((W - W_std) ** 2).mean().item()

    # Compute output reconstruction error (more meaningful than MSE)
    X_sub = X[:min(batch, 64)]
    target_out = X_sub @ W.T
    std_out = X_sub @ W_std.T
    output_error_std = ((target_out - std_out) ** 2).mean().item()

    print(f"\nStandard OBS:")
    print(f"  Weight MSE: {mse_std:.6f}")
    print(f"  Output MSE: {output_error_std:.6f}")

    # QAOBS with different iterations
    for n_iter in [1, 3, 5]:
        W_qaobs, mask_qaobs = qaobs_prune(
            W.float(), X, sparsity, mu1, mu2, nbits, group_size, n_iter=n_iter
        )
        mse_qaobs = ((W - W_qaobs) ** 2).mean().item()

        qaobs_out = X_sub @ W_qaobs.T
        output_error_qaobs = ((target_out - qaobs_out) ** 2).mean().item()

        improvement = (output_error_std - output_error_qaobs) / output_error_std * 100

        print(f"\nQAOBS (n_iter={n_iter}):")
        print(f"  Weight MSE: {mse_qaobs:.6f}")
        print(f"  Output MSE: {output_error_qaobs:.6f}")
        print(f"  Output improvement: {improvement:+.2f}%")


def test_qaobs_real():
    """Test QAOBS on real Qwen-0.5B weights."""
    print("\n" + "="*60)
    print("TEST: QAOBS on Real Qwen-0.5B Weights")
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

        print(f"\nTesting on layer 0 gate_proj [{K}x{N}]")

        batch = 128
        X = torch.randn(batch, N, device=device, dtype=torch.float32)

        _, mu1, mu2 = sinkhorn_log(W, order=16)

        sparsity = 0.35
        nbits = 3
        group_size = 64

        # Standard OBS
        W_std, _ = standard_obs_prune(W, X, sparsity, mu1, mu2, nbits, group_size)

        X_sub = X[:64]
        target_out = X_sub @ W.T
        std_out = X_sub @ W_std.T
        output_error_std = ((target_out - std_out) ** 2).mean().item()

        print(f"\nStandard OBS output MSE: {output_error_std:.6f}")

        # QAOBS
        for n_iter in [1, 3, 5]:
            W_qaobs, _ = qaobs_prune(W, X, sparsity, mu1, mu2, nbits, group_size, n_iter=n_iter)
            qaobs_out = X_sub @ W_qaobs.T
            output_error_qaobs = ((target_out - qaobs_out) ** 2).mean().item()
            improvement = (output_error_std - output_error_qaobs) / output_error_std * 100
            print(f"QAOBS (n_iter={n_iter}) output MSE: {output_error_qaobs:.6f} ({improvement:+.2f}%)")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_qaobs_synthetic()
    test_qaobs_real()
