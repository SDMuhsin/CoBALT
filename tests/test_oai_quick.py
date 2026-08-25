"""
Quick test: Output-Aware Importance (OAI)

Idea: Current importance = |W| × ||X|| (input-based only)
What if we also consider output importance?

importance_oai = |W| × ||X|| × ||Y_row|| / (μ₁ × μ₂)

where Y_row = row norm of output Y = XW
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_oai():
    print("="*70)
    print("OUTPUT-AWARE IMPORTANCE (OAI) QUICK TEST")
    print("="*70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real weights
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}] (K=output, N=input)")

    # Create synthetic activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

    # Compute Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Compute input and output norms
    act_norms = torch.norm(X, dim=0)  # [N] - input column norms

    # Compute output: Y = X @ W.T (since W is [K, N] and X is [batch, N])
    Y = X @ W.T  # [batch, K]
    out_norms = torch.norm(Y, dim=0)  # [K] - output row norms

    # Weight row norms (alternative to output norms)
    W_row_norms = torch.norm(W, dim=1)  # [K]

    # Method 1: Standard inverse μ (input-only)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Method 2: OAI with output norms
    importance_oai_out = importance_std * out_norms.unsqueeze(1)

    # Method 3: OAI with weight row norms
    importance_oai_wrow = importance_std * W_row_norms.unsqueeze(1)

    # Method 4: Input-output balanced
    # Use geometric mean of input and output importance
    importance_io_balanced = importance_std * torch.sqrt(out_norms.unsqueeze(1) / (out_norms.mean() + 1e-6))

    # Create masks (35% sparsity)
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_oai_out = create_mask(importance_oai_out)
    mask_oai_wrow = create_mask(importance_oai_wrow)
    mask_io = create_mask(importance_io_balanced)

    # Evaluate
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]
    n_groups = N // group_size

    def evaluate_mask(mask, name):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        # MSE on non-zero weights
        W_masked = W * mask
        mse = ((W_masked - W_deq * mask) ** 2).sum() / mask.sum()
        return mse.item()

    mse_std = evaluate_mask(mask_std, "Standard")
    mse_oai_out = evaluate_mask(mask_oai_out, "OAI-output")
    mse_oai_wrow = evaluate_mask(mask_oai_wrow, "OAI-wrow")
    mse_io = evaluate_mask(mask_io, "IO-balanced")

    print(f"\n--- Reconstruction MSE ---")
    print(f"Standard (inverse μ):   {mse_std:.6f}")
    print(f"OAI-output:             {mse_oai_out:.6f} ({(mse_std - mse_oai_out) / mse_std * 100:+.2f}%)")
    print(f"OAI-weight-row:         {mse_oai_wrow:.6f} ({(mse_std - mse_oai_wrow) / mse_std * 100:+.2f}%)")
    print(f"IO-balanced:            {mse_io:.6f} ({(mse_std - mse_io) / mse_std * 100:+.2f}%)")

    # Also measure OUTPUT error (more relevant)
    Y_ref = X @ W.T

    def evaluate_output_mse(mask, name):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        Y_approx = X @ W_deq.T
        output_mse = ((Y_ref - Y_approx) ** 2).mean()
        return output_mse.item()

    out_mse_std = evaluate_output_mse(mask_std, "Standard")
    out_mse_oai = evaluate_output_mse(mask_oai_out, "OAI-output")
    out_mse_wrow = evaluate_output_mse(mask_oai_wrow, "OAI-wrow")
    out_mse_io = evaluate_output_mse(mask_io, "IO-balanced")

    print(f"\n--- Output MSE ---")
    print(f"Standard (inverse μ):   {out_mse_std:.6f}")
    print(f"OAI-output:             {out_mse_oai:.6f} ({(out_mse_std - out_mse_oai) / out_mse_std * 100:+.2f}%)")
    print(f"OAI-weight-row:         {out_mse_wrow:.6f} ({(out_mse_std - out_mse_wrow) / out_mse_std * 100:+.2f}%)")
    print(f"IO-balanced:            {out_mse_io:.6f} ({(out_mse_std - out_mse_io) / out_mse_std * 100:+.2f}%)")

    # Analyze norms
    print(f"\n--- Norm Statistics ---")
    print(f"Input norm CV:    {act_norms.std() / act_norms.mean():.2%}")
    print(f"Output norm CV:   {out_norms.std() / out_norms.mean():.2%}")
    print(f"Weight row CV:    {W_row_norms.std() / W_row_norms.mean():.2%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_oai()
