"""
Validate inverse μ importance across multiple layers.

If the improvement is consistent, this could be a genuine finding.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from transformers import AutoModelForCausalLM
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_layer(W, X, layer_name, sparsity=0.35):
    """Test standard vs inverse importance on one layer."""
    K, N = W.shape
    device = W.device

    # Compute Sinkhorn factors
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    act_norms = torch.norm(X, dim=0)

    # Standard importance
    importance_std = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

    # Inverse importance
    importance_inv = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Pure Wanda
    importance_wanda = W.abs() * act_norms.unsqueeze(0)

    # Create masks
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_inv = create_mask(importance_inv)
    mask_wanda = create_mask(importance_wanda)

    # Quantize and measure MSE
    min_max = [0, 7]
    group_size = 64

    def get_mse(mask):
        W_sparse_norm = W_norm * mask
        Q, scales, zeros, _ = quantize_rtn(W_sparse_norm, min_max, group_size=group_size)
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
        W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        W_masked = W * mask
        return ((W_masked - W_deq * mask) ** 2).sum() / mask.sum()

    mse_std = get_mse(mask_std).item()
    mse_inv = get_mse(mask_inv).item()
    mse_wanda = get_mse(mask_wanda).item()

    improvement_inv = (mse_std - mse_inv) / mse_std * 100
    improvement_wanda = (mse_std - mse_wanda) / mse_std * 100

    return {
        'layer': layer_name,
        'mse_std': mse_std,
        'mse_inv': mse_inv,
        'mse_wanda': mse_wanda,
        'improvement_inv': improvement_inv,
        'improvement_wanda': improvement_wanda
    }


def main():
    print("="*70)
    print("MULTI-LAYER INVERSE μ IMPORTANCE VALIDATION")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\nLoading Qwen-0.5B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Test multiple layers and projection types
    test_configs = [
        (0, 'gate_proj'),
        (0, 'up_proj'),
        (0, 'down_proj'),
        (5, 'gate_proj'),
        (10, 'gate_proj'),
        (15, 'gate_proj'),
        (20, 'gate_proj'),
    ]

    torch.manual_seed(42)
    results = []

    for layer_idx, proj_name in test_configs:
        layer = model.model.layers[layer_idx]

        if proj_name == 'gate_proj':
            W = layer.mlp.gate_proj.weight.data.float()
        elif proj_name == 'up_proj':
            W = layer.mlp.up_proj.weight.data.float()
        elif proj_name == 'down_proj':
            W = layer.mlp.down_proj.weight.data.float()

        K, N = W.shape
        X = torch.randn(64, N, device=W.device, dtype=W.dtype) * 0.1

        layer_name = f"layer_{layer_idx}.{proj_name}"
        result = test_layer(W, X, layer_name)
        results.append(result)

        print(f"\n{layer_name} [{K}x{N}]:")
        print(f"  MSE std: {result['mse_std']:.6f}")
        print(f"  MSE inv: {result['mse_inv']:.6f} ({result['improvement_inv']:+.1f}%)")
        print(f"  MSE wanda: {result['mse_wanda']:.6f} ({result['improvement_wanda']:+.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    avg_improvement_inv = sum(r['improvement_inv'] for r in results) / len(results)
    avg_improvement_wanda = sum(r['improvement_wanda'] for r in results) / len(results)

    all_positive_inv = all(r['improvement_inv'] > 0 for r in results)
    all_positive_wanda = all(r['improvement_wanda'] > 0 for r in results)

    print(f"\nAverage improvement (Inverse vs Standard): {avg_improvement_inv:+.2f}%")
    print(f"Average improvement (Wanda vs Standard):   {avg_improvement_wanda:+.2f}%")
    print(f"All layers improved with Inverse: {all_positive_inv}")
    print(f"All layers improved with Wanda:   {all_positive_wanda}")

    if avg_improvement_inv > 5 and all_positive_inv:
        print("\n>>> CONSISTENT SIGNIFICANT IMPROVEMENT with inverse μ!")
        print(">>> This could be a valid research direction.")
    elif avg_improvement_inv > 0:
        print("\n>>> Marginal average improvement, but inconsistent across layers.")
    else:
        print("\n>>> Inverse μ is worse on average.")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
