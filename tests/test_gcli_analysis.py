"""
Analysis: Global Cross-Layer Importance (GCLI)

Question: If we rank ALL weights globally across ALL layers,
does the resulting sparsity distribution differ significantly
from uniform 35% per layer?

This tests whether global ranking would naturally allocate
different sparsity to different layers based on importance.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from transformers import AutoModelForCausalLM
from sinq.sinkhorn import sinkhorn_log
from datasets import load_dataset


def analyze_global_importance():
    print("="*70)
    print("GLOBAL CROSS-LAYER IMPORTANCE ANALYSIS")
    print("="*70)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Collect all layer info
    layer_data = []

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'lm_head' not in name:
            if module.weight.dim() == 2:
                W = module.weight.data.float()
                K, N = W.shape

                # Sinkhorn
                W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
                mu2 = mu2.squeeze()

                # Simple importance (magnitude / μ)
                importance = W.abs() / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

                layer_data.append({
                    'name': name,
                    'shape': (K, N),
                    'n_weights': K * N,
                    'importance': importance.flatten().cpu(),
                    'mu1_mean': mu1.mean().item(),
                    'mu2_mean': mu2.mean().item(),
                    'importance_mean': importance.mean().item(),
                    'importance_std': importance.std().item(),
                })

    print(f"\nFound {len(layer_data)} layers")
    total_weights = sum(d['n_weights'] for d in layer_data)
    print(f"Total weights: {total_weights:,}")

    # Compute global importance ranking
    all_importance = torch.cat([d['importance'] for d in layer_data])
    print(f"All importance tensor shape: {all_importance.shape}")

    # Global threshold for 35% sparsity
    target_sparsity = 0.35
    n_prune = int(total_weights * target_sparsity)
    global_threshold = all_importance.sort().values[n_prune].item()

    print(f"\nGlobal threshold (35% sparsity): {global_threshold:.6f}")

    # Compute per-layer sparsity under global ranking
    print("\n" + "="*70)
    print("PER-LAYER SPARSITY UNDER GLOBAL RANKING")
    print("="*70)
    print(f"{'Layer':<40} {'Shape':<15} {'Sparsity':<10} {'Mean Imp':<12}")
    print("-"*70)

    start_idx = 0
    layer_sparsities = []
    for d in layer_data:
        n = d['n_weights']
        layer_imp = all_importance[start_idx:start_idx + n]
        layer_mask = (layer_imp > global_threshold).float()
        layer_sparsity = 1 - layer_mask.mean().item()
        layer_sparsities.append(layer_sparsity)

        short_name = d['name'][-38:] if len(d['name']) > 38 else d['name']
        print(f"{short_name:<40} {str(d['shape']):<15} {layer_sparsity:<10.1%} {d['importance_mean']:<12.4f}")

        start_idx += n

    # Statistics
    print("\n" + "="*70)
    print("SPARSITY DISTRIBUTION STATISTICS")
    print("="*70)
    sp_tensor = torch.tensor(layer_sparsities)
    print(f"Mean sparsity: {sp_tensor.mean():.1%}")
    print(f"Std sparsity:  {sp_tensor.std():.1%}")
    print(f"Min sparsity:  {sp_tensor.min():.1%}")
    print(f"Max sparsity:  {sp_tensor.max():.1%}")
    print(f"Range:         {sp_tensor.max() - sp_tensor.min():.1%}")

    # Key insight: Does μ correlate with allocated sparsity?
    mu_means = torch.tensor([d['mu1_mean'] * d['mu2_mean'] for d in layer_data])
    imp_means = torch.tensor([d['importance_mean'] for d in layer_data])

    corr_mu_sp = torch.corrcoef(torch.stack([mu_means, sp_tensor]))[0, 1].item()
    corr_imp_sp = torch.corrcoef(torch.stack([imp_means, sp_tensor]))[0, 1].item()

    print(f"\nCorrelation (μ₁×μ₂ vs sparsity): {corr_mu_sp:.4f}")
    print(f"Correlation (mean_importance vs sparsity): {corr_imp_sp:.4f}")

    # Check if any layers would be >50% or <20% sparse
    n_extreme_high = (sp_tensor > 0.5).sum().item()
    n_extreme_low = (sp_tensor < 0.2).sum().item()
    print(f"\nLayers with >50% sparsity: {n_extreme_high}")
    print(f"Layers with <20% sparsity: {n_extreme_low}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    analyze_global_importance()
