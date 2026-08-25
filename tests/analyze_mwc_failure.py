"""
Analyze WHY MWC fails on larger models.

Compare μ₁ distributions between 0.5B (where MWC works) and 1.5B (where MWC fails).
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from transformers import AutoModelForCausalLM
from sinq.sinkhorn import sinkhorn_log
import numpy as np


def analyze_model_mu1(model_name, model, device='cuda'):
    """Analyze μ₁ statistics across all layers."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*70}")

    results = []

    for layer_idx, layer in enumerate(model.model.layers):
        layer = layer.to(device)

        layer_results = {}
        for name, module in [
            ('q_proj', layer.self_attn.q_proj),
            ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj),
            ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj),
            ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            W = module.weight.data.float()
            _, mu1, mu2 = sinkhorn_log(W, order=16)
            mu2 = mu2.squeeze()

            # Compute statistics
            mu1_mean = mu1.mean().item()
            mu1_std = mu1.std().item()
            mu1_cv = mu1_std / mu1_mean  # Coefficient of variation
            mu1_min = mu1.min().item()
            mu1_max = mu1.max().item()
            mu1_range = mu1_max - mu1_min

            # Ratio statistics (important for MWC)
            mu1_ratios = mu1.unsqueeze(0) / mu1.unsqueeze(1)  # [N, N] pairwise ratios
            ratio_max = mu1_ratios.max().item()
            ratio_min = mu1_ratios.min().item()
            ratio_range = ratio_max - ratio_min

            layer_results[name] = {
                'mu1_mean': mu1_mean,
                'mu1_std': mu1_std,
                'mu1_cv': mu1_cv,
                'mu1_min': mu1_min,
                'mu1_max': mu1_max,
                'mu1_range': mu1_range,
                'ratio_max': ratio_max,
                'ratio_min': ratio_min,
                'ratio_range': ratio_range,
            }

        results.append(layer_results)

        # Move layer back to CPU to save memory
        layer = layer.to('cpu')
        torch.cuda.empty_cache()

    return results


def compare_models(results_small, results_large, name_small, name_large):
    """Compare μ₁ statistics between two models."""
    print(f"\n{'='*70}")
    print("COMPARISON: μ₁ Statistics")
    print(f"{'='*70}")

    # Aggregate statistics
    metrics = ['mu1_cv', 'mu1_range', 'ratio_range', 'ratio_max']
    layer_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    for metric in metrics:
        print(f"\n--- {metric} ---")
        print(f"{'Layer Type':<12} | {name_small:>12} | {name_large:>12} | {'Ratio':>10}")
        print("-" * 55)

        for lt in layer_types:
            vals_small = [r[lt][metric] for r in results_small]
            vals_large = [r[lt][metric] for r in results_large]

            avg_small = np.mean(vals_small)
            avg_large = np.mean(vals_large)
            ratio = avg_large / avg_small if avg_small > 0 else float('inf')

            print(f"{lt:<12} | {avg_small:>12.4f} | {avg_large:>12.4f} | {ratio:>10.2f}x")

    # Per-layer analysis for attention layers (where MWC showed most improvement)
    print(f"\n{'='*70}")
    print("Per-Layer μ₁ CV (Coefficient of Variation)")
    print(f"{'='*70}")

    print(f"\n{'Layer':<20} | {name_small + ' q_proj':>15} | {name_large + ' q_proj':>15}")
    print("-" * 55)

    for i in range(min(len(results_small), len(results_large))):
        cv_small = results_small[i]['q_proj']['mu1_cv']
        cv_large = results_large[i]['q_proj']['mu1_cv']
        print(f"Layer {i:<14} | {cv_small:>15.4f} | {cv_large:>15.4f}")

    # Key insight: ratio_range
    print(f"\n{'='*70}")
    print("KEY INSIGHT: μ₁ Ratio Range (max(μ₁[j]/μ₁[k]))")
    print("This is the scaling factor applied in MWC compensation")
    print(f"{'='*70}")

    for lt in ['q_proj', 'k_proj', 'gate_proj']:
        ratios_small = [r[lt]['ratio_range'] for r in results_small]
        ratios_large = [r[lt]['ratio_range'] for r in results_large]

        print(f"\n{lt}:")
        print(f"  {name_small}: mean={np.mean(ratios_small):.4f}, max={np.max(ratios_small):.4f}")
        print(f"  {name_large}: mean={np.mean(ratios_large):.4f}, max={np.max(ratios_large):.4f}")
        print(f"  Ratio: {np.mean(ratios_large)/np.mean(ratios_small):.2f}x larger in {name_large}")


def main():
    print("="*70)
    print("MWC FAILURE ANALYSIS")
    print("Comparing μ₁ distributions: 0.5B vs 1.5B")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load 0.5B model
    print("\nLoading Qwen2.5-0.5B...")
    model_05b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    model_05b.model.embed_tokens = model_05b.model.embed_tokens.to(device)

    results_05b = analyze_model_mu1("Qwen2.5-0.5B", model_05b, device)

    del model_05b
    torch.cuda.empty_cache()

    # Load 1.5B model
    print("\nLoading Qwen2.5-1.5B...")
    model_15b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.float16,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    model_15b.model.embed_tokens = model_15b.model.embed_tokens.to(device)

    results_15b = analyze_model_mu1("Qwen2.5-1.5B", model_15b, device)

    del model_15b
    torch.cuda.empty_cache()

    # Compare
    compare_models(results_05b, results_15b, "0.5B", "1.5B")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Why MWC Fails on 1.5B")
    print(f"{'='*70}")

    # Compute overall statistics
    cv_05b_attn = np.mean([r['q_proj']['mu1_cv'] for r in results_05b])
    cv_15b_attn = np.mean([r['q_proj']['mu1_cv'] for r in results_15b])

    ratio_range_05b = np.mean([r['q_proj']['ratio_range'] for r in results_05b])
    ratio_range_15b = np.mean([r['q_proj']['ratio_range'] for r in results_15b])

    print(f"\nAverage μ₁ CV (q_proj):")
    print(f"  0.5B: {cv_05b_attn:.4f}")
    print(f"  1.5B: {cv_15b_attn:.4f}")
    print(f"  Ratio: {cv_15b_attn/cv_05b_attn:.2f}x")

    print(f"\nAverage μ₁ Ratio Range (q_proj):")
    print(f"  0.5B: {ratio_range_05b:.4f}")
    print(f"  1.5B: {ratio_range_15b:.4f}")
    print(f"  Ratio: {ratio_range_15b/ratio_range_05b:.2f}x")

    if ratio_range_15b > ratio_range_05b * 1.5:
        print("\n*** HYPOTHESIS: MWC fails because μ₁ ratio range is LARGER in 1.5B ***")
        print("The μ₁[j]/μ₁[k] scaling in MWC amplifies errors when ratio range is large.")
    elif cv_15b_attn < cv_05b_attn * 0.7:
        print("\n*** HYPOTHESIS: MWC fails because μ₁ CV is SMALLER in 1.5B ***")
        print("Less μ₁ variance means MWC correction is less beneficial.")
    else:
        print("\n*** HYPOTHESIS: Further investigation needed ***")


if __name__ == '__main__':
    main()
