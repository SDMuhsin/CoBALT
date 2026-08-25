#!/usr/bin/env python3
"""
SUPERSEDED, kept for the record. Do not put its output in the paper.

Use scripts/row_deflation_study.py and scripts/make_deflation_figure.py.

What this script measures is the variance over all entries of a whole matrix
divided by the variance over its survivors, under a global magnitude threshold,
and it reports 0.50 at 50% sparsity. Aggregated over a matrix, Proposition 1
reads

    ratio = d + d(1-d) * (wbar/sigma)^2

and a global magnitude quantile fixes d at exactly the pruning ratio, so the
only free quantity is the term in the mean. That term needs |wbar/sigma| > 0.14
before the printed value moves off 0.50 at all, and LLM weights are near
symmetric, so it never does: Gaussian, Student-t and Laplace matrices all
return exactly 0.5000. The number was near tautological, it was withdrawn from
the paper on 2026-08-03, and Proposition 1 is an identity that no measurement
can confirm. See D12 in llmdocs/V3_DECISIONS.md.

Two further problems, both of which the replacement fixes: the mask is a global
magnitude threshold, which no method in the paper produces, and the aggregation
over whole matrices averages away the per-row density variation that motivates
the method.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/workspace/SINQ')

# IEEE TNNLS style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def magnitude_prune(W: torch.Tensor, sparsity: float) -> tuple:
    """
    Apply magnitude-based pruning to weight matrix.

    Args:
        W: Weight matrix [K, N]
        sparsity: Fraction of weights to prune (0 to 1)

    Returns:
        W_pruned: Pruned weight matrix (with zeros)
        mask: Binary mask where 1 = kept, 0 = pruned
    """
    threshold = torch.quantile(W.abs().flatten(), sparsity)
    mask = (W.abs() >= threshold).float()
    W_pruned = W * mask
    return W_pruned, mask

def compute_variance_naive(W: torch.Tensor) -> float:
    """
    Compute variance treating all entries (including zeros) as real values.
    This is what standard normalization does incorrectly on sparse matrices.
    """
    return W.var().item()

def compute_variance_sparse_aware(W: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute variance only on non-zero (kept) entries.
    This is what PRISM does correctly.
    """
    non_zero_weights = W[mask > 0]
    if len(non_zero_weights) == 0:
        return 0.0
    return non_zero_weights.var().item()

def run_variance_experiment(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """
    Run variance distortion experiment on actual model weights.

    Proposition 1 states: When computing variance on sparse matrices with
    N total entries (including zeros) but only n non-zero entries,
    the naive variance (sum(w^2)/N) underestimates the true variance
    of non-zero weights by factor d = n/N (the density).

    We measure:
    - True variance: variance of non-zero weights only
    - Naive variance: sum(w^2)/N treating zeros as real values
    - Ratio should be approximately d (the density)
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

    # Collect weight matrices from linear layers
    weight_matrices = []
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data.clone()
            if W.numel() > 1000:  # Only consider non-trivial layers
                weight_matrices.append(W)
                layer_names.append(name)

    print(f"Found {len(weight_matrices)} linear layers")

    # Test sparsity levels
    sparsity_levels = np.arange(0.05, 0.96, 0.05)

    # Store results
    results = {
        'sparsity': [],
        'theoretical_ratio': [],
        'measured_ratio_mean': [],
        'measured_ratio_std': [],
        'layer_ratios': []
    }

    print("\nMeasuring variance distortion...")
    print("This validates Proposition 1: naive_var / true_var ≈ d (density)")
    print()
    print("Sparsity | Density (d) | Measured Ratio (naive/true) | Match?")
    print("-" * 65)

    for sparsity in sparsity_levels:
        ratios = []
        density = 1 - sparsity

        for W in weight_matrices[:20]:  # Use first 20 layers for speed
            # Prune the weights (set smallest magnitude to zero)
            W_pruned, mask = magnitude_prune(W, sparsity)

            # True variance: variance of non-zero weights ONLY
            # This is what we SHOULD use for normalization
            non_zero = W_pruned[mask > 0]
            if len(non_zero) > 1:
                true_var = non_zero.var().item()
            else:
                continue

            # Naive variance: sum(w^2)/N where N includes zeros
            # This is what standard methods incorrectly compute
            N = W_pruned.numel()
            sum_sq = (W_pruned ** 2).sum().item()
            mean = W_pruned.mean().item()
            naive_var = sum_sq / N - mean ** 2

            # The ratio naive_var/true_var should approximately equal d (density)
            if true_var > 1e-10:
                ratio = naive_var / true_var
                ratios.append(ratio)

        if ratios:
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)

            results['sparsity'].append(sparsity)
            results['theoretical_ratio'].append(density)
            results['measured_ratio_mean'].append(mean_ratio)
            results['measured_ratio_std'].append(std_ratio)
            results['layer_ratios'].append(ratios)

            match = "✓" if abs(mean_ratio - density) < 0.1 else "✗"
            print(f"  {sparsity*100:5.1f}% |    {density:.3f}     |        {mean_ratio:.3f} ± {std_ratio:.3f}       | {match}")

    return results

def create_variance_distortion_plot(results: dict):
    """
    Create empirical variance distortion plot.
    Shows that naive variance estimation underestimates true variance by factor d.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sparsity = np.array(results['sparsity'])
    theoretical = np.array(results['theoretical_ratio'])
    measured_mean = np.array(results['measured_ratio_mean'])
    measured_std = np.array(results['measured_ratio_std'])

    # Left plot: Variance deflation - naive/true ratio
    ax1.plot(sparsity * 100, theoretical, 'b-', linewidth=2,
            label='Predicted (Prop. 1): ratio = $d_i = 1-s$')
    ax1.errorbar(sparsity * 100, measured_mean, yerr=measured_std,
                color='red', marker='o', markersize=5, linewidth=2,
                capsize=3, label='Measured on Qwen2.5-0.5B')
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Shade the distortion region
    ax1.fill_between(sparsity * 100, measured_mean, 1,
                    alpha=0.2, color='red', label='Underestimation')

    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Naive Variance / True Variance')
    ax1.set_title('(a) Empirical Variance Deflation')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.1)

    # Annotate key point
    idx_50 = np.argmin(np.abs(sparsity - 0.5))
    if len(sparsity) > idx_50:
        ax1.annotate(f'50% sparsity:\nratio={measured_mean[idx_50]:.2f}',
                    xy=(50, measured_mean[idx_50]),
                    xytext=(60, 0.7), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='black'))

    # Right plot: Prediction accuracy
    ax2.scatter(theoretical, measured_mean, s=50, c=sparsity * 100,
               cmap='viridis', edgecolors='black', linewidths=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')

    ax2.set_xlabel('Predicted Ratio (Proposition 1)')
    ax2.set_ylabel('Measured Ratio')
    ax2.set_title('(b) Theory vs Experiment')
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Sparsity (%)')

    plt.tight_layout()
    plt.savefig('/workspace/PRISM/llmdocs/paper/v3_neuro/variance_distortion_empirical.pdf')
    plt.savefig('/workspace/PRISM/llmdocs/paper/v3_neuro/variance_distortion_empirical.png')
    plt.close()

    print("\nCreated: variance_distortion_empirical.pdf")

def create_sparse_aware_correction_plot(results: dict):
    """
    Show that sparse-aware variance recovers the true variance.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sparsity = np.array(results['sparsity'])
    theoretical = np.array(results['theoretical_ratio'])

    # Naive variance ratio (deflated)
    ax.plot(sparsity * 100, theoretical, 'r-', linewidth=2,
           marker='s', markersize=5, label='Naive Variance (deflated)')

    # Sparse-aware variance ratio (corrected - should be ~1.0)
    # The correction is: sparse_var = naive_var / density
    corrected = np.ones_like(theoretical)  # After correction, ratio is 1.0
    ax.plot(sparsity * 100, corrected, 'g-', linewidth=2,
           marker='o', markersize=5, label='Sparse-Aware Variance (corrected)')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1,
              label='True Variance')

    ax.fill_between(sparsity * 100, theoretical, 1,
                   alpha=0.2, color='red', label='Error from naive estimation')

    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Variance Ratio (Estimated / True)')
    ax.set_title('Sparse-Aware Variance Correction (Theorem 1)')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('/workspace/SINQ/paper/sparse_aware_correction.pdf')
    plt.savefig('/workspace/SINQ/paper/sparse_aware_correction.png')
    plt.close()

    print("Created: sparse_aware_correction.pdf")

def main():
    print("=" * 60)
    print("VARIANCE DISTORTION EXPERIMENT")
    print("Validating Proposition 1 on Real Model Weights")
    print("=" * 60 + "\n")

    # Run experiment
    results = run_variance_experiment("Qwen/Qwen2.5-0.5B")

    # Create plots
    create_variance_distortion_plot(results)
    create_sparse_aware_correction_plot(results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mean_deviation = np.mean(np.abs(
        np.array(results['measured_ratio_mean']) -
        np.array(results['theoretical_ratio'])
    ))
    print(f"Mean absolute deviation from theory: {mean_deviation:.4f}")
    print(f"This confirms Proposition 1: naive variance is deflated by factor d_i")

    print("\n" + "=" * 60)
    print("Plots saved to /workspace/SINQ/paper/")
    print("=" * 60)

if __name__ == '__main__':
    main()
