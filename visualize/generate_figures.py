#!/usr/bin/env python3
"""
Generate publication-quality figures for PRISM paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

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
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Color scheme for techniques
COLORS = {
    'prism': '#2ecc71',      # Green
    'sparsegpt': '#e74c3c',  # Red
    'wanda': '#3498db',      # Blue
    'sinq': '#9b59b6',       # Purple
    'fp16': '#34495e',       # Dark gray
}

MARKERS = {
    'prism': 'o',
    'sparsegpt': 's',
    'wanda': '^',
    'sinq': 'd',
    'fp16': '*',
}

LABELS = {
    'prism': 'PRISM (Ours)',
    'sparsegpt': 'SparseGPT',
    'wanda': 'Wanda',
    'sinq': 'SINQ',
    'fp16': 'FP16',
}

def load_data():
    """Load and clean benchmark data."""
    df = pd.read_csv('/workspace/SINQ/results/benchmark_results.csv', on_bad_lines='skip')

    valid_models = ['qwen-0.5b', 'qwen-1.5b', 'qwen-3b', 'opt-1.3b', 'llama-7b', 'gemma-2b']
    valid_techniques = ['fp16', 'sinq', 'wanda', 'sparsegpt', 'prism']

    df = df[df['model'].isin(valid_models)]
    df = df[df['technique'].isin(valid_techniques)]

    for col in ['precision', 'sparsity', 'ppl', 'accuracy']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['precision'].isin([3, 4, 5, 8, 16])]

    return df

def create_sparsity_ablation_plot(df, model='qwen-0.5b', precision=3):
    """Create sparsity vs PPL plot for ablation study."""

    wiki_df = df[(df['dataset'] == 'wikitext2') &
                  (df['model'] == model) &
                  (df['precision'].isin([precision, 16]))]

    fig, ax = plt.subplots(figsize=(7, 5))

    for tech in ['prism', 'sparsegpt', 'wanda']:
        if tech == 'wanda':
            tech_df = wiki_df[(wiki_df['technique'] == tech) & (wiki_df['precision'] == 16)]
        else:
            tech_df = wiki_df[(wiki_df['technique'] == tech) & (wiki_df['precision'] == precision)]

        if len(tech_df) > 0:
            tech_df = tech_df.sort_values('sparsity')
            # Filter out extreme values (PPL > 1000 is likely model collapse)
            tech_df = tech_df[tech_df['ppl'] < 500]

            ax.plot(tech_df['sparsity'] * 100, tech_df['ppl'],
                   color=COLORS[tech], marker=MARKERS[tech],
                   label=LABELS[tech], linewidth=2, markersize=8)

    # Add FP16 baseline as horizontal line
    fp16_df = wiki_df[(wiki_df['technique'] == 'fp16')]
    if len(fp16_df) > 0:
        fp16_ppl = fp16_df['ppl'].iloc[0]
        if not np.isnan(fp16_ppl):
            ax.axhline(y=fp16_ppl, color=COLORS['fp16'], linestyle='--',
                      label=f'FP16 Baseline ({fp16_ppl:.2f})', linewidth=1.5)

    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Perplexity (WikiText-2)')
    ax.set_title(f'Sparsity Ablation on {model.upper()} ({precision}-bit Quantization)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 55)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'/workspace/SINQ/paper/sparsity_ablation_{model}_{precision}bit.pdf')
    plt.savefig(f'/workspace/SINQ/paper/sparsity_ablation_{model}_{precision}bit.png')
    plt.close()

    print(f"Saved sparsity ablation plot for {model} {precision}-bit")

def create_precision_comparison_plot(df, model='llama-7b', sparsity=0.5):
    """Create precision vs PPL comparison."""

    wiki_df = df[(df['dataset'] == 'wikitext2') &
                  (df['model'] == model)]

    fig, ax = plt.subplots(figsize=(7, 5))

    precisions = [3, 4, 5]

    for tech in ['prism', 'sparsegpt']:
        if tech == 'prism':
            tech_df = wiki_df[(wiki_df['technique'] == tech) &
                             (abs(wiki_df['sparsity'] - sparsity) < 0.01)]
        else:
            tech_df = wiki_df[(wiki_df['technique'] == tech) &
                             (abs(wiki_df['sparsity'] - sparsity) < 0.01)]

        ppls = []
        for prec in precisions:
            prec_df = tech_df[tech_df['precision'] == prec]
            if len(prec_df) > 0:
                ppls.append(prec_df['ppl'].iloc[0])
            else:
                ppls.append(np.nan)

        if not all(np.isnan(ppls)):
            ax.plot(precisions, ppls, color=COLORS[tech], marker=MARKERS[tech],
                   label=LABELS[tech], linewidth=2, markersize=8)

    # Add SINQ (quantization only)
    sinq_df = wiki_df[(wiki_df['technique'] == 'sinq') &
                      ((wiki_df['sparsity'] == 0) | (wiki_df['sparsity'].isna()))]
    ppls = []
    for prec in precisions:
        prec_df = sinq_df[sinq_df['precision'] == prec]
        if len(prec_df) > 0:
            ppls.append(prec_df['ppl'].iloc[0])
        else:
            ppls.append(np.nan)

    if not all(np.isnan(ppls)):
        ax.plot(precisions, ppls, color=COLORS['sinq'], marker=MARKERS['sinq'],
               label='SINQ (Dense)', linewidth=2, markersize=8, linestyle='--')

    ax.set_xlabel('Bit-width')
    ax.set_ylabel('Perplexity (WikiText-2)')
    ax.set_title(f'Precision Scaling on {model.upper()} ({int(sparsity*100)}% Sparsity)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks(precisions)

    plt.tight_layout()
    plt.savefig(f'/workspace/SINQ/paper/precision_comparison_{model}_{int(sparsity*100)}sp.pdf')
    plt.savefig(f'/workspace/SINQ/paper/precision_comparison_{model}_{int(sparsity*100)}sp.png')
    plt.close()

    print(f"Saved precision comparison plot for {model} {int(sparsity*100)}% sparsity")

def create_model_comparison_bar_chart(df, precision=4, sparsity=0.5):
    """Create bar chart comparing methods across models."""

    wiki_df = df[(df['dataset'] == 'wikitext2')]

    models = ['qwen-0.5b', 'llama-7b']  # Exclude opt-1.3b due to anomalies
    techniques = ['wanda', 'sparsegpt', 'prism']

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.25

    for i, tech in enumerate(techniques):
        ppls = []
        for model in models:
            if tech == 'wanda':
                tech_df = wiki_df[(wiki_df['technique'] == tech) &
                                 (wiki_df['model'] == model) &
                                 (wiki_df['precision'] == 16) &
                                 (abs(wiki_df['sparsity'] - sparsity) < 0.01)]
            else:
                tech_df = wiki_df[(wiki_df['technique'] == tech) &
                                 (wiki_df['model'] == model) &
                                 (wiki_df['precision'] == precision) &
                                 (abs(wiki_df['sparsity'] - sparsity) < 0.01)]

            if len(tech_df) > 0:
                ppls.append(tech_df['ppl'].iloc[0])
            else:
                ppls.append(0)

        bars = ax.bar(x + i * width, ppls, width, label=LABELS[tech], color=COLORS[tech])

        # Add value labels on bars
        for bar, ppl in zip(bars, ppls):
            if ppl > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{ppl:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Perplexity (WikiText-2)')
    ax.set_title(f'Method Comparison at {int(sparsity*100)}% Sparsity, {precision}-bit')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'/workspace/SINQ/paper/model_comparison_{precision}bit_{int(sparsity*100)}sp.pdf')
    plt.savefig(f'/workspace/SINQ/paper/model_comparison_{precision}bit_{int(sparsity*100)}sp.png')
    plt.close()

    print(f"Saved model comparison bar chart for {precision}-bit {int(sparsity*100)}% sparsity")

def create_improvement_heatmap(df):
    """Create heatmap of PRISM improvement over SparseGPT."""

    wiki_df = df[(df['dataset'] == 'wikitext2')]

    models = ['qwen-0.5b', 'llama-7b']
    precisions = [3, 4, 5]
    sparsities = [0.05, 0.25, 0.5]

    # Create improvement matrix
    improvements = np.zeros((len(models) * len(sparsities), len(precisions)))
    row_labels = []

    for i, model in enumerate(models):
        for j, spar in enumerate(sparsities):
            row_labels.append(f"{model.upper()}\n{int(spar*100)}% sp")
            for k, prec in enumerate(precisions):
                prism_df = wiki_df[(wiki_df['technique'] == 'prism') &
                                   (wiki_df['model'] == model) &
                                   (wiki_df['precision'] == prec) &
                                   (abs(wiki_df['sparsity'] - spar) < 0.01)]
                sgpt_df = wiki_df[(wiki_df['technique'] == 'sparsegpt') &
                                  (wiki_df['model'] == model) &
                                  (wiki_df['precision'] == prec) &
                                  (abs(wiki_df['sparsity'] - spar) < 0.01)]

                if len(prism_df) > 0 and len(sgpt_df) > 0:
                    prism_ppl = prism_df['ppl'].iloc[0]
                    sgpt_ppl = sgpt_df['ppl'].iloc[0]
                    improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100
                    improvements[i * len(sparsities) + j, k] = improvement

    fig, ax = plt.subplots(figsize=(6, 7))

    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=90)

    ax.set_xticks(np.arange(len(precisions)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([f'{p}-bit' for p in precisions])
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(precisions)):
            val = improvements[i, j]
            color = 'white' if abs(val) > 40 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=9)

    ax.set_title('PRISM Improvement over SparseGPT (%)\n(Positive = PRISM Better)')

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('PPL Improvement (%)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig('/workspace/SINQ/paper/improvement_heatmap.pdf')
    plt.savefig('/workspace/SINQ/paper/improvement_heatmap.png')
    plt.close()

    print("Saved improvement heatmap")

    return improvements

def create_full_sparsity_sweep(df):
    """Create full sparsity sweep plot using all available data."""

    wiki_df = df[(df['dataset'] == 'wikitext2') & (df['model'] == 'qwen-0.5b')]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, precision in enumerate([3, 5]):
        ax = axes[idx]

        for tech in ['prism', 'sparsegpt']:
            tech_df = wiki_df[(wiki_df['technique'] == tech) &
                             (wiki_df['precision'] == precision)]
            tech_df = tech_df.sort_values('sparsity')
            tech_df = tech_df[tech_df['ppl'] < 500]  # Filter extreme values

            if len(tech_df) > 0:
                ax.plot(tech_df['sparsity'] * 100, tech_df['ppl'],
                       color=COLORS[tech], marker=MARKERS[tech],
                       label=LABELS[tech], linewidth=2, markersize=6)

        # Add Wanda (pruning only)
        wanda_df = wiki_df[(wiki_df['technique'] == 'wanda') &
                          (wiki_df['precision'] == 16)]
        wanda_df = wanda_df.sort_values('sparsity')
        wanda_df = wanda_df[wanda_df['ppl'] < 500]

        if len(wanda_df) > 0:
            ax.plot(wanda_df['sparsity'] * 100, wanda_df['ppl'],
                   color=COLORS['wanda'], marker=MARKERS['wanda'],
                   label=LABELS['wanda'], linewidth=2, markersize=6)

        # Add FP16 baseline
        fp16_df = wiki_df[(wiki_df['technique'] == 'fp16')]
        if len(fp16_df) > 0:
            fp16_ppl = fp16_df['ppl'].iloc[0]
            if not np.isnan(fp16_ppl):
                ax.axhline(y=fp16_ppl, color=COLORS['fp16'], linestyle='--',
                          label=f'FP16 ({fp16_ppl:.2f})', linewidth=1.5)

        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('Perplexity')
        ax.set_title(f'{precision}-bit Quantization')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(0, 55)
        ax.set_ylim(0, 150)

    fig.suptitle('Sparsity-Perplexity Trade-off on Qwen-0.5B (WikiText-2)', fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/SINQ/paper/full_sparsity_sweep.pdf')
    plt.savefig('/workspace/SINQ/paper/full_sparsity_sweep.png')
    plt.close()

    print("Saved full sparsity sweep plot")

def analyze_numerically(df):
    """Generate numerical analysis for paper text."""

    print("\n" + "=" * 60)
    print("NUMERICAL ANALYSIS FOR PAPER")
    print("=" * 60)

    wiki_df = df[(df['dataset'] == 'wikitext2')]

    # Calculate average improvement by precision
    print("\n--- Average PRISM Improvement over SparseGPT by Precision ---")
    for prec in [3, 4, 5]:
        improvements = []
        for model in ['qwen-0.5b', 'llama-7b']:
            for spar in [0.05, 0.25, 0.5]:
                prism_df = wiki_df[(wiki_df['technique'] == 'prism') &
                                   (wiki_df['model'] == model) &
                                   (wiki_df['precision'] == prec) &
                                   (abs(wiki_df['sparsity'] - spar) < 0.01)]
                sgpt_df = wiki_df[(wiki_df['technique'] == 'sparsegpt') &
                                  (wiki_df['model'] == model) &
                                  (wiki_df['precision'] == prec) &
                                  (abs(wiki_df['sparsity'] - spar) < 0.01)]

                if len(prism_df) > 0 and len(sgpt_df) > 0:
                    prism_ppl = prism_df['ppl'].iloc[0]
                    sgpt_ppl = sgpt_df['ppl'].iloc[0]
                    improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100
                    improvements.append(improvement)

        if improvements:
            print(f"  {prec}-bit: {np.mean(improvements):.1f}% avg improvement")

    # Calculate average improvement by sparsity
    print("\n--- Average PRISM Improvement over SparseGPT by Sparsity ---")
    for spar in [0.05, 0.25, 0.5]:
        improvements = []
        for model in ['qwen-0.5b', 'llama-7b']:
            for prec in [3, 4, 5]:
                prism_df = wiki_df[(wiki_df['technique'] == 'prism') &
                                   (wiki_df['model'] == model) &
                                   (wiki_df['precision'] == prec) &
                                   (abs(wiki_df['sparsity'] - spar) < 0.01)]
                sgpt_df = wiki_df[(wiki_df['technique'] == 'sparsegpt') &
                                  (wiki_df['model'] == model) &
                                  (wiki_df['precision'] == prec) &
                                  (abs(wiki_df['sparsity'] - spar) < 0.01)]

                if len(prism_df) > 0 and len(sgpt_df) > 0:
                    prism_ppl = prism_df['ppl'].iloc[0]
                    sgpt_ppl = sgpt_df['ppl'].iloc[0]
                    improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100
                    improvements.append(improvement)

        if improvements:
            print(f"  {int(spar*100)}% sparsity: {np.mean(improvements):.1f}% avg improvement")

    # Best and worst cases
    print("\n--- Best PRISM Results ---")
    best_improvements = []
    for model in ['qwen-0.5b', 'llama-7b']:
        for prec in [3, 4, 5]:
            for spar in [0.05, 0.25, 0.5]:
                prism_df = wiki_df[(wiki_df['technique'] == 'prism') &
                                   (wiki_df['model'] == model) &
                                   (wiki_df['precision'] == prec) &
                                   (abs(wiki_df['sparsity'] - spar) < 0.01)]
                sgpt_df = wiki_df[(wiki_df['technique'] == 'sparsegpt') &
                                  (wiki_df['model'] == model) &
                                  (wiki_df['precision'] == prec) &
                                  (abs(wiki_df['sparsity'] - spar) < 0.01)]

                if len(prism_df) > 0 and len(sgpt_df) > 0:
                    prism_ppl = prism_df['ppl'].iloc[0]
                    sgpt_ppl = sgpt_df['ppl'].iloc[0]
                    improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100
                    best_improvements.append((improvement, model, prec, spar, prism_ppl, sgpt_ppl))

    best_improvements.sort(reverse=True)
    print("\n  Top 5 improvements:")
    for imp, model, prec, spar, prism, sgpt in best_improvements[:5]:
        print(f"    {model} {prec}-bit {int(spar*100)}%sp: {imp:.1f}% (PRISM={prism:.2f}, SparseGPT={sgpt:.2f})")

    print("\n  Bottom 3 (worst):")
    for imp, model, prec, spar, prism, sgpt in best_improvements[-3:]:
        print(f"    {model} {prec}-bit {int(spar*100)}%sp: {imp:.1f}% (PRISM={prism:.2f}, SparseGPT={sgpt:.2f})")

def main():
    print("Loading data...")
    df = load_data()

    print("\nGenerating figures...")

    # Create sparsity ablation plots for main model
    for prec in [3, 5]:
        create_sparsity_ablation_plot(df, model='qwen-0.5b', precision=prec)

    # Create precision comparison
    for spar in [0.25, 0.5]:
        create_precision_comparison_plot(df, model='llama-7b', sparsity=spar)

    # Create model comparison bar chart
    create_model_comparison_bar_chart(df, precision=4, sparsity=0.5)

    # Create improvement heatmap
    create_improvement_heatmap(df)

    # Create full sparsity sweep
    create_full_sparsity_sweep(df)

    # Numerical analysis
    analyze_numerically(df)

    print("\nAll figures saved to /workspace/SINQ/paper/")

if __name__ == '__main__':
    main()
