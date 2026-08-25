#!/usr/bin/env python3
"""
Generate fine-grained sparsity ablation plots for PRISM paper.
Uses full 19-level sparsity sweep from sparsity_ablation_results.csv.
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
})

COLORS = {
    'prism': '#2ecc71',      # Green
    'sparsegpt': '#e74c3c',  # Red
    'wanda': '#3498db',      # Blue
}

MARKERS = {
    'prism': 'o',
    'sparsegpt': 's',
    'wanda': '^',
}

def load_ablation_data():
    """Load sparsity ablation results."""
    df = pd.read_csv('/workspace/SINQ/results/sparsity_ablation_results.csv')
    # Filter valid rows
    df = df[~df['ppl'].isna() & (df['ppl'] > 0) & (df['ppl'] < 100000)]
    return df

def load_main_data():
    """Load main benchmark results for supplementary data."""
    df = pd.read_csv('/workspace/SINQ/results/benchmark_results.csv', on_bad_lines='skip')
    df = df[~df['ppl'].isna() & (df['ppl'] > 0) & (df['ppl'] < 100000)]
    return df

def create_full_sparsity_sweep():
    """
    Create comprehensive sparsity ablation figure with fine-grained sparsity levels.
    Uses all 19 sparsity levels from the ablation data.
    """
    df = load_ablation_data()
    main_df = load_main_data()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    precisions = [3, 5]
    titles = ['(a) 3-bit Quantization', '(b) 5-bit Quantization']

    for ax, prec, title in zip(axes, precisions, titles):
        # Get Wanda (FP16, precision=16)
        wanda_df = df[(df['technique'] == 'wanda') & (df['precision'] == 16)]
        if len(wanda_df) > 0:
            wanda_sorted = wanda_df.sort_values('sparsity')
            ax.plot(wanda_sorted['sparsity'] * 100, wanda_sorted['ppl'],
                   color=COLORS['wanda'], marker=MARKERS['wanda'],
                   markersize=5, linewidth=2, label='Wanda (FP16)')

        # Get SparseGPT
        sgpt_df = df[(df['technique'] == 'sparsegpt') & (df['precision'] == prec)]
        if len(sgpt_df) > 0:
            sgpt_sorted = sgpt_df.sort_values('sparsity')
            ax.plot(sgpt_sorted['sparsity'] * 100, sgpt_sorted['ppl'],
                   color=COLORS['sparsegpt'], marker=MARKERS['sparsegpt'],
                   markersize=5, linewidth=2, label=f'SparseGPT ({prec}-bit)')

        # Get PRISM
        prism_df = df[(df['technique'] == 'prism') & (df['precision'] == prec)]
        if len(prism_df) > 0:
            prism_sorted = prism_df.sort_values('sparsity')
            ax.plot(prism_sorted['sparsity'] * 100, prism_sorted['ppl'],
                   color=COLORS['prism'], marker=MARKERS['prism'],
                   markersize=5, linewidth=2, label=f'PRISM ({prec}-bit)')

        # FP16 baseline from main data
        qwen_fp16 = main_df[(main_df['model'] == 'qwen-0.5b') &
                           (main_df['technique'] == 'fp16') &
                           (main_df['dataset'] == 'wikitext2')]
        if len(qwen_fp16) > 0:
            ax.axhline(y=qwen_fp16['ppl'].iloc[0], color='gray',
                      linestyle='--', linewidth=1.5, label='FP16 Baseline')

        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('Perplexity (WikiText-2)')
        ax.set_title(title)
        ax.legend(loc='upper left')

        # Set y-axis limit based on precision
        if prec == 3:
            ax.set_ylim(0, 200)  # Higher range for 3-bit
        else:
            ax.set_ylim(0, 50)   # Lower range for 5-bit

        ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig('/workspace/SINQ/paper/full_sparsity_sweep.pdf')
    plt.savefig('/workspace/SINQ/paper/full_sparsity_sweep.png')
    plt.close()

    print("Created: full_sparsity_sweep.pdf")

    # Print numerical analysis
    print("\n=== SPARSITY ABLATION NUMERICAL ANALYSIS ===")
    for prec in [3, 5]:
        print(f"\n--- {prec}-bit Quantization ---")
        prism_df = df[(df['technique'] == 'prism') & (df['precision'] == prec)]
        sgpt_df = df[(df['technique'] == 'sparsegpt') & (df['precision'] == prec)]

        if len(prism_df) > 0 and len(sgpt_df) > 0:
            # Merge on sparsity
            merged = pd.merge(prism_df[['sparsity', 'ppl']],
                            sgpt_df[['sparsity', 'ppl']],
                            on='sparsity', suffixes=('_prism', '_sgpt'))

            merged['improvement'] = (merged['ppl_sgpt'] - merged['ppl_prism']) / merged['ppl_sgpt'] * 100

            print(f"Sparsity | PRISM PPL | SparseGPT PPL | Improvement")
            for _, row in merged.sort_values('sparsity').iterrows():
                print(f"  {row['sparsity']*100:5.0f}% | {row['ppl_prism']:9.2f} | {row['ppl_sgpt']:13.2f} | {row['improvement']:+.1f}%")

def create_critical_sparsity_analysis():
    """
    Analyze at what sparsity level each method breaks down.
    """
    df = load_ablation_data()
    main_df = load_main_data()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Get FP16 baseline for reference threshold
    qwen_fp16 = main_df[(main_df['model'] == 'qwen-0.5b') &
                       (main_df['technique'] == 'fp16') &
                       (main_df['dataset'] == 'wikitext2')]
    fp16_ppl = qwen_fp16['ppl'].iloc[0] if len(qwen_fp16) > 0 else 12.63

    # Define "breakdown" as 2x FP16 perplexity
    threshold = fp16_ppl * 2

    for tech, color in [('wanda', COLORS['wanda']),
                       ('prism', COLORS['prism']),
                       ('sparsegpt', COLORS['sparsegpt'])]:
        if tech == 'wanda':
            tech_df = df[(df['technique'] == tech) & (df['precision'] == 16)]
            label = 'Wanda (FP16)'
        else:
            tech_df = df[(df['technique'] == tech) & (df['precision'] == 3)]
            label = f'{tech.upper()} (3-bit)'

        if len(tech_df) > 0:
            sorted_df = tech_df.sort_values('sparsity')
            ax.plot(sorted_df['sparsity'] * 100, sorted_df['ppl'],
                   color=color, marker='o', markersize=4, linewidth=2, label=label)

    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1,
              label=f'2× FP16 Threshold ({threshold:.1f})')
    ax.axhline(y=fp16_ppl, color='gray', linestyle=':', linewidth=1,
              label=f'FP16 Baseline ({fp16_ppl:.2f})')

    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Perplexity (WikiText-2)')
    ax.set_title('Critical Sparsity Threshold Analysis (Qwen-0.5B)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('/workspace/SINQ/paper/critical_sparsity.pdf')
    plt.savefig('/workspace/SINQ/paper/critical_sparsity.png')
    plt.close()

    print("\nCreated: critical_sparsity.pdf")

    # Find critical sparsity for each method
    print("\n=== CRITICAL SPARSITY ANALYSIS ===")
    print(f"Threshold: 2× FP16 = {threshold:.2f}")

    for tech in ['wanda', 'prism', 'sparsegpt']:
        if tech == 'wanda':
            tech_df = df[(df['technique'] == tech) & (df['precision'] == 16)]
        else:
            tech_df = df[(df['technique'] == tech) & (df['precision'] == 3)]

        if len(tech_df) > 0:
            sorted_df = tech_df.sort_values('sparsity')
            critical = sorted_df[sorted_df['ppl'] > threshold]['sparsity'].min()
            if pd.notna(critical):
                print(f"  {tech}: breaks down at {critical*100:.0f}% sparsity")
            else:
                print(f"  {tech}: stays below threshold at all tested sparsities")

def create_improvement_vs_sparsity():
    """
    Plot PRISM improvement over SparseGPT as function of sparsity.
    """
    df = load_ablation_data()

    fig, ax = plt.subplots(figsize=(8, 5))

    for prec, linestyle in [(3, '-'), (5, '--')]:
        prism_df = df[(df['technique'] == 'prism') & (df['precision'] == prec)]
        sgpt_df = df[(df['technique'] == 'sparsegpt') & (df['precision'] == prec)]

        if len(prism_df) > 0 and len(sgpt_df) > 0:
            merged = pd.merge(prism_df[['sparsity', 'ppl']],
                            sgpt_df[['sparsity', 'ppl']],
                            on='sparsity', suffixes=('_prism', '_sgpt'))

            merged['improvement'] = (merged['ppl_sgpt'] - merged['ppl_prism']) / merged['ppl_sgpt'] * 100
            merged = merged.sort_values('sparsity')

            ax.plot(merged['sparsity'] * 100, merged['improvement'],
                   color=COLORS['prism'], linestyle=linestyle,
                   marker='o', markersize=5, linewidth=2,
                   label=f'{prec}-bit')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.fill_between([0, 100], [0, 0], [100, 100], alpha=0.1, color='green', label='PRISM wins')
    ax.fill_between([0, 100], [-100, -100], [0, 0], alpha=0.1, color='red', label='SparseGPT wins')

    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('PRISM Improvement over SparseGPT (%)')
    ax.set_title('PRISM Advantage vs Sparsity Level (Qwen-0.5B)')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 100)

    plt.tight_layout()
    plt.savefig('/workspace/SINQ/paper/improvement_vs_sparsity.pdf')
    plt.savefig('/workspace/SINQ/paper/improvement_vs_sparsity.png')
    plt.close()

    print("\nCreated: improvement_vs_sparsity.pdf")

def main():
    print("Generating fine-grained sparsity ablation plots...\n")

    create_full_sparsity_sweep()
    create_critical_sparsity_analysis()
    create_improvement_vs_sparsity()

    print("\n" + "=" * 60)
    print("All ablation plots saved to /workspace/SINQ/paper/")
    print("=" * 60)

if __name__ == '__main__':
    main()
