#!/usr/bin/env python3
"""
Analyze PRISM benchmark results and generate tables/figures for paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
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

def load_and_clean_data(filepath):
    """Load CSV and filter out corrupted rows."""
    df = pd.read_csv(filepath, on_bad_lines='skip')

    # Filter valid rows - must have valid model, technique, precision
    valid_models = ['qwen-0.5b', 'qwen-1.5b', 'qwen-3b', 'opt-1.3b', 'llama-7b', 'gemma-2b']
    valid_techniques = ['fp16', 'sinq', 'wanda', 'sparsegpt', 'prism']

    df = df[df['model'].isin(valid_models)]
    df = df[df['technique'].isin(valid_techniques)]

    # Convert numeric columns
    for col in ['precision', 'sparsity', 'ppl', 'accuracy']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter valid precisions
    df = df[df['precision'].isin([3, 4, 5, 8, 16])]

    return df

def summarize_data(df):
    """Print summary of available data."""
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal valid rows: {len(df)}")
    print(f"\nModels: {sorted(df['model'].unique())}")
    print(f"Techniques: {sorted(df['technique'].unique())}")
    print(f"Precisions: {sorted(df['precision'].unique())}")
    print(f"Sparsities: {sorted(df['sparsity'].dropna().unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # Show data availability matrix
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY (WikiText-2 PPL)")
    print("=" * 60)

    wiki_df = df[df['dataset'] == 'wikitext2'].copy()

    for model in sorted(wiki_df['model'].unique()):
        print(f"\n--- {model} ---")
        model_df = wiki_df[wiki_df['model'] == model]

        for tech in ['fp16', 'sinq', 'wanda', 'sparsegpt', 'prism']:
            tech_df = model_df[model_df['technique'] == tech]
            if len(tech_df) > 0:
                precs = sorted(tech_df['precision'].unique())
                spars = sorted(tech_df['sparsity'].dropna().unique())
                print(f"  {tech:12s}: prec={precs}, sparsity={[f'{s:.0%}' for s in spars if not np.isnan(s)]}")

    return wiki_df

def get_main_results_table(df):
    """Extract main comparison table data."""
    wiki_df = df[df['dataset'] == 'wikitext2'].copy()

    # Focus on main sparsities: 5%, 25%, 50%
    main_sparsities = [0.05, 0.25, 0.5]
    main_precisions = [3, 4, 5]

    results = {}

    for model in sorted(wiki_df['model'].unique()):
        model_df = wiki_df[wiki_df['model'] == model]
        results[model] = {}

        # Get FP16 baseline (no sparsity, 16-bit)
        fp16 = model_df[(model_df['technique'] == 'fp16') & (model_df['precision'] == 16)]
        if len(fp16) > 0:
            results[model]['fp16'] = fp16['ppl'].iloc[0]

        # Get SINQ (quantization only, no sparsity)
        for prec in main_precisions:
            sinq = model_df[(model_df['technique'] == 'sinq') &
                           (model_df['precision'] == prec) &
                           ((model_df['sparsity'] == 0) | (model_df['sparsity'].isna()))]
            if len(sinq) > 0:
                results[model][f'sinq_{prec}bit'] = sinq['ppl'].iloc[0]

        # Get sparse methods at each sparsity
        for tech in ['wanda', 'sparsegpt', 'prism']:
            for prec in main_precisions:
                for spar in main_sparsities:
                    key = f"{tech}_{prec}bit_{int(spar*100)}sp"
                    tech_df = model_df[(model_df['technique'] == tech) &
                                       (model_df['precision'] == prec) &
                                       (abs(model_df['sparsity'] - spar) < 0.01)]
                    if len(tech_df) > 0:
                        results[model][key] = tech_df['ppl'].iloc[0]

    return results

def print_latex_table(results):
    """Generate LaTeX table for main results."""
    print("\n" + "=" * 60)
    print("MAIN RESULTS TABLE (WikiText-2 PPL↓)")
    print("=" * 60)

    # Print in readable format first
    models = list(results.keys())

    print("\n--- FP16 Baseline ---")
    for model in models:
        if 'fp16' in results[model]:
            print(f"  {model}: {results[model]['fp16']:.2f}")

    print("\n--- SINQ (Quantization Only) ---")
    for prec in [3, 4, 5]:
        print(f"\n  {prec}-bit:")
        for model in models:
            key = f'sinq_{prec}bit'
            if key in results[model]:
                print(f"    {model}: {results[model][key]:.2f}")

    print("\n--- Joint Pruning + Quantization ---")
    for spar in [5, 25, 50]:
        print(f"\n  Sparsity {spar}%:")
        for prec in [3, 4, 5]:
            print(f"\n    {prec}-bit:")
            for tech in ['wanda', 'sparsegpt', 'prism']:
                key = f"{tech}_{prec}bit_{spar}sp"
                print(f"      {tech:12s}:", end=" ")
                for model in models:
                    if key in results[model]:
                        val = results[model][key]
                        print(f"{model}={val:.2f}", end="  ")
                print()

def analyze_prism_advantage(results):
    """Analyze where PRISM outperforms SparseGPT."""
    print("\n" + "=" * 60)
    print("PRISM vs SparseGPT COMPARISON")
    print("=" * 60)

    comparisons = []

    for model in results.keys():
        for prec in [3, 4, 5]:
            for spar in [5, 25, 50]:
                prism_key = f"prism_{prec}bit_{spar}sp"
                sgpt_key = f"sparsegpt_{prec}bit_{spar}sp"

                if prism_key in results[model] and sgpt_key in results[model]:
                    prism_ppl = results[model][prism_key]
                    sgpt_ppl = results[model][sgpt_key]
                    improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100

                    comparisons.append({
                        'model': model,
                        'precision': prec,
                        'sparsity': spar,
                        'prism_ppl': prism_ppl,
                        'sparsegpt_ppl': sgpt_ppl,
                        'improvement_%': improvement
                    })

    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        print("\nPRISM PPL Improvement over SparseGPT:")
        print(comp_df.to_string(index=False))

        print(f"\nAverage improvement: {comp_df['improvement_%'].mean():.2f}%")
        print(f"Max improvement: {comp_df['improvement_%'].max():.2f}%")
        print(f"Min improvement: {comp_df['improvement_%'].min():.2f}%")

        return comp_df

    return None

def main():
    print("Loading benchmark results...")
    df = load_and_clean_data('/workspace/SINQ/results/benchmark_results.csv')

    wiki_df = summarize_data(df)

    results = get_main_results_table(df)
    print_latex_table(results)

    analyze_prism_advantage(results)

    # Also load ablation data
    print("\n" + "=" * 60)
    print("SPARSITY ABLATION DATA")
    print("=" * 60)

    try:
        ablation_df = load_and_clean_data('/workspace/SINQ/results/sparsity_ablation_results.csv')
        ablation_wiki = ablation_df[ablation_df['dataset'] == 'wikitext2']

        print(f"\nAblation rows: {len(ablation_wiki)}")
        print(f"Sparsities tested: {sorted(ablation_wiki['sparsity'].unique())}")
        print(f"Techniques: {sorted(ablation_wiki['technique'].unique())}")
        print(f"Precisions: {sorted(ablation_wiki['precision'].unique())}")
    except Exception as e:
        print(f"Error loading ablation data: {e}")

if __name__ == '__main__':
    main()
