#!/usr/bin/env python3
"""
Generate LaTeX tables for PRISM paper.
"""
import pandas as pd
import numpy as np

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

def generate_main_results_table(df):
    """Generate main WikiText-2 perplexity comparison table."""

    wiki_df = df[(df['dataset'] == 'wikitext2')]

    models = ['qwen-0.5b', 'llama-7b']
    model_labels = {'qwen-0.5b': 'Qwen-0.5B', 'llama-7b': 'LLaMA-7B'}

    print(r"""
\begin{table*}[t]
\centering
\caption{WikiText-2 Perplexity ($\downarrow$) comparison across models, bit-widths, and sparsity levels. \textbf{Bold} indicates best result in each configuration. PRISM consistently outperforms SparseGPT, with improvements most pronounced at lower bit-widths where variance distortion has greatest impact.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}ll|c|ccc|ccc|ccc@{}}
\toprule
& & & \multicolumn{3}{c|}{\textbf{5\% Sparsity}} & \multicolumn{3}{c|}{\textbf{25\% Sparsity}} & \multicolumn{3}{c}{\textbf{50\% Sparsity}} \\
\textbf{Model} & \textbf{Method} & \textbf{FP16} & 3-bit & 4-bit & 5-bit & 3-bit & 4-bit & 5-bit & 3-bit & 4-bit & 5-bit \\
\midrule""")

    for model in models:
        model_df = wiki_df[wiki_df['model'] == model]

        # Get FP16 baseline
        fp16_df = model_df[(model_df['technique'] == 'fp16') & (model_df['precision'] == 16)]
        fp16_ppl = fp16_df['ppl'].iloc[0] if len(fp16_df) > 0 else np.nan

        # Get SINQ baselines (no sparsity)
        sinq_ppls = {}
        for prec in [3, 4, 5]:
            sinq_df = model_df[(model_df['technique'] == 'sinq') &
                              (model_df['precision'] == prec) &
                              ((model_df['sparsity'] == 0) | (model_df['sparsity'].isna()))]
            sinq_ppls[prec] = sinq_df['ppl'].iloc[0] if len(sinq_df) > 0 else np.nan

        # Print SINQ row
        print(f"{model_labels[model]} & SINQ & ", end="")
        if not np.isnan(fp16_ppl):
            print(f"{fp16_ppl:.2f} & ", end="")
        else:
            print("-- & ", end="")

        for spar in [0.05, 0.25, 0.5]:
            for prec in [3, 4, 5]:
                if spar == 0.05 and prec in sinq_ppls and not np.isnan(sinq_ppls[prec]):
                    print(f"{sinq_ppls[prec]:.2f}", end="")
                else:
                    print("--", end="")
                if not (spar == 0.5 and prec == 5):
                    print(" & ", end="")
        print(r" \\")

        # Print sparse methods
        for tech in ['wanda', 'sparsegpt', 'prism']:
            tech_label = {'wanda': 'Wanda', 'sparsegpt': 'SparseGPT', 'prism': '\\textbf{PRISM}'}[tech]
            print(f" & {tech_label} & -- & ", end="")

            for spar_idx, spar in enumerate([0.05, 0.25, 0.5]):
                for prec_idx, prec in enumerate([3, 4, 5]):
                    if tech == 'wanda':
                        # Wanda uses 16-bit (no quantization)
                        tech_df = model_df[(model_df['technique'] == tech) &
                                          (model_df['precision'] == 16) &
                                          (abs(model_df['sparsity'] - spar) < 0.01)]
                    else:
                        tech_df = model_df[(model_df['technique'] == tech) &
                                          (model_df['precision'] == prec) &
                                          (abs(model_df['sparsity'] - spar) < 0.01)]

                    if len(tech_df) > 0:
                        ppl = tech_df['ppl'].iloc[0]
                        if not np.isnan(ppl):
                            # Find best for this config
                            best_ppl = ppl
                            for t in ['wanda', 'sparsegpt', 'prism']:
                                if t == 'wanda':
                                    t_df = model_df[(model_df['technique'] == t) &
                                                   (model_df['precision'] == 16) &
                                                   (abs(model_df['sparsity'] - spar) < 0.01)]
                                else:
                                    t_df = model_df[(model_df['technique'] == t) &
                                                   (model_df['precision'] == prec) &
                                                   (abs(model_df['sparsity'] - spar) < 0.01)]
                                if len(t_df) > 0 and not np.isnan(t_df['ppl'].iloc[0]):
                                    if t_df['ppl'].iloc[0] < best_ppl:
                                        best_ppl = t_df['ppl'].iloc[0]

                            if abs(ppl - best_ppl) < 0.01:
                                print(f"\\textbf{{{ppl:.2f}}}", end="")
                            else:
                                print(f"{ppl:.2f}", end="")
                        else:
                            print("--", end="")
                    else:
                        print("--", end="")

                    if not (spar_idx == 2 and prec_idx == 2):
                        print(" & ", end="")

            print(r" \\")

        if model != models[-1]:
            print(r"\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table*}
""")

def generate_c4_table(df):
    """Generate C4 perplexity table for supplementary."""

    c4_df = df[(df['dataset'] == 'c4')]

    print(r"""
\begin{table}[t]
\centering
\caption{C4 Perplexity ($\downarrow$) on LLaMA-7B. Results confirm WikiText-2 findings: PRISM provides consistent improvement over SparseGPT.}
\label{tab:c4_results}
\small
\begin{tabular}{@{}l|ccc@{}}
\toprule
\textbf{Method} & 3-bit & 4-bit & 5-bit \\
\midrule""")

    model = 'llama-7b'
    model_df = c4_df[c4_df['model'] == model]

    # FP16
    fp16_df = model_df[(model_df['technique'] == 'fp16')]
    if len(fp16_df) > 0 and not np.isnan(fp16_df['ppl'].iloc[0]):
        print(f"FP16 Baseline & \\multicolumn{{3}}{{c}}{{{fp16_df['ppl'].iloc[0]:.2f}}} \\\\")
        print(r"\midrule")

    # SINQ
    print("SINQ (dense) & ", end="")
    for prec in [3, 4, 5]:
        sinq_df = model_df[(model_df['technique'] == 'sinq') &
                          (model_df['precision'] == prec)]
        if len(sinq_df) > 0 and not np.isnan(sinq_df['ppl'].iloc[0]):
            print(f"{sinq_df['ppl'].iloc[0]:.2f}", end="")
        else:
            print("--", end="")
        if prec != 5:
            print(" & ", end="")
    print(r" \\")
    print(r"\midrule")

    # Sparse methods at 50% sparsity
    for tech in ['sparsegpt', 'prism']:
        tech_label = 'SparseGPT' if tech == 'sparsegpt' else '\\textbf{PRISM}'
        print(f"{tech_label} (50\\%) & ", end="")
        for prec in [3, 4, 5]:
            tech_df = model_df[(model_df['technique'] == tech) &
                              (model_df['precision'] == prec) &
                              (abs(model_df['sparsity'] - 0.5) < 0.01)]
            if len(tech_df) > 0 and not np.isnan(tech_df['ppl'].iloc[0]):
                print(f"{tech_df['ppl'].iloc[0]:.2f}", end="")
            else:
                print("--", end="")
            if prec != 5:
                print(" & ", end="")
        print(r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

def generate_summary_table(df):
    """Generate summary of improvements."""

    wiki_df = df[(df['dataset'] == 'wikitext2')]

    print(r"""
\begin{table}[t]
\centering
\caption{Average PRISM improvement (\%) over SparseGPT across all tested configurations on Qwen-0.5B and LLaMA-7B. The improvement is most pronounced at lower bit-widths, consistent with the theoretical prediction that variance distortion has greater impact when quantization resolution is limited.}
\label{tab:improvement_summary}
\small
\begin{tabular}{@{}l|ccc|c@{}}
\toprule
& \textbf{3-bit} & \textbf{4-bit} & \textbf{5-bit} & \textbf{Avg} \\
\midrule""")

    for spar in [0.05, 0.25, 0.5]:
        improvements = {3: [], 4: [], 5: []}
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
                    if not np.isnan(prism_ppl) and not np.isnan(sgpt_ppl):
                        improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100
                        improvements[prec].append(improvement)

        print(f"{int(spar*100)}\\% Sparsity & ", end="")
        row_avg = []
        for prec in [3, 4, 5]:
            if improvements[prec]:
                avg = np.mean(improvements[prec])
                row_avg.append(avg)
                print(f"{avg:.1f}\\%", end="")
            else:
                print("--", end="")
            print(" & ", end="")

        if row_avg:
            print(f"{np.mean(row_avg):.1f}\\%", end="")
        else:
            print("--", end="")
        print(r" \\")

    # Overall average
    print(r"\midrule")
    print("\\textbf{Overall} & ", end="")
    all_improvements = {3: [], 4: [], 5: []}
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
                    if not np.isnan(prism_ppl) and not np.isnan(sgpt_ppl):
                        improvement = (sgpt_ppl - prism_ppl) / sgpt_ppl * 100
                        all_improvements[prec].append(improvement)

    grand_avg = []
    for prec in [3, 4, 5]:
        if all_improvements[prec]:
            avg = np.mean(all_improvements[prec])
            grand_avg.append(avg)
            print(f"\\textbf{{{avg:.1f}\\%}}", end="")
        else:
            print("--", end="")
        print(" & ", end="")

    if grand_avg:
        print(f"\\textbf{{{np.mean(grand_avg):.1f}\\%}}", end="")
    print(r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

if __name__ == '__main__':
    df = load_data()

    print("% ============================================")
    print("% MAIN RESULTS TABLE")
    print("% ============================================")
    generate_main_results_table(df)

    print("\n% ============================================")
    print("% IMPROVEMENT SUMMARY TABLE")
    print("% ============================================")
    generate_summary_table(df)

    print("\n% ============================================")
    print("% C4 RESULTS TABLE")
    print("% ============================================")
    generate_c4_table(df)
