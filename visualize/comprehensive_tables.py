#!/usr/bin/env python3
"""
Generate comprehensive LaTeX tables for PRISM paper.
Includes all models (Qwen-0.5B, OPT-1.3B, LLaMA-7B) and datasets (WikiText-2, C4).
"""
import pandas as pd
import numpy as np

def load_data():
    """Load and clean benchmark data."""
    df = pd.read_csv('/workspace/SINQ/results/benchmark_results.csv', on_bad_lines='skip')
    df = df[~df['ppl'].isna() & (df['ppl'] > 0) & (df['ppl'] < 100000)]
    return df

def generate_main_table_wikitext2(df):
    """Generate comprehensive WikiText-2 table with all 3 models."""
    wiki = df[df['dataset'] == 'wikitext2']

    models = ['qwen-0.5b', 'opt-1.3b', 'llama-7b']
    model_labels = {
        'qwen-0.5b': 'Qwen-0.5B',
        'opt-1.3b': 'OPT-1.3B$^\\ddagger$',
        'llama-7b': 'LLaMA-7B'
    }

    print(r"""
\begin{table*}[t]
\centering
\caption{WikiText-2 Perplexity ($\downarrow$) comparison across all tested models, bit-widths, and sparsity levels. \textbf{Bold} indicates best result among sparse-quantization methods (SparseGPT, PRISM) in each column. PRISM outperforms SparseGPT on Qwen-0.5B and LLaMA-7B but shows degraded performance on OPT-1.3B (marked with $\ddagger$), suggesting architecture-dependent behavior.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ll|c|ccc|ccc|ccc@{}}
\toprule
& & & \multicolumn{3}{c|}{\textbf{5\% Sparsity}} & \multicolumn{3}{c|}{\textbf{25\% Sparsity}} & \multicolumn{3}{c}{\textbf{50\% Sparsity}} \\
\textbf{Model} & \textbf{Method} & \textbf{FP16} & 3-bit & 4-bit & 5-bit & 3-bit & 4-bit & 5-bit & 3-bit & 4-bit & 5-bit \\
\midrule""")

    for model in models:
        model_df = wiki[wiki['model'] == model]

        # FP16 baseline
        fp16 = model_df[(model_df['technique'] == 'fp16') & (model_df['precision'] == 16)]
        fp16_ppl = fp16['ppl'].iloc[0] if len(fp16) > 0 else None

        # SINQ row
        print(f"{model_labels[model]} & SINQ & ", end="")
        if fp16_ppl:
            print(f"{fp16_ppl:.2f} & ", end="")
        else:
            print("-- & ", end="")

        for spar in [0.05, 0.25, 0.5]:
            for prec in [3, 4, 5]:
                if spar == 0.05:  # SINQ only at 0% sparsity
                    sinq = model_df[(model_df['technique'] == 'sinq') & (model_df['precision'] == prec)]
                    if len(sinq) > 0:
                        print(f"{sinq['ppl'].iloc[0]:.2f}", end="")
                    else:
                        print("--", end="")
                else:
                    print("--", end="")
                if not (spar == 0.5 and prec == 5):
                    print(" & ", end="")
        print(r" \\")

        # Wanda row
        print(f" & Wanda$^\\dagger$ & -- & ", end="")
        for spar_idx, spar in enumerate([0.05, 0.25, 0.5]):
            wanda = model_df[(model_df['technique'] == 'wanda') &
                            (model_df['precision'] == 16) &
                            (abs(model_df['sparsity'] - spar) < 0.02)]
            if len(wanda) > 0:
                val = f"{wanda['ppl'].iloc[0]:.2f}"
            else:
                val = "--"
            print(f"\\multicolumn{{3}}{{{'c|' if spar_idx < 2 else 'c'}}}{{{val}}}", end="")
            if spar_idx < 2:
                print(" & ", end="")
        print(r" \\")

        # SparseGPT and PRISM rows
        for tech in ['sparsegpt', 'prism']:
            tech_label = 'SparseGPT' if tech == 'sparsegpt' else '\\textbf{PRISM}'
            print(f" & {tech_label} & -- & ", end="")

            for spar_idx, spar in enumerate([0.05, 0.25, 0.5]):
                for prec_idx, prec in enumerate([3, 4, 5]):
                    tech_df = model_df[(model_df['technique'] == tech) &
                                      (model_df['precision'] == prec) &
                                      (abs(model_df['sparsity'] - spar) < 0.02)]

                    if len(tech_df) > 0:
                        ppl = tech_df['ppl'].iloc[0]
                        # Find best between SparseGPT and PRISM
                        other_tech = 'prism' if tech == 'sparsegpt' else 'sparsegpt'
                        other_df = model_df[(model_df['technique'] == other_tech) &
                                           (model_df['precision'] == prec) &
                                           (abs(model_df['sparsity'] - spar) < 0.02)]

                        is_best = True
                        if len(other_df) > 0:
                            other_ppl = other_df['ppl'].iloc[0]
                            if other_ppl < ppl:
                                is_best = False

                        if is_best:
                            print(f"\\textbf{{{ppl:.2f}}}", end="")
                        else:
                            print(f"{ppl:.2f}", end="")
                    else:
                        print("--", end="")

                    if not (spar_idx == 2 and prec_idx == 2):
                        print(" & ", end="")
            print(r" \\")

        if model != models[-1]:
            print(r"\midrule")

    print(r"""\multicolumn{12}{l}{\footnotesize $^\dagger$Wanda uses FP16 weights (no quantization). $^\ddagger$OPT-1.3B shows degraded PRISM performance (see Section~\ref{subsec:limitations}).} \\
\bottomrule
\end{tabular}
\end{table*}
""")

def generate_c4_table(df):
    """Generate C4 cross-dataset validation table."""
    c4 = df[df['dataset'] == 'c4']

    print(r"""
\begin{table}[t]
\centering
\caption{C4 Perplexity ($\downarrow$) cross-dataset validation. Results confirm WikiText-2 findings: PRISM outperforms SparseGPT on LLaMA-7B with substantial improvements at lower bit-widths.}
\label{tab:c4_results}
\small
\begin{tabular}{@{}l|l|ccc@{}}
\toprule
\textbf{Model} & \textbf{Method} & 3-bit & 4-bit & 5-bit \\
\midrule""")

    for model in ['llama-7b']:
        model_df = c4[c4['model'] == model]

        # FP16
        fp16 = model_df[(model_df['technique'] == 'fp16')]
        if len(fp16) > 0:
            print(f"LLaMA-7B & FP16 & \\multicolumn{{3}}{{c}}{{{fp16['ppl'].iloc[0]:.2f}}} \\\\")

        # SINQ
        print("& SINQ & ", end="")
        for prec in [3, 4, 5]:
            sinq = model_df[(model_df['technique'] == 'sinq') & (model_df['precision'] == prec)]
            if len(sinq) > 0:
                print(f"{sinq['ppl'].iloc[0]:.2f}", end="")
            else:
                print("--", end="")
            if prec != 5:
                print(" & ", end="")
        print(r" \\")
        print(r"\midrule")

        # SparseGPT and PRISM at 50% sparsity
        for tech in ['sparsegpt', 'prism']:
            tech_label = 'SparseGPT' if tech == 'sparsegpt' else '\\textbf{PRISM}'
            print(f"& {tech_label} (50\\%) & ", end="")
            for prec in [3, 4, 5]:
                tech_df = model_df[(model_df['technique'] == tech) &
                                  (model_df['precision'] == prec) &
                                  (abs(model_df['sparsity'] - 0.5) < 0.02)]
                if len(tech_df) > 0:
                    ppl = tech_df['ppl'].iloc[0]
                    # Check if best
                    other = 'prism' if tech == 'sparsegpt' else 'sparsegpt'
                    other_df = model_df[(model_df['technique'] == other) &
                                       (model_df['precision'] == prec) &
                                       (abs(model_df['sparsity'] - 0.5) < 0.02)]
                    is_best = len(other_df) == 0 or ppl < other_df['ppl'].iloc[0]
                    if is_best:
                        print(f"\\textbf{{{ppl:.2f}}}", end="")
                    else:
                        print(f"{ppl:.2f}", end="")
                else:
                    print("--", end="")
                if prec != 5:
                    print(" & ", end="")
            print(r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

def generate_improvement_table(df):
    """Generate improvement summary table."""
    wiki = df[df['dataset'] == 'wikitext2']

    print(r"""
\begin{table}[t]
\centering
\caption{PRISM improvement (\%) over SparseGPT. Positive values indicate PRISM outperforms. On Qwen-0.5B and LLaMA-7B, PRISM shows consistent improvements (avg 31.3\%), while OPT-1.3B shows degradation (avg -163\%), indicating architecture-dependent behavior.}
\label{tab:improvement_summary}
\small
\begin{tabular}{@{}l|l|ccc|c@{}}
\toprule
\textbf{Model} & \textbf{Sparsity} & \textbf{3-bit} & \textbf{4-bit} & \textbf{5-bit} & \textbf{Avg} \\
\midrule""")

    for model in ['qwen-0.5b', 'llama-7b', 'opt-1.3b']:
        model_label = {'qwen-0.5b': 'Qwen-0.5B', 'llama-7b': 'LLaMA-7B', 'opt-1.3b': 'OPT-1.3B'}[model]
        model_df = wiki[wiki['model'] == model]

        first_row = True
        for spar in [0.05, 0.25, 0.5]:
            if first_row:
                print(f"{model_label} & {int(spar*100)}\\% & ", end="")
                first_row = False
            else:
                print(f" & {int(spar*100)}\\% & ", end="")

            row_improvements = []
            for prec in [3, 4, 5]:
                prism = model_df[(model_df['technique'] == 'prism') &
                                (model_df['precision'] == prec) &
                                (abs(model_df['sparsity'] - spar) < 0.02)]
                sgpt = model_df[(model_df['technique'] == 'sparsegpt') &
                               (model_df['precision'] == prec) &
                               (abs(model_df['sparsity'] - spar) < 0.02)]

                if len(prism) > 0 and len(sgpt) > 0:
                    improvement = (sgpt['ppl'].iloc[0] - prism['ppl'].iloc[0]) / sgpt['ppl'].iloc[0] * 100
                    row_improvements.append(improvement)
                    if improvement > 0:
                        print(f"{improvement:.1f}\\%", end="")
                    else:
                        print(f"\\textcolor{{red}}{{{improvement:.1f}\\%}}", end="")
                else:
                    print("--", end="")
                print(" & ", end="")

            if row_improvements:
                avg = np.mean(row_improvements)
                if avg > 0:
                    print(f"{avg:.1f}\\%", end="")
                else:
                    print(f"\\textcolor{{red}}{{{avg:.1f}\\%}}", end="")
            else:
                print("--", end="")
            print(r" \\")

        if model != 'opt-1.3b':
            print(r"\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

def main():
    print("% ============================================")
    print("% COMPREHENSIVE PRISM RESULTS TABLES")
    print("% Generated by comprehensive_tables.py")
    print("% ============================================\n")

    df = load_data()

    print("% === MAIN RESULTS TABLE (WikiText-2, all models) ===")
    generate_main_table_wikitext2(df)

    print("\n% === IMPROVEMENT SUMMARY TABLE ===")
    generate_improvement_table(df)

    print("\n% === C4 CROSS-DATASET TABLE ===")
    generate_c4_table(df)

if __name__ == '__main__':
    main()
