#!/usr/bin/env python
"""Build the JSQ-vs-PRISM comparison table for qwen-0.5b.

Joins JSQ best-per-config PPLs (results/jsq/jsq_grid_results.tsv, clip mini-search winner)
against the PRISM/SparseGPT/Wanda/SINQ/fp16 baselines (results/benchmark_results.csv),
at matched (precision, sparsity, dataset). Emits results/jsq/comparison_table.md.
"""
import csv, collections, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'results/benchmark_results.csv')
TSV = os.path.join(ROOT, 'results/jsq/jsq_grid_results.tsv')
OUT = os.path.join(ROOT, 'results/jsq/comparison_table.md')

PRECS = ['3', '4', '5']
SPARS = ['0.05', '0.25', '0.5']
DSETS = [('wikitext2', 'WikiText2'), ('c4', 'C4')]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# latest baseline PPL per (technique, precision, sparsity, dataset) for qwen-0.5b
base = {}
for r in csv.DictReader(open(CSV)):
    if r.get('model') != 'qwen-0.5b' or r.get('eval_type') != 'perplexity':
        continue
    p = fnum(r.get('ppl'))
    if p is None:
        continue
    key = (r['technique'], str(int(float(r['precision']))) if r['precision'] else '',
           r['sparsity'], r['dataset'])
    ts = r.get('timestamp', '')
    if key not in base or ts > base[key][1]:
        base[key] = (p, ts)


def b(tech, prec, sp, ds):
    # normalize sparsity formatting: csv stores like '0.5' or '0.05' or '' (quant-only)
    cands = {sp, sp.rstrip('0').rstrip('.') if sp else sp}
    try:
        cands.add(f"{float(sp)}")
    except ValueError:
        pass
    for spk in cands:
        v = base.get((tech, prec, spk, ds))
        if v:
            return v[0]
    return None


# JSQ best-per-config from the grid TSV
jsq = {}
if os.path.exists(TSV):
    for r in csv.DictReader(open(TSV), delimiter='\t'):
        key = (r['variant'], str(int(float(r['precision']))), f"{float(r['sparsity'])}", r['dataset'])
        bp = fnum(r['best_ppl'])
        if bp is not None:
            jsq[key] = (bp, r['best_clip'])


def j(variant, prec, sp, ds):
    v = jsq.get((variant, prec, f"{float(sp)}", ds))
    return v


def cell(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


lines = ["# JSQ vs PRISM and joint baselines — qwen-0.5b (PPL, lower=better)", ""]
lines.append("JSQ columns use the per-config editing-strength (clip_h) mini-search winner "
             "(best of {0,0.005,0.01,0.02}); `c=` notes the winning clip. JSQ uses naive "
             "symmetric per-channel RTN weight quant (paper), so it degrades fast at low bits; "
             "JSQ(WnAn) also quantizes activations to N-bit.")
lines.append("")
for ds_key, ds_name in DSETS:
    fp16 = b('fp16', '16', '0.0', ds_key) or b('fp16', '16', '0', ds_key) or b('fp16', '16', '', ds_key)
    lines.append(f"## {ds_name}  (fp16 ref: {cell(fp16)})")
    lines.append("")
    lines.append("| sparsity | bit | SparseGPT | Wanda(prune-only) | PRISM | JSQ-wo | JSQ-WnAn |")
    lines.append("|---|---|---|---|---|---|---|")
    for sp in SPARS:
        for prec in PRECS:
            sgpt = b('sparsegpt', prec, sp, ds_key)
            wanda = b('wanda', prec, sp, ds_key) or b('wanda', '4', sp, ds_key)  # wanda is bit-independent
            prism = b('prism', prec, sp, ds_key)
            jwo = j('jsq-wo', prec, sp, ds_key)
            jwn = j('jsq', prec, sp, ds_key)
            jwo_s = f"{jwo[0]:.2f} (c={jwo[1]})" if jwo else "—"
            jwn_s = f"{jwn[0]:.2f} (c={jwn[1]})" if jwn else "—"
            lines.append(f"| {int(float(sp)*100)}% | {prec} | {cell(sgpt)} | {cell(wanda)} | "
                         f"{cell(prism)} | {jwo_s} | {jwn_s} |")
    lines.append("")

with open(OUT, 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {OUT}")
print("\n".join(lines))
