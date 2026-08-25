#!/usr/bin/env python3
"""Assemble the ablation DOWNSTREAM comparison from the per-task CSVs.

Merges the new ablation downstream dir with the existing c4_downstream baselines
(prism/sparsegpt/slim were already run there at full split). For each
(technique, precision, sparsity) it reports the discriminative-task accuracies and a
mean, so the 2×2 (abl-base/corr/extras/full=prism) and transfer (*-corr) cells can be
compared on task accuracy, not just PPL.

Usage: python scripts/build_ablation_downstream.py [model]   (default qwen-0.5b)
"""
import csv, os, sys
from collections import defaultdict

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen-0.5b'
ROOT = '/workspace/PRISM/results'
# Downstream dirs to merge (later dirs do NOT override; latest timestamp wins per key).
DS_DIRS = [
    os.path.join(ROOT, f'ablation_{MODEL}_c4_ds', 'downstream'),
    os.path.join(ROOT, 'c4_downstream', 'downstream'),
    os.path.join(ROOT, f'ablation_{MODEL}_c4_ds50', 'downstream'),  # focused llama dir (if present)
]
# Discriminative tasks (accuracy) used for the headline mean, + mrr separately.
ACC_TASKS = ['hellaswag', 'arc_easy', 'arc_challenge', 'lambada', 'mmlu']
ALL_TASKS = ACC_TASKS + ['mrr']

def read(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def key(r):
    return (r.get('technique', '').strip(), int(float(r.get('precision', 0))),
            round(float(r.get('sparsity', 0)), 2))

# task -> key -> (timestamp, metric_value)
data = {t: {} for t in ALL_TASKS}
metric_col = {'mrr': 'mrr'}
for t in ALL_TASKS:
    col = metric_col.get(t, 'accuracy')
    for d in DS_DIRS:
        for r in read(os.path.join(d, f'{t}.csv')):
            if r.get('model') != MODEL:
                continue
            v = r.get(col, '')
            if not v or (r.get('error') or '').strip():
                continue
            try:
                v = float(v)
            except ValueError:
                continue
            k = key(r); ts = r.get('timestamp', '')
            if k not in data[t] or ts > data[t][k][0]:
                data[t][k] = (ts, v)

def g(t, tech, prec, sp):
    v = data[t].get((tech, prec, round(sp, 2)))
    return v[1] if v else None

def meanacc(tech, prec, sp):
    vals = [g(t, tech, prec, sp) for t in ACC_TASKS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def pct(x):
    return f"{100*x:.1f}" if isinstance(x, float) else " - "

SPARS = [0.05, 0.25, 0.50]; BITS = [3, 4, 5]
out = []
def w(s=''): out.append(s)

w(f"# Bias-correction ablation — DOWNSTREAM accuracy ({MODEL}, C4 calib)")
w()
w(f"Headline = mean accuracy over {', '.join(ACC_TASKS)} (%). Higher is better.")
w()

w("## Study A — within-SINQ 2×2 (mean downstream acc %)")
w("| bit | sp | base | +corr only | +extras only | full(PRISM) | corr-only vs PRISM |")
w("|----:|---:|-----:|-----------:|-------------:|------------:|-------------------:|")
for sp in SPARS:
    for bit in BITS:
        b = meanacc('abl-base', bit, sp); c = meanacc('abl-corr', bit, sp)
        e = meanacc('abl-extras', bit, sp)
        f = meanacc('abl-full', bit, sp) or meanacc('prism', bit, sp)
        gap = f"{100*(c-f):+.1f}pt" if (c is not None and f is not None) else " - "
        w(f"| {bit} | {int(sp*100)} | {pct(b)} | {pct(c)} | {pct(e)} | {pct(f)} | {gap} |")
w()

w("## Study B — transfer (mean downstream acc %)")
w("| bit | sp | sparsegpt | →+corr | slim | →+corr | jsq-wo | →+corr | PRISM |")
w("|----:|---:|----------:|-------:|-----:|-------:|-------:|-------:|------:|")
for sp in SPARS:
    for bit in BITS:
        cells = [meanacc('sparsegpt', bit, sp), meanacc('sparsegpt-corr', bit, sp),
                 meanacc('slim', bit, sp), meanacc('slim-corr', bit, sp),
                 meanacc('jsq-wo', bit, sp), meanacc('jsq-wo-corr', bit, sp),
                 meanacc('abl-full', bit, sp) or meanacc('prism', bit, sp)]
        w(f"| {bit} | {int(sp*100)} | " + " | ".join(pct(x) for x in cells) + " |")
w()

w("## Per-task detail @ 50% sparsity (acc %, mrr ×100)")
w("| tech | bit | " + " | ".join(ALL_TASKS) + " |")
w("|------|----:|" + "|".join(["----:"]*len(ALL_TASKS)) + "|")
for tech in ['abl-base', 'abl-corr', 'abl-extras', 'abl-full', 'prism',
             'sparsegpt', 'sparsegpt-corr', 'slim', 'slim-corr', 'jsq-wo', 'jsq-wo-corr', 'wanda-sinq']:
    for bit in BITS:
        vals = [g(t, tech, bit, 0.50) for t in ALL_TASKS]
        if all(v is None for v in vals):
            continue
        w(f"| {tech} | {bit} | " + " | ".join(pct(v) for v in vals) + " |")
w()

rep = "\n".join(out)
dest = os.path.join(ROOT, f'ablation_downstream_report_{MODEL}.md')
with open(dest, 'w') as f:
    f.write(rep + "\n")
print(rep)
print(f"\n[written] {dest}")
