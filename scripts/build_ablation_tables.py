#!/usr/bin/env python3
"""Build the bias-correction ablation report from the grid main.csv files.

Reads results/ablation_<model>_<dataset>/main.csv (same schema as benchmark_results.csv),
dedups to the latest timestamp per (technique, precision, sparsity, dataset), and emits a
markdown report with:

  Study A — the within-SINQ 2x2 component decomposition (correction x extras):
     abl-base   = corr OFF, extras OFF      abl-extras = corr OFF, extras ON  (== prism-nocorr)
     abl-corr   = corr ON,  extras OFF      abl-full   = corr ON,  extras ON  (== prism)
   plus the key question: does correction-only (abl-corr) reach full PRISM (abl-full)?

  Study B — sparse-aware-statistics graft onto foreign quantizers:
     {sparsegpt, slim, jsq-wo} vs their *-corr variants vs PRISM.

Usage: python scripts/build_ablation_tables.py [model]   (default qwen-0.5b)
Writes results/ablation_report_<model>.md and prints it.
"""
import csv, sys, os
from collections import defaultdict

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen-0.5b'
DATASETS = ['wikitext2', 'c4']
ROOT = '/workspace/PRISM/results'

# (technique, precision, sparsity, dataset) -> (timestamp, ppl)
best = {}

def ingest(path, dataset_filter=None):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get('model') != MODEL:
                continue
            if row.get('eval_type') != 'perplexity':
                continue
            ppl = row.get('ppl', '')
            if not ppl:
                continue
            try:
                ppl = float(ppl)
            except ValueError:
                continue
            ds = row.get('dataset')
            if dataset_filter and ds != dataset_filter:
                continue
            key = (row['technique'], int(float(row['precision'])),
                   round(float(row['sparsity']), 2), ds)
            ts = row.get('timestamp', '')
            if key not in best or ts > best[key][0]:
                best[key] = (ts, ppl)

for ds in DATASETS:
    ingest(os.path.join(ROOT, f'ablation_{MODEL}_{ds}', 'main.csv'))
# Fall back to the shared benchmark_results.csv for any missing cells.
ingest(os.path.join(ROOT, 'benchmark_results.csv'))

def g(tech, prec, sp, ds):
    v = best.get((tech, prec, round(sp, 2), ds))
    return v[1] if v else None

def fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "  -  "

SPARS = [0.05, 0.25, 0.50]
BITS = [3, 4, 5]
out = []
def w(s=''): out.append(s)

w(f"# Bias-correction ablation — {MODEL}")
w()
w("PRISM = sparse-aware Sinkhorn (**the correction**) + inverse-μ importance + OBS (**the "
  "other components / 'extras'**), on the SINQ quantizer. This report tests whether the "
  "correction alone explains PRISM's gains.")
w()

# ---------- Study A: 2x2 factorial ----------
w("## Study A — within-SINQ 2×2 (only the toggles differ; same quantizer/eval/seed)")
w()
w("`base`=corr✗extras✗  `corr`=corr✓extras✗  `extras`=corr✗extras✓ (=prism-nocorr)  "
  "`full`=corr✓extras✓ (=PRISM)")
w()
for ds in DATASETS:
    w(f"### {ds} — PPL ↓")
    w("| bit | sp | base | +corr only | +extras only | full(PRISM) | "
      "Δcorr\\|extraON | Δextras\\|corrON | corr-only gap to PRISM |")
    w("|----:|---:|-----:|-----------:|-------------:|------------:|---------------:|"
      "---------------:|----------------------:|")
    for sp in SPARS:
        for bit in BITS:
            base = g('abl-base', bit, sp, ds)
            corr = g('abl-corr', bit, sp, ds)
            extras = g('abl-extras', bit, sp, ds)
            full = g('abl-full', bit, sp, ds)
            if full is None:
                full = g('prism', bit, sp, ds)
            dcorr = (extras - full) if (extras and full) else None      # effect of corr, extras ON
            dext = (corr - full) if (corr and full) else None           # effect of extras, corr ON
            gap = (corr - full) if (corr and full) else None            # how far corr-only is from PRISM
            gap_pct = f"{100*gap/full:+.0f}%" if (gap is not None and full) else "-"
            w(f"| {bit} | {int(sp*100)} | {fmt(base)} | {fmt(corr)} | {fmt(extras)} | "
              f"{fmt(full)} | {fmt(dcorr)} | {fmt(dext)} | {fmt(gap)} ({gap_pct}) |")
    w()

# ---------- Decomposition: share of improvement ----------
w("### Decomposition — share of the total base→PRISM improvement")
w("`corr-only share` = (base−corr)/(base−full); `extras-only share` = (base−extras)/(base−full). "
  "If correction explained PRISM, corr-only share ≈ 100% and the corr-only gap ≈ 0.")
w()
w("| dataset | bit | sp | base→PRISM Δ | corr-only share | extras-only share |")
w("|---------|----:|---:|-------------:|----------------:|------------------:|")
for ds in DATASETS:
    for sp in SPARS:
        for bit in BITS:
            base = g('abl-base', bit, sp, ds)
            corr = g('abl-corr', bit, sp, ds)
            extras = g('abl-extras', bit, sp, ds)
            full = g('abl-full', bit, sp, ds) or g('prism', bit, sp, ds)
            if not (base and corr and extras and full):
                continue
            tot = base - full
            if abs(tot) < 1e-9:
                continue
            cs = 100*(base - corr)/tot
            es = 100*(base - extras)/tot
            w(f"| {ds} | {bit} | {int(sp*100)} | {tot:.2f} | {cs:.0f}% | {es:.0f}% |")
w()

# ---------- Study B: transfer grafts ----------
w("## Study B — graft the correction onto foreign quantizers (does it transfer? reach PRISM?)")
w()
for ds in DATASETS:
    w(f"### {ds} — PPL ↓ (baseline → +correction;  PRISM for reference)")
    w("| bit | sp | sparsegpt | →+corr | slim | →+corr | jsq-wo | →+corr | PRISM |")
    w("|----:|---:|----------:|-------:|-----:|-------:|-------:|-------:|------:|")
    for sp in SPARS:
        for bit in BITS:
            row = [g('sparsegpt', bit, sp, ds), g('sparsegpt-corr', bit, sp, ds),
                   g('slim', bit, sp, ds), g('slim-corr', bit, sp, ds),
                   g('jsq-wo', bit, sp, ds), g('jsq-wo-corr', bit, sp, ds),
                   g('abl-full', bit, sp, ds) or g('prism', bit, sp, ds)]
            w(f"| {bit} | {int(sp*100)} | " + " | ".join(fmt(x) for x in row) + " |")
    w()

report = "\n".join(out)
dest = os.path.join(ROOT, f'ablation_report_{MODEL}.md')
with open(dest, 'w') as f:
    f.write(report + "\n")
print(report)
print(f"\n[written] {dest}")
