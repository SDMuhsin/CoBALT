#!/usr/bin/env python3
"""Collect existing perplexity results for the Table-1 baseline grid.

Scans every results CSV under results/ (recursively) plus the loose per-run
JSON files, and builds a map

    (model, technique, precision, sparsity_pct, dataset) -> ppl

so the runner can skip configs that were already computed (e.g. by the
bias-correction ablation, which already ran wanda-sinq/slim/jsq-wo on
qwen-0.5b and llama-7b for both datasets) and the table builder can assemble
numbers from a union of sources.

Sparsity is normalised to an integer percent (0.05 -> 5, 50 -> 50) so the
0-1 float convention (CSV `sparsity` column) and the *_sp50 filename
convention line up.

Usage:
  collect_ppl.py                      # print coverage report for target grid
  collect_ppl.py --lookup MODEL TECH PREC SP DATASET   # print ppl or NA
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)

TARGET_MODELS = ["qwen-0.5b", "gemma-2b", "llama-7b"]
TARGET_TECHS = ["wanda-sinq", "wanda-awq", "jsq-wo", "slim"]
TARGET_PRECS = [3, 4, 5]
TARGET_SPARS = [5, 25, 50]          # percent
TARGET_DATASETS = ["wikitext2", "c4"]


def _sp_pct(val):
    """Normalise a sparsity value (float 0-1 or percent) to int percent."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f <= 1.0:
        f *= 100.0
    return int(round(f))


def _valid_ppl(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v <= 0 or math.isinf(v) or math.isnan(v):
        return None
    return v


def _add(store, model, tech, prec, sp, dataset, ppl, source):
    ppl = _valid_ppl(ppl)
    if ppl is None:
        return
    sp = _sp_pct(sp)
    if sp is None:
        return
    try:
        prec = int(float(prec))
    except (TypeError, ValueError):
        return
    key = (str(model), str(tech), prec, sp, str(dataset))
    # keep first-seen; prefer CSV sources over jsons (csv scanned first)
    store.setdefault(key, (ppl, source))


def collect(restrict=None):
    """Build the (model,tech,prec,sp,dataset)->ppl map from all result sources.

    If ``restrict`` is given (a list of path substrings), only CSV files whose
    path contains one of those substrings are read, and the loose per-run JSONs
    are skipped entirely. This is used by the runner to reuse ONLY the trusted
    same-settings ablation CSVs (restrict=["ablation_", "table1_baselines"]),
    rather than older mixed-settings runs.
    """
    store = {}

    def _allowed(path):
        if not restrict:
            return True
        return any(sub in path for sub in restrict)

    # --- CSV sources (ablation main.csv, benchmark_results.csv, any *.csv) ---
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "*.csv"), recursive=True)):
        if not _allowed(path):
            continue
        try:
            with open(path, newline="") as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames or "technique" not in rd.fieldnames:
                    continue
                for row in rd:
                    _add(store, row.get("model"), row.get("technique"),
                         row.get("precision"), row.get("sparsity"),
                         row.get("dataset"), row.get("ppl"),
                         os.path.relpath(path, RESULTS_DIR))
        except Exception:
            continue
    if restrict:
        return store
    # --- loose per-run JSONs (results/<model>_<tech>_<prec>bit[_spNN].json) ---
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(path) as fh:
                d = json.load(fh)
            if isinstance(d, list):
                recs = d
            elif isinstance(d, dict) and "results" in d:
                recs = d["results"]
            else:
                recs = [d]
            for r in recs:
                if not isinstance(r, dict):
                    continue
                _add(store, r.get("model"), r.get("technique"),
                     r.get("precision"), r.get("sparsity"),
                     r.get("dataset"), r.get("ppl"),
                     os.path.basename(path))
        except Exception:
            continue
    return store


def lookup(store, model, tech, prec, sp, dataset):
    key = (str(model), str(tech), int(float(prec)), _sp_pct(sp), str(dataset))
    hit = store.get(key)
    return hit[0] if hit else None


def report(store):
    total = missing = 0
    lines = []
    for ds in TARGET_DATASETS:
        for model in TARGET_MODELS:
            for tech in TARGET_TECHS:
                have = []
                miss = []
                for sp in TARGET_SPARS:
                    for prec in TARGET_PRECS:
                        total += 1
                        v = lookup(store, model, tech, prec, sp, ds)
                        tag = f"{prec}b/{sp}%"
                        if v is None:
                            missing += 1
                            miss.append(tag)
                        else:
                            have.append(tag)
                status = "DONE" if not miss else f"MISSING {len(miss)}/9"
                lines.append(f"  {ds:9s} {model:10s} {tech:11s} : {status}"
                             + (f"  [{', '.join(miss)}]" if miss else ""))
    print(f"Target grid: {total} configs  |  present: {total-missing}  |  MISSING: {missing}\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", nargs=5, metavar=("MODEL", "TECH", "PREC", "SP", "DATASET"))
    args = ap.parse_args()
    store = collect()
    if args.lookup:
        v = lookup(store, *args.lookup)
        if v is None:
            print("NA")
            return 1
        print(f"{v:.6f}")
        return 0
    report(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
