#!/usr/bin/env python3
"""Collate the flat camera-ready CSV (results/benchmark_camera/results.csv) into
the camera-ready table: one row per (method, bits, sparsity), columns for the 3
PPL tasks + 4 reasoning tasks. Prints markdown grouped by sparsity, and a
completeness report (missing / errored cells).

Usage:  python src/collect_camera.py [--csv PATH] [--md OUT.md]
"""
import argparse
import csv
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPL = ["wikitext2", "ptb", "c4"]
DS = ["arc_easy", "hellaswag", "piqa", "winogrande"]
TASKS = PPL + DS
COL = {"wikitext2": "wiki↓", "ptb": "ptb↓", "c4": "c4↓",
       "arc_easy": "arc_e↑", "hellaswag": "hella↑", "piqa": "piqa↑", "winogrande": "wino↑"}
METHOD_ORDER = ["fp16", "awq", "sinq", "wanda", "sparsegpt", "jsq-wo", "slim",
                "wanda-awq", "wanda-sinq", "cobalt"]
PRUNE = {"wanda", "sparsegpt", "jsq-wo", "slim", "wanda-awq", "wanda-sinq", "cobalt"}


def load(csv_path):
    """Return {(method,bits,sparsity): {task: (value, error)}}, last row wins."""
    cells = defaultdict(dict)
    if not os.path.exists(csv_path):
        return cells
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["method"], r["bits"], r["sparsity"])
            cells[key][r["task"]] = (r.get("value", ""), (r.get("error") or "").strip())
    return cells


def fmt(task, cell):
    if cell is None:
        return "—"
    val, err = cell
    if err:
        return "ERR"
    if val == "" or val is None:
        return "·"
    try:
        x = float(val)
    except ValueError:
        return str(val)
    if task in PPL:
        return f"{x:.2f}" if x < 1000 else f"{x:.3g}"
    return f"{x*100:.2f}"   # accuracy as %


def method_sort_key(key):
    method, bits, sparsity = key
    mi = METHOD_ORDER.index(method) if method in METHOD_ORDER else 99
    return (float(sparsity), mi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "benchmark_camera", "results.csv"))
    ap.add_argument("--md", default=None)
    ap.add_argument("--model", default="gemma-2b", help="model label for the table header")
    args = ap.parse_args()

    cells = load(args.csv)
    lines = []
    lines.append(f"# Camera-ready benchmark — {args.model}, 3-bit (PPL↓, accuracy%↑)\n")
    lines.append(f"_source: {os.path.relpath(args.csv, _ROOT)} — {len(cells)} cells_\n")

    header = "| method | bits | sp | " + " | ".join(COL[t] for t in TASKS) + " |"
    sep = "|" + "---|" * (3 + len(TASKS))

    # group by sparsity band; flat methods (sparsity 0) shown once at the top.
    flat = sorted([k for k in cells if k[0] not in PRUNE], key=method_sort_key)
    pruned = [k for k in cells if k[0] in PRUNE]
    by_sp = defaultdict(list)
    for k in pruned:
        by_sp[k[2]].append(k)

    def emit(keys, title):
        lines.append(f"\n### {title}\n")
        lines.append(header); lines.append(sep)
        for key in sorted(keys, key=method_sort_key):
            method, bits, sparsity = key
            row = cells[key]
            vals = " | ".join(fmt(t, row.get(t)) for t in TASKS)
            lines.append(f"| {method} | {bits} | {sparsity} | {vals} |")

    if flat:
        emit(flat, "No-pruning references (flat across sparsity)")
    for sp in sorted(by_sp, key=float):
        emit(by_sp[sp], f"Sparsity {sp}")

    # completeness report
    missing = []
    for key in cells:
        for t in TASKS:
            c = cells[key].get(t)
            if c is None:
                missing.append((key, t, "MISSING"))
            elif c[1]:
                missing.append((key, t, c[1][:60]))
    lines.append(f"\n### Completeness\n")
    if not missing:
        lines.append("All present cells fully populated (no missing/errored tasks).")
    else:
        for key, t, why in missing:
            lines.append(f"- {key[0]} sp={key[2]} bits={key[1]} · {t}: {why}")

    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w") as f:
            f.write(out + "\n")
        print(f"\n[wrote {args.md}]")


if __name__ == "__main__":
    main()
