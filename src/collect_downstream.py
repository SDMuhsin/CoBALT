#!/usr/bin/env python3
"""Merge results/downstream_grid/*.csv into per-task comparison tables.

Rows = sparsity, columns = technique, cells = accuracy%. Picks, per
(technique, sparsity), the row with the largest `total` (full split) and no
error. Emits markdown. Usage: python collect_downstream.py [grid_dir]
"""
import csv, os, sys, json
from collections import defaultdict

GRID = sys.argv[1] if len(sys.argv) > 1 else "/workspace/PTQResearch/results/downstream_grid"
TASKS = [("hellaswag", "hellaswag.csv"), ("arc_easy", "arc_easy.csv"),
         ("arc_challenge", "arc_challenge.csv"), ("mmlu", "mmlu.csv")]
# preferred column order for the comparison
TECH_ORDER = ["valor", "prism", "wanda-awq", "wanda-sinq"]
TECH_LABEL = {"valor": "VALOR", "prism": "PRISM",
              "wanda-awq": "Wanda+AWQ", "wanda-sinq": "Wanda+SINQ"}


def load_task(path):
    """Return {(technique, sparsity_float): (acc_pct, total, limit, err)} best row."""
    best = {}
    if not os.path.exists(path):
        return best
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("precision", "")).strip() not in ("3", "3.0"):
                continue
            tech = (r.get("technique") or "").strip()
            try:
                sp = round(float(r.get("sparsity", "nan")), 3)
            except (ValueError, TypeError):
                continue
            err = (r.get("error") or "").strip()
            try:
                total = int(float(r.get("total") or 0))
            except (ValueError, TypeError):
                total = 0
            try:
                acc = float(r.get("accuracy")) * 100.0
            except (ValueError, TypeError):
                continue
            limit = (r.get("limit") or "").strip()
            key = (tech, sp)
            # prefer: no error, then largest total (full split beats subsample)
            cand = (0 if err else 1, total, acc, total, limit, err)
            if key not in best or cand[:2] > best[key][:2]:
                best[key] = cand
    return {k: (v[2], v[3], v[4], v[5]) for k, v in best.items()}


def fmt_table(task, data):
    techs = [t for t in TECH_ORDER if any(k[0] == t for k in data)]
    techs += sorted({k[0] for k in data} - set(TECH_ORDER))
    sparsities = sorted({k[1] for k in data})
    lines = [f"### {task}", ""]
    header = "| sparsity | " + " | ".join(TECH_LABEL.get(t, t) for t in techs) + " |"
    sep = "|" + "---|" * (len(techs) + 1)
    lines += [header, sep]
    for sp in sparsities:
        cells = []
        for t in techs:
            v = data.get((t, sp))
            if v is None:
                cells.append("—")
            else:
                acc, total, limit, err = v
                tag = "" if not limit else f"^{limit}"
                cells.append(f"{acc:.2f}{tag}" if not err else "ERR")
        lines.append(f"| {sp:.2f} | " + " | ".join(cells) + " |")
    # note total N per task (from any full cell)
    tots = {v[1] for v in data.values() if not v[2]}
    n = max(tots) if tots else (max((v[1] for v in data.values()), default=0))
    lines.append("")
    lines.append(f"_N={n} per config (full split unless a cell is annotated ^limit)._")
    lines.append("")
    return "\n".join(lines)


def main():
    print(f"# Downstream comparison @ 3-bit — grid: {GRID}\n")
    any_data = False
    for task, fname in TASKS:
        data = load_task(os.path.join(GRID, fname))
        if not data:
            print(f"### {task}\n\n_(no rows yet)_\n")
            continue
        any_data = True
        print(fmt_table(task, data))
    if not any_data:
        print("_No results in grid yet._")


if __name__ == "__main__":
    main()
