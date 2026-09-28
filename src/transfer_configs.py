"""Table 1 winning configuration per (method, bits, sparsity), for transfer to other suites.

The gemma-2b decoder sweep resolves an argmax per (method, task, bits, sparsity). Its four
tasks have no image in GLUE, so the transfer key is the CELL: for each (method, bits,
sparsity) this module returns the single configuration that maximizes the four-task MEAN in
that cell, which is the quantity Table 1's verdict is scored on. Ties are broken by the
canonical grid order in src/tuned_grids.py, so the choice is deterministic.

Every arm, CoBALT included, is pinned this way, so no arm is tuned on the target suite.
"""
import csv
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import tuned_grids as tg  # noqa: E402

TABLE1_CSV = os.path.join(_ROOT, "results", "benchmark_camera_ds_tuned", "results.csv")
TASKS = ["arc_easy", "piqa", "hellaswag", "winogrande"]


def cell_winners(csv_path=TABLE1_CSV, methods=None):
    """{(method, bits, sparsity): hp_label} -- the four-task-mean argmax in that cell."""
    methods = methods or ["cobalt", "sparsegpt", "wanda-awq", "wanda-sinq", "jsq-wo"]
    order = {m: [lab for lab, _ in tg.variants(m)] for m in methods}
    acc = defaultdict(dict)                      # (m,b,sp) -> {hp: {task: value}}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            m = r.get("method")
            if m not in order or r.get("error"):
                continue
            hp, t, v = r.get("hp") or "default", r.get("task"), r.get("value")
            if hp not in order[m] or t not in TASKS or v in (None, ""):
                continue
            try:
                b, sp = int(r["bits"]), round(float(r["sparsity"]), 2)
                v = float(v) * 100.0
            except (TypeError, ValueError):
                continue
            acc[(m, b, sp)].setdefault(hp, {})[t] = max(
                acc[(m, b, sp)].get(hp, {}).get(t, -1.0), v)
    out = {}
    for key, per_hp in acc.items():
        full = {hp: sum(d.values()) / 4 for hp, d in per_hp.items() if len(d) == 4}
        if not full:
            continue
        rank = {hp: i for i, hp in enumerate(order[key[0]])}
        out[key] = min(full, key=lambda hp: (-full[hp], rank[hp]))
    return out


if __name__ == "__main__":
    w = cell_winners()
    for m in ["cobalt", "sparsegpt", "wanda-awq", "wanda-sinq", "jsq-wo"]:
        print(f"\n{m}")
        for b in (3, 4):
            print("  " + f"{b}-bit " + "  ".join(
                f"sp{sp:.2f}={w.get((m, b, sp), 'MISSING'):<18s}"
                for sp in (0.40, 0.50, 0.60, 0.70, 0.80)))
