#!/usr/bin/env python3
"""Pivot the ViT broad quant+prune grid into per-(model,bits) method x sparsity tables.
Marks the best MATCHED method per cell and the CoBALT vs strongest-matched delta (2026-08-17)."""
import csv, sys, os
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_camera_vit/broad.csv"
MATCHED = ["cobalt", "sparsegpt", "wanda-awq", "wanda-sinq"]      # orthogonal-lever matched set
REFS    = ["fp16", "awq", "sinq", "wanda"]                          # references (uncounted / diff config)
METHOD_ORDER = REFS + MATCHED

# (model,bits) -> {(method,sp) -> val}, and flats (sp=0)
grids = defaultdict(dict)
flats = defaultdict(dict)
sps_by = defaultdict(set)
for r in csv.DictReader(open(CSV)):
    if r.get("task") != "imagenet":
        continue
    key = (r["model"], r["bits"])
    m, sp = r["method"], r["sparsity"]
    v = (r.get("value") or "").strip()
    val = float(v) if v else None
    if sp == "0.00":
        flats[key][m] = val
    else:
        grids[key][(m, sp)] = val
        sps_by[key].add(sp)

for key in sorted(grids):
    model, bits = key
    sps = sorted(sps_by[key])
    print(f"\n# {model} | {bits}-bit | group-128 | ImageNet top-1 (10k subset)")
    hdr = f"{'method':12s} | " + " | ".join(f" sp{s[2:]:>4s}" for s in sps)
    print(hdr); print("-" * len(hdr))
    for m in METHOD_ORDER:
        if m in flats[key]:
            fv = flats[key][m]
            cell = f"{fv:6.2f}" if fv is not None else "   ·  "
            print(f"{m:12s} | " + " | ".join(f"{cell}" for _ in sps) + "   (flat sp=0)")
        elif any((m, s) in grids[key] for s in sps):
            cells = []
            for s in sps:
                v = grids[key].get((m, s))
                cells.append(f"{v:6.2f}" if v is not None else "   ·  ")
            print(f"{m:12s} | " + " | ".join(cells))
    # best matched per sp + cobalt delta
    print("-" * len(hdr))
    best_line, delta_line = [], []
    for s in sps:
        vals = {m: grids[key].get((m, s)) for m in MATCHED if grids[key].get((m, s)) is not None}
        if not vals:
            best_line.append("  ·  "); delta_line.append("  ·  "); continue
        best_m = max(vals, key=vals.get)
        best_line.append(f"{best_m[:6]:>6s}")
        cob = grids[key].get(("cobalt", s))
        if cob is not None:
            # delta of cobalt vs strongest OTHER matched
            others = {m: v for m, v in vals.items() if m != "cobalt"}
            if others:
                sm = max(others.values())
                d = cob - sm
                delta_line.append(f"{d:+6.2f}")
            else:
                delta_line.append("  ·  ")
        else:
            delta_line.append("  ·  ")
    print(f"{'BEST matched':12s} | " + " | ".join(f"{b:>6s}" for b in best_line))
    print(f"{'CoBALT-Δ':12s} | " + " | ".join(f"{d:>6s}" for d in delta_line)
          + "   (CoBALT minus strongest OTHER matched)")
