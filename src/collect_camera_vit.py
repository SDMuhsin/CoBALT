#!/usr/bin/env python3
"""Pivot the ViT camera-ready CSV into a method × sparsity ImageNet-top1 table (2026-08-16)."""
import csv, sys, os

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "benchmark_camera_vit", "results.csv")
METHOD_ORDER = ["fp16", "awq", "sinq", "wanda", "sparsegpt", "jsq-wo", "slim",
                "wanda-awq", "wanda-sinq", "cobalt"]
SPS = ["0.50", "0.60", "0.70", "0.80", "0.90"]

cells = {}   # (method, sparsity) -> value or 'ERR'
flat = {}     # method -> value (non-prune, sparsity 0)
for r in csv.DictReader(open(CSV)):
    if r.get("task") != "imagenet":
        continue
    m, sp = r["method"], r["sparsity"]
    v = (r.get("value") or "").strip()
    val = f"{float(v):.2f}" if v else ("ERR" if (r.get("error") or "").strip() else "")
    if sp == "0.00":
        flat[m] = val
    else:
        cells[(m, sp)] = val

print(f"# ViT-large ImageNet-1k top-1 | 3-bit | group-128 | {CSV}")
hdr = f"{'method':12s} | " + " | ".join(f"sp{s[2:]}" for s in SPS)
print(hdr); print("-" * len(hdr))
for m in METHOD_ORDER:
    if m in flat:                       # non-prune: flat across sparsity
        print(f"{m:12s} | " + " | ".join(f"{flat[m]:>5s}" for _ in SPS) + "   (flat, sp=0)")
    else:
        row = [cells.get((m, s), "·") for s in SPS]
        print(f"{m:12s} | " + " | ".join(f"{c:>5s}" for c in row))
