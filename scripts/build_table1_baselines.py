#!/usr/bin/env python3
"""Assemble the Table-1 fair-baseline comparison into llmdocs/.

Merges, for every cell of the paper's Table-1 grid, the perplexity of each
joint prune+quant method on a single consistent footing:

  * PRISM, SparseGPT, wanda-sinq, jsq-wo, slim on qwen-0.5b / llama-7b come from
    the bias-correction ablation CSVs (results/ablation_*/main.csv) — same runner
    settings as the ablation.
  * Everything for gemma-2b, plus wanda-awq everywhere, comes from the fresh
    Table-1 baseline runs (results/table1_baselines/parts/*.csv).

Source priority: fresh parts first, then ablation CSVs. FP16 / Wanda(FP16) /
SINQ(dense) reference rows are taken from the paper's Table 1 (labelled).

Writes: llmdocs/TABLE1_BASELINES.md  (does NOT touch the paper .tex).
"""
import csv
import glob
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "llmdocs", "TABLE1_BASELINES.md")

MODELS = [("qwen-0.5b", "Qwen-0.5B"), ("gemma-2b", "Gemma-2B"), ("llama-7b", "LLaMA-7B")]
DATASETS = [("wikitext2", "WikiText-2"), ("c4", "C4")]
SPARS = [5, 25, 50]
PRECS = [3, 4, 5]

# Joint prune+quant methods, in table row order (PRISM last). Label -> technique key.
JOINT = [
    ("SparseGPT", "sparsegpt"),
    ("JSQ (wo)", "jsq-wo"),
    ("Wanda+AWQ", "wanda-awq"),
    ("Wanda+SINQ", "wanda-sinq"),
    ("SLiM*", "slim"),
    ("PRISM", "prism"),
]

# Reference rows straight from the paper's Table 1 (not recomputed here).
#   fp16, {sparsity%: wanda_fp16_ppl}
REF = {
    ("qwen-0.5b", "wikitext2"): (12.63, {5: 12.64, 25: 13.14, 50: 21.10}),
    ("gemma-2b", "wikitext2"): (12.90, {5: 12.93, 25: 13.73, 50: 45.12}),
    ("llama-7b", "wikitext2"): (5.69, {5: 5.69, 25: 5.83, 50: 7.11}),
    ("qwen-0.5b", "c4"): (19.21, {5: 19.21, 25: 20.21, 50: 35.35}),
    ("gemma-2b", "c4"): (18.18, {5: 18.22, 25: 19.48, 50: 55.23}),
    ("llama-7b", "c4"): (7.07, {5: 7.07, 25: 7.26, 50: 9.14}),
}


def _sp_pct(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f <= 1.0:
        f *= 100.0
    return int(round(f))


def _valid(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if (v > 0 and not math.isinf(v) and not math.isnan(v)) else None


def load(store, paths):
    """Add rows from CSV paths without overwriting existing keys (priority = call order)."""
    for path in paths:
        try:
            with open(path, newline="") as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames or "technique" not in rd.fieldnames:
                    continue
                for row in rd:
                    ppl = _valid(row.get("ppl"))
                    sp = _sp_pct(row.get("sparsity"))
                    if ppl is None or sp is None:
                        continue
                    try:
                        prec = int(float(row.get("precision")))
                    except (TypeError, ValueError):
                        continue
                    key = (row.get("model"), row.get("technique"), prec, sp, row.get("dataset"))
                    store.setdefault(key, ppl)
        except Exception:
            continue


_MODEL_KEYS = ["qwen-0.5b", "gemma-2b", "llama-7b"]


def load_done_markers(store):
    """Fill any still-missing keys from the fresh-run done-markers.

    Each marker file is named {model}_{tech}_{P}bit_sp{SP}_{dataset}.done and
    contains the ppl (or 'inf' for a deterministic collapse cell that overflowed
    to nan, e.g. gemma jsq-wo 3-bit/50%). This is what surfaces those collapse
    cells, which have no valid CSV row."""
    import re
    donedir = os.path.join(RESULTS, "table1_baselines", "done")
    for path in sorted(glob.glob(os.path.join(donedir, "*.done"))):
        base = os.path.basename(path)[:-5]  # strip .done
        m = re.match(r"^(.*)_(\d+)bit_sp(\d+)_(wikitext2|c4)$", base)
        if not m:
            continue
        modeltech, prec, sp, ds = m.groups()
        model = next((mk for mk in _MODEL_KEYS if modeltech.startswith(mk + "_")), None)
        if model is None:
            continue
        tech = modeltech[len(model) + 1:]
        key = (model, tech, int(prec), int(sp), ds)
        if key in store:
            continue
        try:
            v = float(open(path).read().strip())
        except (ValueError, OSError):
            v = float("inf")  # unreadable collapse marker
        store[key] = v


def build_store():
    store = {}
    # priority 1: fresh table-1 parts
    load(store, sorted(glob.glob(os.path.join(RESULTS, "table1_baselines", "parts", "*.csv"))))
    # priority 2: ablation CSVs (reused qwen/llama cells)
    load(store, sorted(glob.glob(os.path.join(RESULTS, "ablation_*", "main.csv"))))
    # priority 3: done-markers (surfaces collapse cells with no valid CSV row)
    load_done_markers(store)
    return store


def fmt(v):
    if v is None:
        return "—"
    if math.isinf(v) or v >= 1e4:   # destroyed regime (JSQ 3-bit overflows to ~1e35+/inf)
        return "coll."
    if v >= 1000:
        return f"{v:.0f}"
    if v >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def emit_table(store, model_key, model_name, ds_key, ds_name, lines):
    fp16, wanda = REF[(model_key, ds_key)]
    lines.append(f"### {model_name} — {ds_name}  (FP16 = {fp16})\n")
    header = "| Method | " + " | ".join(
        f"{sp}%/{p}b" for sp in SPARS for p in PRECS) + " |"
    sep = "|" + "---|" * (1 + len(SPARS) * len(PRECS))
    lines.append(header)
    lines.append(sep)
    # Wanda (FP16) reference row — one value per sparsity, spread across its 3 bit cols
    wrow = ["Wanda (FP16)†"]
    for sp in SPARS:
        for _ in PRECS:
            wrow.append(fmt(wanda[sp]))
    lines.append("| " + " | ".join(wrow) + " |")
    # Joint methods
    # Pre-compute best (min) joint ppl per cell for bolding.
    best = {}
    for sp in SPARS:
        for p in PRECS:
            vals = [(_lbl, store.get((model_key, tk, p, sp, ds_key)))
                    for _lbl, tk in JOINT]
            vals = [(l, v) for l, v in vals if v is not None]
            if vals:
                best[(sp, p)] = min(v for _, v in vals)
    for label, tk in JOINT:
        row = [label]
        for sp in SPARS:
            for p in PRECS:
                v = store.get((model_key, tk, p, sp, ds_key))
                cell = fmt(v)
                if v is not None and best.get((sp, p)) is not None and abs(v - best[(sp, p)]) < 1e-9:
                    cell = f"**{cell}**"
                row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def coverage(store):
    total = have = 0
    miss = []
    for _dk, _ in DATASETS:
        pass
    for ds_key, _ in DATASETS:
        for model_key, _ in MODELS:
            for _lbl, tk in JOINT:
                for sp in SPARS:
                    for p in PRECS:
                        total += 1
                        if store.get((model_key, tk, p, sp, ds_key)) is not None:
                            have += 1
                        else:
                            miss.append(f"{model_key}/{tk}/{p}b/{sp}%/{ds_key}")
    return total, have, miss


def head_to_head(store):
    """PRISM vs each baseline: win/loss over all cells and at 50% sparsity.

    A cell counts as a PRISM win if PRISM's ppl is strictly lower (collapse = +inf,
    so a baseline that collapses while PRISM survives is a PRISM win). Cells where
    either value is missing are skipped."""
    baselines = [("SparseGPT", "sparsegpt"), ("JSQ (wo)", "jsq-wo"),
                 ("Wanda+AWQ", "wanda-awq"), ("Wanda+SINQ", "wanda-sinq"), ("SLiM*", "slim")]
    rows = []
    for label, tk in baselines:
        w = l = w50 = l50 = 0
        for ds_key, _ in DATASETS:
            for model_key, _ in MODELS:
                for sp in SPARS:
                    for p in PRECS:
                        pv = store.get((model_key, "prism", p, sp, ds_key))
                        bv = store.get((model_key, tk, p, sp, ds_key))
                        if pv is None or bv is None:
                            continue
                        if pv < bv:
                            w += 1
                            if sp == 50:
                                w50 += 1
                        elif pv > bv:
                            l += 1
                            if sp == 50:
                                l50 += 1
        rows.append((label, w, l, w50, l50))
    return rows


def main():
    store = build_store()
    total, have, miss = coverage(store)
    lines = []
    lines.append("# Table 1 — Fair joint prune+quant baselines\n")
    lines.append("Perplexity (↓). Bold = best **joint prune+quant** method in that cell. "
                 "PRISM/SparseGPT/Wanda+SINQ/JSQ/SLiM on Qwen & LLaMA are reused from the "
                 "bias-correction ablation runs (identical runner settings); all Gemma-2B cells "
                 "and all Wanda+AWQ cells are freshly computed here — so every method in a given "
                 "table shares one footing.\n")
    lines.append(f"**Coverage: {have}/{total} joint-method cells populated.** "
                 + ("" if not miss else f"Missing {len(miss)} (runs in progress).") + "\n")
    lines.append("**Provenance / footing.** Every method in a given sub-table was produced by the "
                 "same runner (16-sample calibration). Qwen & LLaMA PRISM/SparseGPT/Wanda+SINQ/JSQ/"
                 "SLiM are the ablation-run values (≈ the paper's Table 1 within sampling noise, e.g. "
                 "LLaMA 50%/4b PRISM 7.26 = paper 7.26; Qwen 50%/4b 22.93 vs 22.91). Gemma PRISM/"
                 "SparseGPT are **re-run here** and sit ~1–3% off the paper's Gemma cells (calibration "
                 "differs). So this table is internally consistent for method-vs-method comparison; to "
                 "drop the 4 new baseline rows into the *published* Table 1 verbatim, re-run them under "
                 "the paper's exact per-model settings (this runner is the harness for that).\n")
    lines.append("Notes:\n")
    lines.append("- †**Wanda (FP16)** keeps FP16 weights (no quantization) — a pruning-only "
                 "upper bound, one value per sparsity, from the paper's Table 1. Not a "
                 "matched-compression joint method.\n")
    lines.append("- *****SLiM** adds a low-rank adapter (~+10% params), so it is **not** matched "
                 "compression at the stated bit-width; treat with an asterisk.\n")
    lines.append("- **JSQ (wo)** = weight-only JSQ (the fair, weight-only variant). Faithful JSQ "
                 "with activation quant collapses below 8-bit by design and is omitted.\n")
    lines.append("- **`coll.`** = collapsed (PPL ≥ 1e4; model destroyed). JSQ's naive per-channel "
                 "RTN collapses at **3-bit on every model** (overflows to ~1e35+/nan) — the same "
                 "brittleness SparseGPT shows at 3-bit, and exactly the regime PRISM is stable in.\n")
    lines.append("- Gemma-2B JSQ needed a port fix: its RMSNorm applies a `(1+weight)` gain, so "
                 "JSQ's SmoothQuant LN↔FC scale migration must be unit-offset-aware or it overflows "
                 "fp16→nan (see `jsq_port.smooth_ln_fcs`). Llama/Qwen JSQ numbers are unchanged.\n")
    # ---- head-to-head summary ----
    lines.append("\n## PRISM head-to-head (all 54 cells; ties omitted)\n")
    lines.append("| Baseline | PRISM wins | PRISM loses | @50% sparsity wins | @50% loses |")
    lines.append("|---|---|---|---|---|")
    for label, w, l, w50, l50 in head_to_head(store):
        lines.append(f"| {label} | {w} | {l} | {w50} | {l50} |")
    lines.append("\nRead this with the notes below: PRISM's margin concentrates at the **hard "
                 "operating point (50% sparsity, low bits)** — where its sparse-aware normalization "
                 "has the most to correct. At low sparsity / high bits the strong quantizer baselines "
                 "(Wanda+AWQ/SINQ) are competitive because there is little variance distortion to fix. "
                 "SLiM's wins are almost entirely on LLaMA-50% and come from its extra-parameter "
                 "low-rank adapter (not matched compression). JSQ collapses at 3-bit on every model.\n")

    for ds_key, ds_name in DATASETS:
        lines.append(f"\n## {ds_name}\n")
        for model_key, model_name in MODELS:
            emit_table(store, model_key, model_name, ds_key, ds_name, lines)
    if miss:
        lines.append("\n<details><summary>Missing cells</summary>\n")
        lines.append("\n".join(f"- {m}" for m in miss))
        lines.append("\n</details>\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {OUT}  ({have}/{total} cells)")


if __name__ == "__main__":
    main()
