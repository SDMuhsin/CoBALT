#!/usr/bin/env python3
"""Assemble the C4 + downstream comparison table from the grid's CSVs.

Reads everything under ``results/c4_downstream/``:
  * main.csv                  -> C4 perplexity per (technique, precision, sparsity)
  * downstream/<task>.csv     -> one metric per (technique, precision, sparsity)

For each config it emits one row with the C4 PPL and the headline metric of
each downstream task:
    hellaswag/arc_easy/arc_challenge/lambada/mmlu/math -> accuracy
    mrr                                                 -> mrr
    humaneval                                           -> pass_at_1

Cell conventions:
  * Missing config / no row in a CSV            -> "NA"
  * Row present but ``error`` is non-empty:
        starts with "SKIPPED"                   -> "SKIP"
        anything else                           -> "FAIL"
  * Accuracies & pass_at_1 shown as percentages, 2 decimals (e.g. 42.13)
  * C4 PPL: 2 decimals; mrr: 4 decimals

Outputs (always re-runnable; partial data -> partial table):
  * results/c4_downstream/comparison_table.md   (GitHub markdown)
  * results/c4_downstream/comparison_table.tsv
and prints the markdown to stdout.

stdlib csv only (pandas is optional and only used opportunistically below).
"""

import csv
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(REPO_ROOT, "results", "c4_downstream")
MAIN_CSV = os.path.join(BASE_DIR, "main.csv")
DOWNSTREAM_DIR = os.path.join(BASE_DIR, "downstream")
OUT_MD = os.path.join(BASE_DIR, "comparison_table.md")
OUT_TSV = os.path.join(BASE_DIR, "comparison_table.tsv")

# ---------------------------------------------------------------------------
# Task -> (csv filename, headline metric column)
# ---------------------------------------------------------------------------
DOWNSTREAM_TASKS = [
    ("hellaswag", "hellaswag.csv", "accuracy"),
    ("arc_easy", "arc_easy.csv", "accuracy"),
    ("arc_challenge", "arc_challenge.csv", "accuracy"),
    ("lambada", "lambada.csv", "accuracy"),
    ("mmlu", "mmlu.csv", "accuracy"),
    # ("math", "math.csv", "accuracy"),  # excluded from this run (full-split MATH ~0 on 0.5B, ~6 days)
    ("mrr", "mrr.csv", "mrr"),
    ("humaneval", "humaneval.csv", "pass_at_1"),
]
# Tasks whose headline metric is a fraction we render as a percentage.
PERCENT_TASKS = {"hellaswag", "arc_easy", "arc_challenge", "lambada", "mmlu",
                 "math", "humaneval"}

COLUMNS = ["technique", "bit", "sparsity", "C4_PPL"] + [t[0] for t in DOWNSTREAM_TASKS]

NA = "NA"


# ---------------------------------------------------------------------------
# Matrix / row order (must mirror the grid). Each entry: (technique, bit, sparsity)
# ---------------------------------------------------------------------------
def matrix_rows():
    rows = []
    rows.append(("fp16", 4, 0.0))
    for prec in (3, 4, 5):
        rows.append(("sinq", prec, 0.0))
    for sp in (0.05, 0.25, 0.50):
        rows.append(("wanda", 4, sp))
    for prec in (3, 4, 5):
        for sp in (0.05, 0.25, 0.50):
            rows.append(("sparsegpt", prec, sp))
    for prec in (3, 4, 5):
        for sp in (0.05, 0.25, 0.50):
            rows.append(("prism", prec, sp))
    # SLiM-LoRA baseline (ICML 2025): joint prune+quant+low-rank. Same grid as PRISM.
    for prec in (3, 4, 5):
        for sp in (0.05, 0.25, 0.50):
            rows.append(("slim", prec, sp))
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _read_csv_rows(path):
    """Yield dict rows from a CSV; empty list if missing/unreadable."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:  # noqa: BLE001 - never crash on a malformed file
        sys.stderr.write("[build_table] WARN reading %s: %s\n" % (path, e))
        return []


def _key(technique, precision, sparsity):
    """Canonical join key: (technique, int precision, float sparsity)."""
    try:
        prec = int(float(precision))
    except (TypeError, ValueError):
        prec = precision
    try:
        sp = float(sparsity)
    except (TypeError, ValueError):
        sp = sparsity
    return (str(technique).strip(), prec, sp)


def _latest_by_timestamp(rows):
    """Collapse rows to one-per-key keeping the LATEST by timestamp string.

    Timestamps from this suite are ISO-ish strings that sort lexicographically;
    we fall back to "last row wins" when a timestamp is missing.
    """
    best = {}
    for i, row in enumerate(rows):
        key = _key(row.get("technique"), row.get("precision"), row.get("sparsity"))
        ts = row.get("timestamp") or ""
        prev = best.get(key)
        if prev is None or (ts, i) >= (prev[0], prev[1]):
            best[key] = (ts, i, row)
    return {k: v[2] for k, v in best.items()}


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------
def _error_cell(err):
    """Map a non-empty error string to SKIP / FAIL; None if no error."""
    if err is None:
        return None
    err = str(err).strip()
    if not err:
        return None
    return "SKIP" if err.upper().startswith("SKIPPED") else "FAIL"


def _fmt_ppl(val):
    try:
        return "%.2f" % float(val)
    except (TypeError, ValueError):
        return NA


def _fmt_pct(val):
    """Fraction (0..1) -> percentage with 2 decimals."""
    try:
        return "%.2f" % (float(val) * 100.0)
    except (TypeError, ValueError):
        return NA


def _fmt_mrr(val):
    try:
        return "%.4f" % float(val)
    except (TypeError, ValueError):
        return NA


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_table():
    # C4 PPL lookup.
    main_rows = _read_csv_rows(MAIN_CSV)
    ppl_by_key = _latest_by_timestamp(main_rows)

    # Downstream lookups: task -> {key -> latest row}.
    ds_by_task = {}
    for task, fname, _metric in DOWNSTREAM_TASKS:
        rows = _read_csv_rows(os.path.join(DOWNSTREAM_DIR, fname))
        ds_by_task[task] = _latest_by_timestamp(rows)

    table = []
    for technique, bit, sparsity in matrix_rows():
        key = _key(technique, bit, sparsity)

        # C4 PPL cell.
        main_row = ppl_by_key.get(key)
        if main_row is None:
            ppl_cell = NA
        else:
            err = _error_cell(main_row.get("error"))
            ppl_cell = err if err is not None else _fmt_ppl(main_row.get("ppl"))

        cells = {
            "technique": technique,
            "bit": str(bit),
            "sparsity": "%.2f" % sparsity,
            "C4_PPL": ppl_cell,
        }

        for task, _fname, metric in DOWNSTREAM_TASKS:
            row = ds_by_task[task].get(key)
            if row is None:
                cells[task] = NA
                continue
            err = _error_cell(row.get("error"))
            if err is not None:
                cells[task] = err
                continue
            raw = row.get(metric)
            if task == "mrr":
                cells[task] = _fmt_mrr(raw)
            elif task in PERCENT_TASKS:
                cells[task] = _fmt_pct(raw)
            else:
                cells[task] = _fmt_pct(raw)

        table.append(cells)
    return table


def render_markdown(table):
    lines = []
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in table:
        lines.append("| " + " | ".join(str(row[c]) for c in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def render_tsv(table):
    out = ["\t".join(COLUMNS)]
    for row in table:
        out.append("\t".join(str(row[c]) for c in COLUMNS))
    return "\n".join(out) + "\n"


def main():
    if not os.path.isdir(BASE_DIR):
        sys.stderr.write(
            "[build_table] no data yet: %s does not exist.\n"
            "[build_table] Run scripts/run_c4_downstream_grid.sh first.\n"
            % BASE_DIR
        )
        # Still emit a full NA table so callers always get a well-formed result.

    table = build_table()
    md = render_markdown(table)
    tsv = render_tsv(table)

    # Best-effort write (directory may not exist yet on a fresh checkout).
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(OUT_MD, "w") as f:
            f.write(md)
        with open(OUT_TSV, "w") as f:
            f.write(tsv)
        sys.stderr.write("[build_table] wrote %s\n" % OUT_MD)
        sys.stderr.write("[build_table] wrote %s\n" % OUT_TSV)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("[build_table] WARN could not write outputs: %s\n" % e)

    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
