"""Downstream-task suite orchestration.

``run_downstream_suite`` runs the requested task adapters on the SAME in-memory
quantized model (right after PPL eval), writes one CSV per benchmark keyed by the
run config, and returns the list of row dicts it wrote (for logging).

Orchestration rules (see DOWNSTREAM_SPEC.md):
- "all" -> ALL_TASKS; otherwise validate the subset against the registry and
  drop unknowns with a warning.
- ARC: a single ``eval_arc.run`` call yields BOTH arc_easy and arc_challenge
  rows (the result dict is split).
- Per task: try/except so one failure never aborts the suite. Time it, call the
  adapter, write the per-task CSV row(s) immediately (partial progress survives).
- humaneval without ``allow_codeexec`` -> SKIP: write a row with
  error="SKIPPED:codeexec-disabled" and blank metrics; no generation.
- On exception: write a row with error="FAILED:<Type>: <msg>" and blank metrics.
- After each task, defensively ``model.to(device)`` (in case CRB code moved it).
"""

import time
import warnings
from pathlib import Path

from .registry import TASK_REGISTRY, ALL_TASKS, CONFIG_COLUMNS, GENERATIVE_TASKS
from .csv_writer import save_task_row


def _effective_limit(task_key, limit, gen_limit):
    """Resolve the per-task sample cap actually passed to an adapter.

    Generative tasks (math/humaneval) use ``gen_limit`` when it is set;
    everything else (and generative tasks when ``gen_limit`` is None) uses
    ``limit``.
    """
    if task_key in GENERATIVE_TASKS and gen_limit is not None:
        return gen_limit
    return limit


def _default_results_dir():
    """<repo>/results/downstream (repo root is two levels above benchmarks/)."""
    # this file: <repo>/benchmarks/downstream/suite.py
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "results" / "downstream"


def _resolve_tasks(tasks):
    """Return the ordered list of concrete task keys to run.

    "all" -> ALL_TASKS. A list is validated against the registry; unknown
    entries are dropped with a warning. ``arc`` is accepted as an alias and
    expanded to [arc_easy, arc_challenge].
    """
    if tasks == "all" or tasks is None:
        return list(ALL_TASKS)

    resolved = []
    for t in tasks:
        t = t.strip()
        if t == "arc":
            for sub in TASK_REGISTRY["arc"]["expands_to"]:
                if sub not in resolved:
                    resolved.append(sub)
            continue
        if t not in TASK_REGISTRY or t == "arc":
            warnings.warn(f"[downstream] unknown task '{t}' — skipping")
            continue
        if TASK_REGISTRY[t].get("csv") is None:
            # Non-CSV umbrella key passed directly; skip.
            warnings.warn(f"[downstream] task '{t}' is not directly runnable — skipping")
            continue
        if t not in resolved:
            resolved.append(t)
    return resolved


def _base_row(config, limit):
    """Build a row pre-filled with config + run columns (blank metrics)."""
    config = config or {}
    row = {}
    for col in CONFIG_COLUMNS:
        row[col] = config.get(col, "")
    row["limit"] = "" if limit is None else limit
    row["n_samples"] = ""
    row["duration_seconds"] = ""
    row["error"] = ""
    return row


def run_downstream_suite(model, tokenizer, device, *,
                         tasks="all",
                         limit=None,
                         gen_limit=None,
                         allow_codeexec=False,
                         config=None,
                         results_dir=None,
                         seqlen=2048,
                         verbose=False):
    """Run downstream benchmarks on an in-memory model; write per-task CSVs.

    Returns a list of the row dicts written (one per CSV row).
    """
    results_dir = Path(results_dir) if results_dir is not None else _default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_tasks(tasks)

    # Determine which CSV-level tasks to produce, and whether ARC needs running.
    # ARC is run ONCE if either arc_easy or arc_challenge is requested.
    arc_subtasks = [t for t in resolved if t in ("arc_easy", "arc_challenge")]
    run_arc = len(arc_subtasks) > 0

    rows = []

    # Iterate in canonical order, but handle ARC specially (one adapter call).
    arc_done = False
    for task_key in resolved:
        if task_key in ("arc_easy", "arc_challenge"):
            if arc_done:
                continue
            arc_done = True
            rows.extend(_run_arc(model, tokenizer, device, arc_subtasks,
                                 limit, gen_limit, config, results_dir, seqlen, verbose))
            # Defensive: restore model device after the (combined) ARC task.
            model.to(device)
            continue

        row = _run_one(model, tokenizer, device, task_key,
                       limit, gen_limit, allow_codeexec, config, results_dir, seqlen, verbose)
        rows.append(row)
        # Defensive: some CRB code paths could move the model; restore.
        model.to(device)

    return rows


def _run_one(model, tokenizer, device, task_key,
             limit, gen_limit, allow_codeexec, config, results_dir, seqlen, verbose):
    """Run a single (non-ARC) task, write its CSV row, return the row dict."""
    spec = TASK_REGISTRY[task_key]
    csv_path = results_dir / spec["csv"]
    eff_limit = _effective_limit(task_key, limit, gen_limit)
    row = _base_row(config, eff_limit)

    # HumanEval gating: skip generation entirely when code exec disabled.
    if task_key == "humaneval" and not allow_codeexec:
        warnings.warn("[downstream] humaneval skipped: code execution disabled "
                      "(pass allow_codeexec=True to enable)")
        row["error"] = "SKIPPED:codeexec-disabled"
        save_task_row(task_key, row, csv_path)
        return row

    t0 = time.time()
    try:
        metrics = spec["adapter"](model, tokenizer, device,
                                  limit=eff_limit, seqlen=seqlen, verbose=verbose)
        duration = time.time() - t0
        row["duration_seconds"] = duration
        for col in spec["metric_columns"]:
            row[col] = metrics.get(col)
        row["n_samples"] = metrics.get("total", "")
    except Exception as e:  # noqa: BLE001 — one task must not abort the suite
        duration = time.time() - t0
        row["duration_seconds"] = duration
        row["error"] = f"FAILED:{type(e).__name__}: {e}"
        if verbose:
            print(f"[downstream] task '{task_key}' failed: {row['error']}")

    save_task_row(task_key, row, csv_path)
    return row


def _run_arc(model, tokenizer, device, arc_subtasks,
             limit, gen_limit, config, results_dir, seqlen, verbose):
    """Run eval_arc ONCE; split result into arc_easy / arc_challenge CSV rows."""
    spec = TASK_REGISTRY["arc"]
    out_rows = []

    # ARC is multiple-choice (never generative), so the effective limit is just
    # ``limit``; resolved via the helper for consistency.
    eff_limit = _effective_limit("arc", limit, gen_limit)

    t0 = time.time()
    metrics = None
    error = None
    try:
        metrics = spec["adapter"](model, tokenizer, device,
                                  limit=eff_limit, seqlen=seqlen, verbose=verbose)
    except Exception as e:  # noqa: BLE001
        error = f"FAILED:{type(e).__name__}: {e}"
        if verbose:
            print(f"[downstream] task 'arc' failed: {error}")
    duration = time.time() - t0

    for sub in arc_subtasks:
        sub_spec = TASK_REGISTRY[sub]
        csv_path = results_dir / sub_spec["csv"]
        row = _base_row(config, eff_limit)
        row["duration_seconds"] = duration

        if error is not None:
            row["error"] = error
        else:
            result_key = sub_spec["arc_result_key"]
            sub_metrics = metrics.get(result_key, {}) if metrics else {}
            for col in sub_spec["metric_columns"]:
                row[col] = sub_metrics.get(col)
            row["n_samples"] = sub_metrics.get("total", "")

        save_task_row(sub, row, csv_path)
        out_rows.append(row)

    return out_rows
