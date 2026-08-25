#!/usr/bin/env python3
"""Run the NEW baselines through the DOWNSTREAM-task suite across the Table-1 grid.

Sibling of run_table1_grid.py (which does perplexity). This runs the accuracy /
downstream benchmarks (HellaSwag, ARC-easy, ARC-challenge, MMLU, LAMBADA, MRR) on
the same quantized models, for the new joint prune+quant baselines:

    models      : qwen-0.5b, gemma-2b, llama-7b
    sparsity    : 5%, 25%, 50%
    precision   : 3, 4, 5 bit
    techniques  : wanda-awq, wanda-sinq, jsq-wo, slim      (the NEW baselines)

    => 3 models x 3 sparsities x 3 bits x 4 techniques = 108 configs

WHY NO DATASET AXIS. Quantization always calibrates on wikitext2 (hardcoded in
the runner), and the downstream tasks evaluate on their OWN fixed datasets — so a
config's downstream accuracy is identical whether the PPL side used wikitext2 or
c4. We therefore run each (model, technique, precision, sparsity) ONCE (with the
PPL dataset pinned to wikitext2, whose result we discard via --no-csv) and it
covers both halves of the paper table. That halves the work vs a per-dataset run.

MATH and HumanEval are excluded by default: MATH is ~days of compute and ~0 on
these models, HumanEval is ~0 and needs a code-exec gate. Add them with --tasks.

--------------------------------------------------------------------------------
Same two guarantees as the PPL runner
--------------------------------------------------------------------------------

1. NEVER OOM. Each MIG slice has dedicated memory (this box: 6x 1g.24gb +
   1x 2g.48gb) and runs one config at a time. A per-model memory requirement
   (MEM_REQ_GB) routes each config only to a slice big enough for it — heaviest
   model to the big slice, light models anywhere. Downstream's peak is bounded by
   the already-loaded quantized model (same as / below the PPL peak), and every
   config is verified to fit a 24 GB slice. No MIG -> full GPUs sized by total
   memory; no GPU -> one serial worker.

2. CRASH-RESUMABLE on "result row present". Each config writes its task CSVs into
   its OWN directory results/table1_downstream_grid/parts/<tag>/ (per-config dirs
   avoid GPFS flock races). A config is DONE iff EVERY requested task has a
   recorded row there — a valid metric, or a deterministic error/collapse (a
   broken model is a real result, not to be recomputed). A transient failure
   (OOM/crash) leaves a task un-recorded, so the config is retried. Re-running the
   script re-scans and skips everything already done.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    env/bin/python scripts/run_table1_downstream_grid.py                 # full split
    env/bin/python scripts/run_table1_downstream_grid.py --dry-run
    env/bin/python scripts/run_table1_downstream_grid.py --ds-limit 200  # faster pilot
    env/bin/python scripts/run_table1_downstream_grid.py --only llama
    env/bin/python scripts/run_table1_downstream_grid.py --models gemma-2b
    env/bin/python scripts/run_table1_downstream_grid.py --techniques slim,wanda-awq
    env/bin/python scripts/run_table1_downstream_grid.py --tasks hellaswag,mmlu
    env/bin/python scripts/run_table1_downstream_grid.py --reuse-existing  # count prior
                                                                          #   ds runs

Env overrides (identical to the PPL runner): MIG_EXCLUDE, NSLICES,
PRISM_HF_HOME_DEFAULT, PRISM_HF_HOME_GEMMA, PRISM_SEED.
"""
import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, "env", "bin", "python")
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.join(RESULTS, "table1_downstream_grid")
PARTSDIR = os.path.join(OUTDIR, "parts")
LOGDIR = os.path.join(OUTDIR, "logs")
PROGRESS = os.path.join(OUTDIR, "progress.tsv")

# ----------------------------------------------------------------------------- grid
MODELS = ["qwen-0.5b", "gemma-2b", "llama-7b"]
SPARS_PCT = [5, 25, 50]
PRECS = [3, 4, 5]
TECHS = ["wanda-awq", "wanda-sinq", "jsq-wo", "slim"]  # the NEW baselines only
# The informative, non-degenerate tasks. (math/humaneval excluded by default.)
DEFAULT_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "mmlu", "lambada", "mrr"]
FIXED_DATASET = "wikitext2"  # PPL side is discarded; downstream is dataset-independent

RETRIES = 3

# Per-model peak GPU memory (GiB). Downstream forwards on the already-loaded
# quantized model, so its peak is <= the PPL peak (measured: llama wanda-sinq PPL
# peaked 16.1 GB on a 24 GB slice). Values carry a safety margin; all <= 24.
MEM_REQ_GB = {"qwen-0.5b": 8.0, "gemma-2b": 12.0, "llama-7b": 22.0}
DEFAULT_REQ_GB = 22.0

# a downstream row is "recorded" if it has a headline metric OR a non-transient
# error. These column names are config/meta/counts, never the headline metric.
_NON_METRIC = {"timestamp", "model", "model_name", "technique", "precision",
               "sparsity", "dataset", "limit", "n_samples", "duration_seconds",
               "error", "correct", "total", "passed", "subject_accuracies"}
_TRANSIENT = ("out of memory", "outofmemory", "cuda error", "illegal memory",
              "killed", "device-side assert", "cublas", "cudnn")

_print_lock = threading.Lock()
_pool_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[t1ds] {msg}", flush=True)


def tag_of(model, tech, prec, sp_pct):
    return f"{model}_{tech}_{prec}bit_sp{sp_pct}"


def req_gb(model):
    return MEM_REQ_GB.get(model, DEFAULT_REQ_GB)


# ----------------------------------------------------------------------------- slices
def discover_slices():
    """[(device_id, gb)]: MIG slices, else physical GPUs, else CPU. (== PPL runner)"""
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL).stdout.decode("utf-8", "replace")
    except Exception:
        listing = ""
    excl = set(filter(None, os.environ.get("MIG_EXCLUDE", "").split(",")))
    slices = []
    for line in listing.splitlines():
        m = re.search(r"MIG\s+\d+g\.(\d+)gb.*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)", line)
        if m and m.group(2) not in excl:
            slices.append((m.group(2), float(m.group(1))))
    if not slices:
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total",
                                "--format=csv,noheader,nounits"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                               ).stdout.decode("utf-8", "replace")
            for line in q.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] not in excl:
                    slices.append((parts[0], int(float(parts[1])) / 1024.0))
        except Exception:
            pass
    nslices = os.environ.get("NSLICES")
    if nslices:
        slices = slices[: int(nslices)]
    return slices


# ----------------------------------------------------------------------------- resume
def _isfloat(s):
    try:
        float(s); return True
    except (TypeError, ValueError):
        return False


def _last_row(path):
    row = None
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                row = r
    except (OSError, csv.Error):
        return None
    return row


def _recorded(row):
    """A task row is recorded iff it has a headline metric, or a deterministic
    (non-transient) error/skip. Transient errors (OOM/crash) are NOT recorded."""
    if row is None:
        return False
    for col, val in row.items():
        if col not in _NON_METRIC and _isfloat(val):
            return True
    err = (row.get("error") or "").lower()
    if err and not any(sig in err for sig in _TRANSIENT):
        return True  # deterministic failure / SKIPPED — a real, non-retryable outcome
    return False


def config_done_parts(tag, tasks):
    """Done iff every requested task has a recorded row in this config's part dir."""
    d = os.path.join(PARTSDIR, tag)
    if not os.path.isdir(d):
        return False
    for task in tasks:
        if not _recorded(_last_row(os.path.join(d, task + ".csv"))):
            return False
    return True


def reuse_done_keys(tasks):
    """(model,tech,prec,sp) configs already covered (valid metric for ALL tasks) by
    prior downstream runs elsewhere. Footing may differ — opt-in via --reuse-existing."""
    prior = [os.path.join(RESULTS, "c4_downstream", "downstream"),
             os.path.join(RESULTS, "llama_downstream", "downstream")]
    prior += glob.glob(os.path.join(RESULTS, "ablation_*_ds*", "downstream"))
    have = {}  # (model,tech,prec,sp) -> set(tasks with a valid metric)
    for d in prior:
        for task in tasks:
            path = os.path.join(d, task + ".csv")
            for r in (_rows(path) or []):
                if any(c not in _NON_METRIC and _isfloat(v) for c, v in r.items()):
                    try:
                        k = (r["model"], r["technique"], int(float(r["precision"])),
                             round(float(r["sparsity"]) * 100))
                    except (KeyError, TypeError, ValueError):
                        continue
                    have.setdefault(k, set()).add(task)
    need = set(tasks)
    return {k for k, ts in have.items() if need.issubset(ts)}


def _rows(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None


# ----------------------------------------------------------------------------- run one
def _hf_token():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    p = os.path.join(ROOT, ".hf_token")
    if os.path.exists(p):
        with open(p) as fh:
            return fh.read().strip()
    return None


def task_env(device_id, model):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PIP_CONFIG_FILE"] = "/dev/null"
    if model == "gemma-2b":
        env["HF_HOME"] = os.environ.get("PRISM_HF_HOME_GEMMA") or "/scratch/ckp908/prism_hf"
    else:
        env["HF_HOME"] = (os.environ.get("PRISM_HF_HOME_DEFAULT")
                          or os.path.join(ROOT, "cache", "huggingface"))
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    tok = _hf_token()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env["PRISM_SEED"] = os.environ.get("PRISM_SEED", "0")
    env["PRISM_LOWMEM"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if device_id != "":
        env["CUDA_VISIBLE_DEVICES"] = device_id
    return env


def run_one(device_id, task, tasks, ds_limit, gen_limit, allow_codeexec):
    """Run one config's full downstream suite as a subprocess. Return (ok, secs, status)."""
    model, tech, prec, sp = task
    tag = tag_of(*task)
    partdir = os.path.join(PARTSDIR, tag)
    os.makedirs(partdir, exist_ok=True)
    logpath = os.path.join(LOGDIR, tag + ".log")
    cmd = [PY, "benchmarks/benchmark_suite.py",
           "--model", model, "--technique", tech,
           "--precision", str(prec), "--sparsity", str(sp / 100.0),
           "--dataset", FIXED_DATASET, "--no-csv",
           "--downstream", "--downstream-tasks", ",".join(tasks),
           "--downstream-csv-dir", partdir]
    if ds_limit is not None:
        cmd += ["--downstream-limit", str(ds_limit)]
    if gen_limit is not None:
        cmd += ["--downstream-gen-limit", str(gen_limit)]
    if allow_codeexec:
        cmd += ["--downstream-allow-codeexec"]
    t0 = time.time()
    with open(logpath, "w") as lf:
        rc = subprocess.run(cmd, cwd=ROOT, env=task_env(device_id, model),
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    ok = config_done_parts(tag, tasks)  # success == every task recorded
    status = "ok" if ok else f"rc={rc},incomplete"
    return ok, dt, status


# ----------------------------------------------------------------------------- schedule
def worker(device_id, gb, pool, counters, cfg):
    while True:
        task = None
        with _pool_lock:
            best_i, best_req = -1, -1.0
            for i, (t, att) in enumerate(pool):
                r = req_gb(t[0])
                if r <= gb and r > best_req:
                    best_req, best_i = r, i
            if best_i < 0:
                if not pool:
                    return
                task = None
            else:
                task, att = pool.pop(best_i)
        if task is None:
            time.sleep(2)
            with _pool_lock:
                if not any(req_gb(t[0]) <= gb for t, _ in pool):
                    return
            continue

        tag = tag_of(*task)
        ok, dt, status = run_one(device_id, task, cfg["tasks"], cfg["ds_limit"],
                                 cfg["gen_limit"], cfg["allow_codeexec"])
        with _print_lock:
            with open(PROGRESS, "a") as pf:
                pf.write(f"{tag}\t{int(dt)}\t{status}\t{device_id[:16]}\tatt{att}\n")
        if ok:
            with _pool_lock:
                counters["done"] += 1
                n = counters["done"]
            log(f"OK    {tag}  ({int(dt)}s)  slice={device_id[:12]}  [{n}/{counters['total']}]")
        elif att < RETRIES:
            log(f"RETRY {tag}  ({status}, {int(dt)}s) -> attempt {att + 1}")
            time.sleep(3)
            with _pool_lock:
                pool.append((task, att + 1))
        else:
            with _pool_lock:
                counters["failed"] += 1
            log(f"FAIL  {tag}  ({status})  gave up after {RETRIES} retries")


def build_tasks(models, techs, spars, precs, tasklist, reuse_keys):
    out = []
    for model in models:
        for tech in techs:
            for sp in spars:
                for prec in precs:
                    tag = tag_of(model, tech, prec, sp)
                    if config_done_parts(tag, tasklist):
                        continue
                    if (model, tech, prec, sp) in reuse_keys:
                        continue
                    out.append((model, tech, prec, sp))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="keep only tasks whose tag contains this")
    ap.add_argument("--exclude", default=None, help="drop tasks whose tag contains this")
    ap.add_argument("--limit", type=int, default=None, help="run at most N configs (smoke)")
    ap.add_argument("--models", default=None, help="comma list override")
    ap.add_argument("--techniques", default=None, help="comma list override")
    ap.add_argument("--tasks", default=None,
                    help=f"comma list of downstream tasks (default: {','.join(DEFAULT_TASKS)})")
    ap.add_argument("--ds-limit", type=int, default=None,
                    help="per-task sample cap (--downstream-limit); omit for FULL split")
    ap.add_argument("--gen-limit", type=int, default=None, help="generative-task cap")
    ap.add_argument("--allow-codeexec", action="store_true",
                    help="enable HumanEval code execution (only if you add humaneval to --tasks)")
    ap.add_argument("--retries", type=int, default=None)
    ap.add_argument("--reuse-existing", action="store_true",
                    help="skip configs already covered by prior downstream runs (footing may differ)")
    args = ap.parse_args()

    global RETRIES
    if args.retries is not None:
        RETRIES = args.retries

    models = args.models.split(",") if args.models else MODELS
    techs = args.techniques.split(",") if args.techniques else TECHS
    tasklist = args.tasks.split(",") if args.tasks else DEFAULT_TASKS

    os.makedirs(PARTSDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    if not os.path.exists(PROGRESS):
        with open(PROGRESS, "w") as f:
            f.write("tag\tseconds\tstatus\tslice\tattempt\n")

    reuse_keys = reuse_done_keys(tasklist) if args.reuse_existing else set()
    # all not-done configs across the full grid (before display filters) — so the
    # "already done" count is accurate regardless of --only/--exclude/--limit.
    all_todo = build_tasks(models, techs, SPARS_PCT, PRECS, tasklist, reuse_keys)
    tasks = list(all_todo)
    if args.only:
        tasks = [t for t in tasks if args.only in tag_of(*t)]
    if args.exclude:
        tasks = [t for t in tasks if args.exclude not in tag_of(*t)]
    if args.limit:
        tasks = tasks[: args.limit]

    slices = discover_slices()
    total_cells = len(models) * len(techs) * len(SPARS_PCT) * len(PRECS)
    log(f"grid: {len(models)}m x {len(techs)}t x {len(SPARS_PCT)}sp x {len(PRECS)}bit "
        f"= {total_cells} configs   tasks/config: {len(tasklist)} ({','.join(tasklist)})")
    log(f"downstream sample cap: {'FULL split' if args.ds_limit is None else args.ds_limit}")
    log(f"already done (all task rows present): {total_cells - len(all_todo)}"
        + (f"  (incl {len(reuse_keys)} reused)" if reuse_keys else ""))
    log("slices: " + (", ".join(f"{d[:12]}:{int(g)}GB" for d, g in slices) if slices else "none (CPU)"))
    log(f"configs to run now: {len(tasks)}")
    from collections import Counter
    for (m, tc), n in sorted(Counter((t[0], t[1]) for t in tasks).items()):
        log(f"    {m:10s} {tc:11s} : {n}")

    if slices:
        max_gb = max(g for _, g in slices)
        toobig = sorted({t[0] for t in tasks if req_gb(t[0]) > max_gb})
        if toobig:
            log(f"ERROR: no slice big enough (max {int(max_gb)}GB) for: {toobig}")
            return 2

    if args.dry_run:
        log("dry-run — exiting")
        return 0
    if not tasks:
        log("nothing to do — all requested configs already have downstream rows")
        return 0
    if not slices:
        slices = [("", DEFAULT_REQ_GB)]

    cfg = {"tasks": tasklist, "ds_limit": args.ds_limit,
           "gen_limit": args.gen_limit, "allow_codeexec": args.allow_codeexec}
    pool = [(t, 0) for t in tasks]
    counters = {"done": 0, "failed": 0, "total": len(tasks)}
    threads = []
    for device_id, gb in slices:
        th = threading.Thread(target=worker, args=(device_id, gb, pool, counters, cfg), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    log(f"ALL DONE  done={counters['done']}  failed={counters['failed']}  total={counters['total']}")
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
