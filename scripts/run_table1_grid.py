#!/usr/bin/env python3
"""Run the NEW joint prune+quant baselines across the full Table-1 grid.

The paper's Table 1 (`llmdocs/paper/v1/results_and_discussion.tex`) currently
reports only Wanda(FP16), SparseGPT and PRISM. This script fills in the four
*new* fair joint prune+quant baselines across the entire Table-1 grid so those
rows can be dropped in later:

    models      : qwen-0.5b, gemma-2b, llama-7b
    datasets    : wikitext2, c4
    sparsity    : 5%, 25%, 50%
    precision   : 3, 4, 5 bit
    techniques  : wanda-awq, wanda-sinq, jsq-wo, slim      (the NEW baselines)

    => 3 models x 2 datasets x 3 sparsities x 3 bits x 4 techniques = 216 configs

It deliberately does NOT run prism / sparsegpt / wanda(fp16): those are already
in the paper draft. It also does NOT touch the paper .tex — it only writes result
CSVs.

--------------------------------------------------------------------------------
Two guarantees this script is built around
--------------------------------------------------------------------------------

1. NEVER OOM.
   The GPUs are MIG-sliced; every slice has *dedicated* memory (this box: 6x
   1g.24gb + 1x 2g.48gb). Each MIG slice runs at most ONE config at a time (one
   worker thread per slice, one subprocess at a time), so the only way to OOM is
   to place a config on a slice too small for it. We prevent that with a static
   per-model memory requirement (MEM_REQ_GB): a worker only ever dequeues a
   config whose requirement fits its slice. Heaviest-fitting config first, so the
   big 48 GB slice soaks up the heavy work and the 24 GB slices take the light
   work. Requirements were set from measured peak usage (see MEM_REQ_GB note) with
   a safety margin, and every config is designed to fit a 24 GB slice.
   If the box is NOT MIG-sliced, each physical GPU is treated as one slice sized
   by its total memory; if there is no GPU at all, a single default-device worker
   runs everything serially.

2. CRASH-RESUMABLE with zero bookkeeping.
   The resume signal is simply: *is the result row already present?* Each config
   writes to its own part-CSV `results/table1_grid/parts/<tag>.csv` (per-config
   files avoid GPFS flock races). A config is considered DONE iff some scanned CSV
   already has a row for it with a parseable perplexity (a finite number, or a
   deterministic nan/inf collapse — those are real results we must not recompute).
   Re-running the script re-scans and skips everything already done, so you can
   kill it / let the box crash / rerun the exact same command as many times as
   you like and it continues from where it stopped. Nothing else (no done-marker
   files, no state db) is needed.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    env/bin/python scripts/run_table1_grid.py                 # run everything left
    env/bin/python scripts/run_table1_grid.py --dry-run       # print the plan only
    env/bin/python scripts/run_table1_grid.py --only llama    # tag-substring filter
    env/bin/python scripts/run_table1_grid.py --exclude c4    # drop matching tags
    env/bin/python scripts/run_table1_grid.py --limit 2       # run at most N (smoke)
    env/bin/python scripts/run_table1_grid.py --models gemma-2b --techniques slim
    env/bin/python scripts/run_table1_grid.py --reuse-existing # also count prior
                                                              #   CSVs as done

Environment overrides:
    MIG_EXCLUDE=MIG-aaa,MIG-bbb   skip specific (e.g. busy / foreign-occupied) slices
    NSLICES=3                     use only the first N discovered slices
    PRISM_HF_HOME_DEFAULT=...     HF cache for qwen/llama (default cache/huggingface)
    PRISM_HF_HOME_GEMMA=...       HF cache for gemma-2b   (default /scratch/ckp908/prism_hf)
    PRISM_SEED=0                  seed passed through to the runner
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
OUTDIR = os.path.join(RESULTS, "table1_grid")
PARTSDIR = os.path.join(OUTDIR, "parts")
LOGDIR = os.path.join(OUTDIR, "logs")
PROGRESS = os.path.join(OUTDIR, "progress.tsv")

# ----------------------------------------------------------------------------- grid
MODELS = ["qwen-0.5b", "gemma-2b", "llama-7b"]
DATASETS = ["wikitext2", "c4"]
SPARS_PCT = [5, 25, 50]
PRECS = [3, 4, 5]
TECHS = ["wanda-awq", "wanda-sinq", "jsq-wo", "slim"]  # the NEW baselines only

RETRIES = 3  # transient glitches (box clock jumps, rc=126) get a few retries

# ----------------------------------------------------------------------------- memory
# Peak GPU memory (GiB) a config needs, keyed by model. The heaviest technique is
# wanda-sinq (full-matrix Sinkhorn on the largest weight matrix, then dense fp16
# eval); the numbers below are its measured peak + a safety margin, and bound the
# other three techniques too. Every value is <= 24 so every config fits a 1g.24gb
# slice; the map is the single knob for the "never OOM" guarantee. See the smoke
# test in the handoff notes for how llama-7b was verified against 24 GB.
MEM_REQ_GB = {
    "qwen-0.5b": 8.0,
    "gemma-2b": 12.0,
    "llama-7b": 22.0,
}
DEFAULT_REQ_GB = 22.0  # unknown model -> assume it needs a full 24 GB slice

_print_lock = threading.Lock()
_pool_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[t1grid] {msg}", flush=True)


def tag_of(model, tech, prec, sp_pct, dataset):
    return f"{model}_{tech}_{prec}bit_sp{sp_pct}_{dataset}"


def req_gb(model):
    return MEM_REQ_GB.get(model, DEFAULT_REQ_GB)


# ----------------------------------------------------------------------------- slices
def discover_slices():
    """Return [(device_id, gb), ...]: MIG slices, else physical GPUs, else CPU.

    device_id is what we put in CUDA_VISIBLE_DEVICES (a MIG-UUID or a GPU index).
    gb is the slice's dedicated memory in GiB, used for the no-OOM routing.
    """
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL).stdout.decode("utf-8", "replace")
    except Exception:
        listing = ""

    excl = set(filter(None, os.environ.get("MIG_EXCLUDE", "").split(",")))

    # Preferred: MIG slices. Lines look like:
    #   MIG 1g.24gb     Device  0: (UUID: MIG-bef7a31e-...)
    slices = []
    for line in listing.splitlines():
        m = re.search(r"MIG\s+\d+g\.(\d+)gb.*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)", line)
        if m:
            uuid = m.group(2)
            if uuid not in excl:
                slices.append((uuid, float(m.group(1))))

    # Fallback: no MIG -> one worker per physical GPU, sized by total memory.
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
def _valid_ppl(s):
    """True if s parses to a real result (finite, or a deterministic nan/inf collapse)."""
    if s is None or str(s).strip() == "":
        return False
    try:
        float(s)  # accepts '22.9', '5.5e+35', 'nan', 'inf' -> all real, recorded results
        return True
    except (TypeError, ValueError):
        return False


def _key(model, tech, prec, sp_pct, dataset):
    return (model, tech, int(prec), int(sp_pct), dataset)


def scan_done(reuse_globs):
    """Build the set of DONE config-keys from every result CSV to be trusted.

    Always scans this script's own parts/. With --reuse-existing, also scans the
    given globs so cells already computed by earlier runs are not recomputed.
    A key is done iff a row for it carries a parseable perplexity.
    """
    done = set()
    paths = sorted(glob.glob(os.path.join(PARTSDIR, "*.csv")))
    for g in reuse_globs:
        paths += sorted(glob.glob(g))
    for path in paths:
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if not _valid_ppl(row.get("ppl")):
                        continue
                    try:
                        done.add(_key(row["model"], row["technique"],
                                      float(row["precision"]),
                                      round(float(row["sparsity"]) * 100),
                                      row["dataset"]))
                    except (KeyError, TypeError, ValueError):
                        continue
        except (OSError, csv.Error):
            continue
    return done


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
    env.pop("PYTHONPATH", None)                 # host shadows the venv otherwise
    env["PIP_CONFIG_FILE"] = "/dev/null"
    # Per-model HF cache: gemma-2b lives on the roomy /scratch bind-mount (gated,
    # too big for the /workspace quota); qwen/llama are in the workspace cache.
    if model == "gemma-2b":
        env["HF_HOME"] = (os.environ.get("PRISM_HF_HOME_GEMMA")
                          or "/scratch/ckp908/prism_hf")
    else:
        env["HF_HOME"] = (os.environ.get("PRISM_HF_HOME_DEFAULT")
                          or os.path.join(ROOT, "cache", "huggingface"))
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    tok = _hf_token()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env["PRISM_SEED"] = os.environ.get("PRISM_SEED", "0")
    env["PRISM_LOWMEM"] = "1"                   # lean eval path (harmless here, helps margin)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if device_id != "":
        env["CUDA_VISIBLE_DEVICES"] = device_id
    return env


def run_one(device_id, task):
    """Launch one config as a subprocess. Return (ok, ppl, seconds, status)."""
    model, tech, prec, sp_pct, dataset = task
    tag = tag_of(*task)
    part_csv = os.path.join("table1_grid", "parts", tag + ".csv")  # rel to results/
    logpath = os.path.join(LOGDIR, tag + ".log")
    cmd = [PY, "benchmarks/benchmark_suite.py",
           "--model", model, "--technique", tech,
           "--precision", str(prec), "--sparsity", str(sp_pct / 100.0),
           "--dataset", dataset, "--csv", part_csv]
    t0 = time.time()
    with open(logpath, "w") as lf:
        rc = subprocess.run(cmd, cwd=ROOT, env=task_env(device_id, model),
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    # Success is defined by the result, not the exit code: run_benchmark() catches
    # OOM/errors internally and can still exit 0 with no perplexity, while a real
    # collapse prints a nan/inf we must accept. So: ok iff the part-CSV now holds a
    # parseable ppl row for this config.
    ppl = None
    try:
        with open(os.path.join(RESULTS, part_csv), newline="") as f:
            for row in csv.DictReader(f):
                if _valid_ppl(row.get("ppl")):
                    ppl = row["ppl"]
    except (OSError, csv.Error):
        pass
    ok = ppl is not None
    status = "ok" if ok else (f"rc={rc},noppl")
    return ok, ppl, dt, status


# ----------------------------------------------------------------------------- schedule
def worker(device_id, gb, pool, counters):
    """Pull the heaviest config this slice can fit, run it, retry on failure."""
    while True:
        task = None
        with _pool_lock:
            # heaviest-fitting first so the big slice takes the heavy models
            best_i = -1
            best_req = -1.0
            for i, (t, att) in enumerate(pool):
                r = req_gb(t[0])
                if r <= gb and r > best_req:
                    best_req, best_i = r, i
            if best_i < 0:
                if not pool:
                    return  # nothing left this slice can (ever) run
                # only oversized-for-this-slice work remains -> let bigger slices take it
                task = None
            else:
                task, att = pool.pop(best_i)
        if task is None:
            time.sleep(2)
            # re-check: if the remaining pool has nothing for us at all, exit
            with _pool_lock:
                if not any(req_gb(t[0]) <= gb for t, _ in pool):
                    return
            continue

        tag = tag_of(*task)
        ok, ppl, dt, status = run_one(device_id, task)
        with _print_lock:
            with open(PROGRESS, "a") as pf:
                pf.write(f"{tag}\t{ppl}\t{int(dt)}\t{status}\t{device_id[:16]}\tatt{att}\n")
        if ok:
            with _pool_lock:
                counters["done"] += 1
                n = counters["done"]
            log(f"OK    {tag}  ppl={ppl}  ({int(dt)}s)  slice={device_id[:12]}  "
                f"[{n}/{counters['total']}]")
        elif att < RETRIES:
            log(f"RETRY {tag}  ({status}, {int(dt)}s) -> attempt {att + 1}")
            time.sleep(3)
            with _pool_lock:
                pool.append((task, att + 1))
        else:
            with _pool_lock:
                counters["failed"] += 1
            log(f"FAIL  {tag}  ({status})  gave up after {RETRIES} retries")


def build_tasks(models, techs, datasets, spars, precs, done):
    tasks = []
    for dataset in datasets:
        for model in models:
            for tech in techs:
                for sp in spars:
                    for prec in precs:
                        if _key(model, tech, prec, sp, dataset) in done:
                            continue
                        tasks.append((model, tech, prec, sp, dataset))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--only", default=None, help="keep only tasks whose tag contains this")
    ap.add_argument("--exclude", default=None, help="drop tasks whose tag contains this")
    ap.add_argument("--limit", type=int, default=None, help="run at most N tasks (smoke test)")
    ap.add_argument("--models", default=None, help="comma list to override models")
    ap.add_argument("--techniques", default=None, help="comma list to override techniques")
    ap.add_argument("--datasets", default=None, help="comma list to override datasets")
    ap.add_argument("--retries", type=int, default=None, help="override retry count")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="also treat prior result CSVs (table1_baselines, benchmark_results) "
                         "as done, to avoid recomputing cells from earlier sessions")
    args = ap.parse_args()

    global RETRIES
    if args.retries is not None:
        RETRIES = args.retries

    models = args.models.split(",") if args.models else MODELS
    techs = args.techniques.split(",") if args.techniques else TECHS
    datasets = args.datasets.split(",") if args.datasets else DATASETS

    os.makedirs(PARTSDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    if not os.path.exists(PROGRESS):
        with open(PROGRESS, "w") as f:
            f.write("tag\tppl\tseconds\tstatus\tslice\tattempt\n")

    reuse_globs = []
    if args.reuse_existing:
        reuse_globs = [
            os.path.join(RESULTS, "table1_baselines", "parts", "*.csv"),
            os.path.join(RESULTS, "benchmark_results.csv"),
        ]
    done = scan_done(reuse_globs)
    tasks = build_tasks(models, techs, datasets, SPARS_PCT, PRECS, done)

    if args.only:
        tasks = [t for t in tasks if args.only in tag_of(*t)]
    if args.exclude:
        tasks = [t for t in tasks if args.exclude not in tag_of(*t)]
    if args.limit:
        tasks = tasks[: args.limit]

    slices = discover_slices()
    total_cells = len(models) * len(techs) * len(datasets) * len(SPARS_PCT) * len(PRECS)
    log(f"grid: {len(models)}m x {len(techs)}t x {len(datasets)}d x "
        f"{len(SPARS_PCT)}sp x {len(PRECS)}bit = {total_cells} cells")
    log(f"already done (result row present): {len(done)}")
    log(f"slices: {len(slices)}  ["
        + ", ".join(f"{d[:12]}:{int(g)}GB" for d, g in slices) + "]" if slices else "slices: none (CPU)")
    log(f"tasks to run now: {len(tasks)}")
    from collections import Counter
    for (m, tc), n in sorted(Counter((t[0], t[1]) for t in tasks).items()):
        log(f"    {m:10s} {tc:11s} : {n}")

    # No-OOM sanity: refuse to start if some task cannot fit ANY available slice.
    if slices:
        max_gb = max(g for _, g in slices)
        toobig = sorted({t[0] for t in tasks if req_gb(t[0]) > max_gb})
        if toobig:
            log(f"ERROR: no slice big enough (max {int(max_gb)}GB) for models: {toobig}")
            return 2

    if args.dry_run:
        log("dry-run — exiting without running anything")
        return 0
    if not tasks:
        log("nothing to do — all requested cells already have a result row")
        return 0
    if not slices:
        slices = [("", DEFAULT_REQ_GB)]  # single default-device worker

    pool = [(t, 0) for t in tasks]
    counters = {"done": 0, "failed": 0, "total": len(tasks)}
    threads = []
    for device_id, gb in slices:
        th = threading.Thread(target=worker, args=(device_id, gb, pool, counters), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    log(f"ALL DONE  done={counters['done']}  failed={counters['failed']}  "
        f"total={counters['total']}")
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
