#!/usr/bin/env python3
"""Parallel MIG work-queue runner for the Table-1 fair-baseline grid.

Fills in the joint prune+quant baselines the paper's Table 1 is missing, across
its full grid:

    models      : qwen-0.5b, gemma-2b, llama-7b
    datasets    : wikitext2, c4
    sparsity    : 5%, 25%, 50%
    precision   : 3, 4, 5 bit
    techniques  : prism, sparsegpt, wanda-sinq, wanda-awq, jsq-wo, slim

Reuse: for qwen-0.5b / llama-7b the bias-correction ablation already computed
prism, sparsegpt, wanda-sinq, jsq-wo and slim under *identical* runner settings,
so those cells are pulled from the ablation CSVs (see collect_ppl.py) and NOT
recomputed. wanda-awq (never in the ablation) and every gemma-2b cell are run
fresh. That leaves 144 fresh configs.

Scheduling: one worker thread per MIG slice, each pinning CUDA_VISIBLE_DEVICES to
its slice and launching benchmark_suite.py as a subprocess. A shared queue load-
balances across slices, so llama-7b (heavy) and gemma-2b/qwen (light) interleave
and every slice stays busy. Every config fits a 1g.24gb slice: wanda-awq stores a
single fp16 weight copy (~13 GB for llama-7b) and PRISM_LOWMEM=1 keeps the
Sinkhorn (wanda-sinq/prism) path lean, both < 24 GB.

Resumable + glitch-guarded: a config is skipped if its done-marker exists; failed
configs are retried up to RETRIES times on any free slice. Each config writes to
its OWN part-CSV under results/table1_baselines/parts/ (no write races).

Usage:
  run_table1_baselines.py                 # run all fresh configs
  run_table1_baselines.py --dry-run       # print the plan, run nothing
  run_table1_baselines.py --only gemma    # only tasks whose tag matches substr
  run_table1_baselines.py --limit 2       # smoke test: run at most N tasks
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from queue import Queue

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import collect_ppl  # noqa: E402

PY = os.path.join(ROOT, "env", "bin", "python")
OUTDIR = os.path.join(ROOT, "results", "table1_baselines")
PARTSDIR = os.path.join(OUTDIR, "parts")
DONEDIR = os.path.join(OUTDIR, "done")
LOGDIR = os.path.join(ROOT, "logs")
PROGRESS = os.path.join(OUTDIR, "progress.tsv")

MODELS = ["qwen-0.5b", "gemma-2b", "llama-7b"]
DATASETS = ["wikitext2", "c4"]
SPARS_PCT = [5, 25, 50]
PRECS = [3, 4, 5]
TECHS = ["prism", "sparsegpt", "wanda-sinq", "wanda-awq", "jsq-wo", "slim"]
# Techniques the ablation already computed for qwen/llama at identical settings.
REUSE_TECHS = {"prism", "sparsegpt", "wanda-sinq", "jsq-wo", "slim"}
REUSE_MODELS = {"qwen-0.5b", "llama-7b"}

RETRIES = 2

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[t1] {msg}", flush=True)


def discover_slices():
    """Return the list of MIG UUIDs (respecting MIG_EXCLUDE / NSLICES env)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout.decode("utf-8", "replace")
    except Exception:
        out = ""
    uuids = re.findall(r"(MIG-[0-9a-fA-F-]+)", out)
    excl = set(filter(None, os.environ.get("MIG_EXCLUDE", "").split(",")))
    uuids = [u for u in uuids if u not in excl]
    nslices = os.environ.get("NSLICES")
    if nslices:
        uuids = uuids[: int(nslices)]
    return uuids


def tag_of(model, tech, prec, sp_pct, dataset):
    return f"{model}_{tech}_{prec}bit_sp{sp_pct}_{dataset}"


def build_tasks():
    """Return the list of fresh configs to run (reuse-eligible cells excluded)."""
    reuse_store = collect_ppl.collect(restrict=["ablation_", "table1_baselines"])
    tasks = []
    for dataset in DATASETS:
        for model in MODELS:
            for tech in TECHS:
                for sp in SPARS_PCT:
                    for prec in PRECS:
                        tag = tag_of(model, tech, prec, sp, dataset)
                        if os.path.exists(os.path.join(DONEDIR, tag + ".done")):
                            continue
                        # Reuse trusted ablation cells (qwen/llama, reuse techs).
                        if tech in REUSE_TECHS and model in REUSE_MODELS:
                            if collect_ppl.lookup(reuse_store, model, tech, prec, sp, dataset) is not None:
                                continue
                        tasks.append((model, tech, prec, sp, dataset))
    return tasks


def _hf_token():
    """Read the HF access token (for gated repos like gemma-2b) from env or .hf_token."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    tokpath = os.path.join(ROOT, ".hf_token")
    if os.path.exists(tokpath):
        with open(tokpath) as fh:
            return fh.read().strip()
    return None


def base_env(uuid):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PIP_CONFIG_FILE"] = "/dev/null"
    # HF cache: default to the /workspace mount, but allow an override (e.g. the
    # roomy /scratch bind-mount) so a gated model too big for the ~24 GB workspace
    # quota can be fetched without evicting the existing llama/qwen cache.
    env["HF_HOME"] = os.environ.get("PRISM_HF_HOME") or os.path.join(ROOT, "cache", "huggingface")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    tok = _hf_token()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env["PRISM_SEED"] = os.environ.get("PRISM_SEED", "0")
    env["PRISM_LOWMEM"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if uuid:
        env["CUDA_VISIBLE_DEVICES"] = uuid
    return env


def run_task(uuid, task):
    """Run one config. Return (ok, ppl, seconds, status)."""
    model, tech, prec, sp, dataset = task
    tag = tag_of(model, tech, prec, sp, dataset)
    sp_frac = sp / 100.0
    part_csv = os.path.join("table1_baselines", "parts", tag + ".csv")  # rel to results/
    logpath = os.path.join(LOGDIR, "table1_" + tag + ".log")
    cmd = [PY, "benchmarks/benchmark_suite.py",
           "--model", model, "--technique", tech,
           "--precision", str(prec), "--sparsity", str(sp_frac),
           "--dataset", dataset, "--csv", part_csv]
    t0 = time.time()
    with open(logpath, "w") as lf:
        rc = subprocess.run(cmd, cwd=ROOT, env=base_env(uuid),
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    ppl = None
    try:
        with open(logpath) as lf:
            m = re.findall(r"Perplexity:\s*([0-9.]+)", lf.read())
            if m:
                ppl = float(m[-1])
    except Exception:
        pass
    ok = (rc == 0 and ppl is not None)
    status = "ok" if ok else (f"rc={rc}" + ("" if ppl is not None else ",noppl"))
    return ok, ppl, dt, status


def worker(uuid, q, counters):
    STOP = counters["STOP"]
    while True:
        item = q.get()
        if item is STOP:
            q.task_done()
            return
        task, att = item
        tag = tag_of(*task)
        ok, ppl, dt, status = run_task(uuid, task)
        with _print_lock:
            with open(PROGRESS, "a") as pf:
                pf.write(f"{tag}\t{ppl}\t{int(dt)}\t{status}\t{uuid[:16]}\tatt{att}\n")
        if ok:
            open(os.path.join(DONEDIR, tag + ".done"), "w").write(f"{ppl}\n")
            counters["done"] += 1
            log(f"OK   {tag}  ppl={ppl}  ({int(dt)}s)  slice={uuid[4:12]}  "
                f"[{counters['done']}/{counters['total']}]")
        else:
            if att < RETRIES:
                log(f"RETRY {tag}  ({status}, {int(dt)}s) att={att+1}")
                q.put((task, att + 1))
            else:
                counters["failed"] += 1
                log(f"FAIL {tag}  ({status})  gave up after {RETRIES} retries")
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on task tag")
    ap.add_argument("--exclude", default=None, help="drop tasks whose tag contains this substr")
    ap.add_argument("--limit", type=int, default=None, help="run at most N tasks")
    args = ap.parse_args()

    os.makedirs(PARTSDIR, exist_ok=True)
    os.makedirs(DONEDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    if not os.path.exists(PROGRESS):
        open(PROGRESS, "w").write("tag\tppl\tseconds\tstatus\tslice\tattempt\n")

    tasks = build_tasks()
    if args.only:
        tasks = [t for t in tasks if args.only in tag_of(*t)]
    if args.exclude:
        tasks = [t for t in tasks if args.exclude not in tag_of(*t)]
    if args.limit:
        tasks = tasks[: args.limit]

    slices = discover_slices()
    log(f"MIG slices: {len(slices)}  ({', '.join(s[4:12] for s in slices) or 'CPU/none'})")
    log(f"fresh tasks to run: {len(tasks)}")
    # brief per-model/tech breakdown
    from collections import Counter
    bd = Counter((t[0], t[1]) for t in tasks)
    for (m, tc), n in sorted(bd.items()):
        log(f"   {m:10s} {tc:11s} : {n}")

    if args.dry_run or not tasks:
        log("dry-run / nothing to do — exiting")
        return 0
    if not slices:
        slices = [""]  # single CPU/default-device worker

    q = Queue()
    for t in tasks:
        q.put((t, 0))
    counters = {"done": 0, "failed": 0, "total": len(tasks), "STOP": object()}

    threads = []
    for uuid in slices:
        th = threading.Thread(target=worker, args=(uuid, q, counters), daemon=True)
        th.start()
        threads.append(th)

    q.join()
    for _ in threads:
        q.put(counters["STOP"])
    for th in threads:
        th.join()

    log(f"ALL DONE  done={counters['done']}  failed={counters['failed']}  total={counters['total']}")
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
