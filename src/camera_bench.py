#!/usr/bin/env python3
"""Camera-ready benchmark — ONE CELL driver.

A "cell" = (method, sparsity) for the fixed model (gemma-2b) at a fixed weight
bit-width. The driver quantizes/prunes the model ONCE, then evaluates:
  * PPL on wikitext2 / ptb / c4  (full test sets; c4 capped to --c4-windows)
  * downstream arc_easy / hellaswag / piqa / winogrande  (full splits)
and UPSERTS one flat row per (task, metric) into a single unified CSV keyed by
(model, method, bits, sparsity, task, metric). Per-task rows already present are
SKIPPED, so an interrupted cell (a MIG slice grabbed by another job) resumes with
no lost work — re-running the cell only re-does the missing evals.

Method -> (sparsity, bits) applicability (decided ONCE, documented here):
  fp16        no compression         -> sparsity=0.00, bits=16   (flat across sparsity axis)
  awq         quant-only  (no prune) -> sparsity=0.00, bits=B    (flat across sparsity axis)
  sinq        quant-only  (no prune) -> sparsity=0.00, bits=B    (flat across sparsity axis)
  wanda       prune-only  (fp16 W)   -> sparsity=S,    bits=16   (3-bit axis N/A -> fp16 weights)
  sparsegpt   joint prune+quant      -> sparsity=S,    bits=B
  jsq-wo      joint prune+quant (WO)  -> sparsity=S,    bits=B
  slim        prune+quant+low-rank   -> sparsity=S,    bits=B
  wanda-awq   prune + AWQ survivors  -> sparsity=S,    bits=B
  wanda-sinq  prune + SINQ survivors -> sparsity=S,    bits=B
  cobalt      OUR column-balanced    -> sparsity=S,    bits=B

Usage:
  python src/camera_bench.py --method cobalt --sparsity 0.7 --bits 3 \
      --csv results/benchmark_camera/results.csv
"""
import argparse
import csv
import fcntl
import gc
import os
import sys
import time
from datetime import datetime, timezone

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa: E402
import nosink as ns  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "gemma-2b"  # overridden by --model in main()
DEV = "cuda"
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]

PPL_TASKS = ["wikitext2", "ptb", "c4"]
# 4 reasoning MCQ benchmarks gemma-2b solves well above chance dense and collapses toward
# chance under aggressive compression (ARC/HellaSwag 4-way ~25%, WinoGrande 2-way 50%).
DS_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]

# Methods that DO prune (sparsity is meaningful). Others are flat at sparsity 0.
PRUNE_METHODS = {"wanda", "sparsegpt", "jsq-wo", "slim", "wanda-awq", "wanda-sinq", "cobalt", "cobalt-awq",
                 "cobaltmask-sgpt", "sgptmask-cobalt", "cobalt-noobs"}
# Methods whose *weights* stay fp16 (bits axis is N/A -> recorded as 16).
FP16_WEIGHT_METHODS = {"fp16", "wanda"}
ALL_METHODS = ["fp16", "awq", "sinq", "wanda", "sparsegpt", "jsq-wo", "slim",
               "wanda-awq", "wanda-sinq", "cobalt", "cobalt-awq",
               "cobaltmask-sgpt", "sgptmask-cobalt", "cobalt-noobs"]

CSV_FIELDS = ["timestamp", "model", "method", "bits", "sparsity", "hp",
              "task", "metric", "value", "correct", "total", "seconds", "error"]


# ----------------------------------------------------------------------------- CSV
def _canon(sparsity, bits):
    return f"{float(sparsity):.2f}", str(int(bits))


def read_done_tasks(csv_path, method, bits, sparsity, hp=""):
    """Return set of task names already present (error-free) for this cell (hp-scoped).

    Rows written before the hp axis existed have an empty `hp`; a caller asking for
    hp="" still matches them, so legacy CSVs resume cleanly."""
    if not os.path.exists(csv_path):
        return set()
    sp_s, b_s = _canon(sparsity, bits)
    done = set()
    try:
        with open(csv_path, "r", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("model") == MODEL and r.get("method") == method
                        and r.get("bits") == b_s and r.get("sparsity") == sp_s
                        and (r.get("hp") or "") == hp
                        and not (r.get("error") or "").strip()):
                    done.add(r.get("task"))
    except Exception:
        return set()
    return done


def read_attempted_tasks(csv_path, method, bits, sparsity, hp=""):
    """Return task names that are SETTLED for this cell — a good OR a FAILED row exists.

    The dispatcher uses this (not read_done_tasks) to decide a cell is finished, so a
    deterministic build failure that wrote FAILED rows is not retried forever."""
    if not os.path.exists(csv_path):
        return set()
    sp_s, b_s = _canon(sparsity, bits)
    seen = set()
    try:
        with open(csv_path, "r", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("model") == MODEL and r.get("method") == method
                        and r.get("bits") == b_s and r.get("sparsity") == sp_s
                        and (r.get("hp") or "") == hp):
                    seen.add(r.get("task"))
    except Exception:
        return set()
    return seen


def append_rows(csv_path, rows):
    """Append rows (list of dicts) under an exclusive flock; write header if new."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            has_header = f.readline().strip().startswith("timestamp")
            f.seek(0, 2)
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not has_header:
                w.writeheader()
            for row in rows:
                w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _row(method, bits, sparsity, task, metric, value, correct="", total="",
         seconds="", error="", hp=""):
    sp_s, b_s = _canon(sparsity, bits)
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": MODEL, "method": method, "bits": b_s, "sparsity": sp_s, "hp": hp,
            "task": task, "metric": metric, "value": value, "correct": correct,
            "total": total, "seconds": f"{seconds:.1f}" if seconds != "" else "",
            "error": error}


# ------------------------------------------------------------------------- model
def build_sbt(sparsity):
    return {t: float(sparsity) for t in TYPES}


def model_tag(model=None):
    """Filesystem-safe short tag for the active model (keys per-model caches)."""
    return (model or MODEL).replace("/", "_")


def ppl_cache_path(cache_dir, task, seq, n):
    # Keyed by model: Llama and gemma use different tokenizers, so their tokenized
    # PPL tensors MUST NOT collide even if they share a cache dir.
    return os.path.join(cache_dir, f"{task}_{model_tag()}_s{seq}_n{n}.pt")


def get_ppl_test(tok, task, ppl_windows, c4_windows, cache_dir):
    """Load the tokenized PPL test tensor for `task`.

    Prefers a pre-materialized cache (identical inputs across every cell, and
    offline-safe). c4 is streaming-only and not cached by HF, so it MUST be
    pre-built by scripts/prep_ppl_cache.py (run with HF offline OFF); wikitext2/
    ptb also load in-process from the HF cache as a fallback.
    """
    seq = bs.EVAL_CONFIG["seq_len"]
    n = c4_windows if task == "c4" else ppl_windows
    cache = ppl_cache_path(cache_dir, task, seq, n)
    if os.path.exists(cache):
        return torch.load(cache)
    if task == "c4":
        raise RuntimeError(
            f"c4 PPL cache missing ({cache}). Build it once with HF online:\n"
            f"  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/prep_ppl_cache.py")
    return bs.get_test_data(tok, seq_len=seq, n_samples=n, dataset_key=task)


def _cobalt_balanced_prune_mask(W, H, sparsity, col_exp=0.5):
    """CoBALT balanced KEEP-mask (row_fair quantile + col_exp quantile + global top-k),
    computed from W and the SparseGPT Hessian H (act_norm ∝ sqrt(diag(H)); the global
    constant cancels in every quantile/threshold op). Returns a bool PRUNE mask
    (True = pruned) matching SparseGPT's fixed_mask convention. Mirrors camera_bench_vit."""
    K, N = W.shape
    if sparsity <= 0.0:
        return torch.zeros_like(W, dtype=torch.bool)
    act = torch.diag(H).clamp(min=0).sqrt()               # [N] ∝ ||X_j||
    imp = W.abs() * act.view(1, -1)
    kr, kc = int(N * sparsity), int(K * sparsity)
    if kr > 0:                                            # row self-normalization
        qr = torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qr
    if col_exp > 0 and kc > 0:                            # column self-normalization
        qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qc.pow(col_exp)
    n_prune = int(K * N * sparsity)
    thr = torch.kthvalue(imp.reshape(-1), n_prune).values
    keep = (imp.reshape(-1) > thr).view(K, N)
    return ~keep


def build_model(method, sparsity, bits, tok, cobalt_group_size=None,
                cobalt_beta=0.5, cobalt_norm="acol", cobalt_per_row=False,
                percdamp=0.01, blocksize=128, jsq_rho=2.1, jsq_clip_h=0.01):
    """Load gemma-2b fp16 and apply the requested compression. Returns model on DEV.

    The bpw-preserving TUNED knobs (cobalt β, sparsegpt percdamp/blocksize, jsq-wo
    rho/clip_h) are threaded here so the dispatcher can land one hp-cell per slice."""
    name = bs.MODELS[MODEL]
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                                  dataset_key="wikitext2")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)
    sbt = build_sbt(sparsity)

    if method == "fp16":
        model = model.to(DEV)
    elif method == "awq":
        model = bs.apply_awq_quantization(model, cal, bits, DEV)
    elif method == "sinq":
        model = bs.apply_sinq_quantization(model, cal, bits, DEV)
    elif method == "wanda":
        model = bs.apply_wanda_pruning(model, cal, float(sparsity), DEV)
    elif method == "sparsegpt":
        model = bs.apply_sparsegpt_pruning(model, cal, float(sparsity), bits, DEV,
                                           percdamp=float(percdamp), blocksize=int(blocksize))
    elif method == "jsq-wo":
        # JSQ weight-only: its two live knobs (rho, clip_h) are read from env by _apply_jsq.
        os.environ["JSQ_RHO"] = f"{float(jsq_rho):g}"
        os.environ["JSQ_CLIPH"] = f"{float(jsq_clip_h):g}"
        model = bs.apply_jsq_weightonly_quantization(model, cal, bits, float(sparsity), DEV)
    elif method == "slim":
        model = bs.apply_slim_quantization(model, cal, bits, float(sparsity), DEV)
    elif method == "wanda-awq":
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV)
    elif method == "wanda-sinq":
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV)
    elif method == "cobalt":
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "cobalt-noobs":
        # ABLATION (Factor C OFF): IDENTICAL to `cobalt` except the OBS error-compensation step is
        # dropped (no_obs=True). Same balanced mask (byte-identical at a given β), same col-norm, same
        # group-RTN. Combined with the β knob this spans the mask×OBS 2x2: (β=0|β*) x (this|cobalt).
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          no_obs=True)
    elif method == "cobaltmask-sgpt":
        # CAUSAL DECOMP: CoBALT's balanced MASK fed into SparseGPT's joint (quant-in-loop)
        # OBS RECIPE. Isolates the recipe by swapping only the mask.
        def _mfn(attr_path, W, H):
            return _cobalt_balanced_prune_mask(W, H, float(sparsity), col_exp=0.5)
        model = bs.apply_sparsegpt_pruning(model, cal, float(sparsity), bits, DEV, mask_fn=_mfn)
    elif method == "sgptmask-cobalt":
        # CAUSAL DECOMP: SparseGPT-style OBS-saliency MASK fed into CoBALT's RECIPE
        # (one-shot prune-OBS + uncompensated final RTN). Isolates the mask.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="obs_saliency", dense_norm="col",
                                          group_size=cobalt_group_size)
    elif method == "cobalt-awq":
        # Generalization variant: balanced mask (tunable beta) + AWQ-style activation-aware
        # survivor quant ('acol' = mu_w^(1-a)/mu_x^a, same OBS activations, non-Sinkhorn).
        # Motivated by the beta-sweep (small beta helps EVERY arch's mask) + Qwen's grid
        # failure being the RTN quantizer, not the mask. See FINDINGS 'OPEN-IDEA RESULT'.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm=cobalt_norm,
                                          mask_mode="balanced", dense_norm=cobalt_norm,
                                          col_balance_exp=cobalt_beta, awq_alpha=0.5,
                                          balance_per_row=cobalt_per_row,
                                          group_size=cobalt_group_size)
    else:
        raise ValueError(f"unknown method {method}")

    bs.move_final_layers_to_device(model, DEV)
    model.eval()
    model.seqlen = bs.EVAL_CONFIG["seq_len"]
    return model


# -------------------------------------------------------------------------- main
def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="model key in benchmark_suite.MODELS")
    ap.add_argument("--method", required=True, choices=ALL_METHODS)
    ap.add_argument("--sparsity", type=float, default=0.0)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "benchmark_camera", "results.csv"))
    ap.add_argument("--ppl-tasks", default=",".join(PPL_TASKS))
    ap.add_argument("--ds-tasks", default=",".join(DS_TASKS))
    ap.add_argument("--c4-windows", type=int, default=256)
    ap.add_argument("--ppl-windows", type=int, default=100000,
                    help="cap for wikitext2/ptb windows (100000 = exhaust full test)")
    ap.add_argument("--limit", type=int, default=None, help="downstream sample cap (None=full split)")
    ap.add_argument("--cobalt-group-size", type=int, default=None,
                    help="override CoBALT RTN group size (default 64; use 128 to MATCH baselines' bpw)")
    ap.add_argument("--cobalt-beta", type=float, default=0.5,
                    help="column-balance strength (method=cobalt tuned β; also cobalt-awq)")
    ap.add_argument("--hp", default="", help="hp label written to the CSV (dispatcher-owned; e.g. 'beta=0.5')")
    ap.add_argument("--sgpt-percdamp", type=float, default=0.01, help="sparsegpt Hessian damping (tuned)")
    ap.add_argument("--sgpt-blocksize", type=int, default=128, help="sparsegpt OBS block width (tuned)")
    ap.add_argument("--jsq-rho", type=float, default=2.1, help="jsq-wo SAR prune<->quant bridge (tuned)")
    ap.add_argument("--jsq-clip-h", type=float, default=0.01, help="jsq-wo activation-clip fraction (tuned)")
    ap.add_argument("--cobalt-norm", default="acol",
                    help="survivor per-col scale for cobalt-awq: 'acol'=AWQ activation-aware, 'col'=weight-std")
    ap.add_argument("--cobalt-per-row", action="store_true",
                    help="cobalt-awq: use the GENTLE balance variant (single per-col reweight + exact "
                         "per-row threshold; unimpeachably non-Sinkhorn) instead of global-topk balance")
    ap.add_argument("--force-true-bits", action="store_true",
                    help="neutralize get_adaptive_nbits so SINQ/Wanda+SINQ stay at true target bits "
                         "(matched-config fix for models where the adaptive bump fires, e.g. Qwen2.5-3B)")
    ap.add_argument("--ppl-cache-dir",
                    default=os.path.join(_ROOT, "results", "benchmark_camera", "ppl_cache"))
    args = ap.parse_args()

    MODEL = args.model
    if MODEL not in bs.MODELS:
        raise SystemExit(f"unknown --model {MODEL}; known: {sorted(bs.MODELS)}")

    if args.force_true_bits:
        # Matched-config: some models (Qwen2.5-3B) trip get_adaptive_nbits, which
        # silently bumps low-variance layers 3->5 bit for SINQ/Wanda+SINQ ONLY.
        # Force every arm to the true target bit-width so no method gets free bits.
        bs.get_adaptive_nbits = lambda W, target_nbits, *a, **k: target_nbits
        print("[cell] --force-true-bits: get_adaptive_nbits neutralized (true bits for all arms)", flush=True)

    method = args.method
    # Normalize the (sparsity, bits) that this method is actually DEFINED at.
    sparsity = args.sparsity if method in PRUNE_METHODS else 0.0
    bits = 16 if method in FP16_WEIGHT_METHODS else args.bits
    sp_s, b_s = _canon(sparsity, bits)

    ppl_tasks = [t.strip() for t in args.ppl_tasks.split(",") if t.strip()]
    ds_tasks = [t.strip() for t in args.ds_tasks.split(",") if t.strip()]
    all_tasks = ppl_tasks + ds_tasks

    done = read_done_tasks(args.csv, method, bits, sparsity, args.hp)
    todo = [t for t in all_tasks if t not in done]
    tag = f"{method} sp={sp_s} bits={b_s} hp={args.hp or 'default'}"
    print(f"[cell] {tag} | csv={args.csv}", flush=True)
    print(f"[cell] done={sorted(done)} todo={todo}", flush=True)
    if not todo:
        print(f"CELL_DONE {tag} (all tasks cached)", flush=True)
        return

    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t_build = time.time()
    try:
        model = build_model(method, sparsity, bits, tok, cobalt_group_size=args.cobalt_group_size,
                            cobalt_beta=args.cobalt_beta, cobalt_norm=args.cobalt_norm,
                            cobalt_per_row=args.cobalt_per_row,
                            percdamp=args.sgpt_percdamp, blocksize=args.sgpt_blocksize,
                            jsq_rho=args.jsq_rho, jsq_clip_h=args.jsq_clip_h)
    except Exception as e:
        # A DETERMINISTIC build failure (e.g. SparseGPT non-PD Cholesky at pd=0.001 on the wide
        # gemma FFN, seeded calibration) would crash the process and be requeued forever. Instead
        # write an honest FAILED row per remaining task and exit 0 so the dispatcher marks the cell
        # settled and moves on (mirrors the GLUE/ViT per-hp-cell FAILED-row behavior).
        for task in todo:
            metric = "perplexity" if task in ppl_tasks else "accuracy"
            append_rows(args.csv, [_row(method, bits, sparsity, task, metric, "",
                                        seconds=time.time() - t_build,
                                        error=f"FAILED:build:{type(e).__name__}: {e}", hp=args.hp)])
        print(f"[FAIL] {tag} build: {type(e).__name__}: {e}", flush=True)
        print(f"CELL_DONE {tag} (build failed -> FAILED rows written)", flush=True)
        return
    print(f"[cell] built in {time.time()-t_build:.1f}s", flush=True)

    # ---- PPL tasks ----
    for task in ppl_tasks:
        if task in done:
            continue
        t0 = time.time()
        try:
            test = get_ppl_test(tok, task, args.ppl_windows, args.c4_windows, args.ppl_cache_dir)
            ppl = bs.evaluate_perplexity(model, test, DEV)
            dt = time.time() - t0
            append_rows(args.csv, [_row(method, bits, sparsity, task, "perplexity",
                                        f"{ppl:.4f}", total=test.shape[0], seconds=dt, hp=args.hp)])
            print(f"RESULT {tag} task={task} ppl={ppl:.4f} n={test.shape[0]} ({dt:.1f}s)", flush=True)
        except Exception as e:
            dt = time.time() - t0
            append_rows(args.csv, [_row(method, bits, sparsity, task, "perplexity", "",
                                        seconds=dt, error=f"FAILED:{type(e).__name__}: {e}", hp=args.hp)])
            print(f"[FAIL] {tag} task={task}: {type(e).__name__}: {e}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    # ---- downstream tasks ----
    from downstream import eval_arc, eval_hellaswag, eval_piqa, eval_winogrande  # noqa
    DS_ADAPTERS = {"hellaswag": eval_hellaswag.run, "piqa": eval_piqa.run,
                   "winogrande": eval_winogrande.run}
    seqlen = bs.EVAL_CONFIG["seq_len"]
    _arc_cache = {}   # eval_arc.run evaluates BOTH configs in one call -> memoize across arc_easy/arc_challenge
    for task in ds_tasks:
        if task in done:
            continue
        t0 = time.time()
        try:
            if task in ("arc_easy", "arc_challenge"):
                if not _arc_cache:
                    _arc_cache.update(eval_arc.run(model, tok, DEV, limit=args.limit,
                                                   seqlen=seqlen, verbose=False))
                key = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
                m = _arc_cache.get(key, {})
            else:
                m = DS_ADAPTERS[task](model, tok, DEV, limit=args.limit, seqlen=seqlen, verbose=False)
            dt = time.time() - t0
            append_rows(args.csv, [_row(method, bits, sparsity, task, "accuracy",
                                        f"{m.get('accuracy'):.4f}", correct=m.get("correct", ""),
                                        total=m.get("total", ""), seconds=dt, hp=args.hp)])
            print(f"RESULT {tag} task={task} acc={m.get('accuracy'):.4f} "
                  f"({m.get('correct')}/{m.get('total')}) ({dt:.1f}s)", flush=True)
        except Exception as e:
            dt = time.time() - t0
            append_rows(args.csv, [_row(method, bits, sparsity, task, "accuracy", "",
                                        seconds=dt, error=f"FAILED:{type(e).__name__}: {e}", hp=args.hp)])
            print(f"[FAIL] {tag} task={task}: {type(e).__name__}: {e}", flush=True)
        model.to(DEV)  # defensive: some adapters can move the model
        gc.collect(); torch.cuda.empty_cache()

    print(f"CELL_DONE {tag}", flush=True)


if __name__ == "__main__":
    main()
