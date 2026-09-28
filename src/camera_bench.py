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

MODEL = "gemma-2b"  # overridden by --model in main
DEV = "cuda"
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]

PPL_TASKS = ["wikitext2", "ptb", "c4"]
# 4 reasoning MCQ benchmarks gemma-2b solves well above chance dense and collapses toward
# chance under aggressive compression (ARC/HellaSwag 4-way ~25%, WinoGrande 2-way 50%).
DS_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]

# Methods that DO prune (sparsity is meaningful). Others are flat at sparsity 0.
PRUNE_METHODS = {"wanda", "sparsegpt", "jsq-wo", "slim", "wanda-awq", "wanda-sinq", "cobalt", "cobalt-blk32", "cobalt-awq", "cobalt-sinq", "cobalt-sinq", "cobalt-seq", "cobalt-seqfix", "cobalt-seqfix-awclip", "cobalt-ec", "cobalt-ecfix",
                 "cobalt-floor", "wanda-floor", "cobalt-exact", "cobalt-span", "wanda-span", "cobalt-perhead", "cobalt-cond", "cobalt-spectral", "cobalt-specblend",
                 "cobaltmask-sgpt", "sgptmask-cobalt", "cobalt-noobs",
                 "cobalt-repack", "cobalt-eout", "cobalt-compand", "cobalt-binc", "cobalt-bincg",
                 "cobalt-awclip", "cobalt-awclipz", "cobalt-gridmask", "cobalt-gridmask-awclip",
                 "wanda-awq-repack", "wanda-sinq-repack", "wanda-awq-binc", "wanda-sinq-binc", "wanda-awq-awclipz", "wanda-sinq-awclipz", "wanda-awq-awclip", "wanda-sinq-awclip"}
# Methods whose *weights* stay fp16 (bits axis is N/A -> recorded as 16).
FP16_WEIGHT_METHODS = {"fp16", "wanda"}
ALL_METHODS = ["fp16", "awq", "sinq", "wanda", "sparsegpt", "jsq-wo", "slim",
               "wanda-awq", "wanda-sinq", "cobalt", "cobalt-blk32", "cobalt-awq", "cobalt-sinq", "cobalt-seq", "cobalt-seqfix", "cobalt-seqfix-awclip", "cobalt-ec", "cobalt-ecfix",
               "cobalt-floor", "wanda-floor", "cobalt-exact", "cobalt-span", "wanda-span", "cobalt-perhead", "cobalt-cond", "cobalt-spectral", "cobalt-specblend",
               "cobaltmask-sgpt", "sgptmask-cobalt", "cobalt-noobs",
               "cobalt-repack", "cobalt-eout", "cobalt-compand", "cobalt-binc", "cobalt-bincg",
               "cobalt-awclip", "cobalt-awclipz", "cobalt-gridmask", "cobalt-gridmask-awclip",
               "wanda-awq-repack", "wanda-sinq-repack", "wanda-awq-binc", "wanda-sinq-binc", "wanda-awq-awclipz", "wanda-sinq-awclipz", "wanda-awq-awclip", "wanda-sinq-awclip"]

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


def _is_transient(e):
    """CUDA OOM on this SHARED, MIG-sliced box is a placement accident, not a property of the cell:
    another project's job (or a second dispatcher) grabbed the slice first. Writing a FAILED row for it
    would mark the cell SETTLED and silently drop it from the grid forever. Re-raise instead, so the
    process exits non-zero and the dispatcher requeues it (its documented transient-failure path)."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return "CUDA out of memory" in str(e) or "CUDA error: out of memory" in str(e)


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
                percdamp=0.01, blocksize=128, jsq_rho=2.1, jsq_clip_h=0.01,
                cobalt_floor_frac=0.0, cobalt_protect=0.0, cobalt_adapt=0.0,
                awq_num_betas=1, awq_use_weightscale=True, awq_l1=False,
                sinq_order=16, sinq_stop=True,
                slim_lora=True, slim_cap_bins=None, slim_sparse_cap=False,
                wanda_act_exp=0.5, wanda_scope="row",
                awq_alpha=0.5, obs_damp=None, mask_scope="global"):
    """Load gemma-2b fp16 and apply the requested compression. Returns model on DEV.

    The bpw-preserving TUNED knobs are threaded here so the dispatcher can land one
    hp-cell per slice.  Every arm carries its own grid: cobalt beta; sparsegpt
    percdamp/blocksize; jsq-wo rho/clip_h; AWQ's scale-search resolution, weight-scale
    division and L1/L2 objective; SINQ's Sinkhorn iteration count and early stop; and
    SLiM's adapter type, cap-search resolution and survivor-only cap.  None of them
    changes what is stored, so every arm stays at the same effective bit budget."""
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
        model = bs.apply_awq_quantization(model, cal, bits, DEV,
                                          awq_num_betas=awq_num_betas,
                                          awq_use_weightscale=awq_use_weightscale,
                                          awq_l1=awq_l1)
    elif method == "sinq":
        model = bs.apply_sinq_quantization(model, cal, bits, DEV,
                                           sinq_order=sinq_order, sinq_stop=sinq_stop)
    elif method == "wanda":
        model = bs.apply_wanda_pruning(model, cal, float(sparsity), DEV,
                                       act_exp=wanda_act_exp, scope=wanda_scope)
    elif method == "sparsegpt":
        model = bs.apply_sparsegpt_pruning(model, cal, float(sparsity), bits, DEV,
                                           percdamp=float(percdamp), blocksize=int(blocksize))
    elif method == "jsq-wo":
        # JSQ weight-only: its two live knobs (rho, clip_h) are read from env by _apply_jsq.
        os.environ["JSQ_RHO"] = f"{float(jsq_rho):g}"
        os.environ["JSQ_CLIPH"] = f"{float(jsq_clip_h):g}"
        model = bs.apply_jsq_weightonly_quantization(model, cal, bits, float(sparsity), DEV)
    elif method == "slim":
        model = bs.apply_slim_quantization(model, cal, bits, float(sparsity), DEV,
                                           slim_lora=slim_lora, cap_bins=slim_cap_bins,
                                           sparse_aware_cap=slim_sparse_cap)
    elif method == "wanda-awq":
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV,
                                                awq_num_betas=awq_num_betas,
                                                awq_use_weightscale=awq_use_weightscale,
                                                awq_l1=awq_l1)
    elif method == "wanda-sinq":
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV,
                                                 sinq_order=sinq_order, sinq_stop=sinq_stop)
    elif method == "cobalt-sinq":
        # MASK-isolation arm (#36): CoBALT balanced mask + the BASELINE's SINQ quantizer (matched quant) ->
        # isolates the MASK from the quantizer. vs wanda-sinq = pure balance-vs-wanda under identical SINQ.
        # group_size is deliberately NOT passed: it inherits apply_wanda_sinq_quantization's 128, the SAME
        # default wanda-sinq gets, so the two arms cannot drift apart on the bit budget.
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV,
                                                 balanced=True, cobalt_beta=float(cobalt_beta),
                                                 sinq_order=sinq_order, sinq_stop=sinq_stop,
                                                 floor_frac=float(cobalt_floor_frac), protect=float(cobalt_protect), adapt=float(cobalt_adapt))
    elif method == "wanda-awq-repack":
        # math4 fairness: AWQ+Wanda with the repacking storage lever (group-128).
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV, repack=True)
    elif method == "wanda-sinq-repack":
        # math4 fairness: SINQ+Wanda with the repacking storage lever (group-128).
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV, repack=True)
    elif method == "wanda-awq-binc":
        # fairness: give the mask-agnostic bin-center survivor lever to the AWQ baseline
        # (its OWN awq normalization). bpw-identical to wanda-awq. See src/eout_quant.py.
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV, binc=True)
    elif method == "wanda-sinq-binc":
        # fairness: give the bin-center survivor lever to the SINQ baseline (its own dual norm).
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV, binc=True)
    elif method == "wanda-awq-awclip":
        # FAIRNESS (attempt-7): give the mask-agnostic awclip SCALE lever to AWQ (its own norm).
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV, awclip=True)
    elif method == "wanda-sinq-awclip":
        # FAIRNESS (attempt-7): give the awclip SCALE lever to SINQ (its own dual norm).
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV, awclip=True)
    elif method == "wanda-awq-awclipz":
        # FAIRNESS (attempt-7): give the mask-agnostic awclipz scale+zero lever to the AWQ
        # baseline on ITS OWN normalization. bpw-identical. The critic's required matched test.
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV, awclipz=True)
    elif method == "wanda-sinq-awclipz":
        # FAIRNESS (attempt-7): give the awclipz lever to the SINQ baseline (its own dual norm).
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV, awclipz=True)
    elif method in ("cobalt-seq", "cobalt-seqfix", "cobalt-seqfix-awclip"):
        # CROSS-LAYER v2 : sequential pass, inputs from the COMPRESSED prefix, per-block beta
        # chosen by the PROPAGATED block-output error (argmin over --seq-betas). cobalt-seqfix = same
        # sequential inputs at the single fixed --cobalt-beta (attribution: prefix compensation alone).
        import cobalt_seq as cs
        _betas = ([float(cobalt_beta)] if method.startswith("cobalt-seqfix")
                  else [float(b) for b in os.environ.get("SEQ_BETAS", "0,0.3,0.5,0.7,1").split(",")])
        _q = "awclip" if method.endswith("-awclip") else "rtn"
        model, _b, _r = cs.apply_cobalt_seq(model, cal, bits, float(sparsity), DEV, betas=_betas,
                                            group_size=(cobalt_group_size or 128), norm="col", quantizer=_q,
                                            log_path=os.path.join(_ROOT, "results", "cobalt_seq", "beta_schedule.tsv"),
                                            tag=f"{MODEL}\t{method}")
    elif method in ("cobalt-ec", "cobalt-ecfix"):
        # SALVAGE v2 : error-CORRECTING sequential CoBALT. Per block, ridge-fit W to map the
        # compressed-prefix input to the DENSE output (strength alpha), then the unchanged mask/OBS/quant.
        # cobalt-ec: alpha in {0,0.5,1} per block chosen by HELD-OUT (ptb) propagated error.
        # cobalt-ecfix: alpha=1 everywhere, no selection (attribution: is the held-out choice needed?).
        import cobalt_seq as cs
        _alphas = [float(a) for a in os.environ.get("EC_ALPHAS", "0,1").split(",")]
        _lams = [float(a) for a in os.environ.get("EC_LAMS", "1,3,10").split(",")]
        _ho = None
        if method == "cobalt-ec":
            _ho = bs.get_calibration_data(tok, n_samples=8, seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                                          dataset_key="ptb")
        else:
            _alphas = [1.0]
        model, _c, _r = cs.apply_cobalt_ec(model, cal, bits, float(sparsity), DEV, alphas=_alphas,
                                           betas=[float(cobalt_beta)], heldout_data=_ho,
                                           group_size=(cobalt_group_size or 128), norm="col", lams=_lams,
                                           log_path=os.path.join(_ROOT, "results", "cobalt_seq", "ec_schedule.tsv"),
                                           tag=f"{MODEL}\t{method}")
    elif method == "cobalt":
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "cobalt-blk32":
        # FIXED-CARDINALITY CoBALT (the served mask): identical recipe to `cobalt` except the single
        # global threshold is replaced by a per-32-column-block top-k, so every aligned block keeps
        # exactly 16 of 32 at sparsity 0.50. Same bpw as `cobalt` at the same bits/group size.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta),
                                          group_size=cobalt_group_size, mask_block=32)
    elif method == "cobalt-floor":
        # NOVEL MASK (attempt-11 #21): CoBALT balanced mask + HARD column-degree FLOOR that rescues the
        # OBS-uncompensable DEAD columns soft-beta leaves (MEASURED). IDENTICAL bpw to `cobalt` (global
        # survivor count held exactly constant by the promote/demote swap). One-shot, non-iterative.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced_floor", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          floor_frac=float(cobalt_floor_frac))
    elif method == "cobalt-cond":
        # NOVEL MASK (#28): non-separable conditioning-modulated column balance. Same bpw as cobalt.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced_cond", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "cobalt-perhead":
        # NOVEL MASK (#30): CoBALT balanced + per-HEAD balance on o_proj (orthogonal granularity). Same bpw.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced_perhead", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "wanda-span":
        # FAIRNESS CONTROL: spanning weight on plain per-row wanda (no col balance) + OBS + RTN.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="wanda_span", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method in ("cobalt-spectral", "cobalt-specblend"):
        # NOVEL MASK (#42): spectral-subspace-preserving survivor set (Jaccard 0.30 vs balanced = genuinely
        # distinct). specblend = geometric mean with magnitude importance. Same bpw as cobalt.
        mm = "balanced_spectral" if method == "cobalt-spectral" else "balanced_specblend"
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode=mm, dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "cobalt-span":
        # NOVEL MASK (#25): softer column balance (beta) + UNIQUE-variance spanning column weight for a
        # well-conditioned survivor set. Same bpw as cobalt. beta passed via --cobalt-beta (deploy 0.4).
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced_span", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "cobalt-exact":
        # NOVEL MASK (attempt-11 #18): EXACT doubly-degree-balanced support -- the HARD form of CoBALT's
        # column balance (keep-rate CV ~4x lower), one-shot greedy, retains magnitude. Same bpw as cobalt.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced_exact", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size)
    elif method == "wanda-floor":
        # S4 ATTRIBUTION control: the SAME column floor on plain per-row Wanda's mask/importance (no
        # column balance) -> isolates whether the FLOOR is a universal anti-starvation lever or needs
        # CoBALT's balanced base. Same bpw.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="wanda_floor", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          floor_frac=float(cobalt_floor_frac))
    elif method in ("cobalt-repack", "cobalt-eout"):
        # math4 arms: IDENTICAL pipeline to `cobalt` (same balanced mask, same OBS,
        # same col-norm, same group-RTN accounting) — only the survivor quantization
        # step differs, at strictly equal total bits (bitmap included).
        #   cobalt-repack: survivor-only codes at budget width b', min-max hull grids
        #   cobalt-eout:   measured-argmin selector incl. E_out code descent (>= never
        #                  worse than deployed RTN on the calibration objective)
        arm = "repack" if method == "cobalt-repack" else "eout"
        mlog = os.path.join(_ROOT, "results", "eout_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer=arm, margin_log=mlog,
                                          margin_ctx={"model": MODEL})
    elif method == "cobalt-compand":
        # NOVEL survivor quantizer: IDENTICAL CoBALT pipeline (same balanced mask,
        # same OBS, same col-norm, same uniform bit-width & group) — only the
        # survivor CODING changes to a power-companded grid fit to CoBALT's
        # post-OBS survivor shape (1 gamma/tensor; storage = same 2 vals/group as
        # RTN). Not repack (uniform b), not multiprecision. See src/eout_quant.py.
        mlog = os.path.join(_ROOT, "results", "compand_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer="compand", margin_log=mlog,
                                          margin_ctx={"model": MODEL})
    elif method == "cobalt-binc":
        # NOVEL survivor quantizer: IDENTICAL CoBALT pipeline (same balanced mask,
        # same OBS, same col-norm, same uniform bit-width & group) — only the survivor
        # CODING changes to a bin-CENTERED uniform grid (alpha-scaled hull) fit to
        # CoBALT's MEASURED near-uniform survivor law. Storage = same 2 vals/group as
        # RTN => bpw-identical to `cobalt`. Not repack, not multiprecision, not companding.
        mlog = os.path.join(_ROOT, "results", "binc_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer="binc", margin_log=mlog,
                                          margin_ctx={"model": MODEL})
    elif method == "cobalt-awclip":
        # NOVEL (attempt-7): IDENTICAL CoBALT pipeline (same balanced mask, OBS, col-norm,
        # uniform bit-width & group) — only the per-group SCALE is chosen by activation-
        # WEIGHTED output error instead of max-abs (the axis every prior arm left at RTN's
        # max-abs). Same dense (scale,zero) decoder => bpw-identical to `cobalt`. Grounded in
        # the measurement that CoBALT's mask keeps the group extreme in a low-||X|| column.
        mlog = os.path.join(_ROOT, "results", "awclip_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer="awclip", margin_log=mlog,
                                          margin_ctx={"model": MODEL})
    elif method == "cobalt-awclipz":
        # NOVEL (attempt-7): joint per-group (SCALE, ZERO) output-weighted grid — awclip + the
        # zero-point DOF. Same dense (scale,zero) decoder => bpw-identical to `cobalt`.
        mlog = os.path.join(_ROOT, "results", "awclipz_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer="awclipz", margin_log=mlog,
                                          margin_ctx={"model": MODEL})
    elif method in ("cobalt-gridmask", "cobalt-gridmask-awclip"):
        # NOVEL (fresh commission): the MASK is grid-aware. IDENTICAL CoBALT pipeline (same balanced
        # importance+beta, OBS, col-norm, uniform bit-width & group) EXCEPT the support is swapped by
        # a deterministic global GRID rule (nosink.grid_balanced_mask_and_obs): prune each group's
        # low-energy scale-setter, promote an interior high-energy candidate => tighter group-RTN
        # hull => finer step for group-mates. Global sparsity unchanged; NOT repack/multiprecision.
        # MEASURED CoBALT-specific + awclip-orthogonal at 2-BIT on gemma+tinyllama (src/probe_gridmask,
        # results/gridmask/); DO NOT use >=3-bit (survivor-loss cost exceeds granularity gain there).
        # -awclip variant stacks the awclip survivor SCALE lever on the grid support (orthogonality).
        arm = "awclip" if method == "cobalt-gridmask-awclip" else "rtn"
        mlog = os.path.join(_ROOT, "results", "gridmask_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="grid_balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer=arm, margin_log=mlog,
                                          margin_ctx={"model": MODEL})
    elif method == "cobalt-bincg":
        # ACTIVATION-GATED binc: per-group endpoint-vs-bincenter by the extreme survivor's column
        # activation (deterministic global rule; targets binc's regime-contingency). bpw == cobalt.
        mlog = os.path.join(_ROOT, "results", "bincg_margins", "margins.csv")
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(cobalt_beta), group_size=cobalt_group_size,
                                          quantizer="bincg", margin_log=mlog,
                                          margin_ctx={"model": MODEL})
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
            return _cobalt_balanced_prune_mask(W, H, float(sparsity), col_exp=float(cobalt_beta))
        model = bs.apply_sparsegpt_pruning(model, cal, float(sparsity), bits, DEV, mask_fn=_mfn,
                                           percdamp=percdamp, blocksize=blocksize)
    elif method == "sgptmask-cobalt":
        # CAUSAL DECOMP: SparseGPT-style OBS-saliency MASK fed into CoBALT's RECIPE
        # (one-shot prune-OBS + uncompensated final RTN). Isolates the mask.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="obs_saliency", dense_norm="col",
                                          mask_scope=mask_scope, obs_damp=obs_damp,
                                          group_size=cobalt_group_size)
    elif method == "cobalt-awq":
        # Generalization variant: balanced mask (tunable beta) + AWQ-style activation-aware
        # survivor quant ('acol' = mu_w^(1-a)/mu_x^a, same OBS activations, non-Sinkhorn).
        # Motivated by the beta-sweep (small beta helps EVERY arch's mask) + Qwen's grid
        # failure being the RTN quantizer, not the mask. See FINDINGS 'OPEN-IDEA RESULT'.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm=cobalt_norm,
                                          mask_mode="balanced", dense_norm=cobalt_norm,
                                          col_balance_exp=cobalt_beta, awq_alpha=float(awq_alpha),
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
    ap.add_argument("--cobalt-adapt", type=float, default=0.0,
                    help="cobalt-sinq: adaptive per-matrix beta from demotion signal (slope; #39)")
    ap.add_argument("--cobalt-protect", type=float, default=0.0,
                    help="cobalt-sinq: saliency-protected fraction of survivor budget (#38)")
    ap.add_argument("--cobalt-floor-frac", type=float, default=0.25,
                    help="cobalt-floor/wanda-floor: hard per-column survivor floor = frac*(1-sp)*K")
    ap.add_argument("--hp", default="", help="hp label written to the CSV (dispatcher-owned; e.g. 'beta=0.5')")
    ap.add_argument("--sgpt-percdamp", type=float, default=0.01, help="sparsegpt Hessian damping (tuned)")
    ap.add_argument("--sgpt-blocksize", type=int, default=128, help="sparsegpt OBS block width (tuned)")
    ap.add_argument("--jsq-rho", type=float, default=2.1, help="jsq-wo SAR prune<->quant bridge (tuned)")
    ap.add_argument("--jsq-clip-h", type=float, default=0.01, help="jsq-wo activation-clip fraction (tuned)")
    # --- baseline bpw-preserving knobs (matched sweep; see docstring of build_model) ---
    ap.add_argument("--awq-num-betas", type=int, default=1,
                    help="awq/wanda-awq: activation-STD exponent grid points (1 = beta search off)")
    ap.add_argument("--awq-no-weightscale", action="store_true",
                    help="awq/wanda-awq: do NOT divide the searched scale by the weight scale")
    ap.add_argument("--awq-l1", action="store_true",
                    help="awq/wanda-awq: L1 instead of L2 reconstruction error in the scale search")
    ap.add_argument("--sinq-order", type=int, default=16,
                    help="sinq/wanda-sinq: Sinkhorn balancing iterations")
    ap.add_argument("--sinq-no-stop", action="store_true",
                    help="sinq/wanda-sinq: keep iterating after the imbalance stops improving")
    ap.add_argument("--slim-naive", action="store_true",
                    help="slim: Naive-LoRA (plain error SVD) instead of saliency-weighted SLiM-LoRA")
    ap.add_argument("--slim-cap-bins", type=int, default=0,
                    help="slim: histogram bins for the MSE-optimal cap search (0 = upstream default)")
    ap.add_argument("--slim-sparse-cap", action="store_true",
                    help="slim: estimate the MSE-optimal cap over surviving weights only")
    ap.add_argument("--wanda-act-exp", type=float, default=0.5,
                    help="wanda: exponent on the activation norm (0.5 = the paper's sqrt)")
    ap.add_argument("--wanda-scope", default="row", choices=["row", "layer"],
                    help="wanda: comparison group the sparsity is enforced over")
    ap.add_argument("--cobalt-norm", default="acol",
                    help="survivor per-col scale for cobalt-awq: 'acol'=AWQ activation-aware, 'col'=weight-std")
    ap.add_argument("--awq-alpha", type=float, default=0.5,
                    help="cobalt-awq: activation exponent a in c = mu_w^(1-a)/mu_x^a (tuned knob, bpw-free)")
    ap.add_argument("--obs-damp", type=float, default=None,
                    help="sgptmask-cobalt: OBS Hessian damping as a FRACTION of mean(diag H) "
                         "(the analog of SparseGPT's percdamp; None = legacy adaptive 1%% with a 1e-2 floor)")
    ap.add_argument("--mask-scope", default="global", choices=["global", "per_row"],
                    help="sgptmask-cobalt: OBS-saliency threshold scope (bpw-identical -- same total survivors)")
    ap.add_argument("--cobalt-per-row", action="store_true",
                    help="cobalt-awq: use the GENTLE balance variant (single per-col reweight + exact "
                         "per-row threshold; unimpeachably non-Sinkhorn) instead of global-topk balance")
    ap.add_argument("--nm", default=None,
                    help="N:M semi-structured sparsity, e.g. '2:4' (forces sparsity=1-n/m; routes every "
                         "arm's mask through top-n-per-m selection; hp label gets 'nm=N:M')")
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
    if args.nm:
        n_, m_ = (int(v) for v in args.nm.split(":"))
        ns.NM_PATTERN = (n_, m_); bs.NM_PATTERN = (n_, m_)
        args.sparsity = 1.0 - n_ / m_
        args.hp = f"nm={args.nm}" + (f",{args.hp}" if args.hp else "")
        print(f"[cell] --nm {args.nm}: N:M masks for all arms, sparsity={args.sparsity:g}", flush=True)
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
                            jsq_rho=args.jsq_rho, jsq_clip_h=args.jsq_clip_h,
                            cobalt_floor_frac=args.cobalt_floor_frac, cobalt_protect=args.cobalt_protect, cobalt_adapt=args.cobalt_adapt,
                            awq_num_betas=args.awq_num_betas,
                            awq_use_weightscale=not args.awq_no_weightscale,
                            awq_l1=args.awq_l1,
                            sinq_order=args.sinq_order, sinq_stop=not args.sinq_no_stop,
                            slim_lora=not args.slim_naive,
                            slim_cap_bins=(args.slim_cap_bins or None),
                            slim_sparse_cap=args.slim_sparse_cap,
                            wanda_act_exp=args.wanda_act_exp, wanda_scope=args.wanda_scope,
                            awq_alpha=args.awq_alpha, obs_damp=args.obs_damp,
                            mask_scope=args.mask_scope)
    except Exception as e:
        if _is_transient(e):
            raise                                    # OOM = slice contention -> requeue, do NOT settle
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
            if _is_transient(e):
                raise                                # OOM = slice contention -> requeue, do NOT settle
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
            if _is_transient(e):
                raise                                # OOM = slice contention -> requeue, do NOT settle
            dt = time.time() - t0
            append_rows(args.csv, [_row(method, bits, sparsity, task, "accuracy", "",
                                        seconds=dt, error=f"FAILED:{type(e).__name__}: {e}", hp=args.hp)])
            print(f"[FAIL] {tag} task={task}: {type(e).__name__}: {e}", flush=True)
        model.to(DEV)  # defensive: some adapters can move the model
        gc.collect(); torch.cuda.empty_cache()

    print(f"CELL_DONE {tag}", flush=True)


if __name__ == "__main__":
    main()
