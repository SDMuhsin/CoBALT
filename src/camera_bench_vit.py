#!/usr/bin/env python3
"""Camera-ready benchmark — ViT-large / ImageNet-1k top-1 (2026-08-16).

The ViT analog of src/camera_bench.py: the SAME method grid the gemma-2b camera-ready used
{fp16,awq,sinq,wanda,sparsegpt,jsq-wo,slim,wanda-awq,wanda-sinq,cobalt} × sparsity {0.5..0.9} at 3-bit,
now on `google/vit-large-patch16-224`, scored on ImageNet-1k top-1 instead of wiki PPL + LLM downstream.

FIDELITY: every baseline runs its ORIGINAL benchmark_suite algorithm unchanged — they are all
architecture-agnostic (iterate get_transformer_layers/get_layer_paths with forward hooks + model(batch),
NO LLM-specific sequential harness, verified). We only teach `bs`'s ~6 structural accessors about ViT
(monkeypatched below), so the LLM benchmark file is untouched. CoBALT = ns.apply_wanda_obs_rtn (balanced
mask). Calibration = in-distribution ImageNet-val images (disjoint from eval). All quant arms share the
group-RTN backbone at group-128 (matched bpw). ONE process: preprocess eval images ONCE, loop all cells,
upsert one CSV row per cell (resumable — re-run skips finished cells)."""
import os, sys, argparse, csv, fcntl, gc, time
from datetime import datetime, timezone
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import benchmark_suite as bs   # noqa: E402
import nosink as ns            # noqa: E402
from eval_vit_imagenet import load_val, stratified, decode, preprocess_all, evaluate  # noqa: E402

DEV = "cuda"
VIT_PATHS = ["attention.attention.query", "attention.attention.key", "attention.attention.value",
             "attention.output.dense", "intermediate.dense", "output.dense",
             "mlp.fc1", "mlp.fc2"]   # last two = DINOv2 MLP (absent paths skipped per-block)


# --------------------------------------------------------------- ViT compat shim for benchmark_suite
_VIT_LIKE = ("vit", "deit", "beit", "dinov2")   # HF image encoders with .{attr}.encoder.layer (dinov2 MLP=mlp.fc*)


def _backbone_attr(model):
    """Return the encoder backbone attr name ('vit'/'deit'/'beit') for a ViT-like model, else None."""
    for attr in _VIT_LIKE:
        if hasattr(model, attr) and hasattr(getattr(model, attr), "encoder"):
            return attr
    return None


def _is_vit(model):
    mt = getattr(getattr(model, "config", None), "model_type", "").lower()
    return mt in _VIT_LIKE and _backbone_attr(model) is not None


def install_vit_compat():
    """Teach bs's structural accessors about ViT (model.vit.encoder.layer / layernorm / classifier).
    All bs.apply_* + ns.apply_wanda_obs_rtn route through these, so the original algorithms then run
    on ViT unchanged. LLM behaviour is preserved (ViT branch is guarded by model_type=='vit')."""
    o_paths, o_layers, o_setl = bs.get_layer_paths, bs.get_transformer_layers, bs.set_transformer_layer
    o_fln, o_emb, o_fin = bs.get_final_layernorm, bs.move_embed_to_device, bs.move_final_layers_to_device

    def get_layer_paths(model):
        return VIT_PATHS if _is_vit(model) else o_paths(model)

    def get_transformer_layers(model):
        a = _backbone_attr(model)
        return getattr(model, a).encoder.layer if a else o_layers(model)

    def set_transformer_layer(model, i, layer):
        a = _backbone_attr(model)
        if a:
            getattr(model, a).encoder.layer[i] = layer
        else:
            o_setl(model, i, layer)

    def get_final_layernorm(model):
        a = _backbone_attr(model)
        if not a:
            return o_fln(model)
        bb = getattr(model, a)
        # BEiT mean-pooling puts the used norm in pooler.layernorm (bb.layernorm is Identity)
        pooler = getattr(bb, "pooler", None)
        if pooler is not None and hasattr(pooler, "layernorm"):
            return pooler.layernorm
        return bb.layernorm

    def move_embed_to_device(model, device):
        a = _backbone_attr(model)
        if a:
            getattr(model, a).embeddings.to(device)
        else:
            o_emb(model, device)

    def move_final_layers_to_device(model, device):
        a = _backbone_attr(model)
        if a:
            bb = getattr(model, a)
            bb.layernorm.to(device)
            if getattr(bb, "pooler", None) is not None:
                bb.pooler.to(device)
            model.classifier.to(device)
        else:
            o_fin(model, device)

    for name, fn in [("get_layer_paths", get_layer_paths), ("get_transformer_layers", get_transformer_layers),
                     ("set_transformer_layer", set_transformer_layer), ("get_final_layernorm", get_final_layernorm),
                     ("move_embed_to_device", move_embed_to_device),
                     ("move_final_layers_to_device", move_final_layers_to_device)]:
        setattr(bs, name, fn)


# --------------------------------------------------------------- method grid (mirror camera_bench.py)
PRUNE_METHODS = {"wanda", "sparsegpt", "jsq-wo", "slim", "wanda-awq", "wanda-sinq", "cobalt",
                 "cobaltmask-sgpt", "sgptmask-cobalt"}
FP16_WEIGHT_METHODS = {"fp16", "wanda"}          # weights stay fp16 -> bits recorded as 16
ALL_METHODS = ["fp16", "awq", "sinq", "wanda", "sparsegpt", "jsq-wo", "slim",
               "wanda-awq", "wanda-sinq", "cobalt"]
CSV_FIELDS = ["timestamp", "model", "method", "bits", "sparsity", "hp", "task", "metric",
              "value", "correct", "total", "seconds", "error"]


def _cobalt_balanced_prune_mask(W, H, sparsity, col_exp=0.5):
    """CoBALT balanced KEEP-mask (row_fair quantile + col_exp quantile + global top-k),
    computed from W and the SparseGPT Hessian H (act_norm ∝ sqrt(diag(H)), the global
    constant cancels in every quantile/threshold op). Returns a bool PRUNE mask
    (True = pruned) matching SparseGPT's fixed_mask convention."""
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


def build_model(model_id, method, sparsity, bits, cal, group_size,
                col_balance_exp=0.5, percdamp=0.01, blocksize=128,
                jsq_rho=2.1, jsq_clip_h=0.01):
    from transformers import AutoModelForImageClassification
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=torch.float16).to(DEV)
    sbt = {ns.type_of(p): float(sparsity) for p in VIT_PATHS}   # uniform sparsity over ViT linear types

    if method == "fp16":
        pass
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
        # JSQ weight-only (bit-matched). Its two live knobs (rho, clip_h) are read from env by
        # _apply_jsq; set them per-cell here. Pixel-value cal passes straight through bs.apply_*'s
        # architecture-agnostic forward-hook path (same as every other ViT arm).
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
                                          col_balance_exp=float(col_balance_exp), group_size=group_size)
    elif method == "cobaltmask-sgpt":
        # CAUSAL DECOMP: CoBALT's balanced MASK fed into SparseGPT's joint (quant-in-loop)
        # OBS RECIPE. Isolates the recipe by swapping only the mask.
        def _mfn(attr_path, W, H):
            return _cobalt_balanced_prune_mask(W, H, float(sparsity), col_exp=0.5)
        model = bs.apply_sparsegpt_pruning(model, cal, float(sparsity), bits, DEV, mask_fn=_mfn)
    elif method == "sgptmask-cobalt":
        # CAUSAL DECOMP: SparseGPT-style OBS-saliency MASK fed into CoBALT's RECIPE
        # (prune-only OBS + uncompensated final RTN). Isolates the mask.
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="obs_saliency", dense_norm="col",
                                          group_size=group_size)
    else:
        raise ValueError(method)
    bs.move_final_layers_to_device(model, DEV)
    model.eval()
    return model


# --------------------------------------------------------------- CSV upsert (mirror camera_bench.py)
def read_done(csv_path, model_id, method, bits, sparsity, hp):
    if not os.path.exists(csv_path):
        return False
    sp_s, b_s = f"{float(sparsity):.2f}", str(int(bits))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if (r.get("model") == model_id and r.get("method") == method and r.get("bits") == b_s
                    and r.get("sparsity") == sp_s and (r.get("hp") or "") == hp
                    and r.get("task") == "imagenet"
                    and not (r.get("error") or "").strip()):
                return True
    return False


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            has_header = f.readline().startswith("timestamp")
            f.seek(0, 2)
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not has_header:
                w.writeheader()
            w.writerow(row)
            f.flush(); os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/vit-large-patch16-224")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--sparsities", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128, help="RTN group size (128 matches AWQ baseline bpw)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=0, help="images/class (<=0 = full 50k)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "benchmark_camera_vit", "results.csv"))
    # ---- bit-MATCHED (bpw-preserving) hyperparameter grids (comma lists; mirror camera_bench_glue) ----
    ap.add_argument("--col-balance-exps", default="0.5", help="cobalt beta grid, comma-sep")
    ap.add_argument("--sgpt-percdamps", default="0.01", help="sparsegpt percdamp grid, comma-sep")
    ap.add_argument("--sgpt-blocksizes", default="128", help="sparsegpt blocksize grid, comma-sep")
    ap.add_argument("--jsq-rhos", default="2.1", help="jsq-wo rho grid, comma-sep")
    ap.add_argument("--jsq-cliphs", default="0.01", help="jsq-wo clip_h grid, comma-sep")
    args = ap.parse_args()

    install_vit_compat()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sparsities = [float(s) for s in args.sparsities.split(",")]
    betas = [float(x) for x in args.col_balance_exps.split(",") if x.strip()]
    percdamps = [float(x) for x in args.sgpt_percdamps.split(",") if x.strip()]
    blocksizes = [int(x) for x in args.sgpt_blocksizes.split(",") if x.strip()]
    jsq_rhos = [float(x) for x in args.jsq_rhos.split(",") if x.strip()]
    jsq_cliphs = [float(x) for x in args.jsq_cliphs.split(",") if x.strip()]

    def hp_variants(method):
        """Expand a method into its (hp_label, build_model_kwargs) grid. Only bpw-preserving knobs
        are swept; wanda-awq/wanda-sinq/slim and the decomp diagnostics have no exposed matched
        knob (single 'default')."""
        if method == "cobalt":
            return [(f"beta={b:g}", {"col_balance_exp": b}) for b in betas]
        if method == "sparsegpt":
            return [(f"pd={pd:g},bs={blk}", {"percdamp": pd, "blocksize": blk})
                    for pd in percdamps for blk in blocksizes]
        if method == "jsq-wo":
            return [(f"rho={r:g},clip={c:g}", {"jsq_rho": r, "jsq_clip_h": c})
                    for r in jsq_rhos for c in jsq_cliphs]
        return [("default", {})]

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    imgs, labs = load_val()
    calib_idx = list(range(args.n_calib))
    eval_idx = stratified(labs, args.n_eval, set(calib_idx))
    cal = proc(decode(imgs, calib_idx), return_tensors="pt")["pixel_values"].half()   # [N,3,224,224]
    print(f"# model={args.model} methods={methods} sparsities={sparsities} bits={args.bits} "
          f"gsize={args.group_size} n_calib={len(calib_idx)} n_eval={len(eval_idx)}", flush=True)

    # Cross-job cache: the eval set is identical for a given (model, n_eval, n_calib, bs), so preprocess
    # it ONCE to disk and let the ~140 separate suite jobs share it (avoids re-decoding 10k images each).
    _msafe = args.model.replace("/", "_")
    cache_dir = os.environ.get("VIT_EVAL_CACHE", "/scratch/root/PTQResearch/cache/vit_eval")
    cache_path = os.path.join(cache_dir, f"{_msafe}_nc{args.n_calib}_ne{args.n_eval}_bs{args.bs}.pt")
    batches = preprocess_all(proc, imgs, eval_idx, args.bs, cache_path=cache_path)  # preprocess ONCE, reuse across all cells + jobs
    print(f"# preprocessed {len(eval_idx)} eval images into {len(batches)} batches "
          f"(cache={cache_path})", flush=True)

    # enumerate cells: non-prune methods are flat (sparsity 0, once); prune methods at each sparsity;
    # each method further expands over its bpw-preserving hp grid (single 'default' if none exposed).
    cells = []   # (method, sparsity, hp_label, hp_kwargs)
    for m in methods:
        sps = sparsities if m in PRUNE_METHODS else [0.0]
        for sp in sps:
            for hp_label, hp_kw in hp_variants(m):
                cells.append((m, sp, hp_label, hp_kw))

    for method, sp, hp_label, hp_kw in cells:
        sparsity = sp if method in PRUNE_METHODS else 0.0
        bits = 16 if method in FP16_WEIGHT_METHODS else args.bits
        tag = f"{method} sp={sparsity:.2f} bits={bits} hp={hp_label}"
        if read_done(args.csv, args.model, method, bits, sparsity, hp_label):
            print(f"[skip] {tag} (cached)", flush=True); continue
        t0 = time.time()
        try:
            model = build_model(args.model, method, sparsity, bits, cal, args.group_size, **hp_kw)
            tb = time.time() - t0
            acc = evaluate(model, batches, labs)
            dt = time.time() - t0
            append_row(args.csv, {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "model": args.model, "method": method, "bits": str(bits),
                                  "sparsity": f"{sparsity:.2f}", "hp": hp_label,
                                  "task": "imagenet", "metric": "top1",
                                  "value": f"{acc:.4f}", "total": len(eval_idx),
                                  "seconds": f"{dt:.1f}"})
            print(f"RESULT {tag} top1={acc:.4f} (build {tb:.0f}s eval {dt-tb:.0f}s)", flush=True)
            del model
        except Exception as e:
            dt = time.time() - t0
            append_row(args.csv, {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "model": args.model, "method": method, "bits": str(bits),
                                  "sparsity": f"{sparsity:.2f}", "hp": hp_label,
                                  "task": "imagenet", "metric": "top1",
                                  "value": "", "seconds": f"{dt:.1f}",
                                  "error": f"FAILED:{type(e).__name__}: {e}"})
            print(f"[FAIL] {tag}: {type(e).__name__}: {e}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
