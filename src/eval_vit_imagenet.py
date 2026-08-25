#!/usr/bin/env python3
"""ImageNet-1k top-1 downstream confirmation of the ViT-large CoBALT (column-balance) lead (2026-08-16).

The ViT screen (src/diag_vit_screen.py) found google/vit-large-patch16-224 sp0.5 is WIN_PREDICTED=Y on
the residual-stream relerr PROXY (wanda 1.125 COLLAPSE -> balanced-b0.5 0.837 RESCUED). IRON LAW:
reconstruction != downstream. This confirms/refutes it on the real metric: PRUNE-ONLY ImageNet-1k top-1,
balanced mask vs wanda mask, fp16 survivors, NO quant. Isolates CoBALT's actual lever (the MASK).

Reuses the EXACT mask/activation math from diag_vit_screen (wanda = per-row Wanda; balanced = row+col
quantile self-norm at strength beta). Calibration = in-distribution ImageNet val images (disjoint from the
eval subset). Dataset = benjamin-paine/imagenet-1k-256x256 val (50k, standard 1000-class label order,
verified aligned to the model head: dense top-1 ~79%)."""
import os, sys, io, argparse, glob
import torch
from PIL import Image
import pyarrow.parquet as pq

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from diag_vit_screen import (iter_targets, collect_activations, apply_scheme, vit_blocks)  # exact same math

DEV = "cuda"
VAL_GLOB = "/scratch/ckp908/prism_hf/hub/datasets--benjamin-paine--imagenet-1k-256x256/snapshots/*/data/validation-*.parquet"


def load_val():
    files = sorted(glob.glob(VAL_GLOB))
    assert files, f"no val parquet found at {VAL_GLOB}"
    imgs, labs = [], []
    for f in files:
        t = pq.ParquetFile(f).read()
        imgs += [d["bytes"] for d in t.column("image").to_pylist()]
        labs += t.column("label").to_pylist()
    return imgs, labs


def stratified(labs, k, exclude):
    """k indices per class (all if k<=0), skipping `exclude` indices."""
    by = {}
    for i, l in enumerate(labs):
        if i in exclude:
            continue
        by.setdefault(l, []).append(i)
    out = []
    for l in sorted(by):
        out += by[l] if k <= 0 else by[l][:k]
    return out


def decode(imgs, idxs):
    return [Image.open(io.BytesIO(imgs[i])).convert("RGB") for i in idxs]


def preprocess_all(proc, imgs, idxs, bs, cache_path=None):
    """Preprocess eval images ONCE (CPU-bound) into cached fp16 batches — reused across all schemes.

    The eval set is identical across every benchmark job for a given (model, n_eval, n_calib, bs),
    so `cache_path` (optional) lets separate jobs share the expensive decode/resize via disk: the
    first job computes and atomically writes it, all others load in seconds. Caching is best-effort —
    any read/write problem silently falls back to recomputing, so it can never fail the run."""
    idxs = list(idxs)
    if cache_path and os.path.exists(cache_path):
        try:
            obj = torch.load(cache_path, map_location="cpu")
            if obj.get("idxs") == idxs:
                return obj["batches"]
        except Exception:
            pass  # corrupt/partial/mismatched cache -> recompute below
    batches = []
    for i in range(0, len(idxs), bs):
        sub = idxs[i:i + bs]
        pv = proc(decode(imgs, sub), return_tensors="pt")["pixel_values"].half().cpu()
        batches.append((pv, sub))
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = f"{cache_path}.tmp.{os.getpid()}"
            torch.save({"idxs": idxs, "batches": batches}, tmp)
            os.replace(tmp, cache_path)  # atomic; concurrent first-wave writers are all identical
        except Exception:
            pass  # best-effort cache; never fail the run over it
    return batches


def evaluate(model, batches, labs):
    correct = n = 0
    for pv, sub in batches:
        with torch.no_grad():
            pred = model(pv.to(DEV)).logits.argmax(-1).cpu().tolist()
        correct += sum(1 for k, p in enumerate(pred) if p == labs[sub[k]])
        n += len(sub)
    return 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/vit-large-patch16-224")
    ap.add_argument("--sparsity", default="0.5", help="comma list of sparsities to sweep")
    ap.add_argument("--beta", default="0.5", help="comma list of balance strengths to sweep")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=5, help="images/class for eval (<=0 = full 50k)")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    sparsities = [float(s) for s in str(args.sparsity).split(",")]
    betas = [float(b) for b in str(args.beta).split(",")]

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

    imgs, labs = load_val()
    calib_idx = list(range(args.n_calib))                    # in-distribution calibration, disjoint from eval
    eval_idx = stratified(labs, args.n_eval, set(calib_idx))
    print(f"# model={args.model} sparsity={sparsities} beta={betas} blocks={len(vit_blocks(model))} "
          f"n_calib={len(calib_idx)} n_eval={len(eval_idx)}", flush=True)

    pv = proc(decode(imgs, calib_idx), return_tensors="pt")["pixel_values"].to(DEV).half()
    acts = collect_activations(model, pv)
    backup = {(li, ap): mod.weight.data.clone().cpu() for li, ap, mod in iter_targets(model)}

    batches = preprocess_all(proc, imgs, eval_idx, args.bs)   # preprocess ONCE, reuse across ALL cells
    print(f"# preprocessed {len(eval_idx)} eval images into {len(batches)} batches", flush=True)

    def restore():
        for li, ap, mod in iter_targets(model):
            mod.weight.data = backup[(li, ap)].to(DEV).to(mod.weight.dtype)

    def run(scheme, sp, beta):
        restore()
        if scheme == "wanda":
            apply_scheme(model, acts, sp, "wanda", 0.0)
        elif scheme == "balanced":
            apply_scheme(model, acts, sp, "balanced", beta)
        return evaluate(model, batches, labs)

    print(f"{'scheme':16s} {'sparsity':>8s} {'top1%':>8s}", flush=True)
    print(f"{'dense':16s} {0.0:8.2f} {run('dense', 0.0, 0.0):8.2f}", flush=True)
    for sp in sparsities:
        print(f"{'wanda':16s} {sp:8.2f} {run('wanda', sp, 0.0):8.2f}", flush=True)
        for beta in betas:
            print(f"{'balanced_b'+str(beta):16s} {sp:8.2f} {run('balanced', sp, beta):8.2f}", flush=True)
    restore()


if __name__ == "__main__":
    main()
