#!/usr/bin/env python3
"""ViT COLLAPSE+RESCUE screen — does CoBALT's gemma-like win regime exist off-LLM? (2026-08-16)

CoBALT (column-balance) wins downstream IFF a model (1) COLLAPSES under Wanda pruning — interior
residual-stream relerr crosses 1.0 (phase transition, validated across 11 LLMs) — AND (2) that
collapse is BALANCE-RESCUABLE (balanced mask pushes interior relerr back <1). 0/13 non-gemma LLMs
qualified. This screens VISION transformers, the best non-LLM bet (documented massive-activation /
LayerNorm-outlier pathology = the column-concentration ingredient rescuability needs).

MASK-ONLY, PRUNE-ONLY (fp16 survivors, NO quant/OBS) so the MASK alone is measured — identical
protocol to src/diag_beta_sweep.py / diag_prop_variants.py (leg-c). Reuses the exact relerr + mask
math. ViT-specific: block/linear accessors + IMAGE calibration. HF ViTs have SEPARATE q/k/v (not
fused) so the per-matrix pipeline applies cleanly.

Reports per (sparsity, scheme): interior max relerr + pruned-vs-dense top-1 agreement on the calib
images (behavioral collapse proxy, ViT analog of prune-only PPL). Verdict: COLLAPSE / RESCUABLE /
WIN_PREDICTED. relerr RANKS/PREDICTS here; a real claim still needs downstream (ImageNet), only run
if WIN_PREDICTED=Y (rule #2)."""
import os, sys, argparse
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"

# linear paths within one encoder block. HF ViT/DeiT/BEiT share the first layout; CLIP vision uses the second.
# DINOv2 shares the attention layout but names its MLP mlp.fc1/mlp.fc2 (union below; iter_targets skips
# paths absent in a given block via hasattr, so one list serves vit/deit/beit AND dinov2).
VIT_PATHS = ["attention.attention.query", "attention.attention.key", "attention.attention.value",
             "attention.output.dense", "intermediate.dense", "output.dense",
             "mlp.fc1", "mlp.fc2"]
CLIP_PATHS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
              "mlp.fc1", "mlp.fc2"]


def paths_for(model):
    return CLIP_PATHS if hasattr(model, "vision_model") else VIT_PATHS


# ---- reused relerr + mask math (identical to diag_beta_sweep) ----
def interior_relerr(h_dense, h_var):
    re = [((h_var[l] - h_dense[l]).norm().item() / (h_dense[l].norm().item() + 1e-20))
          for l in range(len(h_dense))]
    return max(re[:-1]) if len(re) > 1 else max(re)


def _thr(imp, sp, scope):
    K, N = imp.shape
    if scope == 'per_row':
        thr = torch.kthvalue(imp, int(N * sp), dim=1, keepdim=True).values
        return (imp > thr).float()
    thr = torch.kthvalue(imp.reshape(-1), int(K * N * sp)).values
    return (imp.reshape(-1) > thr).view(K, N).float()


def mask_for(W, anorm, sp, kind, beta):
    imp = W.abs() * anorm.view(1, -1)
    if kind == "wanda":
        return _thr(imp, sp, 'per_row')
    K, N = W.shape
    kr, kc = int(N * sp), int(K * sp)
    if kr > 0:
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
    if beta > 0 and kc > 0:
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(beta)
    return _thr(imp, sp, 'global')


# ---- ViT accessors ----
def vit_blocks(model):
    # ViTForImageClassification -> model.vit.encoder.layer ; DeiT -> model.deit... ; generic fallback
    for attr in ("vit", "deit", "beit", "dinov2"):
        if hasattr(model, attr):
            return getattr(model, attr).encoder.layer
    if hasattr(model, "vision_model"):   # CLIP vision tower
        return model.vision_model.encoder.layers
    if hasattr(model, "encoder"):
        return model.encoder.layer
    raise RuntimeError("cannot find ViT blocks")


def iter_targets(model):
    blocks = vit_blocks(model)
    plist = paths_for(model)
    for li, blk in enumerate(blocks):
        for ap in plist:
            mod = blk; ok = True
            for p in ap.split('.'):
                if not hasattr(mod, p):
                    ok = False; break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                yield li, ap, mod


def capture_hidden(model, pv):
    caps = {}; hooks = []
    blocks = vit_blocks(model)
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[i] = o.detach().float().cpu()
        return h
    for i, l in enumerate(blocks):
        hooks.append(l.register_forward_hook(mk(i)))
    model.eval()
    with torch.no_grad():
        model(pv)
    for h in hooks:
        h.remove()
    return [caps[i] for i in range(len(blocks))]


def collect_activations(model, pv):
    """Per-linear INPUT activations (Wanda ||X|| per input channel), on CPU. One forward pass."""
    cache = {}; hooks = []
    def mk(name):
        def h(mod, inp, out):
            x = inp[0].detach()
            cache.setdefault(name, []).append(x.reshape(-1, x.shape[-1]).cpu())
        return h
    for li, ap, mod in iter_targets(model):
        hooks.append(mod.register_forward_hook(mk(f'{li}.{ap}')))
    model.eval()
    with torch.no_grad():
        model(pv)
    for h in hooks:
        h.remove()
    return {k: torch.cat(v, 0) for k, v in cache.items()}


def top1(model, pv):
    with torch.no_grad():
        out = model(pv)
    logits = getattr(out, "logits", None)
    if logits is None:
        return None                       # vision-only model (CLIP): no classification head
    return logits.argmax(-1).cpu()


def apply_scheme(model, acts, sp, kind, beta):
    for li, ap, mod in iter_targets(model):
        X = acts.get(f'{li}.{ap}')
        if X is None:
            continue
        W = mod.weight.data.float().to(DEV)
        Xd = X.to(DEV).float()
        Xd = Xd[:min(Xd.shape[0], 512)]
        anorm = torch.norm(Xd, dim=0)
        m = mask_for(W, anorm, sp, kind, beta)
        mod.weight.data = (W * m).to(mod.weight.dtype)
        del Xd
    torch.cuda.empty_cache()


def get_images(n):
    from datasets import load_dataset
    from transformers import AutoImageProcessor
    ds = load_dataset("zh-plus/tiny-imagenet", split=f"valid[:{n}]")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--n-img", type=int, default=32)
    ap.add_argument("--sparsities", default="0.5,0.6,0.7,0.8")
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    if "clip" in args.model.lower():
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV)
    else:
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV)
    ds = get_images(args.n_img)
    imgs = [im.convert("RGB") for im in ds["image"]]
    pv = proc(imgs, return_tensors="pt")["pixel_values"].to(DEV).half()
    print(f"# model={args.model} n_img={pv.shape[0]} blocks={len(vit_blocks(model))} beta={args.beta}", flush=True)

    h_dense = capture_hidden(model, pv)
    pred_dense = top1(model, pv)
    acts = collect_activations(model, pv)
    backup = {(li, ap): mod.weight.data.clone().cpu() for li, ap, mod in iter_targets(model)}

    print(f"{'sparsity':>8s} {'scheme':16s} {'interior_relerr':>15s} {'top1_agree%':>11s}  verdict", flush=True)
    for sp in [float(s) for s in args.sparsities.split(",")]:
        row = {}
        for kind, beta in [("wanda", 0.0), ("balanced", args.beta)]:
            for li, ap, mod in iter_targets(model):   # restore
                mod.weight.data = backup[(li, ap)].to(DEV).to(mod.weight.dtype)
            apply_scheme(model, acts, sp, kind, beta)
            rel = interior_relerr(h_dense, capture_hidden(model, pv))
            if pred_dense is None:
                agree = float("nan")
            else:
                agree = (top1(model, pv) == pred_dense).float().mean().item() * 100
            row[kind] = rel
            tag = "wanda" if kind == "wanda" else f"balanced_b{beta}"
            print(f"{sp:8.2f} {tag:16s} {rel:15.3f} {agree:11.1f}", flush=True)
        collapse = row["wanda"] > 1.0
        rescuable = row["balanced"] < 1.0
        win = collapse and rescuable
        print(f"{sp:8.2f} {'VERDICT':16s} {'':15s} {'':11s}  "
              f"COLLAPSE={'Y' if collapse else 'N'} RESCUABLE={'Y' if rescuable else 'N'} "
              f"WIN_PREDICTED={'*** Y ***' if win else 'N'}", flush=True)
    # restore
    for li, ap, mod in iter_targets(model):
        mod.weight.data = backup[(li, ap)].to(DEV).to(mod.weight.dtype)


if __name__ == "__main__":
    main()
