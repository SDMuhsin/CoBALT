#!/usr/bin/env python3
"""OBJECTIVE-DIRECTED SCREEN: find a NON-GEMMA model where CoBALT should WIN. A CoBALT win requires BOTH
(a) the naive per-row Wanda prune COLLAPSES (interior max relerr > 1 — error exceeds signal mid-stack; the
rescue regime that gemma has and Llama/Qwen/StarCoder2 lack) AND (b) that collapse is BALANCE-RESCUABLE
(the column-balanced mask pushes interior relerr back < 1, as on the gemma lineage but NOT on pythia).

One model load: capture dense residual hidden states, back up target-linear weights on CPU, then for each of
{wanda per-row, balanced β=0.5} (mask-only, NO OBS) apply the mask, re-measure hidden states, restore. Report
interior (non-final-layer) max relerr for each variant + prune-only wiki PPL (wanda), and the verdict:
  COLLAPSE = wanda interior relerr > 1 ; RESCUABLE = balanced interior relerr < 1 ;
  WIN_PREDICTED = COLLAPSE and RESCUABLE.
Screens conditions (a)+(b) only; a passing candidate still needs a grid cell to confirm the end-to-end win
(and that the survivor RTN quant survives at 3-bit — the axis that sank Qwen)."""
import os, sys, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
import nosink as ns  # noqa

DEV = "cuda"


def capture_hidden(model, batch):
    caps = {}; hooks = []
    layers = bs.get_transformer_layers(model)
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[i] = o.detach().float().cpu()
        return h
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_hook(mk(i)))
    model.eval()
    with torch.no_grad():
        model(batch)
    for h in hooks:
        h.remove()
    return [caps[i] for i in range(len(layers))]


def interior_relerr(h_dense, h_var):
    re = [((h_var[l] - h_dense[l]).norm().item() / (h_dense[l].norm().item() + 1e-20))
          for l in range(len(h_dense))]
    return max(re[:-1]) if len(re) > 1 else max(re), re[-1]


def _thr(imp, sp, scope):
    K, N = imp.shape
    if scope == 'per_row':
        thr = torch.kthvalue(imp, int(N * sp), dim=1, keepdim=True).values
        return (imp > thr).float()
    thr = torch.kthvalue(imp.view(-1), int(K * N * sp)).values
    return (imp.view(-1) > thr).view(K, N).float()


def mask_for(W, anorm, sp, kind):
    imp = W.abs() * anorm.view(1, -1)
    if kind == "wanda":
        return _thr(imp, sp, 'per_row')
    K, N = W.shape
    kr, kc = int(N * sp), int(K * sp)
    if kr > 0:
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
    if kc > 0:
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(0.5)
    return _thr(imp, sp, 'global')


def iter_targets(model):
    layers = ns.get_layers(model)
    paths = bs.get_layer_paths(model)
    for li in range(len(layers)):
        for ap in paths:
            mod = layers[li]; ok = True
            for p in ap.split('.'):
                if not hasattr(mod, p): ok = False; break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                yield li, ap, mod


def apply_masks(model, acts, sp, kind):
    for li, ap, mod in iter_targets(model):
        X = acts.get(f'layer_{li}.{ap}')
        if X is None:
            continue
        W = mod.weight.data.float().to(DEV)
        Xd = X.to(DEV)
        if Xd.dim() == 3:
            Xd = Xd.reshape(-1, Xd.shape[-1])
        Xd = Xd[:min(Xd.shape[0], 256)]
        anorm = torch.norm(Xd.float(), dim=0)
        m = mask_for(W, anorm, sp, kind)
        mod.weight.data = (W * m).to(mod.weight.dtype)
        del Xd
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--n-ppl", type=int, default=20)
    args = ap.parse_args()
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    batch = cal[:args.n_seq].to(DEV)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    h_dense = capture_hidden(model, batch)
    acts = bs.collect_activations(model, cal, DEV)
    backup = {(li, ap): mod.weight.data.clone().cpu() for li, ap, mod in iter_targets(model)}

    # wanda
    apply_masks(model, acts, args.sparsity, "wanda")
    w_int, w_fin = interior_relerr(h_dense, capture_hidden(model, batch))
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                            n_samples=args.n_ppl, dataset_key="wikitext2")
    w_ppl = bs.evaluate_perplexity(model, test, DEV)
    # restore
    for li, ap, mod in iter_targets(model):
        mod.weight.data = backup[(li, ap)].to(DEV).to(mod.weight.dtype)
    # balanced
    apply_masks(model, acts, args.sparsity, "balanced")
    b_int, b_fin = interior_relerr(h_dense, capture_hidden(model, batch))

    collapse = w_int > 1.0
    rescuable = b_int < 1.0
    win = collapse and rescuable
    print(f"SCREEN model={args.model} sp={args.sparsity} "
          f"wanda_interior_relerr={w_int:.3f} balanced_interior_relerr={b_int:.3f} "
          f"prune_only_wiki_ppl={w_ppl:.4g} | COLLAPSE={'Y' if collapse else 'N'} "
          f"RESCUABLE={'Y' if rescuable else 'N'} WIN_PREDICTED={'Y' if win else 'N'}", flush=True)


if __name__ == "__main__":
    main()
