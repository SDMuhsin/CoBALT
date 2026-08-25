#!/usr/bin/env python3
"""OPEN-IDEA TEST: is there ANY column-balance strength beta that HELPS a graceful (non-collapsing) model,
or is Wanda (beta=0) optimal there? CoBALT's win is collapse-rescue; on graceful models uniform beta=0.5
was inert/harmful in the grids. Before building a per-matrix adaptive-beta variant, test the necessary
condition: does the beta->PPL curve dip below Wanda anywhere on Llama/Qwen/StarCoder2? If it is monotone
increasing from beta=0, per-matrix adaptivity can at BEST match Wanda (set beta=0 everywhere) => ceiling
confirmed. If some beta>0 dips below Wanda, that is a real lead worth the per-matrix follow-up.

Mask-ISOLATED (prune-only, fp16 survivors, NO quant/OBS) so we measure the mask alone. One model load,
weight backup on CPU, sweep beta. Reports prune-only wiki PPL (n20) + interior max relerr per beta.
gemma-2b = positive control (expect a dip at beta~0.5). PPL RANKS here; downstream governs a real claim."""
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
    return max(re[:-1]) if len(re) > 1 else max(re)


def _thr(imp, sp, scope):
    K, N = imp.shape
    if scope == 'per_row':
        thr = torch.kthvalue(imp, int(N * sp), dim=1, keepdim=True).values
        return (imp > thr).float()
    thr = torch.kthvalue(imp.view(-1), int(K * N * sp)).values
    return (imp.view(-1) > thr).view(K, N).float()


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


def apply_scheme(model, acts, sp, kind, beta):
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
        m = mask_for(W, anorm, sp, kind, beta)
        mod.weight.data = (W * m).to(mod.weight.dtype)
        del Xd
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--n-ppl", type=int, default=20)
    ap.add_argument("--betas", default="0.1,0.2,0.3,0.5")
    args = ap.parse_args()
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    batch = cal[:args.n_seq].to(DEV)
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                            n_samples=args.n_ppl, dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    h_dense = capture_hidden(model, batch)
    acts = bs.collect_activations(model, cal, DEV)
    backup = {(li, ap): mod.weight.data.clone().cpu() for li, ap, mod in iter_targets(model)}

    schemes = [("wanda", 0.0)] + [("balanced", float(b)) for b in args.betas.split(",")]
    print(f"# model={args.model} sp={args.sparsity} (mask-only, prune-only, no quant/OBS)", flush=True)
    print(f"{'scheme':16s} {'interior_relerr':>15s} {'prune_only_wiki_ppl':>20s}", flush=True)
    best = None
    for kind, beta in schemes:
        for li, ap, mod in iter_targets(model):  # restore
            mod.weight.data = backup[(li, ap)].to(DEV).to(mod.weight.dtype)
        apply_scheme(model, acts, args.sparsity, kind, beta)
        rel = interior_relerr(h_dense, capture_hidden(model, batch))
        ppl = bs.evaluate_perplexity(model, test, DEV)
        tag = f"{kind}" if kind == "wanda" else f"balanced_b{beta}"
        print(f"{tag:16s} {rel:15.3f} {ppl:20.4g}", flush=True)
        if best is None or ppl < best[1]:
            best = (tag, ppl)
    wanda_ppl = None
    # recompute wanda ppl already printed; parse from best is not reliable, so recompute quickly not needed
    print(f"BEST scheme={best[0]} ppl={best[1]:.4g}  "
          f"(if BEST==wanda => beta>0 never helps => per-matrix adaptivity ceiling=match-Wanda)", flush=True)


if __name__ == "__main__":
    main()
