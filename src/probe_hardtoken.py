#!/usr/bin/env python3
"""MEASURE-FIRST for #44 hard-token mask. Reweight the per-column activation norm by per-token LM loss
(hard tokens = high loss) so the mask preserves weights on the channels HARD tokens use. Magnitude-anchored
(keeps |W|) so it stays viable (unlike spectral). Uses per-token loss = NEW info no forward-stat lever used;
mechanism is downstream-aligned (hard examples drive accuracy), not reconstruction/starvation.
Reports Jaccard(balanced, hardtoken) -- moderate (~0.6-0.85) => distinct+viable candidate -> smoke;
~1.0 => absorbed; ~0.3 => far (collapse-risk like spectral)."""
import os, sys, argparse
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.6)
    args = ap.parse_args()
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    model.eval()
    # per-token loss on each calib sequence (forward-only), + capture per-layer inputs with same tokens.
    layers = ns.get_layers(model); paths = bs.get_layer_paths(model)
    # capture inputs to each target linear, per sequence, keeping token dim
    sp = args.sparsity
    js = []
    # process a handful of sequences; accumulate weighted & unweighted col second-moments per matrix
    import torch.nn.functional as F
    cap = {}
    hooks = []
    def mk(key):
        def h(mod, inp, out):
            x = inp[0].detach()
            cap.setdefault(key, []).append(x)  # [B,T,N] fp16
        return h
    # register hooks on each target linear
    handles = []
    layer_paths = paths
    for li, layer in enumerate(layers):
        for ap in layer_paths:
            mod = layer; ok = True
            for p in ap.split('.'):
                if not hasattr(mod, p): ok = False; break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(mk(f'{li}.{ap}')))
    tokw = []   # per-token loss weights, concatenated over sequences (aligned to activation token order)
    with torch.no_grad():
        for s in range(min(8, cal.shape[0])):
            ids = cal[s:s+1].to(DEV)
            cap.clear()
            out = model(ids)
            logits = out.logits[0, :-1].float()      # [T-1, V]
            tgt = ids[0, 1:]                          # [T-1]
            loss_t = F.cross_entropy(logits, tgt, reduction='none')  # [T-1] per-token loss
            # weight per token (align: activations have T tokens; loss has T-1; drop last activation token)
            w = torch.cat([loss_t, loss_t[-1:]])     # pad to T
            w = (w / (w.mean() + 1e-9))              # normalized hard-token weight
            # accumulate col moments for a sample of matrices (first sequence sets the sample)
            if s == 0:
                sample_keys = list(cap.keys())[::max(1, len(cap)//8)][:8]
                acc = {k: {'u': 0.0, 'w': 0.0, 'W': None} for k in sample_keys}
            for k in acc:
                X = cap[k][0][0].float()             # [T,N]
                T = X.shape[0]; wk = w[:T]
                acc[k]['u'] = acc[k]['u'] + (X.pow(2)).sum(0)          # unweighted col second moment
                acc[k]['w'] = acc[k]['w'] + (X.pow(2) * wk.view(-1,1)).sum(0)  # hard-token-weighted
    for h in handles: h.remove()
    # build masks per sampled matrix
    for k in acc:
        li = int(k.split('.')[0]); ap = k.split('.', 1)[1]
        mod = layers[li]
        for p in ap.split('.'): mod = getattr(mod, p)
        W = mod.weight.data.float().to(DEV)
        cn_u = acc[k]['u'].sqrt(); cn_w = acc[k]['w'].sqrt()
        m_bal = ns.balanced_keepmask_local(W.abs()*cn_u.view(1,-1), sp, 0.5).bool()
        m_hard = ns.balanced_keepmask_local(W.abs()*cn_w.view(1,-1), sp, 0.5).bool()
        js.append((m_bal & m_hard).sum().item() / (m_bal | m_hard).sum().item())
    import statistics as st
    print(f"{args.model}: Jaccard(balanced, hardtoken) = {st.mean(js):.4f} "
          f"(~1 absorbed; 0.6-0.85 distinct+viable=>smoke; ~0.3 far/collapse-risk)")


if __name__ == "__main__":
    main()
