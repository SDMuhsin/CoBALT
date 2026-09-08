#!/usr/bin/env python3
"""OBS-COMPENSATION SPECTRUM (attempt-8m, keep-trying/new-object). The survivor = W*mask + comp*mask,
comp = OBS compensation (= -H_inv @ (pruned/diag), CoBALT's compute lever). I've characterized survivor
correlation (pairwise) but NEVER the SPECTRUM of comp itself. If comp is spectrally LOW-RANK, a cheap
low-rank code for comp + RTN on the ~uniform base could beat single-RTN on W_comp -- and if comp is
lower-rank for the BALANCED mask, that's CoBALT-specific. Measure effective rank / spectral energy of
comp: fraction of Frobenius energy in the top-r singular values, and rank@90%/99% energy. Also comp's
share of the survivor norm (how much OBS moves things). balanced vs wanda vs magnitude, sp0.5 & 0.8, 3 fam.
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

BETA = 0.5
SPS = [0.5, 0.8]


def spectrum_stats(comp):
    """rank@90/99% Frobenius energy as FRACTION of min(K,N); top-r energy share hint."""
    try:
        sv = torch.linalg.svdvals(comp.float)
    except Exception:
        return None
    e = (sv ** 2); tot = e.sum.clamp(min=1e-30); cum = torch.cumsum(e, 0) / tot
    d = float(min(comp.shape))
    r90 = float((cum < 0.90).sum.item + 1) / d
    r99 = float((cum < 0.99).sum.item + 1) / d
    # storage-relevant: energy captured by a rank that costs ~1.5% bpw. r_cheap ~ 0.02*min(K,N)
    rc = max(1, int(0.02 * min(comp.shape)))
    ecap = float(cum[min(rc - 1, len(cum) - 1)].item)
    return r90, r99, ecap


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = collect(model, tok, device, "wikitext2", 16)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    acc = {mk: {sp: {"r90": 0.0, "r99": 0.0, "ecap": 0.0, "cshare": 0.0, "n": 0} for sp in SPS} for mk in MK}
    for li in sample:
        layer = layers[li].to(device)
        for ap in lp:
            parts = ap.split('.'); parent = layer; ok = True
            for p in parts[:-1]:
                if not hasattr(parent, p): ok = False; break
                parent = getattr(parent, p)
            if not ok or not hasattr(parent, parts[-1]): continue
            lin = getattr(parent, parts[-1])
            if not isinstance(lin, torch.nn.Linear): continue
            key = f'layer_{li}.{ap}'
            if A.get(key) is None: continue
            W = lin.weight.data.clone.float.to(device); K, N = W.shape
            X = A[key].to(device)
            for mk in MK:
                for sp in SPS:
                    if mk == "balanced":
                        W_comp, mask = ns.balanced_mask_and_obs(W, X, sp, device, col_exp=BETA)
                    elif mk == "wanda":
                        W_comp, mask = ns.wanda_mask_and_obs(W, X, sp, device, scope='per_row')
                    else:
                        W_comp, mask = magnitude_mask_and_obs(W, X, sp, device)
                    comp = (W_comp - W * mask)                 # OBS compensation only
                    st = spectrum_stats(comp)
                    if st is None: continue
                    cshare = float(comp.norm / (W_comp.norm.clamp(min=1e-8)))
                    a = acc[mk][sp]
                    a["r90"] += st[0]; a["r99"] += st[1]; a["ecap"] += st[2]; a["cshare"] += cshare; a["n"] += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  OBS comp spectrum: rank@90/99% energy (frac of min(K,N); LOW=>low-rank), ecap=energy in", flush=True)
    print(f"  rank~2% (storage-cheap), cshare=||comp||/||survivor||:", flush=True)
    for mk in MK:
        for sp in SPS:
            a = acc[mk][sp]; n = max(1, a["n"])
            print(f"    {mk:<10} sp{sp}  r90={a['r90']/n:.2f} r99={a['r99']/n:.2f} ecap@2%={a['ecap']/n*100:.1f}%"
                  f"  cshare={a['cshare']/n:.3f}", flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
