#!/usr/bin/env python3
"""DPCM deployable joint-encoding test (attempt-8j). The oracle KLT coding gain (~0.85 bit at sp0.8)
is balance-agnostic AND non-deployable (stored rotations). DPCM is the DEPLOYABLE, storage-free exploit
of the measured cross-column survivor correlation: within each group, predict column t from the
reconstructed column t-1 with a GLOBAL coefficient rho, quantize the residual at the same nbits (per-
group residual scale/zero = same storage as RTN). Sequential decode, one global rho (~free), non-
iterative. Closed-loop. Measures HELD-OUT (ptb) output-error GAIN vs RTN (rho=0), sp0.5 vs sp0.8,
balanced vs wanda vs magnitude, 3 families. Real & balance-specific => lead; universal/zero => confirms
the joint reopening is not a CoBALT-specific deployable lever.
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import eout_quant as eq  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect, out_err  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

BETA, GROUP, NBITS, CAP = 0.5, 128, 3, 256
NL = 2 ** NBITS - 1
SPS = [0.5, 0.8]
RHOS = [0.0, 0.2, 0.4, 0.6]


def dpcm_quant(W_norm, mask, gsize, rho):
    """Closed-loop DPCM within groups. rho=0 == deployed RTN. Returns W_hat_norm (pre rc/mask)."""
    K, N = W_norm.shape
    if not (N > gsize and N % gsize == 0):
        return None
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize); Mg = mask.view(K, ng, gsize).bool
    # per-group residual scale/zero from OPEN-loop residuals over survivors
    prev = torch.zeros(K, ng, device=W_norm.device)
    R = torch.empty_like(Wg)
    for t in range(gsize):
        R[:, :, t] = Wg[:, :, t] - rho * (Wg[:, :, t - 1] if t > 0 else torch.zeros_like(Wg[:, :, 0]))
    big = torch.finfo(torch.float32).max
    rmin = torch.where(Mg, R, torch.full_like(R, big)).amin(-1, keepdim=True)
    rmax = torch.where(Mg, R, torch.full_like(R, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    rmin = torch.where(empty, torch.zeros_like(rmin), rmin); rmax = torch.where(empty, torch.zeros_like(rmax), rmax)
    s = ((rmax - rmin) / NL).clamp(min=1e-8); z = -torch.round(rmin / s)
    s = s.squeeze(-1); z = z.squeeze(-1)                       # [K,ng]
    # closed-loop reconstruct
    prev_rec = torch.zeros(K, ng, device=W_norm.device)
    Wh = torch.empty_like(Wg)
    for t in range(gsize):
        pred = rho * prev_rec
        r = Wg[:, :, t] - pred
        q = torch.clamp(torch.round(r / s + z), 0, NL)
        rec = (q - z) * s + pred
        Wh[:, :, t] = rec
        prev_rec = rec
    return Wh.reshape(K, N)


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = {"fit": collect(model, tok, device, "wikitext2", 16)}
    try:
        A["ptb"] = collect(model, tok, device, "ptb", 16)
    except Exception as e:
        print(f"  [warn] ptb {repr(e)[:40]}", flush=True)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    HKEY = "ptb" if "ptb" in A else "fit"
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} H={HKEY} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    tot = {mk: {sp: {rho: 0.0 for rho in RHOS} for sp in SPS} for mk in MK}
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
            if A["fit"].get(key) is None: continue
            W = lin.weight.data.clone.float.to(device); K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            Xh = A[HKEY][key].to(device).float
            if Xh.dim == 3: Xh = Xh.reshape(-1, Xh.shape[-1])
            Xh = Xh[:min(Xh.shape[0], CAP)]
            H = Xh.t @ Xh
            X = A["fit"][key].to(device)
            for mk in MK:
                for sp in SPS:
                    if mk == "balanced":
                        W_comp, mask = ns.balanced_mask_and_obs(W, X, sp, device, col_exp=BETA)
                    elif mk == "wanda":
                        W_comp, mask = ns.wanda_mask_and_obs(W, X, sp, device, scope='per_row')
                    else:
                        W_comp, mask = magnitude_mask_and_obs(W, X, sp, device)
                    r, c = ns.compute_norm_scales(W_comp, mask, 'col', device); mkf = mask.float
                    rc = r.view(-1, 1) * c.view(1, -1)
                    W_norm = W_comp / rc
                    for rho in RHOS:
                        Wh = dpcm_quant(W_norm, mkf, block, rho)
                        if Wh is None: continue
                        D = Wh * rc * mkf - W
                        tot[mk][sp][rho] += out_err(D, H)
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  DPCM held-out({HKEY}) GAIN = 1 - e(rho)/e(rho=0); rho=0 is deployed RTN. best rho per cell:", flush=True)
    for mk in MK:
        row = f"    {mk:<10}"
        for sp in SPS:
            base = tot[mk][sp][0.0]
            best = max(((1 - tot[mk][sp][rho] / base), rho) for rho in RHOS if base > 0)
            row += f"  sp{sp}: best={best[0]*100:+.2f}% @rho={best[1]}"
        print(row, flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
