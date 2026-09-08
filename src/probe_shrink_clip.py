#!/usr/bin/env python3
"""SHRINKAGE-CLIP validation (attempt-8, step 1b): does a HELD-OUT-ROBUST clip beat the
calib-optimal awclip? Grounded in probe_heldout_gap: awclip's per-group argmin over the rho
grid overfits the calibration per-column energy ||X_j||^2 (2/3 of its gain is non-transferable,
mostly IN-distribution finite-sample overfit). Shrinking the energy weights toward their
group-mean by a global alpha should trade a little calib gain for more HELD-OUT gain.

  wcol_j(alpha) = c_j^2 * [ (1-alpha) * e_j + alpha * mean_{k in group} e_k ],  e_j = ||X_j||^2
  alpha=0 => awclip (calib-optimal).  alpha=1 => group-flat energy (clip by weight-shape only).

Build the clip on X_fit; score output error tr(D H D^T) on H_fit / H_wiki / H_ptb. If some
alpha>0 gives GAIN[ptb] > GAIN[ptb](alpha=0), a robust clip is real (=> rawclip). Balanced mask,
3 families. Non-iterative, global (one alpha), bpw-identical to RTN.
"""
import argparse, os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import eout_quant as eq  # noqa
from sinq.sparse_quant import quantize_rtn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect, out_err  # reuse  # noqa

SP, BETA, NBITS, GROUP, CAP = 0.5, 0.5, 3, 128, 256
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def shrink_energy(colE, mask, block, alpha):
    """Shrink per-column energy toward its group-mean over SURVIVOR columns, by alpha."""
    N = colE.shape[0]
    if not (N > block and N % block == 0):
        gm = colE.mean
        return (1 - alpha) * colE + alpha * gm
    ng = N // block
    e = colE.view(ng, block)
    # group mean over all columns in the group (energy is per-column, mask is per-row -> use
    # plain column-energy group mean; robust and global)
    gm = e.mean(-1, keepdim=True)
    return ((1 - alpha) * e + alpha * gm).reshape(N)


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = {"fit": collect(model, tok, device, "wikitext2", 16),
         "wiki": collect(model, tok, device, "wikitext2", 16, take_slice=(16, 32))}
    try:
        A["ptb"] = collect(model, tok, device, "ptb", 16)
    except Exception as e:
        print(f"  [warn] ptb unavailable {repr(e)[:60]}", flush=True)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    HK = list(A.keys)
    layer_paths = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} H={HK} ########", flush=True)
    min_max = [0, 2 ** NBITS - 1]
    tot = {a: {hk: 0.0 for hk in HK} for a in ALPHAS}
    rtn_tot = {hk: 0.0 for hk in HK}
    nmat = 0
    for li in sample:
        layer = layers[li].to(device)
        for ap in layer_paths:
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
            H = {}
            for hk in HK:
                a = A[hk].get(key)
                if a is None: H[hk] = None; continue
                Xh = a.to(device).float
                if Xh.dim == 3: Xh = Xh.reshape(-1, Xh.shape[-1])
                Xh = Xh[:min(Xh.shape[0], CAP)]
                H[hk] = Xh.t @ Xh
            Xf = A["fit"][key].to(device).float
            if Xf.dim == 3: Xf = Xf.reshape(-1, Xf.shape[-1])
            Xf = Xf[:min(Xf.shape[0], CAP)]
            colE = (Xf * Xf).sum(0).clamp(min=0)
            W_comp, mask = ns.balanced_mask_and_obs(W, A["fit"][key].to(device), SP, device, col_exp=BETA)
            r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
            mkf = mask.float
            W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
            # rtn anchor
            q, s, z, _ = quantize_rtn(W_norm, min_max, group_size=block)
            s = (s * r.view(-1, 1, 1)) if s.dim == 3 else (s * r.view(-1, 1))
            s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
            D_rtn = eq.dequant_deployed(q, s, z, mkf, c) - W
            for hk in HK:
                if H[hk] is not None: rtn_tot[hk] += out_err(D_rtn, H[hk])
            # shrinkage clip per alpha
            for al in ALPHAS:
                eshr = shrink_energy(colE, mask, block, al)
                wcol = (c.float ** 2) * eshr.float
                W_hat = eq.awclip_quantize(W_norm, mkf, NBITS, block, wcol) * (r.view(-1, 1) * c.view(1, -1)) * mkf
                D = W_hat - W
                for hk in HK:
                    if H[hk] is not None: tot[al][hk] += out_err(D, H[hk])
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  pooled {nmat} matrices. GAIN(alpha,H) = 1 - e/e_rtn.  (alpha=0 == awclip)", flush=True)
    hdr = "    alpha  " + "".join(f"{('GAIN['+hk+']'):>13}" for hk in HK)
    print(hdr, flush=True)
    best_ptb_a, best_ptb_g = None, -9
    for al in ALPHAS:
        row = f"    {al:<5}"
        for hk in HK:
            g = 1 - tot[al][hk] / rtn_tot[hk]
            row += f"{g*100:+11.1f}%"
            if hk == (HK[-1]) and g > best_ptb_g:
                best_ptb_g, best_ptb_a = g, al
        print(row, flush=True)
    g0 = 1 - tot[0.0][HK[-1]] / rtn_tot[HK[-1]]
    verdict = "ROBUST-CLIP WINS" if best_ptb_a not in (0.0,) and best_ptb_g > g0 + 0.003 else "awclip(alpha=0) already best held-out"
    print(f"  >> best held-out({HK[-1]}) alpha={best_ptb_a} gain={best_ptb_g*100:+.1f}% vs awclip {g0*100:+.1f}%  => {verdict}", flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
