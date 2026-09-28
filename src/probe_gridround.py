#!/usr/bin/env python3
"""MEASUREMENT PROBE (grid-aware mask, lane G2 = ROUNDING-AWARE tie-break) at 3-BIT.

The structural insight from scale-setter surgery + G1 (probe_gridmask, probe_gridalloc): a weight's
grid-STRESS (step^2*energy) is POSITIVELY coupled to its keep-value, so any mask that reorders the
importance ranking by grid-friendliness discards signal and collapses. The ONLY grid-aware lane that
can survive is an IMPORTANCE-PRESERVING tie-break: perturb selection by a quantity that is ~INDEPENDENT
of |W|/energy, so it only flips NEAR-TIES.

G2 uses the round-DISTANCE rd(i,j) = |w_norm - nearest_grid_level| / step in [0, 0.5] — the fractional
position of a weight relative to the group-RTN grid. rd is ~uniform and roughly independent of |W|, so:
    grid_importance = balanced_importance * (1 - eps * 2 * rd),
demotes only POORLY-rounding weights (rd~0.5) and only enough to flip weights within eps of the
threshold. Keeping clean-rounding survivors (and OBS-compensating poorly-rounding ones) should reduce
the survivor QUANT error without changing the pruning error (near-tie). eps=0 == balanced control.
Non-iterative, global rule (one eps/model). rd computed vs mask0's grid; out-of-hull pruned weights
get rd=0.5 (discouraged, they would move the grid). MEASURE at 3-bit sp0.5 g128, 3 families:
does any eps>0 reduce output error, CoBALT-specifically, and stack with awclip? Small effect expected.
"""
import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import benchmark_suite as bs  # noqa: E402
import nosink as ns  # noqa: E402
import eout_quant as eq  # noqa: E402
import probe_gridmask as pg  # noqa: E402
from sinq.sparse_quant import compute_hessian_inverse  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SP = 0.5
BETA = 0.5
NBITS = 3
CAP = 256
EPS = [0.0, 0.05, 0.1, 0.2, 0.4]


def round_dist(W_norm, mask, gsize, nbits):
    """Fractional round distance rd[K,N] to the nearest group-RTN level (grid from mask hull).
    rd in [0,0.5]; weights outside the group hull -> 0.5 (promoting them would move the grid)."""
    K, N = W_norm.shape
    m = mask.bool()
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize); Mg = m.view(K, N // gsize, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    step = ((w_max - w_min).clamp(min=1e-8)) / (2 ** nbits - 1)
    q = torch.round((Wg - w_min) / step)
    level = w_min + q.clamp(0, 2 ** nbits - 1) * step
    rd = ((Wg - level).abs() / step).clamp(0, 0.5)
    outside = (Wg < w_min) | (Wg > w_max)
    rd = torch.where(outside, torch.full_like(rd, 0.5), rd)
    return rd.reshape(K, N)


def grid_round_mask(imp, W_norm, mask0, gsize, eps):
    if eps <= 0:
        return mask0
    rd = round_dist(W_norm, mask0, gsize, NBITS)
    grid_imp = imp * (1.0 - eps * 2.0 * rd).clamp(min=1e-6)
    sp = 1.0 - mask0.float().mean().item()
    return ns._threshold_mask(grid_imp, sp, scope='global')


def run_model(MODEL, device="cuda", gsize=128):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    acts_all = bs.collect_activations(model, cal, device)
    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache()
    layer_paths = bs.get_layer_paths(model)
    layers = ns.get_layers(model)
    n_layers = len(layers)
    sample_layers = sorted(set([1, n_layers // 2, n_layers - 2]))
    print(f"\n######## {MODEL} layers={n_layers} sample={sample_layers} sp={SP} beta={BETA} "
          f"nbits={NBITS} g={gsize} (G2 grid-round) ########", flush=True)

    for supp in ["balanced", "wanda"]:
        E = {e: 0.0 for e in EPS}; E_awc = {e: 0.0 for e in EPS}
        base = 0.0; nmat = 0; nflip = {e: 0 for e in EPS}
        for li in sample_layers:
            layer = layers[li].to(device)
            for ap in layer_paths:
                parts = ap.split('.'); parent = layer; ok = True
                for p in parts[:-1]:
                    if not hasattr(parent, p):
                        ok = False; break
                    parent = getattr(parent, p)
                if not ok or not hasattr(parent, parts[-1]):
                    continue
                lin = getattr(parent, parts[-1])
                if not isinstance(lin, torch.nn.Linear):
                    continue
                acts = acts_all.get(f'layer_{li}.{ap}')
                if acts is None:
                    continue
                W = lin.weight.data.float().to(device)
                if W.shape[1] % gsize != 0:
                    continue
                Xa = acts.to(device).float()
                if Xa.dim() == 3:
                    Xa = Xa.reshape(-1, Xa.shape[-1])
                Xa = Xa[:min(Xa.shape[0], CAP)]
                H = Xa.t() @ Xa
                colE = (Xa * Xa).sum(0).clamp(min=0)
                an = torch.norm(Xa, dim=0)
                if supp == "balanced":
                    imp, mask0 = pg.balanced_importance(W, an, SP, BETA)
                else:
                    imp, mask0 = pg.wanda_importance(W, an, SP)
                H_inv = compute_hessian_inverse(Xa, damping=None); H_inv_diag = H_inv.diag()
                Wc0 = pg.obs_compensate(W, mask0, H_inv, H_inv_diag)
                c0 = pg.col_scale(Wc0, mask0)
                Wn0 = Wc0 / c0.view(1, -1)
                base += pg.eout(pg.rtn_dequant(Wn0, mask0, c0, gsize), W, H)
                for e in EPS:
                    m1 = grid_round_mask(imp, Wn0, mask0, gsize, e)
                    nflip[e] += int((m1 != mask0).sum().item()) // 2
                    Wc1 = pg.obs_compensate(W, m1, H_inv, H_inv_diag)
                    c1 = pg.col_scale(Wc1, m1)
                    Wn1 = Wc1 / c1.view(1, -1)
                    E[e] += pg.eout(pg.rtn_dequant(Wn1, m1, c1, gsize), W, H)
                    wcol1 = (c1.float() ** 2) * colE
                    Wn1_ac = eq.awclip_quantize(Wn1, m1, NBITS, gsize, wcol1)
                    E_awc[e] += pg.eout(Wn1_ac * c1.view(1, -1) * m1.float(), W, H)
                nmat += 1
            layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
        b = max(base, 1e-30)
        row = "  ".join(f"e{e}={E[e]/b:.4f}" for e in EPS)
        rowa = "  ".join(f"e{e}={E_awc[e]/b:.4f}" for e in EPS)
        best = min(EPS, key=lambda ee: E[ee])
        print(f"[{MODEL}/{supp}] nmat={nmat} base=1.000 | grid-round RTN: {row}  (best eps={best}, "
              f"{'WIN' if E[best] < base*0.999 else 'no-win'}; flips@{best}={nflip[best]})", flush=True)
        print(f"[{MODEL}/{supp}]                grid-round+awclip: {rowa}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    global SP, BETA, NBITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    ap.add_argument("--gsize", type=int, default=128)
    ap.add_argument("--sp", type=float, default=SP)
    ap.add_argument("--nbits", type=int, default=NBITS)
    ap.add_argument("--beta", type=float, default=BETA)
    args = ap.parse_args()
    SP, BETA, NBITS = args.sp, args.beta, args.nbits
    print(f"=== G2 grid-round probe sp={SP} beta={BETA} nbits={NBITS} g={args.gsize} ===", flush=True)
    for m in args.models.split(","):
        try:
            run_model(m.strip(), gsize=args.gsize)
        except Exception as ex:
            import traceback
            print(f"[{m}] FAILED: {ex}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main()
