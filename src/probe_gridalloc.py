#!/usr/bin/env python3
"""MEASUREMENT PROBE (grid-aware mask, lane G1 = grid-cost sparsity ALLOCATION) at 3-BIT.

User steering : stay grid-aware, but at the REQUIRED 3-bit operating point (2-bit is a
dead/collapse regime that does not count). The scale-setter surgery (probe_gridmask) failed at 3-bit.
G1 is a DIFFERENT grid coupling: instead of the group hull EXTREME, re-rank the WHOLE support by a
grid-aware importance that trades the pruning cost against the QUANTIZATION cost of keeping a weight.

Balanced importance keeps weight (i,j) by |W|.||X|| (row+col self-normalized). It IGNORES that keeping
it as a survivor costs group-RTN quant error proportional to step_g^2 * energy_j, where the per-group
STEP_g = hull_range/(2^b-1) is set by the group's survivor spread. Two survivors of equal pruning
importance are NOT equally cheap to keep: one in a wide-hull (coarse-step) group carries more quant
error than one in a tight-hull group. G1 demotes relatively-expensive-to-quantize survivors:

    grid_importance = balanced_importance / (1 + lambda * qc_rel),
    qc_rel = keep_quant_cost / median_survivor(keep_quant_cost),  keep_quant_cost = step_g^2 * energy_j.

lambda=0 == balanced (control). Deterministic GLOBAL rule (one lambda/model), non-iterative (2 passes:
balanced mask -> step -> re-rank -> re-OBS). NOT scale-setter surgery, NOT multiprecision, NOT repack.
The NOVEL signal is the per-group step^2 factor, which balanced does not see.

MEASURE (calib output error tr((Wq-W) H (Wq-W)^T), H=X^T X), 3-bit sp0.5 beta0.5 g128, 3 layers x
{gemma-2b, tinyllama, qwen-1.5b}: (i) does any lambda>0 reduce output error vs balanced? (ii) is it
CoBALT-specific (helps balanced support more than wanda)? (iii) does it stack with awclip?
If no lambda>0 helps on >=2 incl a healthy model => G1 negative at 3-bit, pivot to G2 (do NOT close).
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
import probe_gridmask as pg  # reuse helpers  # noqa: E402
from sinq.sparse_quant import compute_hessian_inverse  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SP = 0.5
BETA = 0.5
NBITS = 3
CAP = 256
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def group_step(W_norm, mask, gsize, nbits):
    """Per-weight group-RTN step = hull_range/(2^b-1) of the mask's survivors, broadcast [K,N]."""
    K, N = W_norm.shape
    m = mask.bool()
    if not (N > gsize and N % gsize == 0):
        big = torch.finfo(torch.float32).max
        w_min = torch.where(m, W_norm, torch.full_like(W_norm, big)).amin(1, keepdim=True)
        w_max = torch.where(m, W_norm, torch.full_like(W_norm, -big)).amax(1, keepdim=True)
        step = (w_max - w_min).clamp(min=1e-8) / (2 ** nbits - 1)
        return step.expand(K, N)
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize); Mg = m.view(K, ng, gsize)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    rng = torch.where(empty, torch.zeros_like(w_min), (w_max - w_min))
    step = (rng.clamp(min=1e-8) / (2 ** nbits - 1))
    return step.expand(K, ng, gsize).reshape(K, N)


def grid_alloc_mask(imp, W_norm, mask0, energy_norm, gsize, lam):
    """Re-rank the support by grid-aware importance and re-threshold at the SAME sparsity.
    imp: balanced importance [K,N] (pre-threshold). Returns new keep-mask (float)."""
    if lam <= 0:
        return mask0
    step = group_step(W_norm, mask0, gsize, NBITS)                 # [K,N]
    qc = (step ** 2) * energy_norm.view(1, -1)                     # keep-quant-cost [K,N]
    m = mask0.bool()
    med = qc[m].median() if m.any() else qc.median()
    qc_rel = qc / med.clamp(min=1e-30)
    grid_imp = imp / (1.0 + lam * qc_rel)
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
          f"nbits={NBITS} g={gsize} (G1 grid-alloc) ########", flush=True)

    for supp in ["balanced", "wanda"]:
        E = {lam: 0.0 for lam in LAMBDAS}
        E_awc = {lam: 0.0 for lam in LAMBDAS}      # grid-alloc mask + awclip
        base = 0.0
        nmat = 0
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
                energy_norm = (c0.float() ** 2) * colE
                base += pg.eout(pg.rtn_dequant(Wn0, mask0, c0, gsize), W, H)
                for lam in LAMBDAS:
                    m1 = grid_alloc_mask(imp, Wn0, mask0, energy_norm, gsize, lam)
                    Wc1 = pg.obs_compensate(W, m1, H_inv, H_inv_diag)
                    c1 = pg.col_scale(Wc1, m1)
                    Wn1 = Wc1 / c1.view(1, -1)
                    E[lam] += pg.eout(pg.rtn_dequant(Wn1, m1, c1, gsize), W, H)
                    wcol1 = (c1.float() ** 2) * colE
                    Wn1_ac = eq.awclip_quantize(Wn1, m1, NBITS, gsize, wcol1)
                    E_awc[lam] += pg.eout(Wn1_ac * c1.view(1, -1) * m1.float(), W, H)
                nmat += 1
            layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
        b = max(base, 1e-30)
        row = "  ".join(f"l{lam}={E[lam]/b:.4f}" for lam in LAMBDAS)
        rowa = "  ".join(f"l{lam}={E_awc[lam]/b:.4f}" for lam in LAMBDAS)
        best = min(LAMBDAS, key=lambda L: E[L])
        print(f"[{MODEL}/{supp}] nmat={nmat} base=1.000 | grid-alloc RTN: {row}  (best λ={best}, "
              f"{'WIN' if E[best] < base*0.999 else 'no-win'})", flush=True)
        print(f"[{MODEL}/{supp}]                grid-alloc+awclip: {rowa}", flush=True)
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
    print(f"=== G1 grid-alloc probe sp={SP} beta={BETA} nbits={NBITS} g={args.gsize} ===", flush=True)
    for m in args.models.split(","):
        try:
            run_model(m.strip(), gsize=args.gsize)
        except Exception as e:
            import traceback
            print(f"[{m}] FAILED: {e}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main()
