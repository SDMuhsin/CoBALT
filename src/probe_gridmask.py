#!/usr/bin/env python3
"""MEASUREMENT PROBE (fresh commission, step 1): is there a QUANTIZATION-GRID-AWARE MASK lever?

THE NEW DEGREE OF FREEDOM (this commission opens the MASK; attempts 5-8 froze it):
every deployed mask (Wanda |W|.||X||, CoBALT balanced, OBS-saliency, Fisher) chooses the
support by a PRE-quantization importance. NONE of them accounts for the fact that survivors
are then GROUP-RTN quantized, where the per-group step = hull_range / (2^b - 1) is set by the
group's magnitude EXTREME (scale-setter). So keeping a large-|W_norm| survivor forces a COARSE
step on EVERY other survivor in its group -- a grid-cost the importance ignores.

MEASURED ALREADY (attempt-7, probe_scale_energy): CoBALT's balanced mask + OBS parks the group
scale-setter in a BELOW-median-||X|| (low output-energy) column ~95% of the time. awclip only
CLIPPED that scale (soft). The UNTESTED MASK lever: HARD-PRUNE the low-energy scale-setter
(OBS compensates its removal exactly, cheaply) and PROMOTE the best currently-pruned candidate
to hold sparsity -- shrinking the hull => finer step => lower quant error on all group-mates.

This probe MEASURES, at the deployed operating point (sp0.5, beta0.5, 3-bit, g128), on 3
families, whether that support swap actually reduces the post-quantization OUTPUT error
tr(D H D^T), H = X^T X (the real downstream-relevant object, NOT weight MSE), and CRUCIALLY:
  (M1) does it help the CoBALT balanced+OBS support?  E0_bal vs E1_bal  (pooled + per-group)
  (M2) FAIRNESS/attribution: does the SAME swap help an UNBALANCED wanda+OBS support as much?
       If it helps balanced MORE => plausibly CoBALT-specific (the awclip UNIVERSAL trap avoided).
       If it helps both equally => another universal lever, report negative fast.
  (M3) does the MASK swap beat AWCLIP (soft-clip) on the balanced support?  (else it's awclip++)
  (M4) diagnostics: scale-setter low-energy prevalence at THIS operating point; hull shrink ratio;
       fraction of groups where the swap fired / helped.

Everything is one-shot, non-iterative (full batch OBS twice, a deterministic per-group rule).
Calibration output error only (existence of the mechanism); held-out is the later smoke cell.
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
from sinq.sparse_quant import compute_hessian_inverse, quantize_rtn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SP = 0.5
BETA = 0.5
NBITS = 3
CAP256 = 256


def balanced_importance(W, act_norms, sparsity, col_exp):
    """CoBALT balanced importance + GLOBAL top-k mask (mirrors ns.balanced_mask_and_obs,
    row_fair=True, global scope), returns the boolean keep-mask. No OBS here."""
    K, N = W.shape
    imp = W.abs() * act_norms.view(1, -1)
    kr, kc = int(N * sparsity), int(K * sparsity)
    if kr > 0:
        qr = torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qr
    if col_exp > 0 and kc > 0:
        qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qc.pow(col_exp)
    return imp, ns._threshold_mask(imp, sparsity, scope='global')


def wanda_importance(W, act_norms, sparsity):
    imp = W.abs() * act_norms.view(1, -1)
    return imp, ns._threshold_mask(imp, sparsity, scope='global')


def obs_compensate(W, mask, H_inv, H_inv_diag):
    """One-shot batch OBS, vectorized. Identical to the per-row form in every
    ns.*_mask_and_obs path: comp_i = -H_inv @ (W_i*(1-mask_i) / H_inv_diag), and since
    H_inv is symmetric, stacking over rows gives comp = -(P/H_inv_diag) @ H_inv,
    P = W*(1-mask). Then W_comp = (W + comp)*mask (pruned positions -> 0 downstream)."""
    P = (W * (1.0 - mask)) / H_inv_diag.view(1, -1)
    comp = -(P @ H_inv)
    return (W + comp) * mask


def col_scale(W_comp, mask):
    return ns._robust_scale(ns._masked_std(W_comp, mask, dim=0))


def rtn_dequant(W_norm, mask, c, gsize):
    """Deployed endpoint group-RTN over survivors -> dequantized W_hat [K,N] (orig space)."""
    q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** NBITS - 1], group_size=gsize)
    K, N = W_norm.shape
    if scales.dim() == 3:
        ng = scales.shape[1]
        Wd = ((q.view(K, ng, N // ng) - zeros) * scales).reshape(K, N)
    else:
        Wd = (q - zeros) * scales
    return Wd * c.view(1, -1) * mask.float()


def eout(W_hat, W_dense, H):
    D = (W_hat - W_dense).double()
    return float((D @ H.double() * D).sum().item())


def grid_aware_swap_mask(W_norm, mask, energy, gsize, only_lowE=True):
    """Deterministic global GRID-AWARE support rule. In each group:
      - identify the scale-setter s* = survivor with max |W_norm - group_mid|.
      - if s* is in a below-group-median ENERGY position (energy = c^2 ||X||^2), PRUNE s*
        and PROMOTE the highest-|W_norm|-among-pruned candidate p* in the same group.
      This holds the per-group survivor count constant (so global sparsity is unchanged) and
      shrinks the hull (finer step) whenever the removed extreme was low-energy => cheap.
    only_lowE=False: swap the scale-setter unconditionally (upper bound on the lever).
    Returns (new_mask, n_fired). Operates per (row, group) block, vectorized over rows/groups."""
    K, N = W_norm.shape
    m = mask.bool().clone()
    grouped = N > gsize and N % gsize == 0
    if not grouped:
        return mask, 0
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize)
    Mg = m.view(K, ng, gsize)
    Eg = energy.view(1, ng, gsize).expand(K, ng, gsize)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    valid = Mg.any(-1, keepdim=True)
    mid = 0.5 * (w_min + w_max)
    # scale-setter s* = max |w-mid| among survivors; new hull half-range if s* removed = 2nd-largest dev.
    dev = torch.where(Mg, (Wg - mid).abs(), torch.full_like(Wg, -big))
    top2 = dev.topk(2, dim=-1).values                                   # [K,ng,2]
    s_dev = top2[..., 0:1]
    new_half = top2[..., 1:2].clamp(min=0)                              # hull after dropping s*
    s_idx = dev.argmax(-1, keepdim=True)                                # [K,ng,1]
    ext_E = torch.gather(Eg, -1, s_idx)
    med_E = torch.where(Mg, Eg, torch.full_like(Eg, big)).median(-1, keepdim=True).values
    # PROMOTE (corrected): among PRUNED positions that are INTERIOR to the shrunken hull
    # (|w-mid| <= new_half, so they do NOT re-extend the scale), pick the highest OUTPUT-ENERGY
    # one -- adds real output-error capacity while keeping the tight hull. (v1 promoted max-|w|,
    # which re-became the scale-setter and re-extended the hull => measured +33% WORSE on gemma.)
    dev_p = (Wg - mid).abs()
    interior = (~Mg) & (dev_p <= new_half)
    cand_E = torch.where(interior, Eg, torch.full_like(Eg, -big))
    p_idx = cand_E.argmax(-1, keepdim=True)
    has_cand = interior.any(-1, keepdim=True)
    fire = valid & has_cand & (s_dev > 0)
    if only_lowE:
        fire = fire & (ext_E < med_E)
    # apply: drop s*, add p* where fire
    Mg_new = Mg.clone()
    fire_row = fire.squeeze(-1)                                          # [K,ng]
    ar = torch.arange(K, device=W_norm.device).view(-1, 1).expand(K, ng)
    ag = torch.arange(ng, device=W_norm.device).view(1, -1).expand(K, ng)
    fr, fg = ar[fire_row], ag[fire_row]
    si = s_idx.squeeze(-1)[fire_row]
    pi = p_idx.squeeze(-1)[fire_row]
    Mg_new[fr, fg, si] = False
    Mg_new[fr, fg, pi] = True
    return Mg_new.reshape(K, N).float(), int(fire_row.sum().item())


def build_and_score(W, X, imp_fn, gsize, device):
    """Return dict of pooled tr(DHD^T) for: base RTN, grid-swap RTN, grid-swap-unconditional,
    awclip, plus diagnostics, for the given importance/mask function."""
    Xa = X.float()
    if Xa.dim() == 3:
        Xa = Xa.reshape(-1, Xa.shape[-1])
    Xa = Xa[:min(Xa.shape[0], CAP256)]
    H = Xa.t() @ Xa
    energy0 = (Xa * Xa).sum(0).clamp(min=0)                              # ||X_j||^2 [N]
    _, mask0 = imp_fn(W, torch.norm(Xa, dim=0))
    H_inv = compute_hessian_inverse(Xa, damping=None)
    H_inv_diag = H_inv.diag()

    # --- base support: OBS, col-norm, RTN
    Wc0 = obs_compensate(W, mask0, H_inv, H_inv_diag)
    c0 = col_scale(Wc0, mask0)
    Wn0 = Wc0 / c0.view(1, -1)
    Wd0 = rtn_dequant(Wn0, mask0, c0, gsize)
    # dense target = OBS-compensated survivors in original space (what the deployed decoder
    # reconstructs before quant); mask applied. This is the standard nosink target.
    W_target0 = (Wc0 * mask0)
    E_base = eout(Wd0, W_target0, H)

    # energy in NORMALIZED space for the scale-setter test: c^2 * ||X||^2
    energy_norm = (c0.float() ** 2) * energy0

    # --- grid-aware swap (low-energy scale-setter only) ---
    res = {"E_base": E_base}
    import eout_quant as eq
    for tag, lowE in [("grid", True), ("gridU", False)]:
        mask1, nfire = grid_aware_swap_mask(Wn0, mask0, energy_norm, gsize, only_lowE=lowE)
        # NOTE: swap changes the support -> re-OBS, re-col-norm (faithful, non-iterative)
        Wc1 = obs_compensate(W, mask1, H_inv, H_inv_diag)
        c1 = col_scale(Wc1, mask1)
        Wn1 = Wc1 / c1.view(1, -1)
        Wd1 = rtn_dequant(Wn1, mask1, c1, gsize)
        # score BOTH supports against the SAME dense (uncompensated) reference so E is comparable
        # across masks: reference = the true dense weight W (not the OBS target, which differs per
        # mask). tr((Wq - W) H (Wq - W)^T) is the honest output error vs the ORIGINAL layer.
        res[f"E_{tag}"] = eout(Wd1, W.float(), H)
        res[f"nfire_{tag}"] = nfire
        if tag == "grid":
            # ORTHOGONALITY TEST: awclip (soft-clip scale) ON TOP of the grid-swapped support.
            # If this beats awclip-on-base, the grid MASK adds something awclip's scale cannot.
            wcol1 = (c1.float() ** 2) * energy0
            Wn1_ac = eq.awclip_quantize(Wn1, mask1, NBITS, gsize, wcol1)
            Wd1_ac = Wn1_ac * c1.view(1, -1) * mask1.float()
            res["E_grid_awclip_vsW"] = eout(Wd1_ac, W.float(), H)
    # base also scored against true dense W for apples-to-apples
    res["E_base_vsW"] = eout(Wd0, W.float(), H)

    # --- awclip on base support (soft-clip comparison) ---
    wcol = (c0.float() ** 2) * energy0
    Wn_ac = eq.awclip_quantize(Wn0, mask0, NBITS, gsize, wcol)
    Wd_ac = Wn_ac * c0.view(1, -1) * mask0.float()
    res["E_awclip_vsW"] = eout(Wd_ac, W.float(), H)

    # --- diagnostics: scale-setter low-energy prevalence, hull shrink ---
    res["ncols"] = W.shape[1]
    return res


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
    print(f"\n######## {MODEL} layers={n_layers} sample={sample_layers} "
          f"sp={SP} beta={BETA} nbits={NBITS} g={gsize} ########", flush=True)

    def bal_fn(W, an):
        return balanced_importance(W, an, SP, BETA)

    def wan_fn(W, an):
        return wanda_importance(W, an, SP)

    agg = {}
    for supp, fn in [("balanced", bal_fn), ("wanda", wan_fn)]:
        acc = {"E_base_vsW": 0.0, "E_grid": 0.0, "E_gridU": 0.0, "E_awclip_vsW": 0.0,
               "E_grid_awclip_vsW": 0.0,
               "nfire_grid": 0, "nfire_gridU": 0, "nmat": 0,
               "wins_grid": 0, "wins_awclip": 0, "wins_stack": 0}
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
                r = build_and_score(W, acts.to(device), fn, gsize, device)
                acc["E_base_vsW"] += r["E_base_vsW"]
                acc["E_grid"] += r["E_grid"]
                acc["E_gridU"] += r["E_gridU"]
                acc["E_awclip_vsW"] += r["E_awclip_vsW"]
                acc["E_grid_awclip_vsW"] += r["E_grid_awclip_vsW"]
                acc["nfire_grid"] += r["nfire_grid"]
                acc["nfire_gridU"] += r["nfire_gridU"]
                acc["nmat"] += 1
                acc["wins_grid"] += int(r["E_grid"] < r["E_base_vsW"])
                acc["wins_awclip"] += int(r["E_awclip_vsW"] < r["E_base_vsW"])
                acc["wins_stack"] += int(r["E_grid_awclip_vsW"] < r["E_awclip_vsW"])  # stack beats awclip?
            layers[li] = layer.to("cpu")
            torch.cuda.empty_cache()
        base = max(acc["E_base_vsW"], 1e-30)
        awc = max(acc["E_awclip_vsW"], 1e-30)
        stack_vs_awclip = acc["E_grid_awclip_vsW"] / awc      # <1 => grid MASK adds orthogonal gain
        print(f"[{MODEL}/{supp}] nmat={acc['nmat']} "
              f"E_base=1.000  grid={acc['E_grid']/base:.4f}  gridU={acc['E_gridU']/base:.4f}  "
              f"awclip={awc/base:.4f}  grid+awclip={acc['E_grid_awclip_vsW']/base:.4f}  | "
              f"STACK/awclip={stack_vs_awclip:.4f} (stack wins {acc['wins_stack']}/{acc['nmat']})  "
              f"grid wins {acc['wins_grid']}/{acc['nmat']}", flush=True)
        agg[supp] = {k: (v / base if k.startswith("E_") else v) for k, v in acc.items()}
    # differential: does grid help balanced MORE than wanda? (CoBALT-specific test)
    d_bal = 1.0 - agg["balanced"]["E_grid"]
    d_wan = 1.0 - agg["wanda"]["E_grid"]
    print(f"[{MODEL}] GRID output-error reduction  balanced={d_bal*100:+.2f}%  "
          f"wanda={d_wan*100:+.2f}%  DIFFERENTIAL(bal-wan)={100*(d_bal-d_wan):+.2f}pp "
          f"{'<= CoBALT-specific' if d_bal-d_wan > 0.01 else '<= universal/none'}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    global SP, BETA, NBITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    ap.add_argument("--gsize", type=int, default=128)
    ap.add_argument("--sp", type=float, default=SP,
                    help="sparsity operating point (anti-premature-closure sweep: grid-couplings "
                         "strengthen at high sp where hulls are set by few survivors).")
    ap.add_argument("--nbits", type=int, default=NBITS,
                    help="bit-width (awclip scale gain is far bigger at 2-bit => grid-mask may pay).")
    ap.add_argument("--beta", type=float, default=BETA)
    args = ap.parse_args()
    SP, BETA, NBITS = args.sp, args.beta, args.nbits
    print(f"=== gridmask probe  sp={SP} beta={BETA} nbits={NBITS} g={args.gsize} ===", flush=True)
    for m in args.models.split(","):
        try:
            run_model(m.strip(), gsize=args.gsize)
        except Exception as e:
            import traceback
            print(f"[{m}] FAILED: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
