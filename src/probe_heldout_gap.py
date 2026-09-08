#!/usr/bin/env python3
"""HELD-OUT GENERALIZATION-GAP PROBE (attempt-8, step 1 = MEASURE).

Every prior encoding probe (gden/binc/awclip/awclipz/hdiagw, attempts 5-7) screened the
survivor encoding by the *calibration* output error tr(D H_c D^T), where H_c = X_c^T X_c is
built from the SAME activations that fit the encoding. But CoBALT's edge is NOT at
calibration -- the isolation probe found balanced ~= unbalanced at calib output error, and
[[calib-error-blind-to-cobalt]] found CoBALT's win is HELD-OUT / generalization. So the whole
screening apparatus has been blind to the object that actually decides downstream error.

This probe measures what NO prior probe did: the calibration -> held-out GENERALIZATION GAP of
the survivor quantization error D = W_hat - W_dense. Encodings are FIT on a calibration set
(mask + OBS + scale + colE all from X_fit); the fixed error D is then scored on THREE Hessians:
  H_fit  : X_fit^T X_fit          (calibration -- what every prior probe used)
  H_wiki : X_heldwiki^T X_heldwiki (DISJOINT wikitext2 samples -- in-distribution held-out)
  H_ptb  : X_ptb^T X_ptb           (PTB -- cross-distribution held-out, downstream-like shift)

KEY QUESTIONS:
 (1) Do the activation-aware encoding gains (awclip/awclipz over RTN) SHRINK or REVERSE from
     H_fit -> H_wiki -> H_ptb? If they do, the calib-screened lever OVERFITS the calibration
     activation subspace -- explaining why calib-measured levers never transferred downstream,
     and telling us the right target is a held-out-ROBUST encoding, not a calib-optimal one.
 (2) Is the generalization gap SMALLER for the balanced (CoBALT) mask than the wanda
     (unbalanced) mask? A mask-dependent transfer gap = a CoBALT-SPECIFIC handle the encoding
     could exploit (column balance gives reliable per-column stats -> less overfit).

RTN's D is activation-independent (weight-only), so it is the natural non-overfitting anchor;
gain = 1 - e_enc/e_rtn on a FIXED Hessian is the fair per-Hessian normalization. transfer =
gain(H_held) - gain(H_fit): negative => the encoding overfit calibration.
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
from sinq.sparse_quant import quantize_rtn, compute_hessian_inverse  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SP = 0.5
BETA = 0.5
NBITS = 3
GROUP = 128
CAP = 256                       # rows per matrix used to form each Hessian
AWZ_RHOS = (1.0, 0.95, 0.90, 0.85)
AWZ_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)


def out_err(D, H):
    D = D.double
    return float((D @ H.double * D).sum.item)


def awclipz_quantize(W_norm, mask, nbits, gsize, wcol):
    """Joint per-group (SCALE, ZERO) output-weighted selection (fit on wcol=c^2*colE_fit)."""
    K, N = W_norm.shape
    m = mask.bool
    n_levels = 2 ** nbits - 1
    grouped = N > gsize and N % gsize == 0
    if grouped:
        ng = N // gsize
        Wg = W_norm.view(K, ng, gsize); Mg = m.view(K, ng, gsize); wcg = wcol.view(1, ng, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1); wcg = wcol.view(1, 1, N)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    mid0 = 0.5 * (w_min + w_max); half0 = (0.5 * (w_max - w_min)).clamp(min=1e-8)
    best_e = None; best_W = None
    for rho in AWZ_RHOS:
        for dl in AWZ_DELTAS:
            half = half0 * rho
            mid = mid0 + dl * half0
            lo = mid - half; scale = (2.0 * half) / n_levels
            zero = -torch.round(lo / scale)
            q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
            Whn = (q - zero) * scale
            eg = ((Wg - Whn) ** 2 * wcg * Mg).sum(-1, keepdim=True)
            if best_e is None:
                best_e = eg; best_W = Whn
            else:
                take = eg < best_e
                best_e = torch.where(take, eg, best_e); best_W = torch.where(take, Whn, best_W)
    return best_W.reshape(K, N)


def collect(model, tok, device, dataset_key, n_samples, take_slice=None):
    """Collect per-layer activations for a calibration draw. take_slice=(a,b) selects a
    disjoint window of the tokenized stream for an in-distribution held-out split."""
    seq = bs.EVAL_CONFIG["calibration_seq_len"]
    ns_req = n_samples if take_slice is None else take_slice[1]
    cal = bs.get_calibration_data(tok, n_samples=ns_req, seq_len=seq, dataset_key=dataset_key)
    if take_slice is not None:
        cal = cal[take_slice[0]:take_slice[1]]
    return bs.collect_activations(model, cal, device)


def build_encodings(W, W_comp, mask, r, c, colE_fit, block, device):
    """Return {enc_name: D} where D = W_hat - W, all encodings FIT on colE_fit (X_fit)."""
    min_max = [0, 2 ** NBITS - 1]
    mkf = mask.float
    out = {}
    # rtn (activation-independent)
    W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
    q, s, z, _ = quantize_rtn(W_norm, min_max, group_size=block)
    if s.dim == 3:
        s = s * r.view(-1, 1, 1)
    else:
        s = s * r.view(-1, 1)
    s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    W_rtn = eq.dequant_deployed(q, s, z, mkf, c)
    out["rtn"] = W_rtn - W
    # awclip (act-weighted scale, fit on colE_fit)
    W_ac = eq.awclip_only(W_comp, mask, NBITS, block, r, c, colE_fit)
    out["awclip"] = W_ac - W
    # awclipz (act-weighted scale+zero, fit on colE_fit)
    wcol = (c.float ** 2) * colE_fit.float
    W_az = awclipz_quantize(W_norm, mkf, NBITS, block, wcol) * (r.view(-1, 1) * c.view(1, -1)) * mkf
    out["awclipz"] = W_az - W
    return out


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)

    # Three activation draws. FIT = standard cobalt calib (wikitext2, first 16).
    acts_fit = collect(model, tok, device, "wikitext2", 16)
    acts_wiki = collect(model, tok, device, "wikitext2", 16, take_slice=(16, 32))  # disjoint window
    hess_sets = {"fit": acts_fit, "wiki": acts_wiki}
    for xkey in ("ptb", "c4"):
        try:
            hess_sets[xkey] = collect(model, tok, device, xkey, 16)
        except Exception as e:  # offline / streaming hiccup -> skip cross-dist gracefully
            print(f"  [warn] {xkey} held-out unavailable: {repr(e)[:80]}", flush=True)

    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache

    layer_paths = bs.get_layer_paths(model)
    layers = ns.get_layers(model)
    n_layers = len(layers)
    sample_layers = sorted(set([1, n_layers // 2, n_layers - 2]))
    HKEYS = list(hess_sets.keys)
    ENCS = ["rtn", "awclip", "awclipz"]
    MASKS = ["balanced", "wanda"]
    print(f"\n######## {MODEL} layers={n_layers} sample={sample_layers} g={GROUP} "
          f"hessians={HKEYS} ########", flush=True)

    # tot[mask][hkey][enc] = pooled output error
    tot = {mk: {hk: {e: 0.0 for e in ENCS} for hk in HKEYS} for mk in MASKS}
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
            key = f'layer_{li}.{ap}'
            if acts_fit.get(key) is None:
                continue
            W = lin.weight.data.clone.float.to(device)
            K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)

            # Per-Hessian activation matrices (capped) and colE (from FIT only).
            H = {}
            for hk in HKEYS:
                a = hess_sets[hk].get(key)
                if a is None:
                    H[hk] = None
                    continue
                Xh = a.to(device).float
                if Xh.dim == 3:
                    Xh = Xh.reshape(-1, Xh.shape[-1])
                Xh = Xh[:min(Xh.shape[0], CAP)]
                H[hk] = Xh.t @ Xh
            Xf = hess_sets["fit"][key].to(device).float
            if Xf.dim == 3:
                Xf = Xf.reshape(-1, Xf.shape[-1])
            Xf = Xf[:min(Xf.shape[0], CAP)]
            colE_fit = (Xf * Xf).sum(0).clamp(min=0)
            acts_fit_dev = hess_sets["fit"][key].to(device)  # for mask/OBS build

            for mkname in MASKS:
                if mkname == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, acts_fit_dev, SP, device, col_exp=BETA)
                else:
                    W_comp, mask = ns.wanda_mask_and_obs(W, acts_fit_dev, SP, device, scope='per_row')
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                Ds = build_encodings(W, W_comp, mask, r, c, colE_fit, block, device)
                for hk in HKEYS:
                    if H[hk] is None:
                        continue
                    for e in ENCS:
                        tot[mkname][hk][e] += out_err(Ds[e], H[hk])
            nmat += 1
        layers[li] = layer.to("cpu")
        torch.cuda.empty_cache

    # ---- report ----
    print(f"  pooled over {nmat} matrices. GAIN(enc) = 1 - e_enc/e_rtn on each Hessian.", flush=True)
    print(f"  transfer(enc,H) = GAIN(H) - GAIN(fit); negative => overfit calibration.", flush=True)
    for mkname in MASKS:
        print(f"  --- mask = {mkname} ---", flush=True)
        header = "    enc      " + "".join(f"{('GAIN['+hk+']'):>13}" for hk in HKEYS) + \
                 "".join(f"{('trans['+hk+']'):>13}" for hk in HKEYS if hk != "fit")
        print(header, flush=True)
        gains = {}
        for e in ENCS:
            row = f"    {e:<8}"
            gfit = 1 - tot[mkname]["fit"][e] / tot[mkname]["fit"]["rtn"]
            gains[e] = {}
            for hk in HKEYS:
                g = 1 - tot[mkname][hk][e] / tot[mkname][hk]["rtn"]
                gains[e][hk] = g
                row += f"{g*100:+11.1f}%"
            for hk in HKEYS:
                if hk == "fit":
                    continue
                row += f"{(gains[e][hk]-gfit)*100:+11.1f}%"
            print(row, flush=True)
    # mask-differential: does balanced transfer better than wanda?
    print(f"  --- balanced vs wanda TRANSFER differential (bal_trans - wan_trans) ---", flush=True)
    for e in ["awclip", "awclipz"]:
        for hk in HKEYS:
            if hk == "fit":
                continue
            gb = (1 - tot["balanced"][hk][e]/tot["balanced"][hk]["rtn"]) - \
                 (1 - tot["balanced"]["fit"][e]/tot["balanced"]["fit"]["rtn"])
            gw = (1 - tot["wanda"][hk][e]/tot["wanda"][hk]["rtn"]) - \
                 (1 - tot["wanda"]["fit"][e]/tot["wanda"]["fit"]["rtn"])
            flag = "  <== balanced transfers BETTER" if gb > gw + 0.01 else \
                   ("  (transfer-agnostic)" if abs(gb-gw) <= 0.01 else "  (wanda transfers better)")
            print(f"    {e:<8} H={hk:<5} bal_trans={gb*100:+.1f}pt  wan_trans={gw*100:+.1f}pt"
                  f"  Δ={100*(gb-gw):+.1f}pt{flag}", flush=True)


def main:
    P = argparse.ArgumentParser
    P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    A = P.parse_args
    for m in A.models.split(","):
        run_model(m.strip)


if __name__ == "__main__":
    main
