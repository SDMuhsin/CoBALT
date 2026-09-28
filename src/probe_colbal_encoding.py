#!/usr/bin/env python3
"""ISOLATION PROBE (attempt-7b): is the output-aware ENCODING gain AMPLIFIED BY COLUMN
BALANCE — the ONE property CoBALT has that no baseline does?

My earlier fairness screen was CONFOUNDED: it compared cobalt (balanced mask + OBS) vs
wanda-awq (unbalanced, NO OBS), so AWQ's awclip gain was measured on a baseline missing
BOTH balance AND OBS. To isolate column balance, hold OBS + col-norm + RTN pipeline FIXED
and vary ONLY the mask: balanced (CoBALT) vs wanda (unbalanced, per-row) — both +OBS.

For each mask, quantize the OBS-compensated survivors with candidate encodings and measure
the METHOD-AGNOSTIC output error ||X(W_hat - W_dense)||^2. Candidates (all bpw-free,
non-iterative, global rules; scale/zero are already-stored fp16 => no extra storage):
  rtn     : deployed group-RTN (reference)
  awclip  : act-weighted per-group SCALE (attempt-7)
  awclipz : act-weighted per-group SCALE *and* ZERO-POINT jointly (NEW 2nd stored DOF)
  hdiagw  : per-column grid weighted by OBS compensability 1/[H^-1]_jj folded into c (NEW,
            OBS-native — uses CoBALT's Hessian, thrown away after masking today)

KEY OUTPUT: encoding GAIN (1 - e_enc/e_rtn) for balanced vs wanda mask. If a candidate's
gain is LARGER on the balanced mask, column balance AMPLIFIES it => a CoBALT-SPECIFIC
encoding interaction (baselines with unbalanced masks cannot replicate it). If gains are
equal on both masks, the lever is balance-agnostic (universal), confirming the negative.
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
CAP256 = 256
AWZ_RHOS = (1.0, 0.95, 0.90, 0.85)
AWZ_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)   # zero-point shift as fraction of half-range


def out_err(W_hat, W_dense, H):
    D = (W_hat - W_dense).double()
    return float((D @ H.double() * D).sum().item())


def awclipz_quantize(W_norm, mask, nbits, gsize, wcol):
    """Joint per-group (SCALE, ZERO) output-weighted selection: search rho (half-range
    scale) x delta (center shift) on a fixed global grid, pick per-group argmin of
    sum_j wcol_j (W_norm - W_hat)^2. Same 2 stored vals/group as RTN. Non-iterative."""
    K, N = W_norm.shape
    m = mask.bool()
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


def run_model(MODEL, device="cuda"):
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
    min_max = [0, 2 ** NBITS - 1]
    print(f"\n######## {MODEL} layers={n_layers} sample={sample_layers} g={GROUP} ########", flush=True)

    ENCS = ["rtn", "awclip", "awclipz", "hdiagw"]
    tot = {mk: {e: 0.0 for e in ENCS} for mk in ["balanced", "wanda"]}
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
            W = lin.weight.data.clone().float().to(device)
            K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            Xa = acts.to(device).float()
            if Xa.dim() == 3:
                Xa = Xa.reshape(-1, Xa.shape[-1])
            Xa = Xa[:min(Xa.shape[0], CAP256)]
            H = Xa.t() @ Xa
            colE = (Xa * Xa).sum(0).clamp(min=0)
            Hinv = compute_hessian_inverse(Xa, damping=None)
            hdiag_inv = Hinv.diag().clamp(min=1e-8)                 # [N] [H^-1]_jj (compensability)

            for mkname in ["balanced", "wanda"]:
                if mkname == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, acts.to(device), SP, device, col_exp=BETA)
                else:
                    W_comp, mask = ns.wanda_mask_and_obs(W, acts.to(device), SP, device, scope='per_row')
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                mkf = mask.float()
                # rtn
                W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                q, s, z, _ = quantize_rtn(W_norm, min_max, group_size=block)
                if s.dim() == 3:
                    s = s * r.view(-1, 1, 1)
                else:
                    s = s * r.view(-1, 1)
                s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
                W_rtn = eq.dequant_deployed(q, s, z, mkf, c)
                tot[mkname]["rtn"] += out_err(W_rtn, W, H)
                # awclip
                W_ac = eq.awclip_only(W_comp, mask, NBITS, block, r, c, colE)
                tot[mkname]["awclip"] += out_err(W_ac, W, H)
                # awclipz (joint scale+zero)
                wcol = (c.float() ** 2) * colE.float()
                W_az = awclipz_quantize(W_norm, mkf, NBITS, block, wcol) * (r.view(-1, 1) * c.view(1, -1)) * mkf
                tot[mkname]["awclipz"] += out_err(W_az, W, H)
                # hdiagw: fold compensability into per-column scale (finer grid on hard-to-
                # compensate columns). c' = c * (hdiag_inv)^0.25 (mild); decode uses c'.
                c2 = (c * hdiag_inv.pow(0.25)).clamp(min=1e-8)
                Wn2 = W_comp / (r.view(-1, 1) * c2.view(1, -1))
                q2, s2, z2, _ = quantize_rtn(Wn2, min_max, group_size=block)
                if s2.dim() == 3:
                    s2 = s2 * r.view(-1, 1, 1)
                else:
                    s2 = s2 * r.view(-1, 1)
                s2 = torch.nan_to_num(s2, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
                W_hd = eq.dequant_deployed(q2, s2, z2, mkf, c2)
                tot[mkname]["hdiagw"] += out_err(W_hd, W, H)
            nmat += 1
        layers[li] = layer.to("cpu")
        torch.cuda.empty_cache()

    print(f"  encoding GAIN (1 - e/e_rtn), per mask, pooled over {nmat} matrices:", flush=True)
    for e in ["awclip", "awclipz", "hdiagw"]:
        gb = 1 - tot["balanced"][e] / tot["balanced"]["rtn"]
        gw = 1 - tot["wanda"][e] / tot["wanda"]["rtn"]
        flag = "  <== balance AMPLIFIES" if gb > gw + 0.02 else ("  (balance-agnostic)" if abs(gb - gw) <= 0.02 else "  (wanda gains more)")
        print(f"    {e:<8} balanced={gb*100:+.1f}%   wanda={gw*100:+.1f}%   Δ(bal-wan)={100*(gb-gw):+.1f}pt{flag}", flush=True)
    # also the raw cobalt-vs-wanda output error (mask effect) per encoding
    print(f"  cobalt/wanda output-err ratio by encoding (mask effect; <1 => balanced better):", flush=True)
    for e in ENCS:
        print(f"    {e:<8} {tot['balanced'][e]/tot['wanda'][e]:.4f}", flush=True)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    A = P.parse_args()
    for m in A.models.split(","):
        run_model(m.strip())


if __name__ == "__main__":
    main()
