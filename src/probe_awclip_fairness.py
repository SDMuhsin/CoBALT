#!/usr/bin/env python3
"""FAIRNESS SCREEN (attempt-7, gate b): is awclip's edge CoBALT-specific, or does the
strongest matched baseline (AWQ = the exact activation-aware-clip prior art the critic
cited) gain the SAME lever? This is the repack test (BLACKLIST A1): a mask-agnostic lever
that helps the baselines equally is NOT a CoBALT contribution.

For 3 families, sampled layers, matched group=128, sp0.5 3-bit, per matrix we build:
  cobalt   : balanced mask + OBS + col-norm  (W_target=W_comp, r=1, c=col-std)
  wanda-awq: per-row Wanda mask + AWQ scale   (W_target=W_pruned, r=1, c=1/awq_scale)
and quantize survivors two ways each: RTN (deployed) and AWCLIP (act-weighted scale).

Metric = METHOD-AGNOSTIC output error ||X (W_hat - W_dense)||_F^2 (W_dense = ORIGINAL fp16
weight, the SAME reference for every arm) so cross-method comparison is honest (not each
method's own OBS target). Pooled over matrices, per model. Decisions:
  * awclip GAIN per arm = 1 - e_awclip/e_rtn  (does clip help cobalt MORE than awq?)
  * FAIR head-to-head    = e(cobalt-awclip) / e(awq-awclip)  (<1 => cobalt wins clip-equipped)
  * RIGGED head-to-head  = e(cobalt-awclip) / e(awq-rtn)     (the smoke's comparison, for contrast)
If cobalt-awclip does NOT beat awq-awclip pooled on all 3 => repack redux, report null.
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
from sinq.sparse_quant import quantize_rtn  # noqa: E402
from sinq.awq import compute_awq_scale, tiled_fake_quant_rectangle, rtn_fake_quant  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SP = 0.5
BETA = 0.5
NBITS = 3
GROUP = 128
CAP256 = 256


def out_err(W_hat, W_dense, H):
    D = (W_hat - W_dense).double()
    return float((D @ H.double() * D).sum().item())


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

    tot = {k: 0.0 for k in ["cob_rtn", "cob_awc", "awq_rtn", "awq_awc"]}
    win_cob_vs_awqawc = 0; nmat = 0
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
            colE = (Xa * Xa).sum(0).clamp(min=0)                # [N] ||X_j||^2

            # ---- COBALT arm (balanced mask + OBS + col-norm) ----
            W_comp, mask_c = ns.balanced_mask_and_obs(W, acts.to(device), SP, device, col_exp=BETA)
            r_c, c_c = ns.compute_norm_scales(W_comp, mask_c, 'col', device)
            W_norm_c = W_comp / (r_c.view(-1, 1) * c_c.view(1, -1))
            q, s, z, _ = quantize_rtn(W_norm_c, min_max, group_size=block)
            if s.dim() == 3:
                s = s * r_c.view(-1, 1, 1)
            else:
                s = s * r_c.view(-1, 1)
            s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
            W_cob_rtn = eq.dequant_deployed(q, s, z, mask_c.float(), c_c)
            W_cob_awc = eq.awclip_only(W_comp, mask_c, NBITS, block, r_c, c_c, colE)

            # ---- WANDA-AWQ arm (per-row Wanda mask + AWQ scale) ----
            scaler_row = bs._wanda_scaler_row(acts, device)
            mask_a = bs._wanda_row_mask(W, scaler_row, SP)
            W_pruned = W * mask_a
            awq_scales = compute_awq_scale(W_pruned, Xa, min_max, tile=block, method='awq')
            W_awq_rtn = tiled_fake_quant_rectangle(W_pruned, fakequant=rtn_fake_quant,
                                                   min_max=min_max, block=block,
                                                   scales=awq_scales) * mask_a
            r_a = torch.ones(K, device=device)
            c_a = (1.0 / awq_scales.to(device).float().clamp(min=1e-8)).view(-1)
            W_awq_awc = eq.awclip_only(W_pruned, mask_a, NBITS, block, r_a, c_a, colE)

            e = {"cob_rtn": out_err(W_cob_rtn, W, H), "cob_awc": out_err(W_cob_awc, W, H),
                 "awq_rtn": out_err(W_awq_rtn, W, H), "awq_awc": out_err(W_awq_awc, W, H)}
            for k in tot:
                tot[k] += e[k]
            if e["cob_awc"] < e["awq_awc"]:
                win_cob_vs_awqawc += 1
            nmat += 1
        layers[li] = layer.to("cpu")
        torch.cuda.empty_cache()

    g_cob = 1 - tot["cob_awc"] / tot["cob_rtn"]
    g_awq = 1 - tot["awq_awc"] / tot["awq_rtn"]
    print(f"  pooled ||X(W_hat-W_dense)||^2 over {nmat} matrices:", flush=True)
    print(f"    awclip GAIN (rtn->awclip):  COBALT={g_cob*100:+.1f}%   AWQ={g_awq*100:+.1f}%", flush=True)
    print(f"    FAIR   cob-awclip / awq-awclip = {tot['cob_awc']/tot['awq_awc']:.4f}  "
          f"(cob wins {win_cob_vs_awqawc}/{nmat})   [<1 => CoBALT-specific edge survives]", flush=True)
    print(f"    RIGGED cob-awclip / awq-RTN    = {tot['cob_awc']/tot['awq_rtn']:.4f}  "
          f"(the smoke's non-fair comparison, for contrast)", flush=True)
    print(f"    ref    cob-RTN    / awq-RTN    = {tot['cob_rtn']/tot['awq_rtn']:.4f}  "
          f"(mask/OBS effect BEFORE any clip lever)", flush=True)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    A = P.parse_args()
    for m in A.models.split(","):
        run_model(m.strip())


if __name__ == "__main__":
    main()
