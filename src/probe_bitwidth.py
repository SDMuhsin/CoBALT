#!/usr/bin/env python3
"""BIT-WIDTH anti-pitfall sweep (attempt-8k). The whole encoding census pinned 3-bit. 2-bit is a
DISTINCT CoBALT regime (collapse, [[cobalt-2bit-collapse-regime]]) where the coarse grid makes the
SCALE/clip matter far more. Measure awclip held-out(ptb) GAIN vs RTN at nbits {2,3,4}, and its
balance-DIFFERENTIAL (balanced - wanda). If awclip's gain becomes balance-SPECIFIC at 2-bit
(balanced >> wanda/magnitude), the scale axis reopens as CoBALT-specific in the 2-bit regime. sp0.5,
3 families.
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
from probe_heldout_gap import collect, out_err  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

SP = float(os.environ.get("PROBE_SP", "0.5"))
BETA, GROUP, CAP = 0.5, 128, 256
BITS = [2, 3, 4]


def rtn_D(W_comp, mask, r, c, nbits, block, W):
    mm = [0, 2 ** nbits - 1]; mkf = mask.float()
    W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
    q, s, z, _ = quantize_rtn(W_norm, mm, group_size=block)
    s = (s * r.view(-1, 1, 1)) if s.dim() == 3 else (s * r.view(-1, 1))
    s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    return eq.dequant_deployed(q, s, z, mkf, c) - W


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
    torch.cuda.empty_cache()
    HKEY = "ptb" if "ptb" in A else "fit"
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} H={HKEY} sp={SP} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    tot = {mk: {b: {"rtn": 0.0, "awclip": 0.0} for b in BITS} for mk in MK}
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
            W = lin.weight.data.clone().float().to(device); K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            Xh = A[HKEY][key].to(device).float()
            if Xh.dim() == 3: Xh = Xh.reshape(-1, Xh.shape[-1])
            Xh = Xh[:min(Xh.shape[0], CAP)]
            H = Xh.t() @ Xh
            Xf = A["fit"][key].to(device).float()
            if Xf.dim() == 3: Xf = Xf.reshape(-1, Xf.shape[-1])
            Xf = Xf[:min(Xf.shape[0], CAP)]
            colE = (Xf * Xf).sum(0).clamp(min=0)
            X = A["fit"][key].to(device)
            for mk in MK:
                if mk == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, X, SP, device, col_exp=BETA)
                elif mk == "wanda":
                    W_comp, mask = ns.wanda_mask_and_obs(W, X, SP, device, scope='per_row')
                else:
                    W_comp, mask = magnitude_mask_and_obs(W, X, SP, device)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                for b in BITS:
                    D_rtn = rtn_D(W_comp, mask, r, c, b, block, W)
                    D_aw = eq.awclip_only(W_comp, mask, b, block, r, c, colE) - W
                    tot[mk][b]["rtn"] += out_err(D_rtn, H)
                    tot[mk][b]["awclip"] += out_err(D_aw, H)
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
    print(f"  awclip held-out({HKEY}) GAIN vs RTN per nbits, and balance-differential (bal - wanda):", flush=True)
    for b in BITS:
        g = {mk: 1 - tot[mk][b]["awclip"] / tot[mk][b]["rtn"] for mk in MK}
        diff = g["balanced"] - g["wanda"]
        flag = "  <== balance-SPECIFIC" if diff > 0.02 else "  (balance-agnostic)"
        print(f"    {b}-bit  bal={g['balanced']*100:+.1f}%  wan={g['wanda']*100:+.1f}%  mag={g['magnitude']*100:+.1f}%"
              f"  Δ(bal-wan)={diff*100:+.1f}pt{flag}", flush=True)


def main():
    P = argparse.ArgumentParser(); P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args()
    for m in Aa.models.split(","): run_model(m.strip())


if __name__ == "__main__":
    main()
