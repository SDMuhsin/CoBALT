#!/usr/bin/env python3
"""SURVIVOR TRANSFORM-CODING-GAIN headroom (attempt-8i). probe_regime_dist found survivor |adj corr|
rises to ~0.18 at sp0.8 (OBS-induced, CoBALT's winning regime) -- reopening the joint axis. Pairwise
0.18 => only ~1.6% via DPCM, BUT if the correlation is structured GROUP-WIDE a decorrelating transform
(KLT) saves more. This measures the definitive headroom = transform coding gain G of the per-group
survivor covariance: G = mean(diag Sigma)/geomean(eig Sigma); bits saved = 0.5*log2(G). Oracle (ignores
the rotation-storage/format cost => UPPER BOUND on any joint encoding). sp0.5 vs sp0.8, balanced vs
wanda vs magnitude, 3 families. If G_bits is tiny at sp0.8 the reopening has no exploitable headroom;
if large AND balance/OBS-specific, build a deployable approximation.
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

BETA, GROUP = 0.5, 128
SPS = [0.5, 0.8]
MINCO = 8  # min co-survival count to trust a covariance entry / column


def coding_gain_bits(W_norm, mask, gsize):
    """mean over groups of 0.5*log2(G), G=transform coding gain of pairwise-complete survivor cov."""
    K, N = W_norm.shape
    if not (N > gsize and N % gsize == 0):
        return None
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize); Mg = mask.view(K, ng, gsize).float()
    out = []
    for g in range(ng):
        Wc = Wg[:, g, :]; Mc = Mg[:, g, :]                     # [K, gsize]
        cnt = Mc.sum(0)                                         # [gsize] survivors per column
        keep = cnt >= MINCO
        if keep.sum() < 4:
            continue
        Wc = Wc[:, keep]; Mc = Mc[:, keep]; cnt = cnt[keep]
        mu = (Wc * Mc).sum(0) / cnt
        Wcen = (Wc - mu.view(1, -1)) * Mc                      # 0 at pruned
        C = Mc.t() @ Mc                                        # co-survival counts [d,d]
        S = Wcen.t() @ Wcen
        Sigma = S / C.clamp(min=1.0)
        Sigma = 0.5 * (Sigma + Sigma.t())                     # symmetrize
        d = Sigma.shape[0]
        Sigma = Sigma + 1e-6 * Sigma.diag().mean() * torch.eye(d, device=Sigma.device)
        diag = Sigma.diag().clamp(min=1e-12)
        try:
            ev = torch.linalg.eigvalsh(Sigma)
        except Exception:
            continue
        # condition-number floor: eigenvalues below max_eig*1e-3 are noise, not structure
        ev = ev.clamp(min=ev.max() * 1e-3)
        G = diag.mean() / torch.exp(torch.log(ev).mean())
        out.append(0.5 * torch.log2(G.clamp(min=1e-6)))
    if not out:
        return None
    return float(torch.stack(out).mean().item())


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = collect(model, tok, device, "wikitext2", 16)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache()
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    acc = {mk: {sp: [0.0, 0] for sp in SPS} for mk in MK}
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
            W = lin.weight.data.clone().float().to(device); K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            X = A[key].to(device)
            for mk in MK:
                for sp in SPS:
                    if mk == "balanced":
                        W_comp, mask = ns.balanced_mask_and_obs(W, X, sp, device, col_exp=BETA)
                    elif mk == "wanda":
                        W_comp, mask = ns.wanda_mask_and_obs(W, X, sp, device, scope='per_row')
                    else:
                        W_comp, mask = magnitude_mask_and_obs(W, X, sp, device)
                    r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                    W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                    v = coding_gain_bits(W_norm, mask, block)
                    if v is not None: acc[mk][sp][0] += v; acc[mk][sp][1] += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
    print(f"  ORACLE group-KLT coding gain (bits saved / weight; UPPER bound on any joint encoding):", flush=True)
    for mk in MK:
        row = f"    {mk:<10}"
        for sp in SPS:
            g = acc[mk][sp][0]/max(1, acc[mk][sp][1])
            row += f"  sp{sp}={g:.3f}bit"
        print(row, flush=True)


def main():
    P = argparse.ArgumentParser(); P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args()
    for m in Aa.models.split(","): run_model(m.strip())


if __name__ == "__main__":
    main()
