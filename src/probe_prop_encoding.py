#!/usr/bin/env python3
"""PROPAGATION x ENCODING probe (attempt-8e). All single-matrix probes measure tr(D H D^T) on ONE
matrix; downstream error is actually driven by how per-layer quant error PROPAGATES through the
residual stream (collapse/relerr, [[cobalt-win-is-bottleneck-contingent]]). Mechanism-so-far says
encodings can't capture a MASK advantage (balance already fixes propagation). This probe TESTS that:
quantize the FULL model 4 ways {balanced,wanda} x {rtn,awclip}, measure per-layer relerr vs dense +
end-to-end PPL. If awclip reduces PROPAGATED error (final relerr / PPL) MORE for balanced than wanda,
the wall BREAKS and propagation-aware encoding is a real lead. If Δ is balance-agnostic, the wall holds.

relerr[l] = ||h_q[l]-h_dense[l]||_F / ||h_dense[l]||_F on a held-out (ptb) batch. g128/3bit/sp0.5/beta0.5.
"""
import argparse, copy, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"
SP, BETA, NBITS, GROUP = 0.5, 0.5, 3, 128


def capture_hidden(model, batch):
    caps = {}; hooks = []
    layers = bs.get_transformer_layers(model)
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[i] = o.detach().float().cpu()
        return h
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_hook(mk(i)))
    with torch.no_grad():
        model(batch.to(DEV))
    for h in hooks: h.remove()
    return [caps[i] for i in range(len(layers))]


def relerr(hq, hd):
    return [float((a - b).norm() / b.norm().clamp(min=1e-8)) for a, b in zip(hq, hd)]


def load(name):
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    return m


def run_model(MODEL):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=16, seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"], n_samples=20, dataset_key="wikitext2")
    try:
        rb = bs.get_calibration_data(tok, n_samples=2, seq_len=512, dataset_key="ptb")[:1]  # held-out relerr batch
    except Exception:
        rb = bs.get_calibration_data(tok, n_samples=2, seq_len=512, dataset_key="wikitext2")[:1]
    TYPES = ns.TYPES
    sp_by_type = {t: SP for t in TYPES}
    print(f"\n######## {MODEL} sp={SP} nbits={NBITS} g={GROUP} beta={BETA} ########", flush=True)

    # dense reference
    dense = load(name); dense.to(DEV); dense.eval()
    hd = capture_hidden(dense, rb)
    del dense; torch.cuda.empty_cache()

    CONFIGS = [("balanced", "rtn"), ("balanced", "awclip"), ("wanda", "rtn"), ("wanda", "awclip")]
    res = {}
    for mask, enc in CONFIGS:
        m = load(name); bs.move_embed_to_device(m, DEV)
        kw = dict(norm="col", awq_alpha=0.5, dense_norm="col", group_size=GROUP, quantizer=enc)
        if mask == "balanced":
            kw.update(mask_mode="balanced", mask_scope="global", col_balance_exp=BETA)
        else:
            kw.update(mask_mode="wanda", mask_scope="per_row")
        m, _ = ns.apply_wanda_obs_rtn(m, cal, NBITS, sp_by_type, DEV, **kw)
        m.to(DEV); m.eval()
        hq = capture_hidden(m, rb)
        re = relerr(hq, hd)
        ppl = float(bs.evaluate_perplexity(m, test, DEV))
        res[(mask, enc)] = (re, ppl)
        print(f"    {mask:<9} {enc:<7} final_relerr={re[-1]:.4f} mean_relerr={sum(re)/len(re):.4f} ppl={ppl:.4f}", flush=True)
        del m; torch.cuda.empty_cache()

    # does awclip reduce propagated error MORE for balanced?
    def d(mask, i):
        return res[(mask, "rtn")][0][i] - res[(mask, "awclip")][0][i]   # relerr reduction (rtn - awclip)
    fb = d("balanced", -1); fw = d("wanda", -1)
    pplb = res[("balanced", "rtn")][1] - res[("balanced", "awclip")][1]
    pplw = res[("wanda", "rtn")][1] - res[("wanda", "awclip")][1]
    print(f"  awclip PROPAGATED-error reduction (rtn-awclip): final_relerr bal={fb:+.4f} wan={fw:+.4f} Δ={fb-fw:+.4f}", flush=True)
    print(f"  awclip PPL reduction (rtn-awclip): bal={pplb:+.4f} wan={pplw:+.4f} Δ={pplb-pplw:+.4f}"
          f"  {'<== balance BREAKS wall' if (fb>fw+0.005 or pplb>pplw+0.05) else '(balance-agnostic: wall holds)'}", flush=True)


def main():
    P = argparse.ArgumentParser(); P.add_argument("--models", default="gemma-2b")
    A = P.parse_args()
    for m in A.models.split(","): run_model(m.strip())


if __name__ == "__main__":
    main()
