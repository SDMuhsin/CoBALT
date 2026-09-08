TEXT = ("The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms. "
        "Mitochondria generate most of the cell chemical energy in the form of adenosine triphosphate. "
        "Patients presenting with acute chest pain should be evaluated with a twelve lead electrocardiogram. ")

"""Verify src/cobaltkernel/ref_gemma3.py against HF transformers on gemma-3-4b.

Run:
    source env.sh
    /scratch/root/PTQResearch/env_accel_ref/bin/python src/cobaltkernel/verify_ref.py \
        --model /scratch/root/PTQResearch/accel4bit_models/gemma-3-4b/text_bf16 \
        --prompt-len 1152 --steps 32 \
        --out results/cobaltkernel/ref_verify_gemma-3-4b.txt
"""

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel import ref_gemma3 as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=1152)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--noise-floor", action="store_true", default=True)
    a = ap.parse_args()

    dev = "cuda"
    dt = getattr(torch, a.dtype)
    log = []

    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        log.append(s)

    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    text = (open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_ref.py")).read()
            + " ".join(TEXT.split()) * 20)
    ids = torch.tensor(tok(text).input_ids[: a.prompt_len], dtype=torch.long)
    T = ids.shape[0]

    P(f"model            : {a.model}")
    P(f"prompt tokens    : {T}   (sliding_window={cfg.get('sliding_window')})")
    P(f"decode steps     : {a.steps}")
    P(f"attn impl (HF)   : {a.attn}")
    P(f"dtype            : {a.dtype}")
    P("")

    # ---------------- HF reference ----------------
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dt, attn_implementation=a.attn
    ).to(dev).eval()

    def hf_run(model):
        inp = ids[None].to(dev)
        with torch.no_grad():
            o = model(input_ids=inp, use_cache=True)
            pf = o.logits[0].float().cpu()
            past = o.past_key_values
            cur = int(pf[-1].argmax())
            toks = []
            for s in range(a.steps):
                toks.append(cur)
                oo = model(
                    input_ids=torch.tensor([[cur]], device=dev),
                    past_key_values=past,
                    use_cache=True,
                    cache_position=torch.tensor([T + s], device=dev),
                )
                past = oo.past_key_values
                cur = int(oo.logits[0, -1].argmax())
            last = oo.logits[0, -1].float().cpu()
        del o, oo, past
        gc.collect()
        torch.cuda.empty_cache()
        return pf, toks, last

    hf_prefill, hf_tokens, hf_last_logits = hf_run(hf)

    # HF-vs-HF noise floor: the same model, same weights, different kernel /
    # reduction order (sdpa).  Any residual ref-vs-HF gap below this is bf16
    # reduction-order noise, not a spec error.
    nf = None
    if a.noise_floor:
        hf.set_attn_implementation("sdpa")
        nf_prefill, nf_tokens, _ = hf_run(hf)
        nf = (nf_prefill, nf_tokens)

    del hf
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- our reference ----------------
    st = R.load_state(a.model, device=dev, dtype=dt)
    logits, kv = R.prefill(st, ids)
    ref_prefill = logits[0].float().cpu()
    del logits
    torch.cuda.empty_cache()

    cur = int(ref_prefill[-1].argmax())
    ref_tokens = []
    for s in range(a.steps):
        ref_tokens.append(cur)
        lg = R.forward_step(st, cur, T + s, kv)
        cur = int(lg.argmax())
    ref_last_logits = lg.float().cpu()

    # ---------------- compare ----------------
    d = (hf_prefill - ref_prefill).abs()
    scale = hf_prefill.abs().max().item()
    am_hf = hf_prefill.argmax(-1)
    am_ref = ref_prefill.argmax(-1)
    agree = (am_hf == am_ref).float().mean().item()

    P("== prefill ==")
    P(f"  logit range (HF)       : [{hf_prefill.min():.4f}, {hf_prefill.max():.4f}]  |max|={scale:.4f}")
    P(f"  max |diff|             : {d.max().item():.6f}")
    P(f"  mean |diff|            : {d.mean().item():.6f}")
    P(f"  max |diff| / |logit|max: {d.max().item()/scale:.3e}")
    P(f"  argmax agreement       : {agree*100:.4f}%  ({int((am_hf==am_ref).sum())}/{T} positions)")
    if agree < 1.0:
        bad = (am_hf != am_ref).nonzero().flatten().tolist()[:10]
        P(f"  first mismatched pos   : {bad}")
        for b in bad[:3]:
            t1, t2 = int(am_hf[b]), int(am_ref[b])
            P(f"    pos {b}: hf={t1} ({hf_prefill[b,t1]:.5f}/{hf_prefill[b,t2]:.5f})"
              f"  ref={t2} ({ref_prefill[b,t1]:.5f}/{ref_prefill[b,t2]:.5f})")
    # sanity: are local layers actually masking?
    P(f"  positions > window     : {max(0, T - cfg.get('sliding_window', 0))} (local layers must mask)")
    P("")
    P("== 32-step greedy decode ==")
    P(f"  HF  tokens : {hf_tokens}")
    P(f"  ref tokens : {ref_tokens}")
    nmatch = sum(int(x == y) for x, y in zip(hf_tokens, ref_tokens))
    P(f"  token agreement: {nmatch}/{a.steps}")
    dl = (hf_last_logits - ref_last_logits).abs()
    P(f"  last-step max|diff|: {dl.max().item():.6f}  (|logit|max={hf_last_logits.abs().max():.4f})")
    if nf is not None:
        nfp, nft = nf
        dn = (hf_prefill - nfp).abs()
        agn = (am_hf == nfp.argmax(-1)).float().mean().item()
        nfm = sum(int(x == y) for x, y in zip(hf_tokens, nft))
        P("== noise floor: HF-eager vs HF-sdpa (identical weights, identical spec) ==")
        P(f"  max |diff|       : {dn.max().item():.6f}")
        P(f"  mean |diff|      : {dn.mean().item():.6f}")
        P(f"  argmax agreement : {agn*100:.4f}%")
        P(f"  decode agreement : {nfm}/{a.steps}")
        P("")
    # Verdict: exact argmax agreement, OR (in bf16) agreement no worse than the
    # HF-eager-vs-HF-sdpa noise floor -- i.e. our residual gap is bf16
    # reduction-order noise, not a specification error.
    ok = (nmatch == a.steps) and (agree == 1.0 or (nf is not None and agree >= agn))
    P("RESULT: " + ("PASS" if ok else "FAIL"))

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write("\n".join(log) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
