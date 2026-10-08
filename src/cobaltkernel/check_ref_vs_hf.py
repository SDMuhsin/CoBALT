#!/usr/bin/env python
"""Independent check of the pure-torch reference spec (ref_llama / ref_gemma3) against the
transformers implementation, sharing ONE copy of the weights on the GPU.

  python check_ref_vs_hf.py --model <hf snapshot dir> [--tokens 64] [--steps 16]

Reports max |logit diff|, argmax agreement over the prompt, and a greedy-decode token match.
This is the gate that says the *spec* is right before the kernel is held to it.
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel import arch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    R = arch.ref_module(cfg)
    fam = arch.family_of(cfg)
    print(f"family={fam} spec={R.__name__}", flush=True)

    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype),
                                              attn_implementation="sdpa").cuda().eval()
    print(f"HF loaded in {time.time()-t0:.0f}s", flush=True)
    # build the ref state over the SAME tensors (no copy)
    sd = hf.state_dict()
    W = {k: v for k, v in sd.items()}
    if cfg.get("tie_word_embeddings", fam == "gemma3"):
        W["lm_head.weight"] = W["model.embed_tokens.weight"]
    Cfg = R.Gemma3RefConfig if fam == "gemma3" else R.LlamaRefConfig
    rc = Cfg(cfg)
    if fam == "gemma3":
        rope = {lt: R._inv_freq(rc.rope_parameters[lt], rc.head_dim, "cuda") for lt in set(rc.layer_types)}
    else:
        inv = R._inv_freq(rc.rope_theta, rc.head_dim, "cuda")
        rope = {lt: inv for lt in set(rc.layer_types)}
    st = {"cfg": rc, "W": W, "rope": rope, "device": "cuda", "dtype": getattr(torch, a.dtype)}

    tok = AutoTokenizer.from_pretrained(a.model)
    text = open(os.path.join(os.path.dirname(__file__), "ref_gemma3.py")).read()
    ids = tok(text, add_special_tokens=True)["input_ids"][a.seed * 7: a.seed * 7 + a.tokens]
    ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")

    with torch.no_grad():
        hf_logits = hf(ids_t[None]).logits[0].float()
        ref_logits, kv = R.prefill(st, ids_t)
        ref_logits = ref_logits[0].float()
    d = (hf_logits - ref_logits).abs()
    agree = (hf_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    dis = (hf_logits.argmax(-1) != ref_logits.argmax(-1)).nonzero().flatten().tolist()
    for t in dis[:8]:
        h2 = hf_logits[t].topk(2).values; r2 = ref_logits[t].topk(2).values
        print(f"  disagree@{t}: hf top2 margin={float(h2[0]-h2[1]):.4f}  ref margin={float(r2[0]-r2[1]):.4f}  "
              f"|dlogit|@argmax={float((hf_logits[t]-ref_logits[t]).abs().max()):.4f}", flush=True)
    print(f"prefill {len(ids)} tok: max|dlogit|={d.max().item():.4f} mean={d.mean().item():.5f} "
          f"argmax agreement={100*agree:.2f}%  hf_logit_scale={hf_logits.abs().max().item():.1f}", flush=True)

    # greedy decode, both sides from their own prefill
    hf_tokens, ref_tokens = [], []
    with torch.no_grad():
        out = hf.generate(ids_t[None], max_new_tokens=a.steps, do_sample=False)
        hf_tokens = out[0, len(ids):].tolist()
        cur = int(ref_logits[-1].argmax())
        pos = len(ids)
        for s in range(a.steps):
            ref_tokens.append(cur)
            lg = R.forward_step(st, cur, pos, kv)
            cur = int(lg.argmax()); pos += 1
    # HF generate() stops at EOS; compare only up to and including that token
    eos = tok.eos_token_id
    n_cmp = (hf_tokens.index(eos) + 1) if eos in hf_tokens else len(hf_tokens)
    match = sum(int(x == y) for x, y in zip(hf_tokens[:n_cmp], ref_tokens[:n_cmp]))
    print(f"greedy {n_cmp} compared steps (HF stopped at EOS after {n_cmp}): {match}/{n_cmp} tokens identical", flush=True)
    a.steps = n_cmp
    print("hf :", tok.decode(hf_tokens)[:200].replace("\n", " "))
    print("ref:", tok.decode(ref_tokens)[:200].replace("\n", " "))
    ok = agree >= 0.985 and match >= a.steps - 2
    print(f"REFCHECK_DONE {'PASS' if ok else 'FAIL'} agree={100*agree:.2f}% greedy={match}/{a.steps}", flush=True)


if __name__ == "__main__":
    main()
