"""Correctness harness for the Gemma3 decode megakernel vs ref_gemma3.py.

    source scripts/cobaltkernel_env.sh 1g
    python src/cobaltkernel/verify_kernel.py --model .../gemma-3-4b/text_bf16 \
        --prompt-len 1152 --steps 32 --out results/cobaltkernel/kernel_bf16_verify_gemma-3-4b.md

Tests
  (i)   token-by-token kernel prefill vs ref prefill: max|diff| + argmax agreement
        over ALL positions.  Bar = the bf16 HF-eager-vs-HF-sdpa floor recorded in
        results/cobaltkernel/ref_verify_gemma-3-4b.txt (98.61 % argmax, 32/32 greedy).
  (ii)  32 greedy decode steps: token-for-token agreement.
  (ii-b) teacher-forced decode: the kernel fed the reference's own tokens, argmax per step.
  (iii) batch M=4 over 4 different prompts == 4 independent M=1 runs (bit-exact for the
        row-loop build; identical tokens + --batch-tol logits for the tensor-core build).
  An EXACT tie in the reference's top-2 logits counts as agreement everywhere: the
  reference cannot adjudicate it, and bf16 logits near 16-32 tie at a 0.125 grid.
  --bisect: per-layer h dump (kernel) vs ref, to localise a mismatch.
"""

import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel import arch                     # noqa: E402
from cobaltkernel import ref_gemma3 as R          # noqa: E402  (re-bound per model family in main)
from cobaltkernel.runner import KernelRunner      # noqa: E402
from cobaltkernel import dequant_ref as DQ        # noqa: E402

FLOOR_ARGMAX = 0.9861   # HF-eager vs HF-sdpa, bf16, same weights & spec
TEXT = ("The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms. "
        "Mitochondria generate most of the cell chemical energy in the form of adenosine triphosphate. "
        "Patients presenting with acute chest pain should be evaluated with a twelve lead electrocardiogram. ")


def get_ids(model, n, seed=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    here = os.path.dirname(os.path.abspath(__file__))
    txt = open(os.path.join(here, "ref_gemma3.py")).read() + " ".join(TEXT.split()) * 40
    ids = tok(txt).input_ids
    assert len(ids) >= n + seed * 97, "not enough tokens"
    return ids[seed * 97: seed * 97 + n]


def _attention_fp32probs(cfg, W, p, h, cos, sin, kv_layer, layer_type, q_positions, append=True):
    """ref_gemma3._attention with the softmax probabilities kept in FP32 for the
    @V product.  HF (and ref_gemma3) cast them to bf16 first; the CUDA kernel
    uses an fp32 online softmax, i.e. it is deliberately MORE accurate here.
    This variant reproduces the kernel's dtype policy so the two can be compared
    without that one known-and-intended difference in the way."""
    import torch.nn.functional as F
    T = h.shape[1]
    D = cfg.head_dim
    q = F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, T, cfg.num_attention_heads, D).transpose(1, 2)
    k = F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, T, cfg.num_key_value_heads, D).transpose(1, 2)
    v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, T, cfg.num_key_value_heads, D).transpose(1, 2)
    if (p + "self_attn.q_norm.weight") in W:          # QK-norm: gemma3 only
        q = R.rms_norm(q, W[p + "self_attn.q_norm.weight"], cfg.rms_norm_eps)
        k = R.rms_norm(k, W[p + "self_attn.k_norm.weight"], cfg.rms_norm_eps)
    q = R.apply_rope(q, cos, sin)
    k = R.apply_rope(k, cos, sin)
    if append:
        kv_layer["k"] = k if kv_layer["k"] is None else torch.cat([kv_layer["k"], k], 2)
        kv_layer["v"] = v if kv_layer["v"] is None else torch.cat([kv_layer["v"], v], 2)
    K, V = kv_layer["k"], kv_layer["v"]
    S = K.shape[2]
    Kr = R.repeat_kv(K, cfg.num_kv_groups)
    Vr = R.repeat_kv(V, cfg.num_kv_groups)
    attn = torch.matmul(q, Kr.transpose(2, 3)) * cfg.attn_scale
    kp = torch.arange(S, device=q.device)
    qp = q_positions[:, None]
    allowed = kp[None, :] <= qp
    if layer_type == "sliding_attention":
        allowed &= kp[None, :] > qp - cfg.sliding_window
    attn = attn.masked_fill(~allowed[None, None], float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32)          # NO bf16 cast
    out = torch.matmul(attn, Vr.float()).to(q.dtype)
    out = out.transpose(1, 2).reshape(1, T, -1)
    return F.linear(out, W[p + "self_attn.o_proj.weight"])


def ref_layer_dump(state, token_id, pos, kv):
    """ref_gemma3._run with the layer-entry hidden states captured."""
    cfg, W = state["cfg"], state["W"]
    ids = torch.tensor([token_id], device=state["device"], dtype=torch.long)
    positions = torch.tensor([pos], device=state["device"])
    h = R._embed(state, ids)
    cos, sin = {}, {}
    for lt, inv in state["rope"].items():
        cos[lt], sin[lt] = R.rope_cos_sin(inv, positions, state["dtype"])
    dump = [h[0, 0].clone()]
    for i in range(cfg.num_hidden_layers):
        h = R._layer(cfg, W, i, h, cos, sin, kv, positions, True)
        dump.append(h[0, 0].clone())
    return torch.stack(dump)          # [L+1, hidden]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="bf16 HF dir, or a CBK1 packed dir (needs --config)")
    ap.add_argument("--config", default=None,
                    help="dir holding config.json/tokenizer when --model is packed")
    ap.add_argument("--ref-packed", action="store_true",
                    help="build the oracle by dequantizing the packed --model itself "
                         "(the only apples-to-apples oracle for the quantized path)")
    ap.add_argument("--ref-model", default=None,
                    help="HF checkpoint used as the ORACLE (for a packed --model this "
                         "must be the matching fakequant checkpoint)")
    ap.add_argument("--prompt-len", type=int, default=1152)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch-prompt-len", type=int, default=96)
    ap.add_argument("--batch-steps", type=int, default=8)
    ap.add_argument("--bisect", action="store_true")
    ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--batch-tol", type=float, default=0.0,
                    help="(iii) max |logit diff| allowed between M=4 and M=1 (tokens must still be "
                         "identical). 0 = bit-exact, the rule for the row-loop build. The tensor-core "
                         "decode build (COBALT_BLK1632_MMA=1) runs qkv/o_proj on mma with fp32 slice sums "
                         "at M=1 only, so M=4 is a different (correct) reduction order there.")
    ap.add_argument("--skip-prefill-kernel", action="store_true",
                    help="skip test (iv): the PREFILL megakernel + this decode "
                         "megakernel vs decode-only, through KernelRunner.generate()")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log = []

    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        log.append(s)

    T = a.prompt_len
    cfgdir = a.config or a.model
    refdir = a.ref_model or a.model
    global R
    import json as _json
    R = arch.ref_module(_json.load(open(os.path.join(cfgdir, "config.json"))))
    P(f"[ref] family={arch.family_of(_json.load(open(os.path.join(cfgdir, 'config.json'))))} "
      f"spec={R.__name__}")
    ids = get_ids(cfgdir, T, seed=a.prompt_seed)
    max_ctx = T + a.steps + 8

    # ---------------- reference ----------------
    if a.ref_packed:
        st = DQ.load_state_packed(a.model, cfgdir, device="cuda", dtype=torch.bfloat16)
        refdir = f"torch dequant of {a.model}"
    else:
        st = R.load_state(refdir, device="cuda", dtype=torch.bfloat16)
    logits, kv = R.prefill(st, torch.tensor(ids, dtype=torch.long))
    ref_prefill = logits[0].float().cpu()
    del logits
    torch.cuda.empty_cache()
    # Reference run ONE TOKEN AT A TIME.  This is the apples-to-apples control:
    # ref.prefill uses [T,K]x[K,N] GEMMs, the kernel (and this) use GEMVs, and
    # that alone changes the bf16 reduction order.  ref_step-vs-ref_prefill is
    # therefore the honest floor for a token-by-token evaluator.
    kv_s = R.new_kv(st)
    ref_step = torch.empty(T, ref_prefill.shape[1], dtype=torch.float32)
    for t in range(T):
        ref_step[t] = R.forward_step(st, ids[t], t, kv_s).float().cpu()
    del kv_s
    torch.cuda.empty_cache()
    # ... and once more with the kernel's fp32-softmax-probability policy.
    _orig_attn = R._attention
    R._attention = _attention_fp32probs
    kv_f = R.new_kv(st)
    ref_fp32p = torch.empty(T, ref_prefill.shape[1], dtype=torch.float32)
    for t in range(T):
        ref_fp32p[t] = R.forward_step(st, ids[t], t, kv_f).float().cpu()
    del kv_f
    # greedy control: the SAME reference, decoded greedily, differing only in the
    # softmax-probability dtype.  32-step greedy agreement on a near-tie-rich text is
    # a coin flip, so this says how much of a disagreement is achievable at all.
    kv_g = R.new_kv(st)
    lgp, _ = R.prefill(st, torch.tensor(ids, dtype=torch.long), kv_g)
    cur_g = int(lgp[0, -1].argmax())
    del lgp
    fp32p_tokens = []
    fp32p_logits = []
    for s in range(a.steps):
        fp32p_tokens.append(cur_g)
        lg = R.forward_step(st, cur_g, T + s, kv_g)
        fp32p_logits.append(lg.float().cpu().clone())
        cur_g = int(lg.argmax())
    del kv_g, lg
    R._attention = _orig_attn
    torch.cuda.empty_cache()
    cur = int(ref_prefill[-1].argmax())
    ref_tokens = []
    for s in range(a.steps):
        ref_tokens.append(cur)
        lg = R.forward_step(st, cur, T + s, kv)
        cur = int(lg.argmax())
    ref_last = lg.float().cpu()
    del kv, lg
    gc.collect(); torch.cuda.empty_cache()

    if a.bisect:
        bkv = R.new_kv(st)
        bd = None
        for t in range(min(8, T)):
            bd = ref_layer_dump(st, ids[t], t, bkv)
        ref_dump = bd.float().cpu()
    del st
    gc.collect(); torch.cuda.empty_cache()
    P(f"[ref] loaded, prefill {T} tok, {a.steps} greedy steps done")

    # ---------------- kernel ----------------
    r = KernelRunner(a.model, M=1, max_ctx=max_ctx, config_dir=a.config)
    P(f"model     : {a.model}   (quantized={r.quantized})")
    quantized = r.quantized
    P(f"oracle    : {refdir}")
    P(f"grid      : {r.blocks} blocks x 256 thr ({r.blocks_per_sm} blocks/SM), "
      f"{r.smem} B dynamic smem")
    P(f"config    : hidden={r.hidden} L={r.n_layers} H={r.n_heads}/{r.n_kv} D={r.head_dim} "
      f"inter={r.inter} vocab={r.vocab} attn_scale={r.attn_scale:.10f} "
      f"embed_scale={r.embed_scale}")
    P("")

    if a.bisect:
        r.reset()
        for t in range(min(8, T)):
            r.step([ids[t]], [t], debug=True)
        kd = r.dbg_h[:, 0].float().cpu()
        P("== per-layer bisect (h entering layer L, 8th token) ==")
        for L in range(kd.shape[0]):
            d = (kd[L] - ref_dump[L]).abs()
            rel = d.max().item() / max(1e-9, ref_dump[L].abs().max().item())
            P(f"  L{L:3d}  max|diff|={d.max().item():.6f}  rel={rel:.3e}")
        P("")

    # (i) prefill token-by-token
    r.reset()
    stats = {k: dict(n=0, mx=0.0, mean=0.0, bad=[], ties=0) for k in ("prefill", "step", "fp32p")}
    for t in range(T):
        lg, nxt = r.step([ids[t]], [t])
        kl = lg[0].float().cpu()
        ka = int(kl.argmax())
        assert ka == int(nxt[0]), f"kernel argmax {int(nxt[0])} != logits argmax {ka} @ {t}"
        for key, ref in (("prefill", ref_prefill[t]), ("step", ref_step[t]),
                         ("fp32p", ref_fp32p[t])):
            st_ = stats[key]
            d = (kl - ref).abs()
            st_["mx"] = max(st_["mx"], d.max().item())
            st_["mean"] += d.mean().item()
            ra = int(ref.argmax())
            if ka == ra:
                st_["n"] += 1
            elif float(ref[ka]) == float(ref[ra]):
                st_["n"] += 1          # exact tie in the reference: not adjudicable
                st_["ties"] += 1
            elif len(st_["bad"]) < 5:
                st_["bad"].append((t, ra, ka, float(ref[ra]), float(ref[ka])))
    # reference self-consistency: same weights, same spec, GEMM vs GEMV
    ref_self = float((ref_prefill.argmax(-1) == ref_step.argmax(-1)).float().mean())
    ref_self_mx = float((ref_prefill - ref_step).abs().max())
    ref_dtype = float((ref_prefill.argmax(-1) == ref_fp32p.argmax(-1)).float().mean())
    agree = stats["fp32p"]["n"] / T
    P("== (i) prefill positions, kernel (token-by-token) vs reference ==")
    P(f"  logit |max| (ref)           : {ref_prefill.abs().max().item():.4f}")
    for key, label in (("prefill", "vs ref.prefill  (GEMM ref, HF-exact dtypes)"),
                       ("step", "vs ref.forward_step (GEMV ref, HF-exact dtypes)"),
                       ("fp32p", "vs ref.forward_step + fp32 softmax probs "
                                 "(GEMV ref, KERNEL dtypes)  <== the gate")):
        st_ = stats[key]
        P(f"  {label}")
        P(f"     max |diff|       : {st_['mx']:.6f}")
        P(f"     mean |diff|      : {st_['mean']/T:.6f}")
        P(f"     argmax agreement : {st_['n']/T*100:.4f}%  ({st_['n']}/{T}"
          + (f", incl. {st_['ties']} exact ties in the reference" if st_["ties"] else "") + ")")
        for t, ra, ka, l1, l2 in st_["bad"]:
            P(f"       pos {t}: ref={ra} ({l1:.5f}) kernel={ka} (ref logit {l2:.5f})")
    P(f"  CONTROL A  ref.prefill vs ref.forward_step (same code; GEMM vs GEMV):")
    P(f"     max |diff| {ref_self_mx:.6f}   argmax agreement {ref_self*100:.4f}%")
    P(f"  CONTROL B  ref bf16-softmax-probs vs ref fp32-softmax-probs (dtype only):")
    P(f"     argmax agreement {ref_dtype*100:.4f}%   <-- upper bound on agreement with")
    P(f"     an HF-exact reference for ANY fp32-softmax implementation")
    P(f"  bf16 HF-eager-vs-HF-sdpa floor : {FLOOR_ARGMAX*100:.4f}%")
    P("")

    # (ii) greedy decode
    cur = int(ref_prefill[-1].argmax())
    ker_tokens = []
    for s in range(a.steps):
        ker_tokens.append(cur)
        lg, nxt = r.step([cur], [T + s])
        cur = int(nxt[0])
    ker_last = lg[0].float().cpu()
    nmatch = sum(int(x == y) for x, y in zip(ref_tokens, ker_tokens))
    nfp = sum(int(x == y) for x, y in zip(ref_tokens, fp32p_tokens))
    nker_fp = sum(int(x == y) for x, y in zip(fp32p_tokens, ker_tokens))
    P("== (ii) greedy decode ==")
    P(f"  ref (HF-exact) : {ref_tokens}")
    P(f"  ref (fp32 probs, = kernel dtypes) : {fp32p_tokens}")
    P(f"  kernel : {ker_tokens}")
    P(f"  kernel vs HF-exact ref      : {nmatch}/{a.steps}")
    P(f"  kernel vs kernel-dtype ref  : {nker_fp}/{a.steps}   <== the gate")
    P(f"  CONTROL kernel-dtype ref vs HF-exact ref : {nfp}/{a.steps}")
    P(f"  last-step max|diff| : {(ref_last-ker_last).abs().max().item():.6f}")
    P("")

    # (ii-b) TEACHER-FORCED decode: feed the kernel the reference's own tokens for the
    # same `steps` positions PAST the prompt and compare logits position by position.
    # Free-running greedy agreement (ii) cascades from a single near-tie flip, so it is
    # a coin flip on tie-rich text; this is the decode-path test that is not.
    r.reset()
    for t in range(T):
        r.step([ids[t]], [t])
    tf_n, tf_mx, tf_mean, tf_bad, tf_ties = 0, 0.0, 0.0, [], 0
    for s in range(a.steps):
        lg, nxt = r.step([fp32p_tokens[s]], [T + s])
        kl = lg[0].float().cpu()
        ref = fp32p_logits[s][0] if fp32p_logits[s].dim() > 1 else fp32p_logits[s]
        d = (kl - ref).abs()
        tf_mx = max(tf_mx, d.max().item())
        tf_mean += d.mean().item()
        ka, ra = int(kl.argmax()), int(ref.argmax())
        if ka == ra:
            tf_n += 1
        elif float(ref[ka]) == float(ref[ra]):
            tf_n += 1; tf_ties += 1       # exact tie in the reference: not adjudicable
        elif len(tf_bad) < 6:
            tf_bad.append((T + s, ra, ka, float(ref[ra]), float(ref[ka])))
    P("== (ii-b) teacher-forced decode (kernel fed the reference's tokens) ==")
    P(f"  positions {T}..{T + a.steps - 1}")
    P(f"     max |diff|       : {tf_mx:.6f}")
    P(f"     mean |diff|      : {tf_mean / a.steps:.6f}")
    P(f"     argmax agreement : {tf_n}/{a.steps}   <== decode-path gate"
      + (f"   ({tf_ties} exact tie(s) in the reference counted as agreement)" if tf_ties else ""))
    for t, ra, ka, l1, l2 in tf_bad:
        P(f"       pos {t}: ref={ra} ({l1:.5f}) kernel={ka} (ref logit {l2:.5f})")
    P("")

    ok_i = agree >= FLOOR_ARGMAX
    ok_ii = nker_fp == a.steps
    # A free-running greedy stream forks for good at the first disagreement, so a
    # non-adjudicable EXACT tie in the reference's top-2 at that step is a legitimate fork,
    # not a kernel error: the prefix up to it must match, and the teacher-forced gate
    # (ii-b) covers every position past it.
    if not ok_ii:
        s0 = next(i for i in range(a.steps) if ker_tokens[i] != fp32p_tokens[i])
        if s0 >= 1:
            ref0 = fp32p_logits[s0 - 1]
            ref0 = ref0[0] if ref0.dim() > 1 else ref0
            if float(ref0[ker_tokens[s0]]) == float(ref0[fp32p_tokens[s0]]):
                ok_ii = True
                P(f"  (ii) streams fork at step {s0 - 1} on an EXACT tie in the reference "
                  f"({ker_tokens[s0]} vs {fp32p_tokens[s0]} at {float(ref0[ker_tokens[s0]]):.5f}); "
                  f"prefix {s0}/{s0} identical -> counted as PASS, (ii-b) gates the remainder")
                P("")
    ok_iib = tf_n == a.steps
    ok_iii = True
    ok_iv = True

    # (iv) END TO END: prompt through the PREFILL megakernel (ONE launch),
    # generation through this decode megakernel, vs the token-by-token decode-only
    # path above.  The KV-cache contract is `pf::kv_off()` in csrc/prefill_kernel.cuh.
    if r.quantized and not a.skip_prefill_kernel:
        P("== (iv) prefill megakernel + decode megakernel vs decode-only ==")
        try:
            r.reset()
            pf_lg = r.prefill_kernel(ids).float().cpu()[0]
            pf_a = int(pf_lg.argmax())
            dec_a = int(ref_prefill[-1].argmax())
            # decode-only logits at the last prompt position (recomputed, r was reset)
            d_pf = (pf_lg - ref_fp32p[T - 1]).abs()
            P(f"  last-prompt-position logits vs kernel-dtype ref: "
              f"max|diff|={d_pf.max().item():.6f} mean={d_pf.mean().item():.6f} "
              f"argmax {'MATCH' if pf_a == int(ref_fp32p[T-1].argmax()) else 'DIFF'}")
            # (iv-a) teacher-forced decode on the prefill-kernel KV cache
            tn, tmx = 0, 0.0
            for sstep in range(a.steps):
                lg, nxt = r.step([fp32p_tokens[sstep]], [T + sstep])
                kl = lg[0].float().cpu()
                ref = fp32p_logits[sstep][0] if fp32p_logits[sstep].dim() > 1 \
                    else fp32p_logits[sstep]
                tmx = max(tmx, (kl - ref).abs().max().item())
                tn += int(int(kl.argmax()) == int(ref.argmax()))
            P(f"  (iv-a) teacher-forced decode on the prefill-kernel KV : "
              f"{tn}/{a.steps} argmax, max|diff|={tmx:.6f}   (decode-only was "
              f"{tf_n}/{a.steps}, max|diff|={tf_mx:.6f})")
            # (iv-b) free-running greedy through generate()
            r.reset()
            gen_pf = r.generate(ids, a.steps, use_prefill_kernel=True)
            r.reset()
            gen_dec = r.generate(ids, a.steps, use_prefill_kernel=False)
            same = sum(int(x == y) for x, y in zip(gen_pf, gen_dec))
            P(f"  (iv-b) generate(prefill-kernel) vs generate(decode-only) greedy : "
              f"{same}/{a.steps} tokens identical")
            P(f"     prefill-kernel : {gen_pf}")
            P(f"     decode-only    : {gen_dec}")
            nref = sum(int(x == y) for x, y in zip(gen_pf, fp32p_tokens))
            P(f"  (iv-c) generate(prefill-kernel) vs kernel-dtype ref greedy : "
              f"{nref}/{a.steps}")
            # The GATE is (iv-a): the decode path fed by the prefill kernel's KV must
            # agree with the reference as well as the decode-only KV does.  (iv-b) is
            # reported, NOT gated: gemm_tile's bf16x2
            # dequant flips exact logit ties, and one flipped FIRST token cascades
            # through a free-running greedy run.
            ok_iv = tn >= tf_n - 1
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            P(f"  FAILED: {type(e).__name__}: {e}")
            ok_iv = False
        P("")

    # (iii) batch equivalence
    if not a.skip_batch:
        del r
        gc.collect(); torch.cuda.empty_cache()
        BT, BS = a.batch_prompt_len, a.batch_steps
        prompts = [get_ids(cfgdir, BT, seed=i + 1) for i in range(4)]
        r4 = KernelRunner(a.model, M=4, max_ctx=BT + BS + 8, config_dir=a.config)
        r4.reset()
        for t in range(BT):
            lg4, nx4 = r4.step([p[t] for p in prompts], [t] * 4)
        b_logits = lg4.float().cpu().clone()
        cur4 = [int(x) for x in nx4.cpu()]
        b_tokens = [[] for _ in range(4)]
        for s in range(BS):
            for i in range(4):
                b_tokens[i].append(cur4[i])
            lg4, nx4 = r4.step(cur4, [BT + s] * 4)
            cur4 = [int(x) for x in nx4.cpu()]
        del r4
        gc.collect(); torch.cuda.empty_cache()

        r1 = KernelRunner(a.model, M=1, max_ctx=BT + BS + 8, config_dir=a.config)
        s_logits, s_tokens = [], []
        for i in range(4):
            r1.reset()
            for t in range(BT):
                lg1, nx1 = r1.step([prompts[i][t]], [t])
            s_logits.append(lg1[0].float().cpu().clone())
            c = int(nx1[0]); tk = []
            for s in range(BS):
                tk.append(c)
                lg1, nx1 = r1.step([c], [BT + s])
                c = int(nx1[0])
            s_tokens.append(tk)
        del r1
        gc.collect(); torch.cuda.empty_cache()
        P(f"== (iii) batch M=4 vs 4x M=1 ({'bit-exact' if a.batch_tol == 0 else f'max|diff| <= {a.batch_tol}'}"
          f", decode tokens identical) ==")
        for i in range(4):
            d = (b_logits[i] - s_logits[i]).abs().max().item()
            same = b_tokens[i] == s_tokens[i]
            ok_iii &= (d <= a.batch_tol) and same
            P(f"  seq {i}: prefill-last max|diff|={d:.6g}  decode tokens identical={same}")
            if not same:
                P(f"      M=4: {b_tokens[i]}")
                P(f"      M=1: {s_tokens[i]}")
        P("")

    P(f"(i)  prefill argmax >= floor (vs kernel-dtype ref) : {'PASS' if ok_i else 'FAIL'}")
    P(f"(ii) greedy {nker_fp}/{a.steps} tokens{' (forked at an exact tie)' if ok_ii and nker_fp != a.steps else ''}    : "
      f"{'PASS' if ok_ii else 'FAIL'}")
    P(f"(ii-b) teacher-forced decode {tf_n}/{a.steps} : {'PASS' if ok_iib else 'FAIL'}")
    P(f"(iii) batch equivalence{' (bit-exact)' if a.batch_tol == 0 else f' (tol {a.batch_tol})'} : "
      f"{'PASS' if ok_iii else 'FAIL'}")
    P(f"(iv) prefill-kernel + decode : "
      f"{'SKIPPED (--skip-prefill-kernel)' if (a.skip_prefill_kernel or not quantized) else ('PASS' if ok_iv else 'FAIL')}")
    P("RESULT: " + ("PASS" if (ok_i and ok_ii and ok_iib and ok_iii and ok_iv) else "FAIL"))

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write("\n".join(log) + "\n")
    return 0 if (ok_i and ok_ii and ok_iib and ok_iii and ok_iv) else 1


if __name__ == "__main__":
    sys.exit(main())
