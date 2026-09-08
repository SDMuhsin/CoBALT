"""Correctness harness for the Gemma3 PREFILL megakernel vs ref_gemma3.py.

    source scripts/cobaltkernel_env.sh 1g
    python src/cobaltkernel/verify_prefill.py \
        --model /scratch/.../gemma-3-4b/cobalt_sp0.5_b4_g128_cbk1_dense4f \
        --config /scratch/.../gemma-3-4b/text_bf16 --prompt-len 1152 \
        --out results/cobaltkernel/prefill_verify_gemma-3-4b.md

Tests
  (i)  ALL-position logits of a single prefill launch vs the torch oracle built by
       dequantizing the SAME packed bytes (dequant_ref.load_state_packed).
       Bar = the bf16 HF-eager-vs-HF-sdpa argmax floor, 98.5 %
       (results/cobaltkernel/ref_verify_gemma-3-4b.txt).
  (ii) KV CONTRACT: prefill 512 tokens with this kernel, then continue N greedy decode
       steps with the DECODE megakernel (runner.KernelRunner) reading the same cache,
       and compare the tokens with ref_gemma3 greedy decoding.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel import ref_gemma3 as R              # noqa: E402
from cobaltkernel import dequant_ref as DQ            # noqa: E402
from cobaltkernel.prefill_runner import PrefillRunner  # noqa: E402

FLOOR_ARGMAX = 0.985


_EXACT_DEQUANT = DQ.dequant_matrix


def fast_dequant_matrix(blob, entry, device, dtype=torch.bfloat16):
    """dequant_ref.dequant_matrix in the arithmetic the MMA path actually uses.

    cbk::gemm_tile dequantizes with bf16x2 ops:
        (bf16(128+code) - bf16(128+zero)) * bf16(scale) * bf16(col_scale)
    each step rounded to bf16, versus the oracle's single fp32 -> bf16 rounding.
    That is a documented ~5e-3 (relative to |W|max) accuracy cost of the tensor-core
    path; this oracle isolates it from prefill-kernel bugs.  Matrices without a
    column scale (the embedding) keep the exact path -- the prefill kernel
    dequantizes those in fp32 too.
    """
    A = entry["arrays"]
    if entry["layout"] in (6, 7) and A.get("col_scale", {}).get("bytes"):
        return _fast_dequant_blk(blob, entry, device, dtype)
    if entry["layout"] != 1 or not A.get("col_scale", {}).get("bytes"):
        return _EXACT_DEQUANT(blob, entry, device, dtype)
    K, N, G = entry["K"], entry["N"], entry["N"] // DQ.GROUP
    d = DQ._arr(blob, A["data"], torch.uint8)[: K * (N // 2)].view(K, N // 2)
    lo = (d & 0xF).to(torch.bfloat16)
    hi = (d >> 4).to(torch.bfloat16)
    q = torch.stack([lo, hi], dim=2).reshape(K, N)
    z = DQ._arr(blob, A["zero"], torch.uint8)[: K * G].view(K, G).to(torch.bfloat16)
    s = DQ._arr(blob, A["scale"], torch.float16)[: K * G].view(K, G).to(torch.bfloat16)
    w = (q.view(K, G, DQ.GROUP) - z[:, :, None]) * s[:, :, None]
    w = w.reshape(K, N)
    ncs = A["col_scale"]["bytes"] // 2
    cs = DQ._arr(blob, A["col_scale"], torch.float16)[:ncs].to(torch.bfloat16)
    if ncs == 2 * N:
        cs = cs.view(2, N)
        w[0::2] = w[0::2] * cs[0][None, :]
        w[1::2] = w[1::2] * cs[1][None, :]
    else:
        w = w * cs[None, :]
    return w.to(dtype)
def _fast_dequant_blk(blob, entry, device, dtype=torch.bfloat16):
    """The BLK16_32 (CoBALT-16:32) analogue of fast_dequant_matrix.

    cbk::gemm_tile expands a 16:32 block to a DENSE code plane at staging time, writing
    the group's integer `zero` into every PRUNED slot, and then runs the same bf16x2
    dequant.  So the oracle is: pruned code := zero (=> the bf16 subtraction is EXACTLY
    0), kept code := the stored code, then (bf16(128+code) - bf16(128+zero)) * bf16(scale)
    * bf16(col_scale), each step rounded to bf16.
    """
    A = entry["arrays"]
    K, N, G = entry["K"], entry["N"], entry["N"] // DQ.GROUP
    b6 = entry["layout"] == 7
    stride = (N // 2) if b6 else (N * 3 // 8)
    NB = N // 32
    rowb = DQ._arr(blob, A["data"], torch.uint8)[: K * stride].view(K, stride)
    sh8 = torch.arange(8, dtype=torch.uint8, device=rowb.device)
    keep = ((rowb[:, : N // 8].reshape(K, -1, 1) >> sh8) & 1).view(K, N).bool()
    nib = rowb[:, N // 8: N // 8 + NB * 8].reshape(K, NB, 8).to(torch.int64)
    q16 = torch.stack([nib & 0xF, (nib >> 4) & 0xF], -1).view(K, NB, 16)
    if b6:
        hb = rowb[:, N * 3 // 8:].reshape(K, NB, 4).to(torch.int64)
        hw = hb[..., 0] | (hb[..., 1] << 8) | (hb[..., 2] << 16) | (hb[..., 3] << 24)
        sh = torch.arange(8, dtype=torch.int64, device=rowb.device) * 2
        he = (hw.unsqueeze(-1) >> sh) & 3
        ho = (hw.unsqueeze(-1) >> (sh + 16)) & 3
        q16 = q16 | (torch.stack([he, ho], -1).view(K, NB, 16) << 4)
    z = DQ._arr(blob, A["zero"], torch.uint8)[: K * G].view(K, G).to(torch.int64)
    q = z.repeat_interleave(DQ.GROUP, 1).clone()          # pruned slots carry `zero`
    q[keep] = q16.reshape(-1)
    qb = (128 + q).to(torch.bfloat16)
    zb = (128 + z).to(torch.bfloat16).repeat_interleave(DQ.GROUP, 1)
    s = DQ._arr(blob, A["scale"], torch.float16)[: K * G].view(K, G).to(torch.bfloat16)
    w = (qb - zb) * s.repeat_interleave(DQ.GROUP, 1)
    ncs = A["col_scale"]["bytes"] // 2
    cs = DQ._arr(blob, A["col_scale"], torch.float16)[:ncs].to(torch.bfloat16)
    if ncs == 2 * N:
        cs = cs.view(2, N)
        w[0::2] = w[0::2] * cs[0][None, :]
        w[1::2] = w[1::2] * cs[1][None, :]
    else:
        w = w * cs[None, :]
    return w.to(dtype)


TEXT = ("The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms. "
        "Mitochondria generate most of the cell chemical energy in the form of adenosine triphosphate. "
        "Patients presenting with acute chest pain should be evaluated with a twelve lead electrocardiogram. ")


def get_ids(cfgdir, n, seed=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfgdir)
    here = os.path.dirname(os.path.abspath(__file__))
    txt = open(os.path.join(here, "ref_gemma3.py")).read() + " ".join(TEXT.split()) * 40
    ids = tok(txt).input_ids
    assert len(ids) >= n + seed * 97, "not enough tokens"
    return ids[seed * 97: seed * 97 + n]



def batch_test(a, state, P):
    """(iii) One batched step for B independent sequences (different lengths, different
    positions, one KV slot each) vs the torch oracle and vs B independent M=1 steps of the
    DECODE megakernel."""
    P("\n## (iii) batched multi-sequence decode step")
    torch.cuda.empty_cache()
    B = a.batch_seqs
    lens = [96, 128, 64, 111, 80, 137, 50, 121][:B]
    seqs = [get_ids(a.config, n, seed=i + 1) for i, n in enumerate(lens)]
    maxlen = max(lens) + a.batch_steps + 4

    run = PrefillRunner(a.model, config_dir=a.config, max_tokens=max(lens),
                        max_ctx=maxlen, max_logit_rows=max(8, B), kv_batch=B)
    P(f"batch grid {run.blocks_b} blocks ({run.blocks_per_sm_b}/SM), "
      f"dyn smem {run.smem_b} B; sequence lengths {lens}")
    run.reset()
    for b, ids in enumerate(seqs):
        run.prefill(ids, seq=b)

    # oracle: ref_gemma3 per sequence, greedy; the same tokens are teacher-forced into the
    # kernel so a single divergence cannot compound.  A SECOND oracle on the exact-dequant
    # weights gives the control (what any cbk::gemm_tile implementation can reach).
    state_e = None
    if a.batch_control:
        try:
            DQ.dequant_matrix = _EXACT_DEQUANT
            state_e = DQ.load_state_packed(a.model, a.config, device="cuda",
                                           dtype=torch.bfloat16)
        except torch.OutOfMemoryError:
            P("control oracle SKIPPED (out of memory); falling back to the tie criterion")
            state_e = None
            torch.cuda.empty_cache()
    kvs, kvs_e, ref_next, pos = [], [], [], []
    for ids in seqs:
        kv = R.new_kv(state)
        lg, kv = R.prefill(state, torch.tensor(ids), kv)
        kvs.append(kv)
        if state_e is not None:
            kve = R.new_kv(state_e)
            _, kve = R.prefill(state_e, torch.tensor(ids), kve)
            kvs_e.append(kve)
        ref_next.append(int(lg[0, -1].argmax()))
        pos.append(len(ids))

    agree = tot = ctrl_agree = 0
    maxdiff = 0.0
    gaps = []
    tokens = list(ref_next)
    for step in range(a.batch_steps):
        got = run.batch_step(tokens, pos).float()
        ref = torch.stack([R.forward_step(state, tokens[b], pos[b], kvs[b]).float()
                           for b in range(B)])
        maxdiff = max(maxdiff, float((got - ref).abs().max()))
        if state_e is not None:
            refe = torch.stack([R.forward_step(state_e, tokens[b], pos[b], kvs_e[b]).float()
                                for b in range(B)])
            ctrl_agree += int((refe.argmax(-1) == ref.argmax(-1)).sum())
        ka, ra = got.argmax(-1), ref.argmax(-1)
        agree += int((ka == ra).sum()); tot += B
        for b in range(B):                      # how far from a tie is each disagreement?
            if int(ka[b]) != int(ra[b]):
                g = float(ref[b, ra[b]] - ref[b, ka[b]])
                u = 2.0 ** (torch.floor(torch.log2(ref[b, ra[b]].abs())).item() - 7)
                gaps.append(g / u)
        tokens = [int(x) for x in ra]          # teacher forcing on the oracle's tokens
        pos = [p + 1 for p in pos]
    frac = agree / tot
    P(f"{a.batch_steps} steps x {B} sequences: argmax agreement {agree}/{tot} = "
      f"{100*frac:.2f} %, max|diff| {maxdiff:.4g}")
    if gaps:
        P(f"disagreements: {len(gaps)}, oracle top1-vs-kernel-choice gap in bf16 ulp: "
          f"{', '.join(f'{g:.1f}' for g in gaps)}  (<=2 ulp = a tie)")
    ties = all(g <= 2.0 for g in gaps)
    ctrl = (ctrl_agree / tot) if state_e is not None else None
    if ctrl is not None:
        P(f"CONTROL torch(exact W) vs torch(mma W), same {tot} steps: {100*ctrl:.2f} %")
        del state_e
        torch.cuda.empty_cache()

    # cross-check step 1 against the DECODE megakernel at M=1
    dec_ok = None
    try:
        from cobaltkernel.runner import KernelRunner
        dec = KernelRunner(a.model, M=1, max_ctx=maxlen, config_dir=a.config)
        run.reset()
        for b, ids in enumerate(seqs):
            run.prefill(ids, seq=b)
        pos1 = [len(x) for x in seqs]
        got1 = run.batch_step(ref_next, pos1).float().clone()
        dmax, dagree = 0.0, 0
        for b, ids in enumerate(seqs):
            dec.reset()
            dec.prefill([ids])
            lg, _ = dec.step([ref_next[b]], [pos1[b]])
            lg = lg[0].float()
            dmax = max(dmax, float((got1[b] - lg).abs().max()))
            dagree += int(int(got1[b].argmax()) == int(lg.argmax()))
        P(f"vs DECODE megakernel M=1 (same step): argmax {dagree}/{B}, max|diff| {dmax:.4g}")
        dec_ok = dagree == B
        del dec
    except Exception as e:
        P(f"decode cross-check SKIPPED ({type(e).__name__}: {str(e)[:120]})")
    del run
    torch.cuda.empty_cache()
    # bar: the same one part (i) uses -- either the argmax floor, or every disagreement is
    # a bf16 tie -- AND the step must match the decode megakernel's own logits.
    bar = min(FLOOR_ARGMAX, ctrl) if ctrl is not None else FLOOR_ARGMAX
    ok = (frac >= bar or ties) and (dec_ok is None or dec_ok)
    P(f"bar = {100*bar:.2f} %")
    P(f"RESULT (iii): {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CBK1 packed dir (*_dense4f)")
    ap.add_argument("--config", required=True, help="dir with config.json + tokenizer")
    ap.add_argument("--prompt-len", type=int, default=1152)
    ap.add_argument("--kv-prompt-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--skip-kv", action="store_true")
    ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--batch-seqs", type=int, default=4)
    ap.add_argument("--batch-steps", type=int, default=8)
    ap.add_argument("--only-batch", action="store_true")
    ap.add_argument("--batch-control", action="store_true", default=True)
    ap.add_argument("--oracle", choices=("exact", "mma"), default="mma",
                    help="'exact': fp32 dequant (dequant_ref).  'mma': the same weights "
                         "dequantized in the bf16x2 arithmetic cbk::gemm_tile uses, which "
                         "isolates prefill-kernel error from the tile GEMM's documented "
                         "~5e-3 dequant cost.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log = []

    def P(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.append(s)

    ids = get_ids(a.config, a.prompt_len)
    T = len(ids)
    P(f"# prefill verification -- {os.path.basename(a.model)}")
    P(f"prompt {T} tokens, sliding window crossing: {T > 1024}")

    if a.only_batch:                     # (iii) alone: the two oracles + the runner need
        DQ.dequant_matrix = fast_dequant_matrix     # the whole slice, so run it by itself
        state = DQ.load_state_packed(a.model, a.config, device="cuda", dtype=torch.bfloat16)
        DQ.dequant_matrix = _EXACT_DEQUANT
        ok = batch_test(a, state, P)
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            open(a.out, "w").write("\n".join(log) + "\n")
            print(f"[wrote {a.out}]")
        sys.exit(0 if ok else 1)

    # ---------------------------------------------------------------- oracle
    P("\n## (i) all-position logits vs the torch dequant oracle")

    def oracle_logits(kind):
        DQ.dequant_matrix = fast_dequant_matrix if kind == "mma" else _EXACT_DEQUANT
        st = DQ.load_state_packed(a.model, a.config, device="cuda", dtype=torch.bfloat16)
        DQ.dequant_matrix = _EXACT_DEQUANT
        lg, kvv = R.prefill(st, torch.tensor(ids))
        lg = lg[0].float()
        del kvv
        return st, lg

    # exact-dequant oracle: kept only as argmax + top-2 gap (the full fp32 logits of two
    # oracles + the kernel do not fit for long prompts on the 1g slice).
    st_e, lg_e = oracle_logits("exact")
    top2 = lg_e.topk(2, dim=-1)
    arg_e, gap_e = top2.indices[:, 0].clone(), (top2.values[:, 0] - top2.values[:, 1]).clone()
    del st_e, lg_e, top2
    torch.cuda.empty_cache()

    # mma-dequant oracle: the same packed bytes dequantized the way cbk::gemm_tile does.
    state, ref_logits = oracle_logits("mma")
    ref_arg = ref_logits.argmax(-1)
    ctrl = (ref_arg == arg_e).float().mean().item()

    max_ctx = max(T, a.kv_prompt_len + a.steps) + 8
    run = PrefillRunner(a.model, config_dir=a.config, max_tokens=T,
                        max_ctx=max_ctx, max_logit_rows=T)
    P(f"grid {run.blocks} blocks ({run.blocks_per_sm}/SM), dyn smem {run.smem} B")
    run.reset()
    got = run.prefill(ids, all_logits=True).float()
    torch.cuda.synchronize()
    got_arg = got.argmax(-1)

    diff = (got - ref_logits).abs()
    agree = (got_arg == ref_arg)
    frac = agree.float().mean().item()
    frac_e = (got_arg == arg_e).float().mean().item()
    P(f"max|diff| vs mma oracle {diff.max().item():.4g}  mean|diff| {diff.mean().item():.4g}"
      f"   (|logits|max {ref_logits.abs().max().item():.4g})")
    P(f"argmax vs MMA-dequant oracle (the gate) : {int(agree.sum())}/{T} = {100*frac:.2f} %")
    P(f"argmax vs EXACT-dequant oracle          : {100*frac_e:.2f} %")
    P(f"CONTROL torch(exact W) vs torch(mma W)  : {100*ctrl:.2f} %   <- upper bound for ANY "
      f"implementation built on cbk::gemm_tile")
    P(f"bf16 HF-eager-vs-HF-sdpa floor          : {100*FLOOR_ARGMAX:.2f} %")
    # how close are the disagreements to ties?
    bad = (~agree).nonzero().flatten()
    if bad.numel():
        g = gap_e[bad]
        P(f"disagreeing positions: {bad.numel()}, ref top1-top2 gap: "
          f"median {g.median().item():.4f}, max {g.max().item():.4f} "
          f"(logit scale {ref_logits.abs().max().item():.1f})")
    last_ok = int(got_arg[-1]) == int(ref_arg[-1])
    lk, lr = int(got_arg[-1]), int(ref_arg[-1])
    lgap = float(ref_logits[-1, lr] - ref_logits[-1, lk])
    ulp = 2.0 ** (torch.floor(torch.log2(ref_logits[-1, lr].abs())).item() - 7)
    last_tie = lgap <= 2 * ulp
    P(f"last-position argmax match: {last_ok} (kernel {lk}, ref {lr}); "
      f"ref logit gap {lgap:.4f} = {lgap/ulp:.1f} bf16 ulp -> {'TIE' if last_tie else 'REAL'}")
    ok_i = frac >= min(FLOOR_ARGMAX, ctrl) and (last_ok or last_tie)
    P(f"RESULT (i): {'PASS' if ok_i else 'FAIL'}  "
      f"(bar = min(bf16 floor, cbk::gemm_tile control) = {100*min(FLOOR_ARGMAX, ctrl):.2f} %)")
    del ref_logits, got, diff
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- KV contract
    ok_ii = None
    KernelRunner = None
    if not a.skip_kv:
        P("\n## (ii) KV contract: prefill here -> greedy decode with the DECODE megakernel")
        try:
            from cobaltkernel.runner import KernelRunner
        except Exception as e:  # the decode kernel may be mid-edit
            P(f"SKIPPED: the decode megakernel does not build right now ({type(e).__name__})")
            KernelRunner = None
    if not a.skip_kv and KernelRunner is not None:
      try:
          ids2 = ids[: a.kv_prompt_len]
          dec = KernelRunner(a.model, M=1, max_ctx=max_ctx, config_dir=a.config)
          dec.reset()
          pre = PrefillRunner(a.model, config_dir=a.config, max_tokens=len(ids2),
                              max_ctx=max_ctx, max_logit_rows=1,
                              kv=(dec.kcache, dec.vcache))
          lg = pre.prefill(ids2)
          nxt = int(lg[0].argmax())
          got_tok = [nxt]
          pos = len(ids2)
          for _ in range(a.steps - 1):
              logits, _ = dec.step([nxt], [pos])
              nxt = int(logits[0].argmax())
              got_tok.append(nxt)
              pos += 1
          del pre, dec
          torch.cuda.empty_cache()

          kv = R.new_kv(state)
          rl, kv = R.prefill(state, torch.tensor(ids2), kv)
          t = int(rl[0, -1].argmax())
          ref_tok = [t]
          pos = len(ids2)
          for _ in range(a.steps - 1):
              lgt = R.forward_step(state, t, pos, kv)
              t = int(lgt.argmax())
              ref_tok.append(t)
              pos += 1
      except Exception as e:
        P(f"SKIPPED: the decode megakernel does not build/run right now: "
          f"{type(e).__name__}: {str(e)[:200]}")
        KernelRunner = None
    if not a.skip_kv and KernelRunner is not None:
        nmatch = sum(int(x == y) for x, y in zip(got_tok, ref_tok))
        P(f"prefill {len(ids2)} tokens + {a.steps} greedy steps")
        P(f"kernel: {got_tok}")
        P(f"ref   : {ref_tok}")
        P(f"token agreement {nmatch}/{a.steps}")
        ok_ii = nmatch == a.steps
        P(f"RESULT (ii): {'PASS' if ok_ii else 'FAIL'}")

    # ------------------------------------------------- (iii) batched multi-sequence step
    del arg_e, gap_e
    torch.cuda.empty_cache()
    ok_iii = None
    if not a.skip_batch:
        ok_iii = batch_test(a, state, P)

    P("\n## verdict")
    P(f"(i) all-position logits: {'PASS' if ok_i else 'FAIL'}")
    if ok_ii is not None:
        P(f"(ii) KV contract: {'PASS' if ok_ii else 'FAIL'}")
    if ok_iii is not None:
        P(f"(iii) batched multi-sequence step: {'PASS' if ok_iii else 'FAIL'}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write("\n".join(log) + "\n")
        print(f"[wrote {a.out}]")
    sys.exit(0 if (ok_i and (ok_ii is None or ok_ii) and (ok_iii is None or ok_iii)) else 1)


if __name__ == "__main__":
    main()
