"""Speed benchmark for the Gemma3 PREFILL megakernel.

    source scripts/cobaltkernel_env.sh 2g
    python src/cobaltkernel/bench_prefill.py \
        --model /scratch/.../medgemma-27b/cobalt_sp0.5_b4_g128_cbk1_dense4f \
        --config <snapshot with config.json> --M 512 1024 2048 \
        --peak 225.1 --out results/cobaltkernel/medgemma-27b/speed_oneslice/prefill.json

Reports, per prompt length M: TTFT (ms), prefill tok/s, achieved TF/s of the decoder
linear stack against the measured mma roofline, and the per-phase breakdown.
"""

import argparse
import json
import datetime
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel.prefill_runner import PrefillRunner    # noqa: E402


def linear_flops_per_token(man):
    """2 * sum(K*N) over the decoder linear stack of one layer, x n_layers."""
    f = 0
    for lay in man["layers"]:
        for nm in ("qkv", "o_proj", "gateup", "down_proj"):
            e = lay["matrices"][nm]
            f += 2 * e["K"] * e["N"]
    return f



def bench_batched(a, res, lf):
    """M independent sequences, each with `--prompt` tokens already in its own KV slot,
    then `--steps` decode steps -- one cooperative launch per step for the whole batch."""
    import datetime
    res["mode"] = "batched"
    res["protocol"] = {"batched": f"Mx{a.prompt}->{a.steps}, static batch"}
    maxM = max(a.batch)
    ctx = a.prompt + a.steps + 8
    run = PrefillRunner(a.model, config_dir=a.config, max_tokens=max(a.prompt, maxM),
                        max_ctx=ctx, max_logit_rows=maxM, kv_batch=maxM)
    print(f"batch grid {run.blocks_b} blocks ({run.blocks_per_sm_b}/SM), "
          f"dyn smem {run.smem_b} B; prefill grid {run.blocks} ({run.blocks_per_sm}/SM)")
    weights = sum(int(b.numel()) for b in run.blobs.values())
    kv_bytes = run.kcache.numel() * 2 * 2
    ids = [(i * 7919 + 13) % (run.vocab - 1) for i in range(a.prompt)]
    for M in a.batch:
        run.reset()
        for b in range(M):                       # fill the caches (setup, not timed)
            run.prefill(ids, seq=b)
        torch.cuda.synchronize()
        toks = [1 + (i * 131) % 1000 for i in range(M)]
        pos = [a.prompt] * M
        for _ in range(4):                       # warm-up
            run.batch_step(toks, pos)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        p = list(pos)
        for _ in range(a.steps):
            run.batch_step(toks, p)
            p = [x + 1 for x in p]
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        out_tok_s = M * a.steps / ms * 1e3
        cell = {"M": M, "prompt": a.prompt, "steps": a.steps, "ms_total": ms,
                "ms_per_step": ms / a.steps, "output_tok_s": out_tok_s,
                "total_tok_s": (M * (a.prompt + a.steps)) / ms * 1e3,
                "weight_GBs": weights * a.steps / ms * 1e-6,
                "linear_tflops": lf * M * a.steps / ms * 1e-9}
        run.batch_step(toks, p, timings=True)
        torch.cuda.synchronize()
        cell["phases_us"] = run.phase_times_us()
        res["cells"].append(cell)
        ph = cell["phases_us"]; tot = ph["total"]
        print(f"M={M:3d}  {ms/a.steps:7.2f} ms/step  {out_tok_s:8.1f} output tok/s  "
              f"{cell['weight_GBs']:6.1f} GB/s weights  {cell['linear_tflops']:5.1f} TF/s")
        print("   phases: " + "  ".join(
            f"{k}={ph[k]/1e3:.2f}ms({100*ph[k]/tot:.0f}%)"
            for k in list(run.PHASES) + ["gather", "lm_head"]))
    res["batched"] = {"config": f"Mx{a.prompt}->{a.steps}, static batch",
                      "cells": {str(c["M"]): {"output_tok_s": c["output_tok_s"],
                                              "total_tok_s": c["total_tok_s"],
                                              "ms_per_step": c["ms_per_step"]}
                                for c in res["cells"]}}
    best = max(res["cells"], key=lambda c: c["output_tok_s"])
    res["batched"]["output_tok_s"] = best["output_tok_s"]
    res["batched"]["total_tok_s"] = best["total_tok_s"]
    res["batched"]["best_M"] = best["M"]
    res["arm"] = a.arm            # was hard-coded "cobalt_dense4" -- every non-DENSE4
                                  # batched JSON on disk is mislabelled because of it
                                  # (mixed-layout arms, 2026-09-07)
    res["slice"] = res["slice_uuid"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    res["timestamp"] = res["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    res["kernel_dispatched"] = ("pf::batch_kernel (persistent cooperative, 2 blocks/SM, "
                                "gemm_tile MT=32/BR=128/KT=256; one launch per decode step "
                                "for the whole batch)")
    res["weight_bytes"] = weights
    res["kv_bytes_max"] = kv_bytes
    res["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["reserved_mem_mib"] = torch.cuda.max_memory_reserved() / 2**20
    print(f"peak torch alloc {res['peak_mem_mib']:.0f} MiB "
          f"(reserved {res['reserved_mem_mib']:.0f} MiB)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"[wrote {a.out}]")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--M", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--peak", type=float, default=225.1, help="measured bf16 mma TF/s")
    ap.add_argument("--all-logits", action="store_true")
    ap.add_argument("--batch", type=int, nargs="*", default=None,
                    help="batched decode mode: sequence counts to measure (e.g. 8 16 32)")
    ap.add_argument("--prompt", type=int, default=512, help="prompt tokens per sequence")
    ap.add_argument("--steps", type=int, default=128, help="decode steps per sequence")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default="cobalt_dense4",
                    help="arm name recorded in the JSON (batched mode)")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.model, "manifest.json")))
    lf = linear_flops_per_token(man)
    print(f"decoder linear: {lf/2/1e9:.3f} G params, {lf/1e9:.1f} GFLOP/token")

    res = {"model": os.path.basename(a.model), "peak_tflops": a.peak,
           "device": torch.cuda.get_device_name(0), "cells": []}
    if a.batch is not None:
        return bench_batched(a, res, lf)
    maxM = max(a.M)
    run = PrefillRunner(a.model, config_dir=a.config, max_tokens=maxM,
                        max_ctx=maxM + 8, max_logit_rows=maxM if a.all_logits else 8)
    vocab_flops = 2 * run.hidden * run.vocab
    print(f"grid {run.blocks} blocks ({run.blocks_per_sm}/SM), dyn smem {run.smem} B, "
          f"weights {sum(int(b.numel()) for b in run.blobs.values())/1e9:.2f} GB")

    for M in a.M:
        ids = [(i * 7919 + 13) % (run.vocab - 1) for i in range(M)]
        R = M if a.all_logits else 1
        for _ in range(a.warmup):
            run.prefill(ids, all_logits=a.all_logits)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            run.prefill(ids, all_logits=a.all_logits)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ms = min(ts)
        flops = lf * M + vocab_flops * R
        cell = {"M": M, "R": R, "ttft_ms": ms, "prefill_tok_s": M / ms * 1e3,
                "linear_tflops": lf * M / ms * 1e-9,
                "total_tflops": flops / ms * 1e-9,
                "pct_of_peak": 100 * (flops / ms * 1e-9) / a.peak,
                "ms_all": ts}
        # phase breakdown
        run.prefill(ids, all_logits=a.all_logits, timings=True)
        torch.cuda.synchronize()
        cell["phases_us"] = run.phase_times_us()
        res["cells"].append(cell)
        print(f"M={M:5d} R={R:5d}  TTFT {ms:8.2f} ms   {M/ms*1e3:8.1f} tok/s   "
              f"linear {cell['linear_tflops']:6.1f} TF/s   total {cell['total_tflops']:6.1f} "
              f"TF/s = {cell['pct_of_peak']:.1f} % of peak")
        ph = cell["phases_us"]
        tot = ph["total"]
        print("   phases: " + "  ".join(
            f"{k}={ph[k]/1e3:.1f}ms({100*ph[k]/tot:.0f}%)"
            for k in list(run.PHASES) + ["gather", "lm_head"]))
        print(f"   (kernel-internal total {tot/1e3:.1f} ms vs wall {ms:.1f} ms)")

    # top-level fields matching results/cobaltkernel/SPEED_JSON_SCHEMA.md (the 512-token
    # cell is the accel4bit protocol's prefill number; the collector merges this into
    # cobalt_dense4.json)
    for c in res["cells"]:
        if c["M"] == 512:
            res["ttft_ms"] = c["ttft_ms"]
            res["prefill_tok_s"] = c["prefill_tok_s"]
            res["single_stream"] = {"ttft_ms": c["ttft_ms"],
                                    "prefill_tok_s": c["prefill_tok_s"],
                                    "decode_tok_s": None, "e2e_latency_s": None}
    res["arm"] = "cobalt_dense4"
    res["slice"] = res["slice_uuid"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    res["timestamp"] = res["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    res["kernel_dispatched"] = ("pf::prefill_kernel (persistent cooperative, grid.sync phases: "
                                "qkv, rope_kv, attn, o_proj, addnorm, gateup+geglu, down, "
                                "addnorm; lm_head)")
    res["protocol"] = {"prefill_only": "M-token prompt, one cooperative launch, "
                                       "last-position logits; best of --iters"}
    res["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["reserved_mem_mib"] = torch.cuda.max_memory_reserved() / 2**20
    print(f"peak torch alloc {res['peak_mem_mib']:.0f} MiB "
          f"(reserved {res['reserved_mem_mib']:.0f} MiB)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
