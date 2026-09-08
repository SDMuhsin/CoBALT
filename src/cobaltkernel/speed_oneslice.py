"""accel4bit-protocol one-slice speed run for the cobaltkernel megakernel.

Writes results/cobaltkernel/<model>/speed_oneslice/<arm>.json in the schema at
results/cobaltkernel/SPEED_JSON_SCHEMA.md.  Driven by
scripts/run_cobaltkernel_speed_oneslice.sh.

Protocol (accel4bit): 512-token prompt -> 128 generated tokens, greedy, bs=1.
  ttft_ms       : wall time to produce the FIRST generated token, i.e. the whole
                  512-token prompt.  This kernel has no mma.sync prefill path yet, so
                  the prompt is fed ONE TOKEN PER LAUNCH -- reported honestly as such
                  in protocol.prefill_note; it is NOT a batched-prefill number.
  decode_tok_s  : like-for-like, tokens 2..128 only (excludes TTFT).
  e2e_latency_s : full 512->128 wall time.
"""

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel.runner import KernelRunner   # noqa: E402


def nvsmi_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        me = str(os.getpid())
        best = 0
        for line in out.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")[:2]]
            if p == me:
                best = max(best, int(m))
        return best or None
    except Exception:
        return None


def _count_launches(fn):
    """Count DISTINCT CUDA kernel launches made by fn()."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {}
    for e in prof.events():
        # every event whose device_type is CUDA is a real device-side kernel or copy;
        # do NOT filter by name (the prefill kernel's demangled name does not start
        # with "void ", which is what hid it from the v2 proof).
        if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA"):
            names[e.key] = names.get(e.key, 0) + 1
    ev = [(k, v) for k, v in names.items()]
    ev.sort(key=lambda x: -x[1])
    lines = [f"{v} launch(es)/decode step : {k}" for k, v in ev]
    total = sum(v for _, v in ev)
    return ev, total


def kernel_proof(r, tokens, pos):
    ev, total = _count_launches(lambda: r.step(tokens, pos))
    lines = [f"{v} launch(es)/decode step : {k}" for k, v in ev]
    lines.append(f"TOTAL CUDA kernel launches per decode step = {total}")
    return lines, total


def prefill_proof(r, ids):
    """Launch count for ONE whole-prompt prefill (the prefill megakernel)."""
    r.reset()
    ev, total = _count_launches(lambda: r.prefill_kernel(ids))
    lines = [f"{v} launch(es)/prefill of {len(ids)} tokens : {k}" for k, v in ev]
    lines.append(f"TOTAL CUDA kernel launches per {len(ids)}-token prefill = {total}")
    return lines, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--arm", default="cobalt_dense4")
    ap.add_argument("--slice", default=os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
    ap.add_argument("--slice-type", default="2g.48gb")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--batch-M", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--prefill-kernel", dest="pfk", action="store_true", default=True,
                    help="use the PREFILL megakernel for the prompt (default)")
    ap.add_argument("--no-prefill-kernel", dest="pfk", action="store_false",
                    help="feed the prompt one token per decode launch (the v2 protocol)")
    ap.add_argument("--minb-sweep", default=None,
                    help="comma-separated COBALT_MINB values; unused here, see bench")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    rec = {
        "arm": a.arm, "model": a.model_name, "status": "FAILED", "reason": None,
        "slice": a.slice, "slice_uuid": a.slice, "slice_type": a.slice_type,
        "timestamp": ts, "date": ts,
        # layout of record: the BLK16_32 arms carry BLK1632_{4,6}, everything else DENSE4.
        "artifact": {"path": a.model_dir, "bytes": None,
                     "layout": ("BLK1632_" + os.environ["COBALT_BLK1632"]
                                if os.environ.get("COBALT_BLK1632", "0") in ("4", "6")
                                else "DENSE4")},
        "kernel_dispatched": None, "kernel": {"summary": None, "lines": []},
        "protocol": {"single_stream": f"{a.prompt}->{a.gen}, bs=1, greedy",
                     "batched": None,
                     "prefill_note": None},
        "single_stream": {"ttft_ms": None, "prefill_tok_s": None,
                          "decode_tok_s": None, "e2e_latency_s": None},
        "batched": {"output_tok_s": None, "total_tok_s": None, "config": None},
        "peak_gpu_mem_mib": None,
        "peak_mem_MiB": {"max_over_all_stages": None,
                         "note": "max nvidia-smi compute-apps used_memory over the run"},
        "phase_breakdown_ms": {},
        "runtime": {"name": "cobaltkernel", "python": sys.executable,
                    "versions": {"cuda_toolkit": "13.0",
                                 "driver": torch.version.cuda,
                                 "torch": torch.__version__}},
    }
    try:
        n = 0
        for root, _, files in os.walk(a.model_dir):
            for f in files:
                n += os.path.getsize(os.path.join(root, f))
        rec["artifact"]["bytes"] = n

        torch.cuda.reset_peak_memory_stats()
        r = KernelRunner(a.model_dir, M=1, max_ctx=a.prompt + a.gen + 8, config_dir=a.config)
        ids = [(i * 7 + 3) % 1000 + 5 for i in range(a.prompt)]

        # ---- warmup (also pays the one-off JIT/alloc costs) ----
        r.reset()
        for t in range(16):
            r.step([ids[t]], [t])
        torch.cuda.synchronize()
        if a.pfk:
            r.reset()
            r.prefill_kernel(ids)          # JIT-builds + allocates the prefill runner
            torch.cuda.synchronize()

        # ---- single stream: 512 -> 128, ONE end-to-end run ----
        # TTFT / prefill_tok_s come from the PREFILL megakernel (one cooperative launch
        # for the whole prompt); decode_tok_s from tokens 2..128 of the
        # decode megakernel (one launch each).
        r.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if a.pfk:
            lg = r.prefill_kernel(ids)
            cur = int(lg[0].argmax())
        else:
            for t in range(a.prompt):
                lg, nxt = r.step([ids[t]], [t])
            cur = int(nxt.cpu()[0])
        torch.cuda.synchronize()
        t_ttft = time.perf_counter() - t0          # prompt done, first token in hand
        t1 = time.perf_counter()
        first = True
        t_after_first = None
        for s in range(a.gen - 1):
            lg, nxt = r.step([cur], [a.prompt + s])
            cur = int(nxt.cpu()[0])
            if first:
                torch.cuda.synchronize()
                t_after_first = time.perf_counter()
                first = False
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t1
        e2e = time.perf_counter() - t0
        # like-for-like decode = tokens 2..128
        n_lfl = a.gen - 1
        rec["single_stream"]["ttft_ms"] = round(t_ttft * 1e3, 3)
        rec["single_stream"]["prefill_tok_s"] = round(a.prompt / t_ttft, 3)
        rec["single_stream"]["decode_tok_s"] = round(n_lfl / t_gen, 3)
        rec["single_stream"]["e2e_latency_s"] = round(e2e, 4)
        rec["protocol"]["prefill_note"] = (
            "ttft_ms / prefill_tok_s = the PREFILL megakernel, ONE "
            "cudaLaunchCooperativeKernel for the whole 512-token prompt; "
            "decode_tok_s = tokens 2..128 of the SAME end-to-end run through the decode "
            "megakernel, one cooperative launch per token. Comparable to "
            "llama.cpp/vLLM prefill." if a.pfk else
            "NO prefill kernel used in this run: the prompt is fed one token per "
            "decode launch, so ttft_ms / prefill_tok_s are decode-rate numbers.")

        # ---- phase breakdown on one instrumented step ----
        r.step([cur], [a.prompt + a.gen], timings=True)
        torch.cuda.synchronize()
        ph = r.phase_times_us()
        rec["phase_breakdown_ms"] = {
            "qkv": round(ph["qkv"] / 1e3, 4),
            "attn": round((ph["attn"] + ph["attn_reduce"]) / 1e3, 4),
            "o_proj": round(ph["o_proj"] / 1e3, 4),
            "gateup_geglu": round(ph["gateup"] / 1e3, 4),
            "down_proj": round(ph["down"] / 1e3, 4),
            "norms_and_misc": round((ph["h"] + ph["h2"] + ph["tail_h"]) / 1e3, 4),
            "lm_head": round(ph["lm_head"] / 1e3, 4),
            "argmax": round(ph["argmax"] / 1e3, 4),
            "total_measured": round(ph["total"] / 1e3, 4),
        }
        # ---- kernel dispatch proof ----
        lines, total = kernel_proof(r, [cur], [a.prompt + a.gen])
        pf_lines, pf_total, pf_kern = ([], 0, 0)
        if a.pfk:
            pf_lines, pf_total = prefill_proof(r, ids)
            # of those, how many are real KERNELS (the rest are tiny HtoD memcpys)
            pf_kern = sum(int(l.split()[0]) for l in pf_lines
                          if "Memcpy" not in l and "TOTAL" not in l)
            lines = pf_lines + lines
            rec["launch_count"] = {
                "prefill_kernel_launches_for_%d_tokens" % a.prompt: pf_kern,
                "prefill_htod_memcpys": pf_total - pf_kern,
                "decode_launches_per_token": total,
                "total_kernel_launches_for_%d_to_%d" % (a.prompt, a.gen):
                    pf_kern + total * (a.gen - 1),
                "note": ("ONE cooperative KERNEL launch for the whole prompt (plus a "
                         "couple of tiny HtoD memcpys of the token-id vector from the "
                         "prefill runner) + ONE cooperative launch per generated token "
                         "(tokens/positions travel in the decode kernel's parameter "
                         "block, so the decode step has no memcpy at all)."),
            }
        pf_txt = (f"pf::prefill_kernel -- {pf_kern} cooperative kernel launch(es) "
                  f"(+{pf_total - pf_kern} tiny HtoD memcpys) for the whole "
                  f"{a.prompt}-token prompt; " if a.pfk else "")
        summary = (pf_txt + f"cobalt_megakernel::megakernel<M,KG,MINB> -- ONE persistent "
                   f"cooperative launch per decode step ({total} CUDA kernel launch(es) "
                   f"measured, tokens/positions travel in the kernel parameter block so "
                   f"there is no HtoD memcpy), 10 grid.sync phases per layer: h, "
                   f"prep_qkv, qkv(fused), attn (QK-norm+RoPE+KV-append FOLDED IN), "
                   f"attn_reduce, o_proj, h2, prep_gateup, gateup+geglu, down_proj")
        rec["kernel_dispatched"] = summary
        rec["kernel"]["summary"] = summary
        rec["kernel"]["lines"] = lines
        rec["kernel"]["grid_blocks"] = r.blocks
        rec["kernel"]["blocks_per_sm"] = r.blocks_per_sm
        rec["kernel"]["threads_per_block"] = 256
        rec["kernel"]["dynamic_smem_bytes"] = r.smem
        rec["peak_gpu_mem_mib"] = nvsmi_used_mib() or int(
            torch.cuda.max_memory_reserved() / 2**20)
        rec["peak_mem_MiB"]["max_over_all_stages"] = rec["peak_gpu_mem_mib"]
        r._pf = None                  # drop the prefill runner's buffers first
        del r
        gc.collect()
        torch.cuda.empty_cache()

        # ---- in-kernel batch ----
        best = None
        rec["batched_by_M"] = {}
        for M in a.batch_M:
            try:
                rm = KernelRunner(a.model_dir, M=M, max_ctx=a.prompt + a.gen + 8,
                                  config_dir=a.config)
                rm.reset()
                for t in range(a.prompt):
                    lg, nxt = rm.step([ids[t]] * M, [t] * M)
                cur4 = [int(x) for x in nxt.cpu()]
                torch.cuda.synchronize()
                tb = time.perf_counter()
                for s in range(a.gen - 1):
                    lg, nxt = rm.step(cur4, [a.prompt + s] * M)
                    cur4 = [int(x) for x in nxt.cpu()]
                torch.cuda.synchronize()
                dt = time.perf_counter() - tb
                ots = M * (a.gen - 1) / dt
                pk = nvsmi_used_mib()
                if pk:
                    rec["peak_gpu_mem_mib"] = max(rec["peak_gpu_mem_mib"] or 0, pk)
                    rec["peak_mem_MiB"]["max_over_all_stages"] = rec["peak_gpu_mem_mib"]
                rec["batched_by_M"][str(M)] = {
                    "output_tok_s": round(ots, 3),
                    "per_seq_tok_s": round(ots / M, 3),
                    "peak_mem_MiB": pk}
                if best is None or ots > best[1]:
                    best = (M, ots)
                del rm
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                rec["batched_by_M"][str(M)] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[batch M={M}] {e}", flush=True)
        if best:
            M, ots = best
            rec["batched"]["output_tok_s"] = round(ots, 3)
            rec["batched"]["total_tok_s"] = round(
                M * (a.prompt + a.gen - 1) / ((a.prompt + a.gen - 1) * M / ots), 3)
            rec["batched"]["config"] = (
                f"in-kernel batch M={M} (max supported by the decode megakernel), "
                f"{a.prompt}->{a.gen}, all sequences same length. NOT the vLLM "
                f"'256 prompts, concurrency 32' protocol -- no continuous batching.")
            rec["protocol"]["batched"] = rec["batched"]["config"]
        rec["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["status"] = "FAILED"
        rec["reason"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    return 0 if rec["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
