#!/usr/bin/env python
"""Sub-phase profile of ONE decode step with a COBALT_PROF=1 build: grid.sync-delimited phase
times (block 0 stamps) and block 0's OWN work inside the attention / GEMV phases (CBK_PROF
accumulators), so phase time - own work = barrier wait / skew.

  COBALT_PROF=1 COBALT_BLK1632=4 python prof_phases.py --model <packed> --config <hf> [--ctx 640]
"""
import argparse, os, subprocess, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel.runner import KernelRunner  # noqa: E402


def sm_clock_hz():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.split()
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gl = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
        gi = 0
        for line in gl.splitlines():
            if line.startswith("GPU "):
                gi = int(line.split()[1].rstrip(":"))
            if uuid and uuid[:16] in line:
                break
        return float(q[gi]) * 1e6
    except Exception:
        return 2.43e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ctx", type=int, default=640)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    assert os.environ.get("COBALT_PROF"), "build with COBALT_PROF=1"
    r = KernelRunner(a.model, M=1, max_ctx=a.ctx + 16, config_dir=a.config)
    ids = [(i * 7 + 3) % 1000 + 5 for i in range(a.ctx)]
    r.reset()
    for t in range(a.ctx):
        r.step([ids[t]], [t])
    torch.cuda.synchronize()
    hz = sm_clock_hz()
    agg_ph, agg_pr = {}, {}
    for k in range(a.reps):
        r.timings.zero_()
        r.step([ids[0]], [a.ctx + k], timings=True)
        torch.cuda.synchronize()
        ph = r.phase_times_us(sm_clock_hz=hz)
        pr = r.prof_times_us(sm_clock_hz=hz)
        for kk, v in ph.items():
            agg_ph[kk] = agg_ph.get(kk, 0) + v / a.reps
        for kk, v in pr.items():
            agg_pr[kk] = agg_pr.get(kk, 0) + v / a.reps
    L = r.n_layers
    print(f"SM clock {hz/1e6:.0f} MHz, {L} layers, ctx {a.ctx}, blocks {r.blocks}, KPB {r.KEYS_PER_BLOCK}, "
          f"split {r._split_for(a.ctx)}")
    print(f"{'phase (grid.sync-delimited, per token)':42} {'us/token':>9} {'us/layer':>9}")
    for kk, v in agg_ph.items():
        print(f"  {kk:40} {v:9.1f} {v/L:9.2f}")
    print(f"{'block-0 OWN work inside phases (per token)':42} {'us/token':>9} {'us/layer':>9}")
    for kk, v in agg_pr.items():
        print(f"  {kk:40} {v:9.1f} {v/L:9.2f}")
    attn = agg_ph.get("attn", 0) + agg_ph.get("attn_reduce", 0)
    own = sum(agg_pr.get(k, 0) for k in ("attn_prep", "attn_sq", "attn_walk", "attn_combine", "attn_reduce"))
    print(f"ATTENTION: phase {attn:.1f} us/token, block-0 own work {own:.1f} us/token -> "
          f"barrier/skew {attn-own:.1f} us/token ({(attn-own)/max(attn,1e-9)*100:.0f}%)")
    for ph_k, pr_k in (("qkv", "qkv_gemv"), ("o_proj", "o_gemv"), ("gateup", "gateup_gemv"), ("down", "down_gemv"), ("lm_head", "lm_head_gemv")):
        if ph_k in agg_ph and pr_k in agg_pr:
            print(f"{ph_k:8}: phase {agg_ph[ph_k]:8.1f}  own {agg_pr[pr_k]:8.1f}  skew {agg_ph[ph_k]-agg_pr[pr_k]:7.1f} us/token")
    print("PROF_DONE")


if __name__ == "__main__":
    main()
