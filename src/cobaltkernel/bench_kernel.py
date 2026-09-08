"""Decode throughput + phase breakdown for the BF16-dense Gemma3 megakernel.

    source scripts/cobaltkernel_env.sh 1g      # or 2g for the headline number
    python src/cobaltkernel/bench_kernel.py --model .../gemma-3-4b/text_bf16 \
           --M 1 4 8 --prompt 512 --gen 128
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel.runner import KernelRunner   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--bw", type=float, default=None,
                    help="measured slice read BW in GB/s (2g=770.8, 1g=385.5)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log = []

    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        log.append(s)

    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    bw = a.bw if a.bw else (770.8 if sms > 60 else 385.5)
    P(f"device: {props.name}  SMs={sms}  mem={props.total_memory/2**30:.1f} GiB  "
      f"assumed read BW={bw} GB/s")
    P(f"protocol: {a.prompt} prompt (fed token-by-token through the decode kernel) "
      f"-> {a.gen} generated tokens")
    P("")

    rows = []
    for M in a.M:
        r = KernelRunner(a.model, M=M, max_ctx=a.prompt + a.gen + 8, config_dir=a.config)
        mb = r.weight_bytes()
        ceil_toks = bw * 1e9 / mb
        r.reset()
        ids = [[(i * 7 + 3) % 1000 + 5 for i in range(a.prompt)] for _ in range(M)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(a.prompt):
            lg, nxt = r.step([x[t] for x in ids], [t] * M)
        torch.cuda.synchronize()
        t_pf = time.perf_counter() - t0

        cur = [int(x) for x in nxt.cpu()]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(a.gen):
            lg, nxt = r.step(cur, [a.prompt + s] * M)
            cur = [int(x) for x in nxt.cpu()]
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0

        # phase breakdown on a single instrumented step
        r.step(cur, [a.prompt + a.gen] * M, timings=True)
        torch.cuda.synchronize()
        ph = r.phase_times_us()

        step_ms = t_gen / a.gen * 1e3
        toks = M * a.gen / t_gen
        eff_bw = mb / (t_gen / a.gen) / 1e9
        P(f"== M={M} ==")
        P(f"  grid                 : {r.blocks} blocks ({r.blocks_per_sm}/SM) x 256 thr")
        P(f"  weight bytes / step  : {mb/2**30:.3f} GiB")
        P(f"  decode               : {step_ms:.3f} ms/step  {toks:.2f} tok/s "
          f"({M*a.gen} tokens in {t_gen:.2f} s)")
        P(f"  bandwidth ceiling    : {ceil_toks*M:.2f} tok/s (M={M}) "
          f"| achieved {eff_bw:.1f} GB/s = {eff_bw/bw*100:.1f}% of {bw}")
        P(f"  prompt feed (1 tok/launch): {a.prompt/t_pf:.1f} tok/s")
        tot = ph["total"]
        P(f"  phase breakdown (clock64, block0, one step, total {tot:.0f} us):")
        for k in r.PHASES + ["tail_h", "lm_head", "argmax"]:
            P(f"      {k:12s} {ph[k]:9.1f} us  {ph[k]/tot*100:5.1f}%")
        P(f"      {'barriers/etc':12s} {tot-sum(ph[k] for k in ph if k!='total'):9.1f} us")
        P("")
        rows.append(dict(M=M, tok_s=toks, step_ms=step_ms, ceil=ceil_toks * M,
                         eff_bw=eff_bw, blocks=r.blocks, phases=ph))
        del r
        gc.collect(); torch.cuda.empty_cache()

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write("\n".join(log) + "\n")
        open(a.out + ".json", "w").write(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
