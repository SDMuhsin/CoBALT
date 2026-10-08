#!/usr/bin/env python
"""Dump the decode megakernel's logits for a fixed token sequence, so two builds of the
kernel (e.g. pre-/post-port source trees) can be compared bit-for-bit with torch.equal.

  python dump_logits.py --model <packed dir> --config <hf dir> --steps 48 --out <file.pt>
"""
import argparse, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel.runner import KernelRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    r = KernelRunner(a.model, M=1, max_ctx=a.steps + 8, config_dir=a.config)
    ids = [(i * 7 + 3) % 1000 + 5 for i in range(a.steps)]   # same deterministic prompt as speed_oneslice
    out = []
    for t, tok in enumerate(ids):
        lg, _ = r.step([tok], [t])
        out.append(lg.detach().float().cpu().clone())
    torch.save({"ids": ids, "logits": torch.stack(out), "model": a.model,
                "blocks": r.blocks}, a.out)
    print(f"DUMP_DONE {a.out} steps={a.steps} blocks={r.blocks} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
