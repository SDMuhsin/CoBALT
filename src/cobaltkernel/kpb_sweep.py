"""COBALT_KPB sweep -- attention key-split width, a PURE RUNTIME knob.

`KernelRunner.KEYS_PER_BLOCK` only feeds `_split_for`, which sets `a.split` (the number of
cross-block key splits per (sequence, kv-head)) in the kernel PARAMETER block.  Nothing is
recompiled, so one model load can sweep the whole axis -- and, unlike speed_oneslice.py,
this reports `attn` (the walk) and `attn_reduce` (the cross-block reduce) SEPARATELY, which
is what says whether a win comes from warp utilisation in the walk or from fewer partials.

At ctx 640 with KPB=128: split = 5, so each block's range is ~128 keys = nch = 4 chunks of
32, handed to 8 warps -> warps 4..7 are IDLE for the whole walk.  KPB=256 -> split = 3,
nch = 8, exactly one chunk per warp.

Protocol matches speed_oneslice.py's decode measurement: 512-token prompt fed through the
decode kernel, then tokens 2..128 timed.  Absolute tok/s is therefore the decode-only
("--no-prefill-kernel") number and is comparable to that column, not to the prefill-kernel one.
"""
import os, sys, time, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import runner as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--kpb", type=int, nargs="+", default=[64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    r = R.KernelRunner(a.model_dir, M=1, max_ctx=a.prompt + a.gen + 8, config_dir=a.config)
    ids = [(i * 7 + 3) % 1000 + 5 for i in range(a.prompt)]

    rows = []
    # interleave the sweep with repeats so drift shows up as disagreement between passes
    order = [k for _ in range(a.repeat) for k in a.kpb]
    for kpb in order:
        r.KEYS_PER_BLOCK = kpb
        split = r._split_for(a.prompt + a.gen - 1)
        r.reset()
        for t in range(16):
            r.step([ids[t]], [t])
        torch.cuda.synchronize()

        r.reset()
        for t in range(a.prompt):
            lg, nxt = r.step([ids[t]], [t])
        cur = int(nxt.cpu()[0])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for s in range(a.gen - 1):
            lg, nxt = r.step([cur], [a.prompt + s])
            cur = int(nxt.cpu()[0])
        torch.cuda.synchronize()
        tok_s = (a.gen - 1) / (time.perf_counter() - t1)

        r.step([cur], [a.prompt + a.gen], timings=True)
        torch.cuda.synchronize()
        ph = r.phase_times_us()
        row = {"kpb": kpb, "split": split, "decode_tok_s": round(tok_s, 3),
               "attn_walk_ms": round(ph["attn"] / 1e3, 4),
               "attn_reduce_ms": round(ph["attn_reduce"] / 1e3, 4),
               "gateup_ms": round(ph["gateup"] / 1e3, 4),
               "down_ms": round(ph["down"] / 1e3, 4),
               "total_ms": round(ph["total"] / 1e3, 4),
               "first_token": cur}
        rows.append(row)
        print(json.dumps(row), flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)
    print("KPBSWEEPDONE")


if __name__ == "__main__":
    main()
