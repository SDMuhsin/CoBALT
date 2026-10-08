"""Command line for the CoBALT deployment package.

    python -m prod doctor                       # is this machine ready?
    python -m prod recipes [name]               # what can I build?
    python -m prod targets                      # which models ship, and what did each score?
    python -m prod quantize  -r blk1632_b4 --model <hf> --out <raw>
    python -m prod pack      -r blk1632_b4 --raw <raw> --out <packed> --model <hf>
    python -m prod verify    <packed> [--config <hf>] [--kernel]
    python -m prod bench     <packed> [--config <hf>] [--prompt 512 --gen 128]
    python -m prod generate  <packed> [--config <hf>] --prompt "text" [-n 64]

`--config` is only needed when the artifact directory has no config.json beside the
weights (the shipped overlays put it there).  Quantizing a 27B model takes hours; run it
detached.
"""
from __future__ import annotations

import argparse
import json
import sys


def _p_common(p):
    p.add_argument("-r", "--recipe", default="blk1632_b4",
                   help="shipped arm (see `python -m prod recipes`)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prod", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check toolchain and device prerequisites")

    p = sub.add_parser("recipes", help="list the shipped arms")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("targets", help="list the shipped models and their reference numbers")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("quantize", help="stage 1: HF checkpoint -> raw artifact")
    _p_common(p)
    p.add_argument("--model", required=True, help="HF snapshot dir")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="print the command, run nothing")

    p = sub.add_parser("pack", help="stage 2: raw artifact -> CBK1 packed artifact")
    _p_common(p)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None,
                   help="HF snapshot (supplies the tied embedding and the norms)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("verify", help="stage 4: check a packed artifact")
    p.add_argument("artifact")
    p.add_argument("--config", default=None)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--kernel", action="store_true",
                   help="also run the megakernel against the torch reference (slow)")
    # 1152 is the length the 98.5% argmax bar was established at. Shorter runs are
    # noisy -- at 128 positions a single disagreement moves the figure by 0.8 pt.
    p.add_argument("--prompt-len", type=int, default=1152)
    p.add_argument("--steps", type=int, default=32)

    p = sub.add_parser("bench", help="stage 5: measure decode throughput")
    p.add_argument("artifact")
    p.add_argument("--config", default=None)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--batch-M", type=int, nargs="*", default=[])
    p.add_argument("--out", default=None, help="write the result JSON here")

    p = sub.add_parser("generate", help="greedy generation from a packed artifact")
    p.add_argument("artifact")
    p.add_argument("--config", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("-n", "--max-new-tokens", type=int, default=64)
    p.add_argument("--max-ctx", type=int, default=1024)

    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        from . import env
        rep = env.preflight(strict=False)
        print(json.dumps(rep, indent=2))
        return 1 if rep["blockers"] else 0

    if a.cmd == "recipes":
        from . import recipes
        if a.json:
            import dataclasses
            rs = ([recipes.get(a.name)] if a.name else list(recipes.RECIPES.values()))
            print(json.dumps([dataclasses.asdict(r) for r in rs], indent=2))
        else:
            print(recipes.describe(a.name))
            print("`python -m prod targets` lists the models and the measurement protocol.")
        return 0

    if a.cmd == "targets":
        from . import recipes
        if a.json:
            import dataclasses
            print(json.dumps([dataclasses.asdict(t) for t in recipes.TARGETS.values()],
                             indent=2))
        else:
            print(recipes.describe_targets())
        return 0

    if a.cmd == "quantize":
        from . import quantize, recipes
        r = recipes.get(a.recipe)
        if a.dry_run:
            print("quantize_cobalt.py " + " ".join(
                str(x) for x in quantize.build_argv(r, a.model, a.out, a.device)))
            return 0
        quantize.run(r, a.model, a.out, a.device)
        print(f"raw artifact -> {a.out}")
        return 0

    if a.cmd == "pack":
        from . import pack, recipes
        r = recipes.get(a.recipe)
        if a.dry_run:
            print("pack_cobalt.py " + " ".join(
                str(x) for x in pack.build_argv(r, a.raw, a.out, a.model, a.device)))
            return 0
        bpw = pack.run(r, a.raw, a.out, a.model, a.device)
        print(f"packed artifact -> {a.out}\n{json.dumps(bpw, indent=2)}")
        return 0

    if a.cmd == "verify":
        from . import verify
        rep = verify.check_artifact(a.artifact, layers=a.layers, strict=False)
        print(json.dumps(rep, indent=2))
        if not rep["ok"]:
            return 1
        if a.kernel:
            import os
            if not a.config and not os.path.exists(os.path.join(a.artifact, "config.json")):
                ap.error("--kernel needs --config <hf snapshot> when the artifact has no "
                         "config.json beside the weights")
            verify.check_kernel(a.artifact, a.config, a.prompt_len, a.steps)
        return 0

    if a.cmd == "bench":
        from . import bench
        res = bench.decode(a.artifact, config_dir=a.config, prompt=a.prompt, gen=a.gen,
                           batch_M=tuple(a.batch_M))
        print(json.dumps(res, indent=2))
        print()
        print(bench.compare_to_reference(res))
        if a.out:
            json.dump(res, open(a.out, "w"), indent=2)
        return 0

    if a.cmd == "generate":
        from . import CobaltModel
        m = CobaltModel.load_pretrained(a.artifact, config_dir=a.config,
                                        max_ctx=a.max_ctx)
        print(m.generate_text(a.prompt, a.max_new_tokens))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
