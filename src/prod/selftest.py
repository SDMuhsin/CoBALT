"""End-to-end self-test of the deployment package.

    python -m prod.selftest                       # offline checks only, ~1 s
    python -m prod.selftest --artifact DIR        # + verify that artifact
    python -m prod.selftest --artifact DIR --config HF --kernel   # + build and generate

The offline tier needs no GPU and no artifact: it checks that every recipe reconstructs
the exact command line that built the shipped arms, that the diagnostic knobs really are
cleared, and that the modules import.  Run it after touching anything in `prod`.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import recipes

# The command lines that produced the shipped artifacts, as recorded in their manifests.
EXPECTED_QUANT = {
    "dense4": "--sparsity 0.5 --bits 4 --beta 0.5 --group-size 128 --hull survivor",
    "blk1632_b4": "--mask-block 32",
    "blk1632_b4_oproj": "--mask-block 32 --mask-block-exclude self_attn.o_proj",
    "blk1632_b6_oproj4": "--bits 6 --mask-block 32 --mask-block-exclude self_attn.o_proj "
                         "--bits-override self_attn.o_proj=4",
}
EXPECTED_PACK = {
    "dense4": "--layout DENSE4 --bits 4",
    "blk1632_b4": "--layout BLK1632_4 --bits 4",
    "blk1632_b4_oproj": "--layout BLK1632_4 --bits 4",
    "blk1632_b6_oproj4": "--layout BLK1632_6 --bits 6 "
                         "--fuse-qkv --layout-override o_proj=DENSE4 --bits-override o_proj=4",
}


def _fragments(spec: str) -> list[str]:
    """Split a recorded command line into the '--flag value' fragments to look for."""
    return ["--" + part.strip().lstrip("-") for part in spec.split(" --") if part.strip()]


def _fail(cond, msg, fails):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def offline() -> list[str]:
    from . import pack, quantize
    fails: list[str] = []

    print("recipes reconstruct the shipped command lines")
    for name in recipes.names():
        r = recipes.get(name)
        q = " ".join(str(x) for x in quantize.build_argv(r, "HF", "OUT"))
        p = " ".join(str(x) for x in pack.build_argv(r, "RAW", "OUT", "HF"))
        for frag in _fragments(EXPECTED_QUANT[name]):
            _fail(frag in q, f"{name}: quantizer argv has {frag!r}", fails)
        for frag in _fragments(EXPECTED_PACK[name]):
            _fail(frag in p, f"{name}: packer argv has {frag!r}", fails)

    print("diagnostic knobs are cleared by activate()")
    for knob in recipes.DIAGNOSTIC_KNOBS:
        os.environ[knob] = "1"
    with recipes.activate("blk1632_b4"):
        leaked = [k for k in recipes.DIAGNOSTIC_KNOBS if k in os.environ]
        _fail(not leaked, f"no diagnostic leaks into the run (leaked: {leaked})", fails)
        _fail(os.environ.get("COBALT_BLK1632") == "4", "pinned knob is set", fails)
    _fail(all(os.environ.get(k) == "1" for k in recipes.DIAGNOSTIC_KNOBS),
          "the caller's environment is restored on exit", fails)
    for knob in recipes.DIAGNOSTIC_KNOBS:
        os.environ.pop(knob, None)

    print("bpw accounting matches the format spec")
    for name, want in (("dense4", 4.1875), ("blk1632_b4", 3.1875),
                       ("blk1632_b6_oproj4", 4.1875)):
        _fail(recipes.get(name).bpw == want, f"{name} bpw == {want}", fails)
    return fails


def with_artifact(artifact, config, kernel) -> list[str]:
    from . import bench, model, verify
    fails: list[str] = []

    print("artifact")
    r = model.recipe_for_artifact(artifact)
    _fail(True, f"recipe inferred from the manifest: {r.name}", fails)
    rep = verify.check_artifact(artifact, layers=1, strict=False)
    _fail(rep["ok"], f"structural + numerical check ({rep.get('problems')})", fails)
    _fail(abs(rep["bpw_recomputed"] - r.bpw) < 0.05,
          f"bpw {rep['bpw_recomputed']} matches recipe {r.bpw}", fails)

    if kernel:
        print("kernel")
        m = model.CobaltModel.load(artifact, config_dir=config, max_ctx=192)
        ids = [(i * 7 + 3) % 1000 + 5 for i in range(64)]
        out = m.generate(ids, 16)
        _fail(len(out) == 16 and all(isinstance(t, int) for t in out),
              f"generate() returned 16 token ids", fails)
        del m
        res = bench.decode(artifact, config_dir=config, prompt=64, gen=16, warmup=4)
        _fail(res["decode_tok_s"] > 0, f"bench decode {res['decode_tok_s']} tok/s", fails)
        print(bench.compare_to_reference(res))
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--kernel", action="store_true")
    a = ap.parse_args(argv)

    fails = offline()
    if a.artifact:
        fails += with_artifact(a.artifact, a.config, a.kernel)
    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print("  - " + f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
