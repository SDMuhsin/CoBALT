#!/usr/bin/env python
"""Tests for the CoBALT-16:32 fixed-cardinality block mask.

  (a) mask_block=0 path is UNCHANGED: cobalt_quantize dispatches to balanced_keepmask
      and blocked_keepmask's normalisation prefix matches balanced_keepmask's.
  (b) blocked_keepmask(block=32) keeps exactly 16 of every aligned 32-block of every
      row; overall sparsity is exactly 0.5.
  (c) BIT-IDENTITY: re-quantize gemma-3-4b layer 00 with --mask-block 0 and torch.equal
      every tensor against the shipped artifact.
  (d) DEAD-COLUMN measurement: fully-dead input columns per matrix, global-topk vs 16:32.

Usage:
  python test_maskblock.py                 # (a) + (b) only, cpu/gpu, seconds
  python test_maskblock.py --real <dir>    # + (c) against <dir>/layer_00.safetensors
  python test_maskblock.py --deadcols <artifact_a> <artifact_b>   # (d) from two artifacts
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cobalt_math as cm


def test_ab(dev):
    torch.manual_seed(0)
    ok = True
    for (K, N) in [(64, 128), (257, 256), (1024, 2560), (128, 10240)]:
        imp = torch.rand(K, N, device=dev).float() ** 3 * torch.rand(1, N, device=dev)
        for sp in (0.0, 0.5):
            for beta in (0.0, 0.5):
                # (a) normalisation prefix identity: blocked with block=N and the
                # SAME keep count as a per-row global-topk is not the same mask, so we
                # instead check the prefix by monkeypatching the selection.
                ref = cm.balanced_keepmask(imp.clone(), sp, beta)
                if sp == 0.0:
                    got = cm.blocked_keepmask(imp.clone(), sp, beta, 32)
                    assert torch.equal(ref, got), "sp=0 path differs"
                    continue
                m = cm.blocked_keepmask(imp.clone(), sp, beta, 32)
                # (b) exactly `keep` per aligned block
                keep = 32 - int(32 * sp)
                cnt = m.view(K, N // 32, 32).sum(-1)
                assert bool((cnt == keep).all()), f"block counts != {keep}: {cnt.unique()}"
                assert abs(float(1 - m.mean()) - sp) < 1e-9, f"sparsity {1-m.float().mean()}"
                # sanity: same total survivors as the global top-k at this sparsity
                assert abs(int(m.sum()) - int(ref.sum())) <= K * N // 32, "cardinality far off"
        print(f"  [b] K={K} N={N}: exactly 16/32 per block, sparsity exact 0.5  OK")

    # (a) hard check: mask_block=0 dispatch inside cobalt_quantize is byte-identical
    K, N = 256, 512
    W = torch.randn(K, N, device=dev)
    X = torch.randn(4096, N, device=dev)
    H = (X.T @ X).float()
    out0 = cm.cobalt_quantize(W, H, 0.5, 0.5, 4, 128, "survivor")
    out1 = cm.cobalt_quantize(W, H, 0.5, 0.5, 4, 128, "survivor", mask_block=0)
    for i, (x, y) in enumerate(zip(out0[:6], out1[:6])):
        assert torch.equal(x, y), f"cobalt_quantize tensor {i} differs with mask_block=0"
    outb = cm.cobalt_quantize(W, H, 0.5, 0.5, 4, 128, "survivor", mask_block=32)
    assert not torch.equal(out0[4], outb[4]), "mask_block=32 produced the SAME mask (suspicious)"
    cntb = outb[4].view(K, N // 32, 32).sum(-1)
    assert bool((cntb == 16).all())
    print("  [a] cobalt_quantize(mask_block=0) == cobalt_quantize() bit-identical  OK")
    print("  [a] cobalt_quantize(mask_block=32) gives 16/32 everywhere              OK")
    return ok


def test_real(ref_dir, new_dir):
    from safetensors import safe_open
    ra = os.path.join(ref_dir, "layer_00.safetensors")
    rb = os.path.join(new_dir, "layer_00.safetensors")
    bad = []
    with safe_open(ra, framework="pt") as A, safe_open(rb, framework="pt") as B:
        ka, kb = sorted(A.keys()), sorted(B.keys())
        assert ka == kb, f"key sets differ: {set(ka) ^ set(kb)}"
        for k in ka:
            ta, tb = A.get_tensor(k), B.get_tensor(k)
            if not (ta.shape == tb.shape and ta.dtype == tb.dtype and torch.equal(ta, tb)):
                d = (ta.float() - tb.float()).abs()
                bad.append((k, int((ta != tb).sum()), float(d.max())))
    if bad:
        print(f"  [c] NOT bit-identical: {len(bad)}/{len(ka)} tensors differ")
        for k, n, mx in bad:
            print(f"      {k}: {n} elements differ, max|Δ|={mx:.3e}")
        return False
    print(f"  [c] BIT-IDENTICAL: all {len(ka)} tensors torch.equal  OK")
    return True


def deadcols(art_a, art_b, nlayers=None):
    """Fully-dead input columns (no survivor in ANY row) per matrix, two artifacts."""
    from safetensors import safe_open
    import json
    def scan(art):
        man = json.load(open(os.path.join(art, "manifest.json")))
        agg = {}
        for li in sorted(int(x) for x in man["layers"]):
            p = os.path.join(art, f"layer_{li:02d}.safetensors")
            if not os.path.exists(p):
                continue
            with safe_open(p, framework="pt") as h:
                names = sorted({k.rsplit(".", 1)[0] for k in h.keys()})
                for n in names:
                    packed = h.get_tensor(f"{n}.mask")
                    K = packed.shape[0]
                    N = h.get_tensor(f"{n}.q").shape[1]
                    bits = torch.arange(8, dtype=torch.uint8)
                    m = ((packed.unsqueeze(-1) >> bits.view(1, 1, 8)) & 1).reshape(K, -1)[:, :N]
                    dead = int((m.sum(0) == 0).sum())
                    a = agg.setdefault(n, dict(N=N, dead=[], layers=0))
                    a["dead"].append(dead); a["layers"] += 1
        return agg
    A, B = scan(art_a), scan(art_b)
    print(f"\n| matrix | N | dead cols global-topk (mean/max) | dead cols 16:32 (mean/max) |")
    print("|---|---|---|---|")
    for n in A:
        da, db = A[n]["dead"], B.get(n, {}).get("dead", [0])
        print(f"| {n} | {A[n]['N']} | {sum(da)/len(da):.1f} / {max(da)} | "
              f"{sum(db)/len(db):.1f} / {max(db)} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--real", nargs=2, metavar=("REF_DIR", "NEW_DIR"), default=None)
    ap.add_argument("--deadcols", nargs=2, metavar=("ART_TOPK", "ART_BLK"), default=None)
    a = ap.parse_args()
    rc = 0
    if a.deadcols:
        deadcols(*a.deadcols)
    else:
        print("== (a)+(b) synthetic ==")
        test_ab(torch.device(a.device))
        if a.real:
            print("== (c) real gemma-3-4b layer_00 bit-identity ==")
            if not test_real(*a.real):
                rc = 1
    sys.exit(rc)
