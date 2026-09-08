"""Correctness + roofline/throughput harness for cbk::gemm_tile (bf16 mma.sync path).

  source scripts/cobaltkernel_env.sh 1g     # or 2g if free
  python src/cobaltkernel/test_gemm.py --roofline
  python src/cobaltkernel/test_gemm.py --test
  python src/cobaltkernel/test_gemm.py --bench
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_cobalt as P
from pack_cobalt import (GROUP, ARR_ALIGN, BLOB_TAIL, DENSE4, DENSE8, LAYOUT_NAME, _align,
                          BLK1632_4, BLK1632_6, BLK1632, BLK1632_BITS, BLOCK, BLOCK_KEEP)

HERE = os.path.dirname(os.path.abspath(__file__))
# (MT, WM, S, BR, KT) configurations compiled into test_gemm.cu -- keep in sync with CBK_CFGS
CONFIGS = [(32, 1, 4, 64, 64, 1, 1), (32, 1, 4, 128, 128, 1, 1), (32, 1, 4, 256, 128, 1, 1),
           (32, 1, 4, 64, 256, 1, 1), (32, 1, 4, 64, 256, 2, 1),
           (32, 1, 4, 128, 256, 1, 1), (32, 1, 4, 128, 256, 2, 1),
           (16, 1, 6, 64, 64, 1, 1),
           (64, 1, 4, 64, 64, 1, 1), (64, 1, 3, 128, 128, 1, 1), (64, 1, 3, 256, 128, 1, 1),
           (64, 1, 3, 256, 128, 2, 1),
           (128, 1, 2, 128, 64, 1, 1), (128, 1, 2, 128, 128, 1, 1), (128, 1, 2, 128, 128, 2, 1),
           (128, 1, 2, 128, 128, 3, 1), (128, 2, 2, 256, 128, 1, 1), (128, 1, 2, 256, 128, 1, 1),
           # round 4: occupancy configs, last field = __launch_bounds__ min blocks/SM
           (64, 1, 3, 128, 128, 1, 2), (64, 2, 3, 128, 128, 1, 2), (64, 2, 3, 128, 128, 1, 1),
           (128, 1, 2, 64, 128, 1, 2), (128, 2, 2, 128, 128, 1, 2), (128, 2, 2, 128, 128, 1, 1),
           (64, 2, 3, 256, 128, 1, 2), (64, 2, 3, 256, 128, 1, 1),
           (128, 2, 2, 256, 128, 1, 2),
           (64, 1, 3, 64, 128, 1, 3), (32, 1, 4, 128, 256, 1, 2), (32, 1, 4, 64, 256, 1, 3),
           (128, 2, 2, 128, 256, 1, 1)]
SHAPES = [("q_proj", 4096, 5376), ("k_proj", 2048, 5376), ("qkv_fused", 8192, 5376),
          ("o_proj", 5376, 4096),
          ("gateup", 43008, 5376), ("down_proj", 5376, 21504), ("lm_head", 262144, 5376)]


# BLK16_32 tile-path configs -- keep in sync with the CBK_TEST_BLK branch of CBK_CFGS
# in csrc/test_gemm.cu.
BLK_CONFIGS = [(16, 1, 6, 64, 64, 1, 1), (32, 1, 4, 64, 64, 1, 1), (32, 1, 4, 128, 128, 1, 1),
               (32, 1, 4, 64, 256, 1, 1), (32, 1, 4, 128, 256, 1, 2),
               (64, 1, 3, 128, 128, 1, 1), (64, 1, 3, 256, 128, 1, 1),
               (64, 2, 3, 128, 128, 1, 2),
               (128, 1, 2, 128, 128, 1, 1), (128, 2, 2, 256, 128, 1, 1),
               (128, 1, 2, 256, 128, 1, 1)]


def build_ext(extra=()):
    from torch.utils.cpp_extension import load
    name = "cbk_test_gemm"
    if extra:
        name += "_" + "".join(c for c in "".join(extra) if c.isalnum())
    return load(name=name, sources=[os.path.join(HERE, "csrc", "test_gemm.cu")],
                extra_include_paths=[os.path.join(HERE, "csrc")],
                extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math",
                                   "-gencode", "arch=compute_120a,code=sm_120a"] + list(extra),
                verbose=False)


def make_blob(pk, device="cuda"):
    parts, offs, pos = [], {}, 0

    def add(name, arr):
        nonlocal pos
        pad = _align(pos, ARR_ALIGN) - pos
        if pad:
            parts.append(np.zeros(pad, np.uint8)); pos += pad
        offs[name] = pos
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        parts.append(b); pos += b.size
    add("data", pk["data"]); add("scale", pk["scale"]); add("zero", pk["zero"])
    return torch.from_numpy(np.concatenate(parts)).to(device), offs, pos


def synth(K, N, layout, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    G = N // GROUP
    stride = N if layout == DENSE8 else N // 2
    data = torch.randint(0, 256, (K * stride + BLOB_TAIL,), generator=g, device=device,
                         dtype=torch.int32).to(torch.uint8).cpu().numpy()
    scale = (torch.rand(K, G, generator=g, device=device) * 0.02 + 1e-3).half().cpu().numpy()
    nl = 255 if layout == DENSE8 else 15
    zero = torch.randint(0, nl + 1, (K, G), generator=g, device=device,
                         dtype=torch.int32).to(torch.uint8).cpu().numpy()
    cs = (torch.rand(N, generator=g, device=device) * 1.0 + 0.5).half()
    return dict(data=data, scale=scale, zero=zero), cs


def ref_matmul(pk, cs, K, N, layout, X, cs_pairs=0, bf16_W=False):
    """torch reference: dequant -> (optionally round W to bf16) -> fp32 matmul."""
    G = N // GROUP
    dev = X.device
    d = torch.from_numpy(pk["data"][:K * (N if layout == DENSE8 else N // 2)]).to(dev)
    sc = torch.from_numpy(pk["scale"].astype(np.float32)).to(dev).view(K, G)
    zi = torch.from_numpy(pk["zero"].astype(np.float32)).to(dev).view(K, G)
    if layout == DENSE8:
        q = d.view(K, N).float()
    else:
        b = d.view(K, N // 2)
        q = torch.stack([(b & 0xF), (b >> 4)], -1).view(K, N).float()
    W = (q - zi.repeat_interleave(GROUP, 1)) * sc.repeat_interleave(GROUP, 1)
    if cs is not None:
        c = cs.float().to(dev)
        if cs_pairs:
            c = c.view(2, N)
            sel = (torch.arange(K, device=dev) & 1).view(K, 1)
            W = W * torch.where(sel == 0, c[0].view(1, N), c[1].view(1, N))
        else:
            W = W * c.view(1, N)
    if bf16_W:
        W = W.to(torch.bfloat16).float()
    return X.float() @ W.t()


# ---------------------------------------------------------------- BLK16_32 (sec.13)
def synth_blk(K, N, layout, seed=0, device="cuda", zero_off_grid=False):
    """A VALID BLK16_32 packed matrix: exactly 16 survivors per aligned 32-column block,
    random code planes.  Structure is what the decoder reads; the torch oracle is
    pack_cobalt.dequant_reference on the SAME bytes."""
    bits = BLK1632_BITS[layout]
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)
    G, NB = N // GROUP, N // BLOCK
    stride = (N * 3 // 8) if bits == 4 else (N // 2)
    scale = (torch.rand(K, G, generator=g, device=dev) * 0.02 + 1e-3).half().cpu().numpy()
    nl = (1 << bits) - 1
    if zero_off_grid:                       # sec.13.3: a zero point OFF the code grid
        zero = torch.full((K, G), nl + 37, device=dev, dtype=torch.int32)
    else:
        zero = torch.randint(0, nl + 1, (K, G), generator=g, device=dev, dtype=torch.int32)
    zero = zero.to(torch.uint8).cpu().numpy()
    r = torch.rand(K, NB, BLOCK, generator=g, device=dev)
    idx = torch.topk(r, BLOCK_KEEP, dim=-1).indices
    keep = torch.zeros_like(r, dtype=torch.bool).scatter_(-1, idx, True).view(K, N)
    sh = torch.arange(8, device=dev, dtype=torch.uint8)
    mb = (keep.view(K, N // 8, 8).to(torch.uint8) << sh).sum(-1).to(torch.uint8)
    body = torch.randint(0, 256, (K, stride - N // 8), generator=g, device=dev,
                         dtype=torch.int32).to(torch.uint8)
    data = torch.cat([mb, body], 1).reshape(-1)
    data = torch.cat([data, torch.zeros(BLOB_TAIL, dtype=torch.uint8, device=dev)])
    cs = (torch.rand(N, generator=g, device=dev) * 1.0 + 0.5).half()
    pk = dict(data=data.cpu().numpy(), row_off=None, scale=scale, zero=zero)
    return pk, cs, keep


def ref_matmul_blk(pk, cs, K, N, layout, X, cs_pairs=0, bf16_W=True):
    W = P.dequant_reference(pk, K, N, layout).to(X.device)
    if cs is not None:
        c = cs.float().to(X.device)
        if cs_pairs:
            c = c.view(2, N)
            sel = (torch.arange(K, device=X.device) & 1).view(K, 1)
            W = W * torch.where(sel == 0, c[0].view(1, N), c[1].view(1, N))
        else:
            W = W * c.view(1, N)
    if bf16_W:
        W = W.to(torch.bfloat16).float()
    return X.float() @ W.t()


def run_test_blk(a):
    """Correctness of cbk::gemm_tile's BLK16_32 path against the torch dequant oracle,
    at the coverage the DENSE4 path has, plus the pruned-positions-contribute-exactly-zero
    test (FORMAT.md sec.13.3) THROUGH the tile."""
    dev = "cuda"
    rows, worst = [], 0.0
    for layout in (BLK1632_4, BLK1632_6):
        ext = build_ext(tuple(a.define) + (f"-DCBK_TEST_BLK={BLK1632_BITS[layout]}",))
        cases = [(256, 5376, 0), (512, 5376, 1), (256, 21504, 0), (320, 4096, 0),
                 (128, 1024, 0)]
        for K, N, csp in cases:
            pk, cs1, keep = synth_blk(K, N, layout, seed=K + N, device=dev)
            cs = cs1 if not csp else torch.cat([cs1, (cs1 * 0.7 + 0.2).half()]).view(-1)
            blob, offs, nb = make_blob(pk, dev)
            for M in (16, 32, 64, 100, 512):
                for MT in (16, 32, 64, 128):
                    cfgs = [c for c in BLK_CONFIGS if c[0] == MT and N % c[4] == 0
                            and c[3] <= max(64, K)]
                    if not cfgs:
                        continue
                    X = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1
                    R32 = ref_matmul_blk(pk, cs.to(dev), K, N, layout, X, csp, bf16_W=False)
                    Rbf = ref_matmul_blk(pk, cs.to(dev), K, N, layout, X, csp, bf16_W=True)
                    for (_, WMc, S, BR, KT, PF, MB) in cfgs:
                        Y = ext.gemm(blob, X, cs.to(dev), K, N, layout, offs["data"],
                                     offs["scale"], offs["zero"], M, MT, 0, csp, WMc, S,
                                     BR, KT, PF, MB)
                        e32 = float((Y - R32).abs().max() / R32.abs().max())
                        ebf = float((Y - Rbf).abs().max() / Rbf.abs().max())
                        worst = max(worst, ebf)
                        rows.append(dict(layout=LAYOUT_NAME[layout], K=K, N=N, M=M, MT=MT,
                                         WM=WMc, BR=BR, KT=KT, MINB=MB, cs_pairs=csp,
                                         rel_vs_fp32=e32, rel_vs_bf16W=ebf))
                        assert ebf < 1.5e-2, rows[-1]
            # ---- GeGLU epilogue
            if csp:
                M, MT = 32, 32
                X = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1
                _, WMc, S, BR, KT, PF, MB = next(c for c in BLK_CONFIGS
                                                 if c[0] == MT and c[1] == 1 and N % c[4] == 0)
                Yg = ext.gemm(blob, X, cs.to(dev), K, N, layout, offs["data"], offs["scale"],
                              offs["zero"], M, MT, 2, csp, WMc, S, BR, KT, PF, MB).float()
                R = ref_matmul_blk(pk, cs.to(dev), K, N, layout, X, csp, bf16_W=True)
                gate, up = R[:, 0::2], R[:, 1::2]
                Rg = torch.nn.functional.gelu(gate, approximate="tanh") * up
                eg = float((Yg - Rg).abs().max() / Rg.abs().max())
                print(f"  GEGLU {LAYOUT_NAME[layout]} K={K} N={N}: rel err = {eg:.3e}")
                assert eg < 2e-2, eg
    print(f"{'layout':<11}{'K':>7}{'N':>7}{'M':>5}{'MT':>4}{'BR':>5}{'KT':>5}{'cs':>4}"
          f"{'rel vs fp32':>14}{'rel vs bf16-W':>15}")
    for r in rows:
        print(f"{r['layout']:<11}{r['K']:>7}{r['N']:>7}{r['M']:>5}{r['MT']:>4}{r['BR']:>5}"
              f"{r['KT']:>5}{r['cs_pairs']:>4}"
              f"{r['rel_vs_fp32']:>14.3e}{r['rel_vs_bf16W']:>15.3e}")
    print(f"\nALL PASS ({len(rows)} cases)  worst rel vs fp32 = "
          f"{max(r['rel_vs_fp32'] for r in rows):.3e}   "
          f"worst vs bf16-rounded-W = {worst:.3e}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


def run_prune_zero(a):
    """FORMAT.md sec.13.3 THROUGH the tile GEMM: a HUGE activation at every pruned column
    must not move the output by a single bit, whatever the zero point is.  DENSE4 is the
    control (it stores `zero` at pruned positions too, so through this path it is also
    exact -- unlike its GEMV accumulator)."""
    dev = "cuda"
    K, N, M, MT = 256, 5376, 64, 64
    torch.manual_seed(0)
    print(f"{'layout':<11}{'zero':>10}{'max|dY|':>12}{'bit-identical':>15}"
          f"{'max|W| pruned':>15}")
    ok_all = True
    for layout in (BLK1632_4, BLK1632_6):
        ext = build_ext(tuple(a.define) + (f"-DCBK_TEST_BLK={BLK1632_BITS[layout]}",))
        for off in (False, True):
            pk, cs, keep = synth_blk(K, N, layout, seed=7, device=dev, zero_off_grid=off)
            blob, offs, nb = make_blob(pk, dev)
            W = P.dequant_reference(pk, K, N, layout)
            wp = float(W.abs()[~keep.cpu()].max())
            X0 = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1
            colkeep = keep.any(0)                    # a column pruned in EVERY row
            X1 = X0.clone()
            X1[:, ~colkeep] = torch.tensor(1.0e4, dtype=torch.bfloat16)
            args = (K, N, layout, offs["data"], offs["scale"], offs["zero"], M, MT, 0, 0,
                    1, 3, 128, 128, 1, 1)
            Y0 = ext.gemm(blob, X0, cs.to(dev), *args)
            Y1 = ext.gemm(blob, X1, cs.to(dev), *args)
            d = float((Y1.float() - Y0.float()).abs().max())
            bit = bool(torch.equal(Y0, Y1))
            ok_all &= bit and d == 0.0
            print(f"{LAYOUT_NAME[layout]:<11}{'OFF grid' if off else 'on grid':>10}"
                  f"{d:>12.3e}{('yes' if bit else 'NO'):>15}{wp:>15.3e}")
    print("PRUNE-ZERO THROUGH THE TILE: " + ("PASS" if ok_all else "FAIL"))
    assert ok_all


def run_roofline(ext, a):
    props = torch.cuda.get_device_properties(0)
    print(f"# SMs={props.multi_processor_count}")
    best = 0.0
    for blocks in (props.multi_processor_count, 2 * props.multi_processor_count,
                   4 * props.multi_processor_count):
        t = ext.mma_peak(blocks, 4096, 5)
        best = max(best, t)
        print(f"mma.sync m16n8k16 bf16 peak: blocks={blocks:<5} {t:8.1f} TFLOP/s")
    print(f"ROOFLINE bf16 mma peak = {best:.1f} TFLOP/s")
    return best


def run_test(ext, a):
    dev = "cuda"
    rows = []
    cases = [(DENSE4, 256, 5376, 0), (DENSE4, 512, 5376, 1), (DENSE4, 256, 21504, 0),
             (DENSE4, 320, 4096, 0), (DENSE8, 256, 5376, 0)]
    for layout, K, N, csp in cases:
        pk, cs1 = synth(K, N, layout, seed=K + N, device=dev)
        cs = cs1 if not csp else torch.cat([cs1, (cs1 * 0.7 + 0.2).half()]).view(-1)
        blob, offs, nb = make_blob(pk, dev)
        for M in (16, 32, 64, 100, 512):
            for MT in (16, 32, 64):
                X = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1
                cfgs = [c for c in CONFIGS if c[0] == MT
                        and N % c[4] == 0 and (layout == DENSE4 or c[4] == 64)]
                R32 = ref_matmul(pk, cs.to(dev), K, N, layout, X, csp, bf16_W=False)
                Rbf = ref_matmul(pk, cs.to(dev), K, N, layout, X, csp, bf16_W=True)
                for (_, WMc, S, BR, KT, PF, MB) in cfgs:
                    Y = ext.gemm(blob, X, cs.to(dev), K, N, layout, offs["data"],
                                 offs["scale"], offs["zero"], M, MT, 0, csp, WMc, S, BR, KT,
                                 PF, MB)
                    e32 = float((Y - R32).abs().max() / R32.abs().max())
                    ebf = float((Y - Rbf).abs().max() / Rbf.abs().max())
                    rows.append(dict(layout=LAYOUT_NAME[layout], K=K, N=N, M=M, MT=MT,
                                     WM=WMc, BR=BR, KT=KT, PF=PF, MINB=MB, cs_pairs=csp,
                                     rel_vs_fp32=e32,
                                     rel_vs_bf16W=ebf))
                    assert ebf < 1.5e-2, rows[-1]
        # GeGLU epilogue check (needs cs_pairs semantics: rows 2i gate, 2i+1 up)
        if csp:
            M, MT = 32, 32
            X = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1
            S, BR, KT, PF, MB = next((c[2], c[3], c[4], c[5], c[6]) for c in CONFIGS
                                     if c[0] == MT and c[1] == 1)
            Yg = ext.gemm(blob, X, cs.to(dev), K, N, layout, offs["data"], offs["scale"],
                          offs["zero"], M, MT, 2, csp, 1, S, BR, KT, PF, MB).float()
            R = ref_matmul(pk, cs.to(dev), K, N, layout, X, csp, bf16_W=True)
            gate, up = R[:, 0::2], R[:, 1::2]
            Rg = torch.nn.functional.gelu(gate, approximate="tanh") * up
            eg = float((Yg - Rg).abs().max() / Rg.abs().max())
            print(f"  GEGLU epilogue K={K} N={N}: rel err vs torch gelu_tanh(gate)*up = {eg:.3e}")
            assert eg < 2e-2, eg
    print(f"{'layout':<8}{'K':>7}{'N':>7}{'M':>5}{'MT':>4}{'BR':>5}{'KT':>5}{'cs':>4}"
          f"{'rel vs fp32':>14}{'rel vs bf16-W':>15}")
    for r in rows:
        print(f"{r['layout']:<8}{r['K']:>7}{r['N']:>7}{r['M']:>5}{r['MT']:>4}{r['BR']:>5}"
              f"{r['KT']:>5}{r['cs_pairs']:>4}"
              f"{r['rel_vs_fp32']:>14.3e}{r['rel_vs_bf16W']:>15.3e}")
    print(f"\nALL PASS ({len(rows)} cases)  worst rel vs fp32 = "
          f"{max(r['rel_vs_fp32'] for r in rows):.3e}   "
          f"worst vs bf16-rounded-W = {max(r['rel_vs_bf16W'] for r in rows):.3e}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


def run_bench(ext, a):
    dev = "cuda"
    peak = a.peak
    ceil_bw = a.ceiling
    print(f"# roofline: {peak:.1f} TFLOP/s bf16 mma, {ceil_bw:.1f} GB/s read")
    rows = []
    shapes = SHAPES
    if a.shapes:
        shapes = [(t.split(":")[0], int(t.split(":")[1]), int(t.split(":")[2]))
                  for t in a.shapes.split(",")]
    for (name, K, N) in shapes:
        pk, cs = synth(K, N, DENSE4, seed=hash(name) % 997, device=dev)
        blob, offs, nb = make_blob(pk, dev)
        wbytes = K * (N // 2) + K * (N // GROUP) * 3
        for M in a.M:
            X = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1
            best = None
            for (MT, wm, S, BR, KT, PF, MB) in CONFIGS:
                if MT > max(16, M) or BR > max(64, K) or N % KT:
                    continue
                if a.only and (MT, BR, KT) != tuple(a.only):
                    continue
                try:
                    ms = ext.bench_gemm(blob, X, cs.to(dev), K, N, DENSE4, offs["data"],
                                        offs["scale"], offs["zero"], M, MT, 1, 0, wm, S, BR,
                                        KT, PF, MB, a.warmup, a.iters)
                except Exception:
                    continue
                if a.only:
                    fl_ = 2.0 * M * K * N
                    print(f"  [only] {name:<10}M={M:<5}MT={MT} BR={BR} KT={KT} PF={PF} "
                          f"WM={wm} minb={MB} "
                          f"{ms:8.3f} ms {fl_/(ms*1e-3)/1e12:7.1f} TF/s "
                          f"{wbytes/(ms*1e-3)/1e9:8.1f} GB/s", flush=True)
                if best is None or ms < best[0]:
                    best = (ms, MT, wm, S, BR, KT, PF, MB)
            if best is None:
                continue
            ms, MT, wm, S, BR, KT, PF, MB = best
            fl = 2.0 * M * K * N
            tf = fl / (ms * 1e-3) / 1e12
            gbs = wbytes / (ms * 1e-3) / 1e9
            rows.append(dict(shape=name, K=K, N=N, M=M, MT=MT, WM=wm, S=S, BR=BR, KT=KT,
                             PF=PF, MINB=MB, ms=ms, TFLOPs=tf, GBs=gbs,
                             pct_peak=100 * tf / peak,
                             pct_bw=100 * gbs / ceil_bw))
            print(f"{name:<10}M={M:<5}MT={MT:<4}BR={BR:<4}KT={KT:<4}PF={PF} WM={wm} minb={MB} {ms:8.3f} ms {tf:7.1f} TF/s "
                  f"({100*tf/peak:5.1f}% peak) {gbs:8.1f} GB/s ({100*gbs/ceil_bw:5.1f}% bw)",
                  flush=True)
        del blob
        torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return rows


def run_occ(ext, a):
    print(f"{'MT':>4}{'WM':>4}{'BR':>5}{'KT':>5}{'PF':>4}{'minb':>6}"
          f"{'regs':>7}{'spillB':>8}{'smem':>8}{'blk/SM':>8}")
    for (MT, wm, S, BR, KT, PF, MB) in CONFIGS:
        r = ext.cfg_info(MT, wm, S, BR, KT, PF, MB, 0)
        if not len(r):
            continue
        print(f"{MT:>4}{wm:>4}{BR:>5}{KT:>5}{PF:>4}{MB:>6}"
              f"{r[0]:>7}{r[1]:>8}{r[2]:>8}{r[3]:>8}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofline", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test-blk", action="store_true",
                    help="BLK16_32 (CoBALT-16:32) tile-path correctness vs the torch dequant")
    ap.add_argument("--prune-zero-test", action="store_true",
                    help="pruned columns must contribute EXACTLY 0 through the tile GEMM")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--occ", action="store_true", help="registers/smem/blocks-per-SM report")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--peak", type=float, default=200.0)
    ap.add_argument("--ceiling", type=float, default=770.8)
    ap.add_argument("--wm", type=int, default=1)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--BR", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shapes", default=None,
                    help="override SHAPES: name:K:N,name:K:N (wave-quantisation probes)")
    ap.add_argument("--only", type=int, nargs=3, default=None,
                    metavar=("MT", "BR", "KT"),
                    help="restrict the sweep to this tile (all PF variants of it)")
    ap.add_argument("--define", nargs="*", default=[],
                    help="extra -D flags, e.g. --define -DCBK_GEMM_NO_LDMATRIX")
    a = ap.parse_args()
    if a.test_blk or a.prune_zero_test:
        if a.test_blk:
            run_test_blk(a)
        if a.prune_zero_test:
            run_prune_zero(a)
        return
    ext = build_ext(tuple(a.define))
    print("smem bytes per (MT,BR,KT): " + "  ".join(
        "%d/%d/%d=%d" % (mt, BR, KT, ext.smem_bytes(mt, S, BR, KT, 0))
        for (mt, _, S, BR, KT, _pf, _mb) in sorted(set((c[0], c[1], c[2], c[3], c[4], 1, 1)
                                                       for c in CONFIGS))))
    if a.occ:
        run_occ(ext, a)
    if a.roofline:
        a.peak = run_roofline(ext, a)
    if a.test:
        run_test(ext, a)
    if a.bench:
        run_bench(ext, a)


if __name__ == "__main__":
    main()
