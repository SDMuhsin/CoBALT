"""Correctness + bandwidth harness for the CBK1 device GEMV.

  source scripts/cobaltkernel_env.sh 2g
  python src/cobaltkernel/test_gemv.py --test          # correctness (all layouts x M)
  python src/cobaltkernel/test_gemv.py --bench         # 27B-shape bandwidth table
  python src/cobaltkernel/test_gemv.py --pack-synth DIR # write a synthetic CBK1 artifact
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_cobalt as P
from pack_cobalt import (GROUP, ROW_ALIGN, ARR_ALIGN, BLOB_TAIL, SPARSE4, DENSE4, DENSE8,
                         SPARSE4X, BF16, SPARSE4E, BLK1632_4, BLK1632_6, BLK1632,
                         BLK1632_BITS, BLOCK, BLOCK_KEEP, LAYOUT_NAME, _align)

HERE = os.path.dirname(os.path.abspath(__file__))
CEIL_2G, CEIL_1G = 770.8, 385.5   # measured read ceilings (ENV.md §3)

# MedGemma-27B linear shapes (K = out rows, N = in cols)
SHAPES = [("q_proj", 4096, 5376), ("k_proj", 2048, 5376), ("v_proj", 2048, 5376),
          ("o_proj", 5376, 4096), ("gateup", 43008, 5376), ("down_proj", 5376, 21504),
          ("lm_head", 262144, 5376)]


def build_ext(interleaved=False, bpg=1, zfill=False, minb=1, lut=0, zf=0):
    from torch.utils.cpp_extension import load
    nm = "cbk_test_gemv_x" if interleaved else "cbk_test_gemv"
    if bpg != 1:
        nm += f"_bpg{bpg}"
    if zfill:
        nm += "_zf"
    if minb != 1:
        nm += f"_mb{minb}"
    if lut:
        nm += f"_lut{lut}"
    if zf:
        nm += "_zfm"
    bias = int(os.environ.get("COBALT_BLK1632_BIAS", 0))
    if bias:
        nm += "_bias"
    return load(name=nm,
                sources=[os.path.join(HERE, "csrc", "test_gemv.cu")],
                extra_include_paths=[os.path.join(HERE, "csrc")],
                extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math",
                                   "-gencode", "arch=compute_120a,code=sm_120a",
                                   f"-DCBK_BLK1632_BPG={bpg}",
                                   f"-DCBK_TEST_MINB={minb}",
                                   f"-DCBK_BLK1632_LUT={lut}",
                                   f"-DCBK_BLK1632_ZF={zf}",
                                   f"-DCBK_BLK1632_BIAS={bias}"]
                                  + (["-DCBK_X_INTERLEAVED"] if interleaved else [])
                                  + (["-DCBK_BLK1632_ZFILL"] if zfill else []),
                verbose=False)


# ------------------------------------------------------------------ blob assembly
def make_blob(packed, K, N, layout, device="cuda"):
    """Concatenate {data,row_off,scale,zero} into one 256-B aligned blob (the VRAM image)."""
    parts, offs, pos = [], {}, 0

    def add(name, arr):
        nonlocal pos
        pad = _align(pos, ARR_ALIGN) - pos
        if pad:
            parts.append(np.zeros(pad, np.uint8)); pos += pad
        offs[name] = pos
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        parts.append(b); pos += b.size

    add("data", packed["data"])
    add("row_off", packed["row_off"] if packed.get("row_off") is not None else np.zeros(4, np.uint32))
    add("scale", packed["scale"])
    add("zero", packed["zero"])
    pad = _align(pos, ARR_ALIGN) - pos
    if pad:
        parts.append(np.zeros(pad, np.uint8)); pos += pad
    blob = torch.from_numpy(np.concatenate(parts)).to(device)
    return blob, offs, pos


# ------------------------------------------------------------------ synthetic packed data
def synth_packed(K, N, layout, sparsity=0.5, seed=0, device="cuda"):
    """Directly synthesize a VALID packed matrix (bit-exact structure, random content).
    Used for the big-shape bandwidth benchmark, where running the real packer would be
    dominated by host-side quantization work irrelevant to the measurement."""
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)
    G = N // GROUP
    scale = (torch.rand(K, G, generator=g, device=dev) * 0.01 + 1e-3).half().cpu().numpy()
    zero = torch.randint(0, 16, (K, G), generator=g, device=dev, dtype=torch.int32
                         ).to(torch.uint8).cpu().numpy()
    if layout in BLK1632:
        bts = BLK1632_BITS[layout]
        stride = (N // 2) if bts == 6 else (N * 3 // 8)
        NB = N // BLOCK
        # a valid 16-of-32 mask: random permutation per block, first 16 kept
        r = torch.rand(K, NB, BLOCK, generator=g, device=dev)
        idx = torch.topk(r, BLOCK_KEEP, dim=-1).indices
        kb = torch.zeros_like(r, dtype=torch.bool).scatter_(-1, idx, True)
        keep = kb.view(K, N)
        sh = torch.arange(8, device=dev, dtype=torch.uint8)
        mb = (keep.view(K, N // 8, 8).to(torch.uint8) << sh).sum(-1).to(torch.uint8)
        body = torch.randint(0, 256, (K, stride - N // 8), generator=g, device=dev,
                             dtype=torch.int32).to(torch.uint8)
        data = torch.cat([mb, body], 1).reshape(-1)
        data = torch.cat([data, torch.zeros(BLOB_TAIL, dtype=torch.uint8, device=dev)])
        return dict(data=data.cpu().numpy(), row_off=None, scale=scale, zero=zero,
                    n_survivors=int(keep.sum()), n_zero_fixed=0)
    if layout in (DENSE4, DENSE8, BF16):
        stride = int(P._align(0, 1) + (N // 2 if layout == DENSE4 else (N if layout == DENSE8 else 2 * N)))
        data = torch.randint(0, 256, (K * stride + BLOB_TAIL,), generator=g, device=dev,
                             dtype=torch.int32).to(torch.uint8).cpu().numpy()
        return dict(data=data, row_off=None, scale=scale, zero=zero,
                    n_survivors=K * N, n_zero_fixed=0)
    keep = (torch.rand(K, N, generator=g, device=dev) > sparsity)
    sh = torch.arange(8, device=dev, dtype=torch.uint8)
    mb = (keep.view(K, N // 8, 8).to(torch.uint8) << sh).sum(-1).to(torch.uint8)
    cnt = keep.view(K, G, GROUP).sum(-1).to(torch.int32)
    gb = (cnt + 1) // 2
    gcum = torch.cumsum(gb, 1) - gb
    hdr = (N // 8) + ((_align(G * 2, 16)) if layout in (SPARSE4X, SPARSE4E) else 0)
    hdr_a = _align(hdr, ROW_ALIGN)
    rb = ((hdr_a + gb.sum(1).to(torch.int64) + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN
    row_off = torch.zeros(K + 1, dtype=torch.int64, device=dev)
    row_off[1:] = torch.cumsum(rb, 0)
    total = int(row_off[K])
    data = torch.randint(0, 256, (total + BLOB_TAIL,), generator=g, device=dev,
                         dtype=torch.int32).to(torch.uint8)
    cols = torch.arange(N // 8, device=dev, dtype=torch.int64)
    data[(row_off[:K].view(K, 1) + cols.view(1, -1)).reshape(-1)] = mb.reshape(-1)
    if layout in (SPARSE4X, SPARSE4E):
        g16 = gcum.to(torch.int32).to(torch.int16).view(torch.uint8).view(K, G * 2)
        gc = torch.arange(G * 2, device=dev, dtype=torch.int64) + (N // 8)
        data[(row_off[:K].view(K, 1) + gc.view(1, -1)).reshape(-1)] = g16.reshape(-1)
    return dict(data=data.cpu().numpy(), row_off=row_off.to(torch.uint32).cpu().numpy(),
                scale=scale, zero=zero, n_survivors=int(keep.sum()), n_zero_fixed=0)


# ------------------------------------------------------------------ correctness
def run_test(ext, args):
    dev = "cuda"
    torch.manual_seed(0)
    cases = [(256, 5376), (128, 21504), (192, 4096), (64, 1024)]
    rows = []
    for layout in (SPARSE4, SPARSE4X, SPARSE4E, DENSE4, DENSE8, BLK1632_4, BLK1632_6):
        bits = 8 if layout == DENSE8 else BLK1632_BITS.get(layout, 4)
        for (K, N) in cases:
            raw = P.synth_raw_matrix(K, N, sparsity=0.5, bits=bits, seed=K + N + layout,
                                     device=dev, block=(BLOCK if layout in BLK1632 else 0))
            pk = P.pack_matrix(raw["mask"], raw["q"], raw["scale"], raw["zero"], layout,
                               bits=bits, device=torch.device(dev), return_ref=True)
            blob, offs, nb = make_blob(pk, K, N, layout, dev)
            Wr = pk["W_ref"].to(dev)
            for M in (1, 2, 4, 8):
                x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
                smem = (M * N * 2) <= 98304
                xflat = (x.t().contiguous() if args.interleaved_x else x).reshape(-1)
                out = ext.gemv(blob, xflat, K, N, layout, nb, offs["data"],
                               offs["row_off"], offs["scale"], offs["zero"], M, 1, 188, smem)
                ref = (Wr @ x.float().t())
                err = (out - ref).abs()
                den = ref.abs().max().clamp(min=1e-12)
                rows.append(dict(layout=LAYOUT_NAME[layout], K=K, N=N, M=M,
                                 max_abs=float(err.max()), max_rel=float(err.max() / den)))
                assert float(err.max() / den) < 2e-5, rows[-1]
            # dequant_row check
            dq = ext.dequant_rows(blob, K, N, layout, offs["data"], offs["row_off"],
                                  offs["scale"], offs["zero"], 0, min(K, 32))
            dref = Wr[:min(K, 32)].to(torch.bfloat16).float()
            e2 = (dq.float() - dref).abs().max()
            rows[-1]["dequant_row_max_abs"] = float(e2)
            assert float(e2) < 1e-3 * float(dref.abs().max().clamp(min=1e-9)) + 1e-6, (layout, K, N, float(e2))
    print(f"{'layout':<10}{'K':>7}{'N':>7}{'M':>4}{'max_abs':>13}{'max_rel':>12}")
    for r in rows:
        print(f"{r['layout']:<10}{r['K']:>7}{r['N']:>7}{r['M']:>4}{r['max_abs']:>13.3e}{r['max_rel']:>12.3e}")
    print(f"\nALL PASS  ({len(rows)} cases)  worst rel = {max(r['max_rel'] for r in rows):.3e}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
    return rows


# ------------------------------------------------------------------ benchmark
def run_bench(ext, args):
    dev = "cuda"
    ceil = CEIL_2G if args.ceiling is None else args.ceiling
    props = torch.cuda.get_device_properties(0)
    blocks = min(args.blocks, 2 * props.multi_processor_count)
    print(f"# device SMs={props.multi_processor_count} 256 thr/block, 8 warps each; "
          f"grid swept over {args.block_sweep}; ceiling={ceil} GB/s")
    rows = []
    layouts = [DENSE4, SPARSE4E, BLK1632_4, BLK1632_6] if args.b1632 \
        else [SPARSE4, SPARSE4X, SPARSE4E, DENSE4]
    for (name, K, N) in SHAPES:
        for layout in ([DENSE4] if name == "lm_head" else layouts):
            pk = synth_packed(K, N, layout, 0.5, seed=hash(name) % 1000, device=dev)
            blob1, offs, nb = make_blob(pk, K, N, layout, dev)
            stream = (pk["data"].size - BLOB_TAIL) + pk["scale"].nbytes + pk["zero"].nbytes \
                + (pk["row_off"].nbytes if pk.get("row_off") is not None else 0)
            ncopy = max(1, min(args.max_bytes // nb, args.max_copies))
            blob = blob1.repeat(ncopy) if ncopy > 1 else blob1
            del blob1
            torch.cuda.empty_cache()
            for M in args.m_list:
                x = torch.randn(M * N, device=dev, dtype=torch.bfloat16)
                fits = (M * N * 2) <= 98304
                best = None
                for sm in ([m for m in args.smem_modes if (m == 0 or fits)] if True else []):
                    sm = bool(sm)
                    for bl in args.block_sweep:
                        ms = ext.bench(blob, x, K, N, layout, nb, offs["data"], offs["row_off"],
                                       offs["scale"], offs["zero"], M, ncopy, bl, sm,
                                       args.warmup, args.iters)
                        if best is None or ms < best[0]:
                            best = (ms, bl, sm)
                ms, bl, smem = best
                gbs = ncopy * stream / (ms * 1e-3) / 1e9
                rows.append(dict(shape=name, K=K, N=N, layout=LAYOUT_NAME[layout], M=M,
                                 bpw=8.0 * stream / (K * N), copies=ncopy, blocks=bl,
                                 total_MB=ncopy * nb / 2**20, ms=ms, GBs=gbs,
                                 ns_per_row=ms * 1e6 / (ncopy * K),
                                 pct_ceiling=100.0 * gbs / ceil, smem_x=smem))
                print(f"{name:<10}{LAYOUT_NAME[layout]:<10}M={M:<2} bpw={rows[-1]['bpw']:5.3f} "
                      f"blocks={bl:<5}{ms:8.3f} ms  {gbs:8.1f} GB/s  "
                      f"{rows[-1]['pct_ceiling']:5.1f}%  {rows[-1]['ns_per_row']:6.2f} ns/row "
                      f"smem_x={smem}", flush=True)
            del blob
            torch.cuda.empty_cache()
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
    return rows


# ------------------------------------------------------------------ multi-row bench
def _pick_R(K, GW, rmax=4, rtol=11):
    """Mirror of megakernel.cu::pick_R (COBALT_RTOL=11 default) for the harness."""
    cd = lambda a, b: -(-a // b)
    bc = min(cd(cd(K, r), GW) * r for r in (1, 2, 4) if r <= rmax)
    best = 1
    for r in (1, 2, 4):
        if r <= rmax and cd(cd(K, r), GW) * r * 10 <= bc * rtol:
            best = r
    return best


def run_multi(ext, args, bench=True):
    """Head-to-head of gemv_dense4_multi vs gemv_blk1632_multi -- the routines the DECODE
    MEGAKERNEL actually runs (R consecutive rows per column pass, x in GLOBAL memory in
    the interleaved layout, uint4 activation reads, __ldcg weights, PF granule prefetch)."""
    dev = "cuda"
    ceil = CEIL_2G if args.ceiling is None else args.ceiling
    props = torch.cuda.get_device_properties(0)
    print(f"# MULTI-ROW harness  SMs={props.multi_processor_count}  ceiling={ceil} GB/s  "
          f"R={args.multi_R or 'auto(pick_R)'}  NX={args.multi_NX}")
    rows = []
    shapes = [s for s in SHAPES if args.shapes is None or s[0] in args.shapes]
    for (name, K, N) in shapes:
        for layout in ([DENSE4] if name == "lm_head" else [DENSE4, BLK1632_4, BLK1632_6]):
            pk = synth_packed(K, N, layout, 0.5, seed=hash(name) % 1000, device=dev)
            blob1, offs, nb = make_blob(pk, K, N, layout, dev)
            stream = (pk["data"].size - BLOB_TAIL) + pk["scale"].nbytes + pk["zero"].nbytes
            ncopy = max(1, min(args.max_bytes // nb, args.max_copies))
            blob = blob1.repeat(ncopy) if ncopy > 1 else blob1
            del blob1
            torch.cuda.empty_cache()
            M = 1
            x = torch.randn(M * N, device=dev, dtype=torch.bfloat16)
            xb = torch.randn(M * N, device=dev, dtype=torch.bfloat16)
            best = None
            for bl in args.block_sweep:
                GW = bl * 8
                R = args.multi_R or _pick_R(K, GW)
                for ft in args.flattail:
                    ms = ext.bench_multi(blob, x, xb, K, N, layout, nb, offs["data"],
                                         offs["row_off"], offs["scale"], offs["zero"],
                                         M, R, args.multi_NX, ncopy, bl, bool(ft),
                                         args.warmup, args.iters)
                    if best is None or ms < best[0]:
                        best = (ms, bl, R, ft)
            ms, bl, R, ft = best
            gbs = ncopy * stream / (ms * 1e-3) / 1e9
            rows.append(dict(shape=name, K=K, N=N, layout=LAYOUT_NAME[layout], M=M, R=R,
                             NX=args.multi_NX, flattail=int(ft), blocks=bl,
                             copies=ncopy, ms=ms, GBs=gbs,
                             ns_per_row=ms * 1e6 / (ncopy * K),
                             pct_ceiling=100.0 * gbs / ceil,
                             bpw=8.0 * stream / (K * N)))
            print(f"{name:<10}{LAYOUT_NAME[layout]:<11}R={R} ft={ft} blocks={bl:<5}"
                  f"{ms:8.3f} ms {gbs:8.1f} GB/s {rows[-1]['pct_ceiling']:5.1f}% "
                  f"{rows[-1]['ns_per_row']:6.2f} ns/row", flush=True)
            del blob
            torch.cuda.empty_cache()
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
    return rows


def run_multi_test(ext, args):
    """Correctness of the MULTI routines against the torch reference (and DENSE4-multi as
    the known-good control).  x is interleaved; M=1."""
    dev = "cuda"
    torch.manual_seed(0)
    # 5376 -> NT=168, NTAIL=8 ; 2688 -> NT=84, NTAIL=20 (R*NTAIL spans >1 tail round) ;
    # 21504 / 4096 / 1024 -> NTAIL=0, i.e. FLATTAIL must be a provable no-op there.
    cases = [(256, 5376), (128, 21504), (192, 4096), (64, 1024), (96, 2688)]
    worst = 0.0
    n = 0
    for layout in (DENSE4, BLK1632_4, BLK1632_6):
        bits = BLK1632_BITS.get(layout, 4)
        for (K, N) in cases:
            raw = P.synth_raw_matrix(K, N, sparsity=0.5, bits=bits, seed=K + N + layout,
                                     device=dev, block=(BLOCK if layout in BLK1632 else 0))
            pk = P.pack_matrix(raw["mask"], raw["q"], raw["scale"], raw["zero"], layout,
                               bits=bits, device=torch.device(dev), return_ref=True)
            blob, offs, nb = make_blob(pk, K, N, layout, dev)
            Wr = pk["W_ref"].to(dev)
            x = torch.randn(N, device=dev, dtype=torch.bfloat16)
            xb = torch.randn(N, device=dev, dtype=torch.bfloat16)
            for R in (1, 2, 4):
              for ft in (0, 1):
                for NX in ((1, 2) if R % 2 == 0 else (1,)):
                    out = ext.gemv_multi(blob, x, xb, K, N, layout, nb, offs["data"],
                                         offs["row_off"], offs["scale"], offs["zero"],
                                         1, R, NX, 1, 188, bool(ft))
                    if NX == 1:
                        ref = (Wr @ x.float()).view(-1, 1)
                    else:
                        ra = (Wr @ x.float()); rb2 = (Wr @ xb.float())
                        sel = (torch.arange(K, device=dev) % 2 == 1)
                        ref = torch.where(sel, rb2, ra).view(-1, 1)
                    den = ref.abs().max().clamp(min=1e-12)
                    rel = float((out - ref).abs().max() / den)
                    worst = max(worst, rel)
                    n += 1
                    assert rel < 2e-5, (LAYOUT_NAME[layout], K, N, R, NX, ft, rel)
            print(f"{LAYOUT_NAME[layout]:<11}{K:>7}x{N:<7} R=1,2,4 NX=1,2 flattail=0,1  OK",
                  flush=True)
    print(f"\nMULTI ALL PASS ({n} cases)  worst rel = {worst:.3e}")


# ------------------------------------------------------------------ synthetic artifact
def pack_synth_artifact(out_dir, layers=2, layout=SPARSE4, embed_layout=DENSE4,
                        hidden=5376, inter=21504, nq=4096, nkv=2048, vocab=262144,
                        small=False, device="cuda"):
    """Write a complete synthetic CBK1 artifact (RAW schema -> packed), for the kernel-B
    integration tests. `small` shrinks the shapes so it fits in a few hundred MB."""
    if small:
        hidden, inter, nq, nkv, vocab = 512, 1024, 512, 256, 4096
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    mats = [("q_proj", nq, hidden), ("k_proj", nkv, hidden), ("v_proj", nkv, hidden),
            ("o_proj", hidden, nq), ("gateup", 2 * inter, hidden), ("down_proj", hidden, inter)]
    man = dict(format="CBK1", version=1, group_size=GROUP, bits=4, layout=LAYOUT_NAME[layout],
               embed_layout=LAYOUT_NAME[embed_layout], fuse_gateup=True, synthetic=True,
               shapes=dict(hidden=hidden, intermediate=inter, vocab=vocab,
                           n_layers=layers), layers=[])
    tot_p = tot_stream = tot_dense = 0
    for li in range(layers):
        bw = P.BlobWriter(os.path.join(out_dir, f"layer_{li:02d}.bin"))
        ent = dict(file=f"layer_{li:02d}.bin", matrices={}, norms={})
        for (nm, K, N) in mats:
            raw = P.synth_raw_matrix(K, N, 0.5, 4, seed=li * 17 + hash(nm) % 97, device=dev)
            pk = P.pack_matrix(raw["mask"], raw["q"], raw["scale"], raw["zero"], layout,
                               device=dev)
            cs = raw["col_scale"].half().cpu().numpy()
            if nm == "gateup":
                cs = np.stack([cs, cs], 0)
            ent["matrices"][nm] = P.write_matrix(bw, nm, K, N, layout, pk, cs)
            tot_p += K * N
            tot_stream += ent["matrices"][nm]["stream_bytes"]
            tot_dense += K * N // 2 + K * (N // GROUP) * 3
            del pk, raw
            torch.cuda.empty_cache()
        for nn in P.NORMS:
            n = 128 if nn.endswith("_norm") else hidden
            ent["norms"][nn] = bw.put(nn, torch.randn(n).float().numpy())
        ent["bytes"] = bw.close()
        man["layers"].append(ent)
        print(f"[synth] layer {li} -> {ent['bytes']/2**20:.1f} MiB", flush=True)
    # embedding / lm_head (tied)
    bw = P.BlobWriter(os.path.join(out_dir, "embed.bin"))
    W = torch.randn(vocab, hidden, device=dev, dtype=torch.bfloat16)
    if embed_layout == BF16:
        e = dict(K=vocab, N=hidden, G=hidden // GROUP, layout=BF16, layout_name="BF16",
                 arrays=dict(data=bw.put("embed.data", W.cpu().view(torch.uint8).numpy())))
    else:
        bits = 8 if embed_layout == DENSE8 else 4
        q, sc, zr = P.rtn_dense(W.float(), bits)
        pk = P.pack_matrix(None, q, sc, zr, embed_layout, bits=bits, device=dev)
        e = P.write_matrix(bw, "embed", vocab, hidden, embed_layout, pk, None)
    man["embed"] = e
    man["embed"]["file"] = "embed.bin"
    bw.put("model_norm", torch.randn(hidden).float().numpy())
    man["embed"]["bytes"] = bw.close()
    man["bpw"] = dict(params=tot_p, packed_bytes=tot_stream,
                      bpw_packed=8.0 * tot_stream / tot_p, bpw_dense4=8.0 * tot_dense / tot_p)
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(man["bpw"], indent=1))
    return man


def run_prune_zero_test(ext, args):
    """BLK16_32: pruned columns must contribute EXACTLY 0.0 -- two independent probes.

    (1) COLUMN-UNIFORM mask (the same 16 of every 32 columns kept in every row) so that
        "pruned" is a property of the column: feed x with HUGE values at the pruned
        columns and require the output to be BIT-IDENTICAL to x=0 there.
    (2) OFF-GRID zero point (`fix_zero=False`, zero > nlevels): the DENSE4 layout must
        break (it encodes pruned positions as `code = zero`), BLK16_32 must not (it
        encodes nothing and applies `zero` only over kept slots).
    """
    dev = "cuda"
    K, N = 256, 5376
    g = torch.Generator(device="cpu").manual_seed(7)
    pat = torch.zeros(N // BLOCK, BLOCK, dtype=torch.bool)
    pat.scatter_(-1, torch.topk(torch.rand(N // BLOCK, BLOCK, generator=g),
                                BLOCK_KEEP, dim=-1).indices, True)
    keep = pat.view(1, N).expand(K, N).contiguous()
    sh = torch.arange(8, dtype=torch.uint8)
    mb = (keep.view(K, N // 8, 8).to(torch.uint8) << sh).sum(-1).to(torch.uint8).to(dev)
    pruned = ~pat.view(N).to(dev)
    print(f"# prune-zero test  K={K} N={N}  pruned cols={int(pruned.sum())}/{N}")
    for layout in (BLK1632_4, BLK1632_6, DENSE4):
        bits = BLK1632_BITS.get(layout, 4)
        nl = (1 << bits) - 1
        q = torch.randint(0, nl + 1, (K, N), generator=g, dtype=torch.uint8).to(dev)
        scale = (torch.rand(K, N // GROUP, generator=g) * 0.01 + 1e-3).half().to(dev)
        for zmode in ("ongrid", "offgrid"):
            if zmode == "ongrid":
                zr = torch.randint(0, nl + 1, (K, N // GROUP), generator=g).half().to(dev)
                fz = True
            else:
                zr = torch.full((K, N // GROUP), float(nl + 37)).half().to(dev)   # > nlevels
                fz = False
            pk = P.pack_matrix(mb, q, scale, zr, layout, bits=bits,
                               device=torch.device(dev), return_ref=True, fix_zero=fz)
            blob, offs, nb = make_blob(pk, K, N, layout, dev)
            x0 = torch.randn(N, device=dev, dtype=torch.bfloat16)
            xh = x0.clone(); xh[pruned] = torch.tensor(1.0e4, dtype=torch.bfloat16)
            o0 = ext.gemv(blob, x0, K, N, layout, nb, offs["data"], offs["row_off"],
                          offs["scale"], offs["zero"], 1, 1, 188, True)
            oh = ext.gemv(blob, xh, K, N, layout, nb, offs["data"], offs["row_off"],
                          offs["scale"], offs["zero"], 1, 1, 188, True)
            dq = ext.dequant_rows(blob, K, N, layout, offs["data"], offs["row_off"],
                                  offs["scale"], offs["zero"], 0, 32)
            wpr = dq.float()[:, pruned].abs().max().item()
            same = torch.equal(o0, oh)
            dmax = (o0 - oh).abs().max().item()
            print(f"{LAYOUT_NAME[layout]:<11} zero={zmode:<8} n_zero_fixed={pk['n_zero_fixed']:<5} "
                  f"bit-identical={str(same):<6} max|d|={dmax:.3e}  "
                  f"max|W| at pruned (dequant_row)={wpr:.3e}")
            if layout in BLK1632:
                assert same and wpr == 0.0, "BLK16_32 pruned positions are NOT exactly zero"
    print("BLK16_32: pruned positions contribute EXACTLY 0 under BOTH zero-point modes.")


def run_sparsity_sweep(ext, args):
    """Where does the sparse path break even against DENSE4 in WALL CLOCK?"""
    dev = "cuda"
    K, N = 4096, 5376
    print(f"# sparsity break-even, shape {K}x{N}, M=1")
    for sp in (0.0, 0.25, 0.5, 0.625, 0.75, 0.875):
        for layout in (SPARSE4, SPARSE4X, SPARSE4E, DENSE4):
            if layout == DENSE4 and sp not in (0.0,):
                continue
            pk = synth_packed(K, N, layout, sp, seed=3, device=dev)
            blob1, offs, nb = make_blob(pk, K, N, layout, dev)
            stream = (pk["data"].size - BLOB_TAIL) + pk["scale"].nbytes + pk["zero"].nbytes \
                + (pk["row_off"].nbytes if pk.get("row_off") is not None else 0)
            nc = max(1, min(args.max_bytes // nb, args.max_copies))
            blob = blob1.repeat(nc); del blob1; torch.cuda.empty_cache()
            x = torch.randn(N, device=dev, dtype=torch.bfloat16)
            best = min(ext.bench(blob, x, K, N, layout, nb, offs["data"], offs["row_off"],
                                 offs["scale"], offs["zero"], 1, nc, bl, True,
                                 args.warmup, args.iters) for bl in args.block_sweep)
            print(f"sp={sp:<6} {LAYOUT_NAME[layout]:<9} bpw={8.0*stream/(K*N):5.3f} "
                  f"{best*1e6/(nc*K):6.2f} ns/row  {nc*stream/(best*1e-3)/1e9:7.1f} GB/s", flush=True)
            del blob; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-multi", action="store_true",
                    help="head-to-head of the MULTI-ROW routines the megakernel runs")
    ap.add_argument("--test-multi", action="store_true")
    ap.add_argument("--multi-R", type=int, default=0, help="0 = pick_R like the megakernel")
    ap.add_argument("--multi-NX", type=int, default=1)
    ap.add_argument("--flattail", type=int, nargs="+", default=[0, 1],
                    help="BLK16_32 warp-tail fix: flatten the leftover (row,granule) items")
    ap.add_argument("--shapes", nargs="+", default=None)
    ap.add_argument("--pack-synth", default=None)
    ap.add_argument("--synth-small", action="store_true")
    ap.add_argument("--synth-layers", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=188)
    ap.add_argument("--block-sweep", type=int, nargs="+", default=[188, 376, 564, 1128])
    ap.add_argument("--sparsity-sweep", action="store_true")
    ap.add_argument("--b1632", action="store_true",
                    help="bench/test only the head-to-head set DENSE4 / SPARSE4E / BLK16_32")
    ap.add_argument("--prune-zero-test", action="store_true",
                    help="BLK16_32: assert pruned columns contribute EXACTLY 0")
    ap.add_argument("--interleaved-x", action="store_true")
    ap.add_argument("--m-list", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--smem-modes", type=int, nargs="+", default=[1, 0],
                    help="stage x in shared memory: 1=yes 0=no (both by default)")
    ap.add_argument("--zfill", action="store_true",
                    help="DIAGNOSTIC: BLK16_32 fills pruned slots with `zero` "
                         "(DENSE4 inner loop) instead of the exact masked-xs form")
    ap.add_argument("--minb", type=int, default=1,
                    help="min blocks/SM in __launch_bounds__ (2 = the megakernel's register cap)")
    ap.add_argument("--bpg", type=int, default=1,
                    help="BLK16_32 blocks decoded per lane per granule (1 or 2)")
    ap.add_argument("--lut", type=int, default=0,
                    help="BLK16_32 selector source: 0 global table (shipped), 1 __constant__, "
                         "2 __shared__ staged per block, 3 arithmetic (no table)")
    ap.add_argument("--zf", type=int, default=0,
                    help="BLK16_32 multi-row: DENSE4-style zero-fill of pruned slots "
                         "(drops the per-row masked x-sum; changes rounding)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-bytes", type=int, default=2_000_000_000)
    ap.add_argument("--max-copies", type=int, default=64)
    ap.add_argument("--ceiling", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ext = build_ext(a.interleaved_x, a.bpg, a.zfill, a.minb, a.lut, a.zf)
    if a.test:
        run_test(ext, a)
    if a.prune_zero_test:
        run_prune_zero_test(ext, a)
    if a.test_multi:
        run_multi_test(ext, a)
    if a.bench:
        run_bench(ext, a)
    if a.bench_multi:
        run_multi(ext, a)
    if a.sparsity_sweep:
        run_sparsity_sweep(ext, a)
    if a.pack_synth:
        pack_synth_artifact(a.pack_synth, a.synth_layers, small=a.synth_small)


if __name__ == "__main__":
    main()
