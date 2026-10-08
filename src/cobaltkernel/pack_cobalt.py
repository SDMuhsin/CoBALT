"""CBK1 packer: RAW CoBALT artifact -> packed VRAM-image artifact.

RAW input (produced by src/cobaltkernel/quantize_cobalt.py), per layer file
`layer_XX.safetensors`, per linear name in {q_proj,k_proj,v_proj,o_proj,
gate_proj,up_proj,down_proj}:
    <name>.mask       uint8  [K, N/8]   packed bits, LSB-first (1 = survivor)
    <name>.q          uint8  [K, N]     group-RTN codes, pruned positions = 0
    <name>.scale      fp16   [K, N/128]
    <name>.zero       fp16 or uint8 [K, N/128]
    <name>.col_scale  fp16   [N]
    dequant: W_hat[k,j] = (q - zero) * scale * col_scale[j] * keep(k,j)

OUTPUT: a directory with manifest.json + per-layer .bin blobs whose byte layout is
EXACTLY the VRAM layout (load = read + cudaMemcpy).  See docs/FORMAT.md.

Also usable as a library:
    from pack_cobalt import pack_matrix, synth_raw_matrix, CBK
"""
import argparse, glob, json, math, os, struct, sys, time
import numpy as np
import torch

GROUP = 128
ROW_ALIGN = 16
ARR_ALIGN = 256
BLOB_TAIL = 64

SPARSE4, DENSE4, DENSE8, SPARSE4X, BF16, SPARSE4E = 0, 1, 2, 3, 4, 5
BLK1632_4, BLK1632_6 = 6, 7
BLK1632 = (BLK1632_4, BLK1632_6)
BLK1632_BITS = {BLK1632_4: 4, BLK1632_6: 6}
BLOCK = 32                       # CoBALT-16:32 mask block (input columns)
BLOCK_KEEP = 16                  # survivors per block -- FIXED
LAYOUT_NAME = {0: "SPARSE4", 1: "DENSE4", 2: "DENSE8", 3: "SPARSE4X", 4: "BF16",
               5: "SPARSE4E", 6: "BLK1632_4", 7: "BLK1632_6"}
LAYOUT_ID = {v: k for k, v in LAYOUT_NAME.items()}

LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NORMS = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
         "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"]


def _align(x, a):
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------- helpers
def _to_int_zero(zero, nlevels):
    """RAW `zero` may be fp16 (a rounded integer) or uint8. Return int32 tensor."""
    if zero.dtype == torch.uint8:
        return zero.to(torch.int32)
    return torch.round(zero.float()).to(torch.int32)


def _fix_zero_range(qf, scale, zero, keep, nlevels):
    """Guarantee zero in [0, nlevels].

    CoBALT's group RTN uses zero = -round(w_min/scale), which leaves [0,nlevels]
    only for a group whose hull does not straddle 0 (all-positive or all-negative
    survivors) -- rare.  For such groups we re-derive (scale, zero, q) from the
    group's dequantized values with a 0-inclusive hull, which makes zero
    representable by construction.  Returns (qf, scale, zero_int, n_fixed).
    """
    K, N = qf.shape
    G = N // GROUP
    z = _to_int_zero(zero, nlevels)
    bad = (z < 0) | (z > nlevels)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return qf, scale.float(), z, 0
    sc = scale.float()
    v = (qf.view(K, G, GROUP) - z.view(K, G, 1).float()) * sc.view(K, G, 1)   # pre-col-scale W
    kv = keep.view(K, G, GROUP)
    big = torch.finfo(torch.float32).max
    vmin = torch.where(kv, v, torch.full_like(v, big)).amin(-1)
    vmax = torch.where(kv, v, torch.full_like(v, -big)).amax(-1)
    empty = ~kv.any(-1)
    vmin = torch.where(empty, torch.zeros_like(vmin), vmin).clamp(max=0.0)   # 0-inclusive hull
    vmax = torch.where(empty, torch.zeros_like(vmax), vmax).clamp(min=0.0)
    s2 = ((vmax - vmin) / nlevels).clamp(min=1e-12)
    z2 = torch.round(-vmin / s2).clamp(0, nlevels)
    q2 = torch.clamp(torch.round(v / s2.unsqueeze(-1) + z2.unsqueeze(-1)), 0, nlevels)
    b3 = bad.unsqueeze(-1).expand_as(qf.view(K, G, GROUP))
    qf = torch.where(b3, q2, qf.view(K, G, GROUP)).view(K, N)
    sc = torch.where(bad, s2, sc)
    z = torch.where(bad, z2.to(torch.int32), z)
    return qf, sc, z, n_bad


def _unpack_bits(mask_bits, N):
    """[K, N/8] uint8 LSB-first -> bool [K, N]."""
    K = mask_bits.shape[0]
    sh = torch.arange(8, device=mask_bits.device, dtype=torch.uint8)
    return ((mask_bits.unsqueeze(-1) >> sh) & 1).to(torch.bool).view(K, N)


# --------------------------------------------------------------------------- pack one matrix
def pack_matrix(mask_bits, qf, scale, zero, layout, bits=4, chunk_rows=4096, device=None,
                return_ref=False, fix_zero=True):
    """Pack one matrix.

    mask_bits : uint8 [K, N/8] (or None for a dense/unmasked matrix)
    qf        : float or uint8 [K, N] codes
    scale     : [K, G] fp16/fp32       zero : [K, G] fp16/uint8
    layout    : SPARSE4 | SPARSE4X | DENSE4 | DENSE8

    Returns dict(data=uint8 numpy, row_off=uint32 numpy or None,
                 scale=fp16 numpy, zero=uint8 numpy, n_survivors=int, n_zero_fixed=int)
    """
    dev = device or (qf.device if qf.is_cuda else torch.device("cpu"))
    K, N = qf.shape
    G = N // GROUP
    assert N % GROUP == 0
    nlevels = (1 << bits) - 1
    qf = qf.to(dev).float()
    scale = scale.to(dev)
    zero = zero.to(dev)
    if mask_bits is None:
        keep = torch.ones(K, N, dtype=torch.bool, device=dev)
        mask_bits = torch.full((K, N // 8), 255, dtype=torch.uint8, device=dev)
    else:
        mask_bits = mask_bits.to(dev)
        keep = _unpack_bits(mask_bits, N)

    if fix_zero:
        qf, sc, zi, n_fixed = _fix_zero_range(qf, scale, zero, keep, nlevels)
    else:
        # DIAGNOSTIC ONLY (see FORMAT.md sec.13 / sec.11): keep the raw zero point even
        # when it is off the quantization grid.  Used to MEASURE whether a layout needs
        # the 0-inclusive-hull re-derivation at all.
        sc, zi, n_fixed = scale.float(), _to_int_zero(zero, nlevels), 0
    qc = torch.clamp(qf, 0, nlevels).to(torch.int32)
    # pruned positions must dequantize to 0 in the DENSE layouts
    if layout in (DENSE4, DENSE8):
        qc = torch.where(keep, qc, zi.repeat_interleave(GROUP, dim=1))

    out_scale = sc.half().cpu().numpy()
    out_zero = zi.to(torch.uint8).cpu().numpy()
    n_surv = int(keep.sum())
    ref = None
    if return_ref:
        # exact fp32 weight the packed matrix represents (BEFORE the column scale)
        sc16 = torch.from_numpy(out_scale).to(dev).float()
        ref = (qc.float() - zi.repeat_interleave(GROUP, dim=1).float()) * \
            sc16.repeat_interleave(GROUP, dim=1)
        if layout in (SPARSE4, SPARSE4X, SPARSE4E) + BLK1632:
            ref = ref * keep.float()
        ref = ref.cpu()

    if layout == DENSE8:
        data = qc.to(torch.uint8).cpu().numpy().reshape(-1)
        data = np.concatenate([data, np.zeros(BLOB_TAIL, np.uint8)])
        return dict(data=data, row_off=None, scale=out_scale, zero=out_zero,
                    n_survivors=n_surv, n_zero_fixed=n_fixed, W_ref=ref)
    if layout == DENSE4:
        lo = qc[:, 0::2].to(torch.uint8)
        hi = qc[:, 1::2].to(torch.uint8)
        data = (lo | (hi << 4)).cpu().numpy().reshape(-1)
        data = np.concatenate([data, np.zeros(BLOB_TAIL, np.uint8)])
        return dict(data=data, row_off=None, scale=out_scale, zero=out_zero,
                    n_survivors=n_surv, n_zero_fixed=n_fixed, W_ref=ref)

    if layout in BLK1632:
        want = BLK1632_BITS[layout]
        if bits != want:
            raise ValueError(f"{LAYOUT_NAME[layout]} requires bits={want}, got bits={bits}")
        if N % BLOCK:
            raise ValueError(f"BLK16_32 requires N % {BLOCK} == 0, got N={N}")
        NBk = N // BLOCK
        kb = keep.view(K, NBk, BLOCK)
        cnt = kb.sum(-1)
        if not bool((cnt == BLOCK_KEEP).all()):
            bad = (cnt != BLOCK_KEEP)
            nb = int(bad.sum())
            r, c = [int(v) for v in torch.nonzero(bad)[0]]
            raise ValueError(
                f"BLK16_32 needs EXACTLY {BLOCK_KEEP} survivors in every aligned block of "
                f"{BLOCK} columns: {nb}/{K*NBk} blocks violate it "
                f"(first at row {r}, block {c}: {int(cnt[r, c])} survivors). "
                f"Re-quantize with quantize_cobalt.py --mask-block 32.")
        # survivor codes in COLUMN order, 16 per block  (boolean indexing is row-major)
        cod = qc[keep].view(K, NBk, BLOCK_KEEP).to(torch.int32)
        # nibble plane: 8 B/block, code 2j -> low nibble of byte j
        nib = ((cod[:, :, 0::2] & 0xF) | ((cod[:, :, 1::2] & 0xF) << 4)).to(torch.uint8)
        planes = [mask_bits.view(K, N // 8), nib.reshape(K, NBk * 8)]
        if want == 6:
            # hi2 plane: 4 B/block; bits [0,16) = high-2-bits of EVEN within-block survivor
            # indices (0,2,..,14) in order, bits [16,32) = the ODD ones.
            sh = (torch.arange(BLOCK_KEEP // 2, device=dev, dtype=torch.int64) * 2)
            he = ((((cod[:, :, 0::2] >> 4) & 3).to(torch.int64)) << sh).sum(-1)
            ho = ((((cod[:, :, 1::2] >> 4) & 3).to(torch.int64)) << (sh + 16)).sum(-1)
            hw = (he | ho).to(torch.int64)
            hb = torch.stack([(hw >> 0) & 0xFF, (hw >> 8) & 0xFF,
                              (hw >> 16) & 0xFF, (hw >> 24) & 0xFF], -1).to(torch.uint8)
            planes.append(hb.reshape(K, NBk * 4))
        data = torch.cat(planes, dim=1).reshape(-1).cpu().numpy()
        data = np.concatenate([data, np.zeros(BLOB_TAIL, np.uint8)])
        return dict(data=data, row_off=None, scale=out_scale, zero=out_zero,
                    n_survivors=n_surv, n_zero_fixed=n_fixed, W_ref=ref)

    # ---------------- SPARSE4 / SPARSE4X ----------------
    has_goff = layout in (SPARSE4X, SPARSE4E)
    hdr = (N // 8) + ((_align(G * 2, 16)) if has_goff else 0)
    cnt = keep.view(K, G, GROUP).sum(-1).to(torch.int32)          # [K,G]
    gbytes = (cnt + 1) // 2                                        # padded to even nibbles
    gcum = torch.cumsum(gbytes, dim=1) - gbytes                    # exclusive, per row
    row_code_bytes = gbytes.sum(1)                                 # [K]
    row_bytes = torch.tensor([_align(hdr, ROW_ALIGN)], device=dev, dtype=torch.int64) + \
        row_code_bytes.to(torch.int64)
    row_bytes = ((row_bytes + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN
    row_off = torch.zeros(K + 1, dtype=torch.int64, device=dev)
    row_off[1:] = torch.cumsum(row_bytes, 0)
    total = int(row_off[K])
    assert total < 2**32, "matrix blob exceeds 4 GiB; uint32 row_off insufficient"
    if has_goff:
        assert int(row_code_bytes.max()) < 2**16, "per-row codes exceed 64 KiB; goff must be uint32"

    data = torch.zeros(total + BLOB_TAIL, dtype=torch.uint8, device=dev)

    # bitmap region (identical to the RAW mask row) + optional goff table
    bm = mask_bits.view(K, N // 8)
    ridx = row_off[:K]
    bcols = torch.arange(N // 8, device=dev, dtype=torch.int64)
    data[(ridx.view(K, 1) + bcols.view(1, -1)).reshape(-1)] = bm.reshape(-1)
    if has_goff:
        g16 = gcum.to(torch.int32).to(torch.int16).view(torch.uint8).view(K, G * 2)
        gcols = torch.arange(G * 2, device=dev, dtype=torch.int64) + (N // 8)
        data[(ridx.view(K, 1) + gcols.view(1, -1)).reshape(-1)] = g16.reshape(-1)

    # codes: nibble index of survivor (k,j) = 2*gcum[k,g] + rank-within-group
    hdr_a = _align(hdr, ROW_ALIGN)
    k0 = 0
    while k0 < K:
        k1 = min(K, k0 + chunk_rows)
        kp = keep[k0:k1]
        excl = torch.cumsum(kp.to(torch.int32), dim=1) - kp.to(torch.int32)      # within row
        gstart = excl.view(-1, G, GROUP)[:, :, 0]                                # [c,G]
        rank = excl - gstart.repeat_interleave(GROUP, dim=1)
        nib = 2 * gcum[k0:k1].repeat_interleave(GROUP, dim=1) + rank             # [c,N]
        base = (row_off[k0:k1] + hdr_a).view(-1, 1)
        byte_idx = (base + (nib.to(torch.int64) >> 1))[kp]
        shift = ((nib & 1) << 2).to(torch.uint8)[kp]
        vals = (qc[k0:k1].to(torch.uint8)[kp] << shift)
        data.scatter_add_(0, byte_idx, vals)   # nibble slots are disjoint -> add == or
        del excl, gstart, rank, nib, byte_idx, shift, vals, kp
        k0 = k1

    return dict(data=data.cpu().numpy(), row_off=row_off.to(torch.uint32).cpu().numpy(),
                scale=out_scale, zero=out_zero, n_survivors=n_surv, n_zero_fixed=n_fixed,
                W_ref=ref)


def pack_matrix_auto(mask_bits, qf, scale, zero, layout, bits=4, device=None,
                     max_elems=1 << 27, return_ref=False):
    """pack_matrix, but row-chunked for the fixed-stride layouts (rows are independent).
    Keeps the 262144x5376 embedding off the 24 GiB smoke slice's memory cliff."""
    K, N = qf.shape
    if layout not in (DENSE4, DENSE8) + BLK1632 or (K * N) <= max_elems or return_ref:
        return pack_matrix(mask_bits, qf, scale, zero, layout, bits=bits, device=device,
                           return_ref=return_ref)
    step = max(1, max_elems // N)
    dat, scs, zrs, ns, nf = [], [], [], 0, 0
    for k0 in range(0, K, step):
        sl = slice(k0, min(K, k0 + step))
        pk = pack_matrix(None if mask_bits is None else mask_bits[sl], qf[sl], scale[sl],
                         zero[sl], layout, bits=bits, device=device)
        dat.append(pk["data"][:-BLOB_TAIL])
        scs.append(pk["scale"]); zrs.append(pk["zero"])
        ns += pk["n_survivors"]; nf += pk["n_zero_fixed"]
        del pk
    return dict(data=np.concatenate(dat + [np.zeros(BLOB_TAIL, np.uint8)]), row_off=None,
                scale=np.concatenate(scs), zero=np.concatenate(zrs),
                n_survivors=ns, n_zero_fixed=nf, W_ref=None)


# --------------------------------------------------------------------------- reference dequant
def dequant_reference(packed, K, N, layout, col_scale=None, bits=4):
    """Torch fp32 reference dequant of a packed matrix (host side, for tests)."""
    G = N // GROUP
    sc = torch.from_numpy(packed["scale"].astype(np.float32)).view(K, G)
    zi = torch.from_numpy(packed["zero"].astype(np.float32)).view(K, G)
    data = torch.from_numpy(packed["data"])
    if layout == DENSE8:
        q = data[:K * N].view(K, N).float()
        W = (q - zi.repeat_interleave(GROUP, 1)) * sc.repeat_interleave(GROUP, 1)
    elif layout == DENSE4:
        b = data[:K * N // 2].view(K, N // 2)
        q = torch.stack([(b & 0xF), (b >> 4)], -1).view(K, N).float()
        W = (q - zi.repeat_interleave(GROUP, 1)) * sc.repeat_interleave(GROUP, 1)
    elif layout in BLK1632:
        bts = BLK1632_BITS[layout]
        stride = (N * 3 // 8) if bts == 4 else (N // 2)
        NBk = N // BLOCK
        rowb = data[:K * stride].view(K, stride)
        mb = rowb[:, : N // 8]
        keep = ((mb.view(K, -1, 1) >> torch.arange(8, dtype=torch.uint8)) & 1).view(K, N).bool()
        nib = rowb[:, N // 8: N // 8 + NBk * 8].reshape(K, NBk, 8).int()
        q16 = torch.stack([nib & 0xF, (nib >> 4) & 0xF], -1).view(K, NBk, 16)
        if bts == 6:
            hb = rowb[:, N * 3 // 8:].reshape(K, NBk, 4).long()
            hw = hb[..., 0] | (hb[..., 1] << 8) | (hb[..., 2] << 16) | (hb[..., 3] << 24)
            sh = torch.arange(8, dtype=torch.int64) * 2
            he = (hw.unsqueeze(-1) >> sh) & 3
            ho = (hw.unsqueeze(-1) >> (sh + 16)) & 3
            hi2 = torch.stack([he, ho], -1).view(K, NBk, 16)
            q16 = q16 | (hi2.int() << 4)
        # exactly 16 survivors per block, in column order -> row-major boolean scatter
        W = torch.zeros(K, N, dtype=torch.float32)
        W[keep] = q16.reshape(-1).float()
        W = (W - zi.repeat_interleave(GROUP, 1)) * sc.repeat_interleave(GROUP, 1)
        W = W * keep.float()          # pruned positions: EXACTLY 0.0
    else:
        ro = torch.from_numpy(packed["row_off"].astype(np.int64))
        has_goff = layout in (SPARSE4X, SPARSE4E)
        hdr = _align((N // 8) + (_align(G * 2, 16) if has_goff else 0), ROW_ALIGN)
        W = torch.zeros(K, N, dtype=torch.float32)
        for k in range(K):
            rb = int(ro[k])
            mb = data[rb:rb + N // 8]
            keep = ((mb.view(-1, 1) >> torch.arange(8, dtype=torch.uint8)) & 1).view(-1).bool()
            cols = torch.nonzero(keep).view(-1)
            cnt = keep.view(G, GROUP).sum(1)
            gb = torch.cumsum((cnt + 1) // 2, 0) - (cnt + 1) // 2
            cb = data[rb + hdr:]
            nib = (2 * gb.repeat_interleave(GROUP)[cols] +
                   (torch.cumsum(keep.int(), 0) - keep.int())[cols] -
                   (torch.cumsum(keep.int(), 0) - keep.int()).view(G, GROUP)[:, 0].repeat_interleave(GROUP)[cols])
            byt = cb[nib // 2]
            q = torch.where((nib % 2) == 0, byt & 0xF, byt >> 4).float()
            g = cols // GROUP
            W[k, cols] = (q - zi[k, g]) * sc[k, g]
    if col_scale is not None:
        W = W * torch.as_tensor(col_scale, dtype=torch.float32).view(1, N)
    return W


# --------------------------------------------------------------------------- synthetic RAW
def synth_raw_matrix(K, N, sparsity=0.5, bits=4, seed=0, device="cpu", balanced=True,
                     block=0):
    """Random RAW-schema matrix with a row/column-balanced ~`sparsity` mask.

    `block > 0` switches the final selection from a global top-k to a per-aligned-block
    top-`round(block*(1-sparsity))` (the CoBALT-16:32 mask, cf. cobalt_math.blocked_keepmask).
    `block = 0` (default) is the canonical path and is unchanged."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    imp = torch.rand(K, N, generator=g)
    if balanced:
        # row-quantile then column-quantile self-normalisation, then global top-k (CoBALT-shaped)
        kr = max(1, int(N * sparsity)); kc = max(1, int(K * sparsity))
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(0.5)
    if block:
        assert N % block == 0, f"N={N} not a multiple of block={block}"
        nk = block - int(block * sparsity)
        ib = imp.view(K, N // block, block)
        idx = torch.topk(ib, nk, dim=-1).indices
        keep = torch.zeros_like(ib, dtype=torch.bool).scatter_(-1, idx, True).view(K, N)
    else:
        thr = torch.kthvalue(imp.reshape(-1), max(1, int(K * N * sparsity))).values
        keep = (imp > thr)
    G = N // GROUP
    nl = (1 << bits) - 1
    q = torch.randint(0, nl + 1, (K, N), generator=g, dtype=torch.uint8)
    q = q * keep.to(torch.uint8)
    scale = (torch.rand(K, G, generator=g) * 0.01 + 0.001).half()
    zero = torch.randint(0, nl + 1, (K, G), generator=g).half()
    col_scale = (torch.rand(N, generator=g) * 1.5 + 0.5).half()
    sh = torch.arange(8, dtype=torch.uint8)
    mb = (keep.view(K, N // 8, 8).to(torch.uint8) << sh).sum(-1).to(torch.uint8)
    d = torch.device(device)
    return dict(mask=mb.to(d), q=q.to(d), scale=scale.to(d), zero=zero.to(d), col_scale=col_scale.to(d))


# --------------------------------------------------------------------------- embedding RTN
def rtn_dense(W, bits, group=GROUP):
    """Plain asymmetric group min-max RTN with a 0-inclusive hull (no mask, no col scale)."""
    K, N = W.shape
    G = N // group
    nl = (1 << bits) - 1
    Wg = W.float().view(K, G, group)
    wmin = Wg.amin(-1).clamp(max=0.0)
    wmax = Wg.amax(-1).clamp(min=0.0)
    scale = ((wmax - wmin) / nl).clamp(min=1e-12)
    zero = torch.round(-wmin / scale).clamp(0, nl)
    q = torch.clamp(torch.round(Wg / scale.unsqueeze(-1) + zero.unsqueeze(-1)), 0, nl).view(K, N)
    return q, scale.half(), zero.to(torch.uint8)


# --------------------------------------------------------------------------- container writer
class BlobWriter:
    """Accumulates 256-B aligned arrays into one .bin file; records offsets."""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.pos = 0

    def put(self, name, arr):
        pad = _align(self.pos, ARR_ALIGN) - self.pos
        if pad:
            self.f.write(b"\0" * pad); self.pos += pad
        off = self.pos
        b = np.ascontiguousarray(arr).tobytes()
        self.f.write(b); self.pos += len(b)
        return dict(off=off, bytes=len(b))

    def close(self):
        pad = _align(self.pos, ARR_ALIGN) - self.pos
        if pad:
            self.f.write(b"\0" * pad); self.pos += pad
        self.f.close()
        return self.pos


def write_matrix(bw, name, K, N, layout, packed, col_scale):
    e = dict(K=int(K), N=int(N), G=int(N // GROUP), layout=int(layout),
             layout_name=LAYOUT_NAME[layout], arrays={})
    e["arrays"]["data"] = bw.put(f"{name}.data", packed["data"])
    if packed.get("row_off") is not None:
        e["arrays"]["row_off"] = bw.put(f"{name}.row_off", packed["row_off"])
    e["arrays"]["scale"] = bw.put(f"{name}.scale", packed["scale"])
    e["arrays"]["zero"] = bw.put(f"{name}.zero", packed["zero"])
    if col_scale is not None:
        cs = np.ascontiguousarray(col_scale)
        e["arrays"]["col_scale"] = bw.put(f"{name}.col_scale", cs)
        e["cs_rows"] = int(cs.shape[0]) if cs.ndim == 2 else 1
    else:
        e["cs_rows"] = 0
    e["n_survivors"] = int(packed["n_survivors"])
    e["n_zero_fixed"] = int(packed["n_zero_fixed"])
    # weight bytes actually streamed by a GEMV over this matrix
    e["stream_bytes"] = int(e["arrays"]["data"]["bytes"] - BLOB_TAIL
                            + e["arrays"]["scale"]["bytes"] + e["arrays"]["zero"]["bytes"])
    e["bpw"] = 8.0 * (e["stream_bytes"] + (e["arrays"].get("row_off", {"bytes": 0})["bytes"])) / (K * N)
    return e


# --------------------------------------------------------------------------- fused QKV
def dense_row_stride(layout, N):
    """Bytes per row of a fixed-stride layout (the only ones a row-concat fusion works on)."""
    if layout == DENSE4:
        return N // 2
    if layout == DENSE8:
        return N
    if layout == BLK1632_4:
        return N * 3 // 8
    if layout == BLK1632_6:
        return N // 2
    raise ValueError(
        f"row-concat fusion needs a fixed-stride layout, got {LAYOUT_NAME[layout]}")


def fuse_rows_packed(parts):
    """Row-CONCATENATE already-packed DENSE4/DENSE8 matrices into one.

    parts: [(name, K, packed_dict), ...].  The dense layouts have a fixed row stride and
    are row-independent, so concatenating the packed rows is byte-identical to packing the
    row-concatenated matrix -- the fused blob IS the DENSE4 layout of a K=sum(K_i) matrix.
    Returns (packed_fused, row_ranges dict name -> [k0, k1]).
    """
    data = np.concatenate([p["data"][:-BLOB_TAIL] for _, _, p in parts]
                          + [np.zeros(BLOB_TAIL, np.uint8)])
    scale = np.concatenate([p["scale"].reshape(-1) for _, _, p in parts])
    zero = np.concatenate([p["zero"].reshape(-1) for _, _, p in parts])
    rr, k0 = {}, 0
    for nm, K, _ in parts:
        rr[nm] = [int(k0), int(k0 + K)]
        k0 += K
    return dict(data=data, row_off=None, scale=scale, zero=zero,
                n_survivors=sum(int(p["n_survivors"]) for _, _, p in parts),
                n_zero_fixed=sum(int(p["n_zero_fixed"]) for _, _, p in parts),
                W_ref=None), rr


def alias_matrix(bw, name, K, N, layout, packed, col_scale, fused_name, fused_entry, k_start):
    """Manifest entry for a matrix that ALIASES rows [k_start, k_start+K) of a fused matrix.

    No weight bytes are duplicated: data/scale/zero are byte ranges INSIDE the fused
    arrays (the dense row stride makes every sub-matrix a contiguous slice).  Only the
    matrix's own fp16 col_scale[N] is written (2N bytes).
    """
    G = N // GROUP
    fa = fused_entry["arrays"]
    stride = dense_row_stride(layout, N)
    e = dict(K=int(K), N=int(N), G=int(G), layout=int(layout),
             layout_name=LAYOUT_NAME[layout], arrays={})
    e["arrays"]["data"] = dict(off=int(fa["data"]["off"] + k_start * stride),
                               bytes=int(K * stride))
    e["arrays"]["scale"] = dict(off=int(fa["scale"]["off"] + k_start * G * 2),
                                bytes=int(K * G * 2))
    e["arrays"]["zero"] = dict(off=int(fa["zero"]["off"] + k_start * G),
                               bytes=int(K * G))
    if col_scale is not None:
        cs = np.ascontiguousarray(col_scale)
        e["arrays"]["col_scale"] = bw.put(f"{name}.col_scale", cs)
        e["cs_rows"] = int(cs.shape[0]) if cs.ndim == 2 else 1
    else:
        e["cs_rows"] = 0
    e["n_survivors"] = int(packed["n_survivors"])
    e["n_zero_fixed"] = int(packed["n_zero_fixed"])
    e["alias_of"] = fused_name
    e["alias_row_start"] = int(k_start)
    e["stream_bytes"] = int(e["arrays"]["data"]["bytes"] + e["arrays"]["scale"]["bytes"]
                            + e["arrays"]["zero"]["bytes"])
    e["bpw"] = 8.0 * e["stream_bytes"] / (K * N)
    return e


# --------------------------------------------------------------------------- CLI
class ShardReader:
    """Reads tensors by name from an HF safetensors snapshot (index or loose shards)."""

    def __init__(self, model_dir):
        from safetensors import safe_open
        self._open = safe_open
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        self.map = {}
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            for k, f in wm.items():
                self.map[k] = os.path.join(model_dir, f)
        else:
            for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        self.map[k] = f
        self._h = {}

    def get(self, name):
        f = self.map[name]
        if f not in self._h:
            self._h[f] = self._open(f, framework="pt")
        return self._h[f].get_tensor(name)

    def has(self, name):
        return name in self.map


def pack_embedding(bw, W, embed_layout, bits_embed=4, device="cuda"):
    """Quantize the tied embedding / lm_head matrix [vocab, hidden] and append it."""
    K, N = W.shape
    if embed_layout == BF16:
        return dict(K=int(K), N=int(N), G=int(N // GROUP), layout=BF16, layout_name="BF16",
                    arrays=dict(data=bw.put("embed.data",
                                            W.to(torch.bfloat16).cpu().view(torch.uint8).numpy())),
                    n_survivors=int(K * N), n_zero_fixed=0,
                    stream_bytes=int(K * N * 2), bpw=16.0)
    dev = torch.device(device)
    e, chunks = None, []
    q, sc, zr = [], [], []
    step = max(1, 2 ** 26 // max(N, 1))
    for k0 in range(0, K, step):
        w = W[k0:k0 + step].to(dev).float()
        a, b, c = rtn_dense(w, bits_embed)
        q.append(a.to(torch.uint8).cpu()); sc.append(b.cpu()); zr.append(c.cpu())
        del w, a, b, c
    q = torch.cat(q); sc = torch.cat(sc); zr = torch.cat(zr)
    pk = pack_matrix_auto(None, q, sc, zr, embed_layout, bits=bits_embed, device=dev)
    return write_matrix(bw, "embed", K, N, embed_layout, pk, None)


def pack_artifact(raw_dir, out_dir, layout=SPARSE4, embed_layout=DENSE4, fuse_gateup=True,
                  bits=4, device="cuda", limit_layers=None, model_path=None,
                  embed_bits=4, fuse_qkv=False, layout_over=None, bits_over=None):
    """`layout_over` / `bits_over`: {LINEAR name -> layout id / bit width} for MIXED-LAYOUT
    artifacts (the o_proj-hybrid arms).  Both default to
    empty = every matrix uses `layout` / `bits`, which reproduces the previous behaviour
    byte-for-byte.  q/k/v must agree with each other (they are row-concatenated) and gate/up
    must agree with each other (they are row-interleaved)."""
    layout_over = dict(layout_over or {})
    bits_over = dict(bits_over or {})
    for k in list(layout_over) + list(bits_over):
        assert k in LINEARS, f"layout/bits override for unknown linear '{k}'"

    def _lay(nm):
        return layout_over.get(nm, layout)

    def _bits(nm):
        b = bits_over.get(nm, bits)
        if _lay(nm) in BLK1632:
            assert b == BLK1632_BITS[_lay(nm)], f"{nm}: {LAYOUT_NAME[_lay(nm)]} needs bits={BLK1632_BITS[_lay(nm)]}"
        return b

    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import load_file
    raw_manifest = {}
    p = os.path.join(raw_dir, "manifest.json")
    if os.path.exists(p):
        raw_manifest = json.load(open(p))
    files = sorted(glob.glob(os.path.join(raw_dir, "layer_*.safetensors")))
    if limit_layers:
        files = files[:limit_layers]
    man = dict(format="CBK1", version=1, group_size=GROUP, bits=bits,
               layout=LAYOUT_NAME[layout], embed_layout=LAYOUT_NAME[embed_layout],
               layout_override={k: LAYOUT_NAME[v] for k, v in layout_over.items()},
               bits_override=dict(bits_over),
               mixed_layout=bool(layout_over or bits_over),
               fuse_gateup=bool(fuse_gateup), fuse_qkv=bool(fuse_qkv), raw_dir=raw_dir,
               raw_config=raw_manifest.get("config", {}), layers=[], embed=None, misc=None)
    tot_p = tot_stream = tot_dense = 0
    for li, f in enumerate(files):
        t = load_file(f)
        bw = BlobWriter(os.path.join(out_dir, f"layer_{li:02d}.bin"))
        ent = dict(file=f"layer_{li:02d}.bin", matrices={}, norms={})
        names = [n for n in LINEARS]
        if fuse_gateup:
            names = [n for n in names if n not in ("gate_proj", "up_proj")] + ["gateup"]

        def _load(nm):
            key = [k for k in t if k.endswith(f"{nm}.q")][0][:-2]
            K, N = t[f"{key}.q"].shape
            return (K, N, t[f"{key}.mask"], t[f"{key}.q"], t[f"{key}.scale"],
                    t[f"{key}.zero"], t[f"{key}.col_scale"])

        # ---- fused QKV: ONE DENSE matrix, rows q|k|v row-CONCATENATED (see FORMAT.md §2.4b)
        if fuse_qkv:
            qlay, qbits = _lay("q_proj"), _bits("q_proj")
            assert all(_lay(n) == qlay and _bits(n) == qbits
                       for n in ("k_proj", "v_proj")), \
                "--fuse-qkv row-concatenates q|k|v: they must share layout and bits"
            assert qlay in (DENSE4, DENSE8) + BLK1632, \
                "--fuse-qkv requires a fixed-row-stride layout (DENSE4/DENSE8/BLK1632_*)"
            parts, css, Nq = [], [], None
            for nm in ("q_proj", "k_proj", "v_proj"):
                K, N, mb, q, sc, zr, cs = _load(nm)
                assert Nq in (None, N); Nq = N
                pk = pack_matrix_auto(mb, q, sc, zr, qlay, bits=qbits,
                                      device=torch.device(device))
                parts.append((nm, K, pk))
                css.append(cs.half().cpu().numpy().reshape(-1))
            fpk, rr = fuse_rows_packed(parts)
            Kf = sum(K for _, K, _ in parts)
            fe = write_matrix(bw, "qkv", Kf, Nq, qlay, fpk, np.stack(css, 0))   # [3,N]
            fe["fused_from"] = ["q_proj", "k_proj", "v_proj"]
            fe["row_ranges"] = rr
            fe["cs_row_of"] = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
            ent["matrices"]["qkv"] = fe
            tot_p += Kf * Nq
            tot_stream += fe["stream_bytes"]
            tot_dense += Kf * Nq // 2 + Kf * (Nq // GROUP) * 3
            for i, (nm, K, pk) in enumerate(parts):
                ent["matrices"][nm] = alias_matrix(bw, nm, K, Nq, qlay, pk, css[i],
                                                   "qkv", fe, rr[nm][0])
            del fpk, parts
            names = [n for n in names if n not in ("q_proj", "k_proj", "v_proj")]

        for nm in names:
            if nm == "gateup":
                assert _lay("gate_proj") == _lay("up_proj") and \
                       _bits("gate_proj") == _bits("up_proj"), \
                    "fused gateup interleaves gate/up rows: they must share layout and bits"
                nlay, nbits = _lay("gate_proj"), _bits("gate_proj")
                gk = [k for k in t if k.endswith("gate_proj.q")][0][:-2]
                uk = [k for k in t if k.endswith("up_proj.q")][0][:-2]
                K, N = t[f"{gk}.q"].shape
                mb = torch.stack([t[f"{gk}.mask"], t[f"{uk}.mask"]], 1).view(2 * K, N // 8)
                q = torch.stack([t[f"{gk}.q"], t[f"{uk}.q"]], 1).view(2 * K, N)
                sc = torch.stack([t[f"{gk}.scale"], t[f"{uk}.scale"]], 1).view(2 * K, N // GROUP)
                zr = torch.stack([t[f"{gk}.zero"], t[f"{uk}.zero"]], 1).view(2 * K, N // GROUP)
                cs = torch.stack([t[f"{gk}.col_scale"], t[f"{uk}.col_scale"]], 0)  # [2,N]
                K = 2 * K
            else:
                K, N, mb, q, sc, zr, cs = _load(nm)
                nlay, nbits = _lay(nm), _bits(nm)
            pk = pack_matrix_auto(mb, q, sc, zr, nlay, bits=nbits, device=torch.device(device))
            ent["matrices"][nm] = write_matrix(bw, nm, K, N, nlay, pk,
                                               cs.half().cpu().numpy())
            tot_p += K * N
            tot_stream += ent["matrices"][nm]["stream_bytes"]
            tot_dense += K * N // 2 + K * (N // GROUP) * 3
            del pk
        ent["bytes"] = bw.close()
        man["layers"].append(ent)
        print(f"[pack] layer {li:02d} -> {ent['bytes']/2**20:.1f} MiB", flush=True)
    # ---- embedding / lm_head + norms from the original checkpoint ----
    if model_path:
        rd = ShardReader(model_path)
        # norms: append fp32 copies to each layer blob is impossible after close(), so they
        # go into misc.bin together with the final norm.
        bw = BlobWriter(os.path.join(out_dir, "embed.bin"))
        W = rd.get("model.embed_tokens.weight")
        man["embed"] = pack_embedding(bw, W, embed_layout, embed_bits, device)
        man["embed"]["file"] = "embed.bin"
        del W
        man["embed"]["bytes"] = bw.close()
        # untied lm_head (llama family): packed with the SAME layout/bits as the embedding
        cfg_p = os.path.join(model_path, "config.json")
        cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
        cfg = cfg.get("text_config", cfg)
        tied = cfg.get("tie_word_embeddings", not rd.has("lm_head.weight"))
        if not tied and rd.has("lm_head.weight"):
            bw = BlobWriter(os.path.join(out_dir, "lm_head.bin"))
            W = rd.get("lm_head.weight")
            man["lm_head"] = pack_embedding(bw, W, embed_layout, embed_bits, device)
            man["lm_head"]["file"] = "lm_head.bin"
            del W
            man["lm_head"]["bytes"] = bw.close()
            print(f"[pack] untied lm_head -> lm_head.bin ({man['lm_head']['bytes']/2**20:.1f} MiB)", flush=True)
        else:
            man["lm_head"] = None
        bw = BlobWriter(os.path.join(out_dir, "misc.bin"))
        misc = {}
        if rd.has("model.norm.weight"):
            misc["model.norm"] = bw.put("model.norm", rd.get("model.norm.weight").float().numpy())
        for li in range(len(man["layers"])):
            for nn in NORMS:
                key = f"model.layers.{li}.{nn}.weight"
                if rd.has(key):
                    misc[f"{li}.{nn}"] = bw.put(key, rd.get(key).float().numpy())
        man["misc"] = dict(file="misc.bin", arrays=misc, bytes=bw.close())
    man["bpw"] = dict(params=tot_p, packed_bytes=tot_stream,
                      bpw_packed=8.0 * tot_stream / max(tot_p, 1),
                      bpw_dense4=8.0 * tot_dense / max(tot_p, 1))
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", default="SPARSE4", choices=list(LAYOUT_ID))
    ap.add_argument("--embed-layout", default="DENSE4", choices=["DENSE4", "DENSE8", "BF16"])
    ap.add_argument("--no-fuse-gateup", action="store_true")
    ap.add_argument("--fuse-qkv", action="store_true",
                    help="also emit a row-concatenated q|k|v DENSE matrix (col_scale [3,N]); "
                         "the separate q/k/v entries are kept and alias into it")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-layers", type=int, default=None)
    ap.add_argument("--model-path", default=None,
                    help="HF snapshot dir; adds the tied embedding/lm_head and the norms")
    ap.add_argument("--embed-bits", type=int, default=4)
    ap.add_argument("--layout-override", default="",
                    help="per-matrix layout, e.g. 'o_proj=DENSE4' (comma-separated). "
                         "Empty = uniform --layout (bit-identical to before this flag).")
    ap.add_argument("--bits-override", default="",
                    help="per-matrix code width, e.g. 'o_proj=4' (comma-separated)")
    a = ap.parse_args()
    lov, bov = {}, {}
    for kv in [s for s in a.layout_override.split(",") if s.strip()]:
        k, v = kv.split("="); lov[k.strip()] = LAYOUT_ID[v.strip()]
    for kv in [s for s in a.bits_override.split(",") if s.strip()]:
        k, v = kv.split("="); bov[k.strip()] = int(v)
    if a.embed_layout == "DENSE8":
        a.embed_bits = 8
    m = pack_artifact(a.raw, a.out, LAYOUT_ID[a.layout], LAYOUT_ID[a.embed_layout],
                      not a.no_fuse_gateup, a.bits, a.device, a.limit_layers,
                      a.model_path, a.embed_bits, a.fuse_qkv, lov, bov)
    print(json.dumps(m["bpw"], indent=1))


if __name__ == "__main__":
    main()
