"""Torch dequantizer for a CBK1 packed artifact -> a ref_gemma3 weight dict.

This is the ORACLE for the quantized megakernel path: it reads exactly the bytes the
kernel reads and reconstructs W_hat = (q - zero) * scale * col_scale in torch, so a
kernel/reference disagreement can only be a kernel bug, not a quantization difference.
(It also independently checks the packer.)

    st = load_state_packed(packed_dir, config_dir)      # same shape as ref_gemma3.load_state
"""

import json
import os

import torch

from . import arch

LAYOUT_SPARSE4, LAYOUT_DENSE4, LAYOUT_DENSE8, LAYOUT_SPARSE4X, LAYOUT_BF16 = 0, 1, 2, 3, 4
LAYOUT_SPARSE4E = 5
LAYOUT_BLK1632_4, LAYOUT_BLK1632_6 = 6, 7
GROUP = 128
BLOCK = 32


def _blob(path, device):
    with open(path, "rb") as f:
        b = f.read()
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).to(device)


def _arr(blob, e, dtype):
    v = blob[e["off"]: e["off"] + e["bytes"]]
    return v.view(dtype)


def dequant_matrix(blob, entry, device, dtype=torch.bfloat16):
    """Return W_hat [K, N] in `dtype`."""
    K, N, G, lay = entry["K"], entry["N"], entry["N"] // GROUP, entry["layout"]
    A = entry["arrays"]
    if lay == LAYOUT_BF16:
        w = _arr(blob, A["data"], torch.bfloat16)[: K * N].view(K, N)
        return w.to(dtype)
    if lay == LAYOUT_DENSE4:
        d = _arr(blob, A["data"], torch.uint8)[: K * (N // 2)].view(K, N // 2)
        lo = (d & 0xF).to(torch.float32)
        hi = (d >> 4).to(torch.float32)
        q = torch.stack([lo, hi], dim=2).reshape(K, N)   # col 2t = low nibble
    elif lay == LAYOUT_DENSE8:
        q = _arr(blob, A["data"], torch.uint8)[: K * N].view(K, N).to(torch.float32)
    elif lay in (LAYOUT_BLK1632_4, LAYOUT_BLK1632_6):
        # CoBALT-16:32 planes (FORMAT.md sec.13): [mask N/8][nibble N/4][hi2 N/8 if b=6]
        b6 = lay == LAYOUT_BLK1632_6
        stride = (N // 2) if b6 else (N * 3 // 8)
        NB = N // BLOCK
        rowb = _arr(blob, A["data"], torch.uint8)[: K * stride].view(K, stride)
        sh8 = torch.arange(8, dtype=torch.uint8, device=rowb.device)
        keep = ((rowb[:, : N // 8].reshape(K, -1, 1) >> sh8) & 1).view(K, N).bool()
        nib = rowb[:, N // 8: N // 8 + NB * 8].reshape(K, NB, 8).to(torch.int64)
        q16 = torch.stack([nib & 0xF, (nib >> 4) & 0xF], -1).view(K, NB, 16)
        if b6:
            hb = rowb[:, N * 3 // 8:].reshape(K, NB, 4).to(torch.int64)
            hw = hb[..., 0] | (hb[..., 1] << 8) | (hb[..., 2] << 16) | (hb[..., 3] << 24)
            sh = torch.arange(8, dtype=torch.int64, device=rowb.device) * 2
            he = (hw.unsqueeze(-1) >> sh) & 3
            ho = (hw.unsqueeze(-1) >> (sh + 16)) & 3
            q16 = q16 | (torch.stack([he, ho], -1).view(K, NB, 16) << 4)
        q = torch.zeros(K, N, dtype=torch.float32, device=rowb.device)
        q[keep] = q16.reshape(-1).to(torch.float32)   # 16 survivors/block, column order
        s = _arr(blob, A["scale"], torch.float16)[: K * G].view(K, G).to(torch.float32)
        z = _arr(blob, A["zero"], torch.uint8)[: K * G].view(K, G).to(torch.float32)
        w = ((q.view(K, G, GROUP) - z[:, :, None]) * s[:, :, None]).reshape(K, N)
        w = w * keep.to(torch.float32)                # pruned positions: EXACTLY 0
        if "col_scale" in A and A["col_scale"]["bytes"]:
            ncs = A["col_scale"]["bytes"] // 2
            cs = _arr(blob, A["col_scale"], torch.float16)[:ncs].to(torch.float32)
            if ncs == 2 * N:
                cs = cs.view(2, N)
                w[0::2] *= cs[0][None, :]
                w[1::2] *= cs[1][None, :]
            else:
                assert ncs == N, f"col_scale has {ncs} entries, N={N}"
                w = w * cs[None, :]
        return w.to(dtype)
    else:
        raise NotImplementedError(
            f"layout {lay}: the torch oracle only implements the fixed-stride layouts")
    s = _arr(blob, A["scale"], torch.float16)[: K * G].view(K, G).to(torch.float32)
    z = _arr(blob, A["zero"], torch.uint8)[: K * G].view(K, G).to(torch.float32)
    w = (q.view(K, G, GROUP) - z[:, :, None]) * s[:, :, None]
    w = w.reshape(K, N)
    if "col_scale" in A and A["col_scale"]["bytes"]:
        ncs = A["col_scale"]["bytes"] // 2
        cs = _arr(blob, A["col_scale"], torch.float16)[:ncs].to(torch.float32)
        if ncs == 2 * N:
            # FUSED gate/up: col_scale is [2, N] (row 0 = gate, row 1 = up) and the
            # weight rows are interleaved 2i = gate_i, 2i+1 = up_i.
            cs = cs.view(2, N)
            w[0::2] *= cs[0][None, :]
            w[1::2] *= cs[1][None, :]
        else:
            assert ncs == N, f"col_scale has {ncs} entries, N={N}"
            w = w * cs[None, :]
    return w.to(dtype)


def load_state_packed(packed_dir, config_dir, device="cuda", dtype=torch.bfloat16,
                      layers_limit=None):
    cfg_raw = json.load(open(os.path.join(config_dir, "config.json")))
    R = arch.ref_module(cfg_raw)
    cfg = R.Gemma3RefConfig(cfg_raw) if arch.family_of(cfg_raw) == "gemma3" else R.LlamaRefConfig(cfg_raw)
    man = json.load(open(os.path.join(packed_dir, "manifest.json")))
    W = {}

    eb = _blob(os.path.join(packed_dir, man["embed"]["file"]), device)
    emb = dequant_matrix(eb, man["embed"], device, dtype)[: cfg.vocab_size]
    del eb
    W["model.embed_tokens.weight"] = emb
    if man.get("lm_head"):
        hb = _blob(os.path.join(packed_dir, man["lm_head"]["file"]), device)
        W["lm_head.weight"] = dequant_matrix(hb, man["lm_head"], device, dtype)[: cfg.vocab_size]
        del hb
    else:
        W["lm_head.weight"] = emb

    mb = _blob(os.path.join(packed_dir, man["misc"]["file"]), device)
    norms = {}
    for name, e in man["misc"]["arrays"].items():
        norms[name] = _arr(mb, e, torch.float32)[: e["bytes"] // 4].to(dtype).clone()
    del mb
    W["model.norm.weight"] = norms["model.norm"]

    n = layers_limit or len(man["layers"])
    for i in range(n):
        lay = man["layers"][i]
        b = _blob(os.path.join(packed_dir, lay["file"]), device)
        p = f"model.layers.{i}."
        mm = lay["matrices"]
        for nm, key in (("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"),
                        ("v_proj", "self_attn.v_proj"), ("o_proj", "self_attn.o_proj"),
                        ("down_proj", "mlp.down_proj")):
            W[p + key + ".weight"] = dequant_matrix(b, mm[nm], device, dtype).clone()
        if "gateup" in mm:
            gu = dequant_matrix(b, mm["gateup"], device, dtype)
            W[p + "mlp.gate_proj.weight"] = gu[0::2].clone()
            W[p + "mlp.up_proj.weight"] = gu[1::2].clone()
            del gu
        else:
            W[p + "mlp.gate_proj.weight"] = dequant_matrix(b, mm["gate_proj"], device, dtype).clone()
            W[p + "mlp.up_proj.weight"] = dequant_matrix(b, mm["up_proj"], device, dtype).clone()
        for nn in arch.all_norm_names():           # whichever norms this family has
            if f"{i}.{nn}" in norms:
                W[p + nn + ".weight"] = norms[f"{i}.{nn}"]
        del b
        torch.cuda.empty_cache()

    rope = {}
    for lt in sorted(set(cfg.layer_types)):
        if arch.family_of(cfg_raw) == "gemma3":
            rope[lt] = R._inv_freq(cfg.rope_parameters[lt], cfg.head_dim, device)
        else:
            rope[lt] = R._inv_freq(cfg.rope_theta, cfg.head_dim, device)
    return {"cfg": cfg, "W": W, "rope": rope, "device": device, "dtype": dtype}
