#!/usr/bin/env python
"""Stream a CoBALT raw artifact back into a bf16 HF checkpoint (fake-quantised
W_hat with exact structural zeros), shard by shard, never holding the model.

  python export_fakequant.py --src <hf snapshot> --art <raw artifact dir> --out <dir>

Only the 7 quantised linears per decoder layer are rewritten; norms are copied verbatim.
`--embed-bits {16,8,4}` additionally fake-quantises the tied embedding / lm_head the way the
kernel arm serves it (DENSE4/DENSE8: plain asymmetric group-128 min-max RTN, no mask/OBS/
col-scale, via pack_cobalt.rtn_dense); 16 (default) keeps it bf16.
"""
import argparse, json, os, re, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LIN_RE = re.compile(r"^model\.layers\.(\d+)\.((?:self_attn|mlp)\.\w+)\.weight$")
# tied embedding / lm_head: the kernel arm serves these as DENSE4/DENSE8 (plain group-128
# min-max RTN, no mask/OBS/col-scale) -- see docs/FORMAT.md sec.7.
EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")

COPY_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
              "chat_template.jinja", "preprocessor_config.json"]


def unpack_mask(packed, N):
    K = packed.shape[0]
    bits = torch.arange(8, dtype=torch.uint8, device=packed.device)
    m = (packed.unsqueeze(-1) >> bits.view(1, 1, 8)) & 1
    return m.reshape(K, -1)[:, :N].float()


def dequant_embed(W, bits, dev, group=128):
    """Fake-quantise the tied embedding with pack_cobalt.rtn_dense (identical numerics to
    the packer's DENSE4/DENSE8 embedding path), streamed in row chunks."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pack_cobalt import rtn_dense
    K, N = W.shape
    assert N % group == 0, f"embedding N={N} not divisible by {group}"
    out = torch.empty_like(W)
    step = max(1, 2 ** 26 // max(N, 1))
    for k0 in range(0, K, step):
        w = W[k0:k0 + step].to(dev).float()
        q, sc, zr = rtn_dense(w, bits, group)
        g = w.shape[1] // group
        deq = (q.view(-1, g, group) - zr.float().unsqueeze(-1)) * sc.float().unsqueeze(-1)
        out[k0:k0 + step] = deq.view(w.shape).to(W.dtype).cpu()
        del w, q, sc, zr, deq
    return out


def dequant_layer(art_dir, li, dev):
    p = os.path.join(art_dir, f"layer_{li:02d}.safetensors")
    if not os.path.exists(p):
        return None
    out = {}
    with safe_open(p, framework="pt") as h:
        names = sorted({k.rsplit(".", 1)[0] for k in h.keys()})
        for n in names:
            q = h.get_tensor(f"{n}.q").to(dev)
            scale = h.get_tensor(f"{n}.scale").to(dev).float()
            zero = h.get_tensor(f"{n}.zero").to(dev).float()
            c = h.get_tensor(f"{n}.col_scale").to(dev).float()
            mask = unpack_mask(h.get_tensor(f"{n}.mask").to(dev), q.shape[1])
            K, N = q.shape
            g = N // scale.shape[1]
            W = (q.view(K, N // g, g).float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
            out[n] = (W.view(K, N) * c.view(1, -1) * mask).to(torch.bfloat16).cpu()
            del q, scale, zero, c, mask, W
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embed-bits", type=int, choices=[16, 8, 4], default=16,
                    help="16 = keep the tied embedding/lm_head bf16 (default); 8/4 = DENSE8/DENSE4 RTN "
                         "matching what the kernel arm serves")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.art, "manifest.json")))
    t0 = time.time()

    for f in COPY_FILES:
        s = os.path.join(a.src, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.out, f))
    idx_p = os.path.join(a.src, "model.safetensors.index.json")
    shards = sorted({v for v in json.load(open(idx_p))["weight_map"].values()}) if os.path.exists(idx_p) \
        else [os.path.basename(x) for x in sorted(os.listdir(a.src)) if x.endswith(".safetensors")]
    if os.path.exists(idx_p):
        shutil.copy2(idx_p, os.path.join(a.out, "model.safetensors.index.json"))

    cache_li, cache = -1, None
    n_rep, maxrel, n_emb = 0, 0.0, 0
    for sh in shards:
        tensors = {}
        with safe_open(os.path.join(a.src, sh), framework="pt") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                m = LIN_RE.match(k)
                if m:
                    li, name = int(m.group(1)), m.group(2)
                    if li != cache_li:
                        cache, cache_li = dequant_layer(a.art, li, dev), li
                    if cache is not None and name in cache:
                        w_hat = cache[name]
                        rel = float((w_hat.float() - t.float()).norm() / t.float().norm())
                        maxrel = max(maxrel, rel)
                        exp = man["layers"].get(str(li), {}).get(name, {}).get("relerr")
                        if exp is not None and abs(rel - exp) > 5e-3:
                            print(f"[warn] layer {li} {name} relerr {rel:.4f} != manifest {exp:.4f}", flush=True)
                        t = w_hat
                        n_rep += 1
                elif a.embed_bits < 16 and k in EMBED_KEYS:
                    w_ref = t
                    t = dequant_embed(w_ref, a.embed_bits, dev)
                    rel = float((t.float() - w_ref.float()).norm() / w_ref.float().norm())
                    print(f"[{time.strftime('%F %T')}] embed {k} {tuple(t.shape)} -> DENSE{a.embed_bits} "
                          f"relerr={rel:.5f}", flush=True)
                    n_emb += 1
                    del w_ref
                tensors[k] = t.contiguous()
        save_file(tensors, os.path.join(a.out, sh), metadata={"format": "pt"})
        print(f"[{time.strftime('%F %T')}] wrote {sh} ({len(tensors)} tensors, {n_rep} replaced so far)", flush=True)
        del tensors
    print(f"DONE replaced {n_rep} matrices ({n_emb} embedding, embed_bits={a.embed_bits}), "
          f"max linear relerr {maxrel:.4f}, {time.time()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
