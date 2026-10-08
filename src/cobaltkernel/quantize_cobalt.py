#!/usr/bin/env python
"""Memory-streamed, layer-wise CoBALT quantizer for Gemma3 / Mistral / Llama decoder models.

Model family is read from config.json `model_type` (see `family_of`); the Gemma3 path is
unchanged, the Llama-style path (mistral, llama) differs only in how the skeleton layer is
built and called -- the quantization math is identical.

Designed to quantize MedGemma-27B (62 layers, bf16 54 GB) inside a single MIG
slice: the model is NEVER materialised -- decoder layers are constructed on
`meta` and their weights streamed from the snapshot's safetensors shards one
layer at a time.  Calibration hidden states live on pinned CPU memory and are
streamed sample-by-sample to the GPU.

Math: canonical CoBALT (src/cobaltkernel/cobalt_math.py) --
  Wanda importance -> row+col quantile balance (beta) -> ONE global top-k mask
  -> OBS compensation with the FULL-calibration Hessian -> sparse-aware
  per-column scale -> group RTN (survivor or all hull).

Output: one `layer_XX.safetensors` per decoder layer + `manifest.json`.
"""
import argparse, copy, gc, glob, json, os, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cobalt_math as cm  # noqa: E402

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
# linears that share the same input activation -> one Hessian each
HGROUPS = {"attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
           "attn_out": ["self_attn.o_proj"],
           "mlp_in": ["mlp.gate_proj", "mlp.up_proj"],
           "mlp_out": ["mlp.down_proj"]}
HOOK_SRC = {"attn_in": "self_attn.q_proj", "attn_out": "self_attn.o_proj",
            "mlp_in": "mlp.gate_proj", "mlp_out": "mlp.down_proj"}


def materialize(mod, dev):
    """module.to_empty() re-allocates BUFFERS too (uninitialised) -- for Gemma3 that
    silently zeroes embed_scale. Snapshot real buffers, to_empty, restore them."""
    bufs = {n: b for n, b in mod.named_buffers() if b is not None and b.device.type != "meta"}
    mod.to_empty(device=dev)
    for n, b in bufs.items():
        *pre, last = n.split(".")
        sub = mod.get_submodule(".".join(pre)) if pre else mod
        setattr(sub, last, b.to(dev))
    return mod


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


# ------------------------------------------------------------------ config
def load_config(model_path):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    # normalise the NEW-style (transformers >=5.14) rope_parameters dict into the
    # old attributes that transformers 4.5x's Gemma3RotaryEmbedding reads, otherwise
    # the global rope's linear factor 8.0 is silently dropped (verified for gemma-3-4b).
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "full_attention" in rp:
        g, s = rp["full_attention"], rp["sliding_attention"]
        cfg.rope_theta = g["rope_theta"]
        cfg.rope_scaling = {"rope_type": g.get("rope_type", "default"), "factor": g.get("factor", 1.0)}
        cfg.rope_local_base_freq = s["rope_theta"]
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    return cfg


def family_of(cfg):
    """'gemma3' (dual RoPE, 4 norms, QK-norm, tied embed*scale) or 'llama' (mistral/llama:
    one RoPE, 2 norms, no QK-norm, untied lm_head, no embed scale)."""
    mt = getattr(cfg, "model_type", "")
    if mt.startswith("gemma3"):
        return "gemma3"
    if mt in ("mistral", "llama"):
        return "llama"
    raise NotImplementedError(f"model_type={mt!r}: only gemma3 / mistral / llama are wired")


def build_skeleton(cfg, family):
    from accelerate import init_empty_weights
    if family == "gemma3":
        from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel as Cls
    elif cfg.model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralModel as Cls
    else:
        from transformers.models.llama.modeling_llama import LlamaModel as Cls
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights():
        skel = Cls(cfg)
    torch.set_default_dtype(torch.float32)
    return skel


# ------------------------------------------------------------------ weight streaming
class ShardReader:
    def __init__(self, model_dir):
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wmap = json.load(open(idx))["weight_map"]
        else:
            self.wmap = {}
            for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        self.wmap[k] = os.path.basename(f)
        self.dir = model_dir
        self._open = {}

    def _h(self, shard):
        if shard not in self._open:
            self._open[shard] = safe_open(os.path.join(self.dir, shard), framework="pt")
        return self._open[shard]

    def keys(self):
        return self.wmap.keys()

    def get(self, name):
        return self._h(self.wmap[name]).get_tensor(name)

    def prefix(self, pfx, strip=True):
        out = {}
        for k in self.wmap:
            if k.startswith(pfx):
                out[k[len(pfx):] if strip else k] = self.get(k)
        return out


# ------------------------------------------------------------------ calibration
def build_calib(model_path, calib, n_calib, seq_len, calib_file, offset=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if calib == "ultrachat":
        raw = open(calib_file, encoding="utf-8").read()
        samples = [s for s in raw.split("\n\n") if s.strip()]
        bos = tok.bos_token_id
        stream = []
        for s in samples:
            stream.append(bos)
            stream.extend(tok(s, add_special_tokens=False)["input_ids"])
            if len(stream) >= (offset + n_calib + 1) * seq_len:
                break
        src = f"{calib_file} ({'ultrachat_200k/train_sft seed42, ' if 'ultrachat' in calib_file else ''}packed text, block offset {offset})"
    elif calib == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        stream = tok("\n\n".join(ds["text"]), add_special_tokens=False)["input_ids"]
        src = f"wikitext2-raw train (packed, block offset {offset})"
    else:
        raise ValueError(calib)
    avail = len(stream) // seq_len - offset
    if avail <= 0:
        raise RuntimeError(f"calib offset {offset} beyond the {len(stream)//seq_len} available "
                           f"{seq_len}-token blocks")
    n = min(n_calib, avail)
    lo = offset * seq_len
    ids = torch.tensor(stream[lo: lo + n * seq_len], dtype=torch.long).view(n, seq_len)
    return ids, src, n


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.5, help="col_balance_exp")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--hull", choices=["all", "survivor"], default="survivor")
    ap.add_argument("--calib", choices=["ultrachat", "wikitext2"], default="ultrachat")
    ap.add_argument("--calib-file", default="results/accel4bit/calib_ultrachat_512x2048.txt")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib-offset", type=int, default=0,
                    help="skip this many seq_len-token blocks of the packed calibration stream "
                         "(use n-calib to get a DISJOINT calibration realization)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers-limit", type=int, default=0)
    ap.add_argument("--seq-inputs", action="store_true",
                    help="propagate through ALREADY-QUANTIZED layers (non-canonical)")
    ap.add_argument("--acts-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--damping-frac", type=float, default=0.01)
    ap.add_argument("--bits-override", default="",
                    help="per-matrix bit-width overrides, comma-separated 'module=bits' "
                         "(e.g. 'self_attn.o_proj=4'). Default empty = every matrix uses --bits, i.e. "
                         "bit-identical to a run without this flag.")
    ap.add_argument("--mask-block-exclude", default="",
                    help="comma-separated module names (e.g. 'self_attn.o_proj') that KEEP the canonical "
                         "global top-k even when --mask-block is set. Default empty = no exclusions, i.e. "
                         "bit-identical to a plain --mask-block run.")
    ap.add_argument("--sparsity-override", default="",
                    help="per-matrix sparsity, comma-separated 'module=sp' (e.g. 'self_attn.o_proj=0.0,"
                         "mlp.down_proj=0.0' keeps those DENSE: no mask, no OBS). Default empty = every "
                         "matrix uses --sparsity, bit-identical to a run without this flag.")
    ap.add_argument("--quant-mode", choices=["rtn", "gptq"], default="rtn",
                    help="survivor quantizer: 'rtn' = shipped group min-max RTN (bit-identical default); "
                         "'gptq' = sequential column-wise OBS error feedback under the FIXED CoBALT mask "
                         "(exact joint-OBS pruning compensation + rounding-error compensation)")
    ap.add_argument("--clip", choices=["none", "mse", "aw"], default="none",
                    help="per-group (scale, zero) search on a fixed global grid: none = min-max (default, "
                         "bit-identical); mse = unweighted; aw = activation-weighted (output-error diagonal)")
    ap.add_argument("--actorder", action="store_true",
                    help="gptq only: process columns by descending Hessian diagonal with STATIC groups")
    ap.add_argument("--gptq-block", type=int, default=128)
    ap.add_argument("--heldout-calib", type=int, default=0,
                    help="extra calibration blocks (taken AFTER the n-calib fit blocks) whose Gram is kept "
                         "separate and used only to report a HELD-OUT output error per matrix (eout_ho); "
                         "0 = off (bit-identical to the shipped path)")
    ap.add_argument("--mask-block", type=int, default=0,
                    help="fixed-cardinality block mask: keep exactly block*(1-sparsity) of every "
                         "aligned block of <block> input columns (32 => CoBALT-16:32). "
                         "0 (default) = canonical ONE global top-k, bit-identical to the shipped path.")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32
    torch.backends.cudnn.allow_tf32 = not a.no_tf32
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cfg = load_config(a.model_path)
    family = family_of(cfg)
    nlayers = cfg.num_hidden_layers if a.layers_limit <= 0 else min(a.layers_limit, cfg.num_hidden_layers)
    log(f"model={a.model_path} family={family} ({cfg.model_type}) layers={cfg.num_hidden_layers} "
        f"(processing {nlayers}) hidden={cfg.hidden_size} inter={cfg.intermediate_size}")

    ids, calib_src, n_tot = build_calib(a.model_path, a.calib, a.n_calib + a.heldout_calib, a.seq_len,
                                        a.calib_file, a.calib_offset)
    n_calib = min(a.n_calib, n_tot)
    n_ho = n_tot - n_calib
    log(f"calib: {n_calib} x {a.seq_len} tokens from {calib_src}" + (f" + {n_ho} HELD-OUT blocks" if n_ho else ""))

    reader = ShardReader(a.model_path)
    skel = build_skeleton(cfg, family)

    # ---- embeddings + rotary (real tensors) ----
    embed = materialize(skel.embed_tokens, dev)
    embed.load_state_dict({"weight": reader.get("model.embed_tokens.weight")})
    if family == "gemma3":
        assert float(embed.embed_scale) > 0, "embed_scale lost"
        rot_g, rot_l = skel.rotary_emb.to(dev), skel.rotary_emb_local.to(dev)
    else:
        rot_g, rot_l = skel.rotary_emb.to(dev), None

    # ---- hidden state buffers (pinned CPU or GPU) ----
    H0, T = cfg.hidden_size, a.seq_len
    buf_dev = "cpu" if a.acts_device == "cpu" else dev
    inp = torch.empty(n_tot, T, H0, dtype=torch.bfloat16,
                      device=buf_dev, pin_memory=(buf_dev == "cpu"))
    with torch.no_grad():
        for i in range(n_tot):
            inp[i].copy_(embed(ids[i].to(dev)))
    embed.to("cpu"); del embed; torch.cuda.empty_cache()
    out = torch.empty_like(inp) if a.seq_inputs else None

    # ---- shared position embeddings / masks (all samples have identical length) ----
    dummy = torch.zeros(1, T, H0, dtype=torch.bfloat16, device=dev)
    cache_position = torch.arange(T, device=dev)
    position_ids = cache_position.unsqueeze(0)
    mk = dict(config=cfg, input_embeds=dummy, attention_mask=None,
              cache_position=cache_position, past_key_values=None, position_ids=position_ids)
    with torch.no_grad():
        pe_g = rot_g(dummy, position_ids)
        if family == "gemma3":
            pe_l = rot_l(dummy, position_ids)
            masks = {"full_attention": create_causal_mask(**mk),
                     "sliding_attention": create_sliding_window_causal_mask(**mk)}
        else:
            # mistral: uniform sliding window when config.sliding_window is set (HF: same rule)
            sw = getattr(cfg, "sliding_window", None)
            masks = {"llama": (create_sliding_window_causal_mask(**mk) if sw
                               else create_causal_mask(**mk))}
            log(f"llama-family mask: {'sliding window ' + str(sw) if sw else 'full causal'} "
                f"(seq_len {T}{' < window -> no effect' if sw and T <= sw else ''})")
    del dummy

    def call_layer(layer, h, amask):
        if family == "gemma3":
            o = layer(h, position_embeddings_global=pe_g, position_embeddings_local=pe_l,
                      attention_mask=amask, position_ids=position_ids,
                      past_key_value=None, use_cache=False, cache_position=cache_position)
        else:
            o = layer(h, attention_mask=amask, position_ids=position_ids,
                      past_key_value=None, use_cache=False, cache_position=cache_position,
                      position_embeddings=pe_g)
        return o[0] if isinstance(o, tuple) else o

    def mask_for(layer):
        return masks[layer.attention_type] if family == "gemma3" else masks["llama"]

    manifest = dict(model_path=a.model_path, config=dict(
        sparsity=a.sparsity, bits=a.bits, beta=a.beta, group_size=a.group_size, hull=a.hull,
        calib=a.calib, calib_source=calib_src, n_calib=n_calib, seq_len=a.seq_len,
        calib_offset=a.calib_offset, mask_block=a.mask_block,
        quant_mode=a.quant_mode, clip=a.clip, actorder=bool(a.actorder), gptq_block=a.gptq_block,
        heldout_calib=n_ho,
        mask_block_exclude=[x for x in a.mask_block_exclude.split(",") if x.strip()],
        bits_override=a.bits_override,
        seq_inputs=bool(a.seq_inputs), damping_frac=a.damping_frac, tf32=not a.no_tf32,
        method=("cobalt: wanda-imp + row/col-quantile balance + "
                + (f"per-{a.mask_block}-block top-{a.mask_block - int(a.mask_block * a.sparsity)}"
                   + (f" (EXCEPT {a.mask_block_exclude} = global topk)" if a.mask_block_exclude else "")
                   if a.mask_block > 0 else "global topk")
                + " + OBS + col-scale + "
                + ("group-RTN" if a.quant_mode == "rtn" else
                   f"GPTQ-seq({'actorder,static-groups' if a.actorder else 'dynamic-groups'})")
                + (f" + clip={a.clip}" if a.clip != "none" else ""))),
        hidden_size=H0, num_hidden_layers=cfg.num_hidden_layers, layers_processed=nlayers,
        model_type=cfg.model_type, family=family,
        quantized_modules=LINEARS,
        unquantized=("embed_tokens, lm_head (tied), all RMSNorms kept bf16" if family == "gemma3"
                     else "embed_tokens, lm_head (untied), all RMSNorms kept bf16"),
        layers={}, artifact_schema=dict(
            mask="uint8 [K, N/8] bitmap, LSB-first: keep(k,j) = (mask[k, j//8] >> (j%8)) & 1",
            q="uint8 [K, N] group-RTN codes, pruned positions = 0",
            scale="fp16 [K, N/g]", zero="fp16 [K, N/g]", col_scale="fp16 [N]",
            dequant="W_hat[k,j] = (q[k,j] - zero[k,j//g]) * scale[k,j//g] * col_scale[j] * keep(k,j)"))

    bover = {}
    for tok in a.bits_override.split(","):
        tok = tok.strip()
        if not tok:
            continue
        mn, _, bv = tok.partition("=")
        assert mn in LINEARS, f"--bits-override name not in LINEARS: {mn}"
        bover[mn] = int(bv)
    if bover:
        log(f"per-matrix BITS overrides: {bover}")

    sover = {}
    for tok in a.sparsity_override.split(","):
        tok = tok.strip()
        if not tok:
            continue
        mn, _, sv = tok.partition("=")
        assert mn in LINEARS, f"--sparsity-override name not in LINEARS: {mn}"
        sover[mn] = float(sv)
    if sover:
        log(f"per-matrix SPARSITY overrides: {sover}")
        manifest["config"]["sparsity_override"] = sover

    excl = set(x for x in a.mask_block_exclude.split(",") if x.strip())
    if excl:
        unknown = excl - set(LINEARS)
        assert not unknown, f"--mask-block-exclude names not in LINEARS: {unknown}"
        log(f"mask-block EXCLUSIONS (canonical global top-k): {sorted(excl)}")

    tot_num, tot_den_dense, tot_den_surv, tot_params, tot_surv = 0, 0, 0, 0, 0
    tot_bits_dense = tot_bits_surv = 0
    peak_mem = 0
    ho_num = ho_den = 0.0

    for li in range(nlayers):
        t0 = time.time()
        layer = materialize(skel.layers[li], dev)
        sd = reader.prefix(f"model.layers.{li}.")
        missing = layer.load_state_dict(sd, strict=True)
        del sd
        layer.eval()
        amask = mask_for(layer)

        # ---- Hessian accumulation over ALL calibration tokens ----
        Hs, Hs_ho, hooks = {}, {}, []
        ho_flag = [False]

        def mk_hook(gname):
            def hook(mod, args):
                X = args[0]
                X = X.reshape(-1, X.shape[-1]).float()
                tgt = Hs_ho if ho_flag[0] else Hs
                if gname not in tgt:
                    tgt[gname] = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float32, device=dev)
                tgt[gname] += X.T @ X
            return hook

        for gname, src in HOOK_SRC.items():
            mod = layer.get_submodule(src)
            hooks.append(mod.register_forward_pre_hook(mk_hook(gname)))

        with torch.no_grad():
            for i in range(n_tot):
                ho_flag[0] = i >= n_calib
                h = inp[i].to(dev).unsqueeze(0)
                o = call_layer(layer, h, amask)
                dst = out if a.seq_inputs else inp
                dst[i].copy_(o.squeeze(0))          # SYNC: async D2H into the calib buffer was measured unreliable
                del h, o
        ho_flag[0] = False
        torch.cuda.synchronize()
        for hk in hooks:
            hk.remove()
        t_fwd = time.time() - t0
        hstat = {g: (float(H.abs().amax()), bool(torch.isfinite(H).all())) for g, H in Hs.items()}
        for g, (mx, fin) in hstat.items():
            if (not fin) or mx == 0.0:
                zs = [i for i in range(n_calib) if float(inp[i].abs().max()) == 0.0]
                raise RuntimeError(
                    f"layer {li}: Hessian '{g}' degenerate (max={mx}, finite={fin}); "
                    f"inp.absmax={float(inp.abs().max()):.4e} zero_samples={len(zs)}/{n_calib} "
                    f"first={zs[:5]} hstat={hstat} pinned={inp.is_pinned()}")

        # ---- quantize the 7 matrices ----
        tensors, lstats = {}, {}
        for gname, names in HGROUPS.items():
            Hm = Hs.pop(gname)
            for name in names:
                mod = layer.get_submodule(name)
                W = mod.weight.data
                K, N = W.shape
                assert N % 8 == 0 and N % a.group_size == 0, f"{name}: N={N}"
                sp = sover.get(name, a.sparsity)
                mb = 0 if (name in excl or sp == 0.0) else a.mask_block
                nb = bover.get(name, a.bits)
                q, scale, zero, c, mask, W_hat, st = cm.cobalt_quantize(
                    W, Hm, sp, a.beta, nb, a.group_size, a.hull, a.damping_frac,
                    mask_block=mb, quant_mode=a.quant_mode, clip=a.clip, actorder=a.actorder,
                    gptq_block=a.gptq_block, H_ho=Hs_ho.get(gname))
                if "eout_ho" in st:
                    ho_num += st["eout_ho_num"]; ho_den += st["eout_ho_den"]
                st["mask_block"] = mb
                st["sparsity_target"] = sp
                st["bits"] = nb
                # pack
                mu = mask.to(torch.uint8).view(K, N // 8, 8)
                wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=dev, dtype=torch.int32).view(1, 1, 8)
                packed = (mu.to(torch.int32) * wts).sum(-1).to(torch.uint8)
                tensors[f"{name}.mask"] = packed.cpu()
                tensors[f"{name}.q"] = q.to(torch.uint8).cpu()
                tensors[f"{name}.scale"] = scale.half().cpu()
                tensors[f"{name}.zero"] = zero.half().cpu()
                tensors[f"{name}.col_scale"] = c.half().cpu()
                st["shape"] = [K, N]
                lstats[name] = st
                nsurv = int(round((1.0 - st["sparsity_achieved"]) * K * N))
                tot_params += K * N; tot_surv += nsurv
                tot_bits_dense += nb * K * N; tot_bits_surv += nb * nsurv
                if a.seq_inputs:
                    mod.weight.data = W_hat.to(mod.weight.dtype)
                del q, scale, zero, c, mask, W_hat, mu, packed
                torch.cuda.empty_cache()
            del Hm
            Hs_ho.pop(gname, None)
            torch.cuda.empty_cache()

        # ---- optional sequential (compressed-prefix) propagation ----
        if a.seq_inputs:
            with torch.no_grad():
                for i in range(n_calib):
                    h = inp[i].to(dev).unsqueeze(0)
                    o = call_layer(layer, h, amask)
                    inp[i].copy_(o.squeeze(0))
                    del h, o
            torch.cuda.synchronize()

        save_file(tensors, os.path.join(a.out, f"layer_{li:02d}.safetensors"))
        manifest["layers"][str(li)] = lstats
        del tensors
        skel.layers[li] = torch.nn.Identity()
        del layer
        gc.collect(); torch.cuda.empty_cache()
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() // 2**20)
        rel = {k: round(v["relerr"], 4) for k, v in lstats.items()}
        eo = {k: round(v["eout_ratio"], 4) for k, v in lstats.items()}
        eho = {k: round(v["eout_ho"], 4) for k, v in lstats.items() if "eout_ho" in v}
        log(f"layer {li:02d}/{nlayers} done in {time.time()-t0:.1f}s (fwd {t_fwd:.1f}s) "
            f"peakGPU={peak_mem}MiB Hmax={ {g: '%.2e' % v[0] for g, v in hstat.items()} } "
            f"relerr={rel} eout={eo}" + (f" eout_ho={eho}" if eho else ""))
        if ho_den > 0:
            # energy-weighted total (dominated by the high-||X|| matrices) + the plain mean of per-matrix ratios
            manifest["eout_ho_total"] = ho_num / ho_den
            ehs = [v["eout_ho"] for L in manifest["layers"].values() for v in L.values() if "eout_ho" in v]
            ecs = [v["eout_ratio"] for L in manifest["layers"].values() for v in L.values()]
            manifest["eout_ho_mean"] = sum(ehs) / len(ehs)
            manifest["eout_calib_mean"] = sum(ecs) / len(ecs)
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)

    # ---- bpw accounting ----
    g = a.group_size
    n_groups = sum(s["shape"][0] * s["shape"][1] // g for L in manifest["layers"].values() for s in L.values())
    n_cols = sum(s["shape"][1] for L in manifest["layers"].values() for s in L.values())
    bits_meta = 32 * n_groups + 16 * n_cols          # fp16 scale + fp16 zero per group; fp16 col_scale
    bpw_dense = (tot_bits_dense + tot_params + bits_meta) / max(tot_params, 1)
    bpw_surv = (tot_bits_surv + tot_params + bits_meta) / max(tot_params, 1)
    manifest["bpw"] = dict(dense_codes=bpw_dense, survivor_codes_plus_bitmap=bpw_surv,
                           params=tot_params, survivors=tot_surv,
                           global_sparsity=1 - tot_surv / max(tot_params, 1))
    manifest["peak_gpu_mib"] = peak_mem
    manifest["wall_seconds"] = time.time() - t_all
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"DONE {nlayers} layers in {manifest['wall_seconds']:.0f}s peakGPU={peak_mem}MiB "
        f"bpw dense={bpw_dense:.3f} survivor={bpw_surv:.3f} sparsity={manifest['bpw']['global_sparsity']:.4f}"
        + (f" eout_ho_total={manifest['eout_ho_total']:.5f} eout_ho_mean={manifest['eout_ho_mean']:.5f} "
           f"eout_calib_mean={manifest['eout_calib_mean']:.5f}" if ho_den > 0 else ""))


if __name__ == "__main__":
    main()
