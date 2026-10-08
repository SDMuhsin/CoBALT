#!/usr/bin/env python
"""Block-wise reconstruction of a CoBALT raw artifact (zero-byte, zero-kernel-cost quantizer stage).

Takes a finished CoBALT artifact (mask + OBS-compensated, GPTQ/RTN-rounded codes + group scale/zero +
per-column scale) and, decoder block by decoder block, minimises the block output error against the bf16
model on calibration tokens by adjusting ONLY quantities the artifact already stores:

  * survivor codes      q    (integer 0..2^b-1, learned through a straight-through rounding estimator)
  * group scales        scale (fp16 per 128-group)
  * per-column scales   col_scale (fp16 per input column)

The CoBALT keep-mask is FIXED (pruned entries stay exactly 0), the group zero-point is FIXED (uint8 in the
kernel), the artifact schema / byte count / kernel layout are unchanged, so the packed kernel is byte-for-byte
the same size and runs at the same speed. Inputs to block i are the outputs of the already-reconstructed
blocks 0..i-1 (sequential propagation); targets are the bf16 block applied to bf16 inputs. Held-out calibration
blocks select the epoch (early stopping) -- the calibration output error is known to be ~2/3 overfit.

Output: a new artifact dir with the same layer_XX.safetensors schema + manifest (recon section appended).
"""
import argparse, copy, json, os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_cobalt as qc  # noqa: E402

LINEARS = qc.LINEARS


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def unpack_mask(packed, N):
    K = packed.shape[0]
    bits = torch.arange(8, dtype=torch.uint8, device=packed.device)
    m = (packed.unsqueeze(-1) >> bits.view(1, 1, 8)) & 1
    return m.reshape(K, -1)[:, :N].bool()


def pack_mask(mask_bool):
    K, N = mask_bool.shape
    mu = mask_bool.to(torch.uint8).view(K, N // 8, 8)
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=mask_bool.device, dtype=torch.int32).view(1, 1, 8)
    return (mu.to(torch.int32) * wts).sum(-1).to(torch.uint8)


def round_ste(x):
    return x + (torch.round(x) - x).detach()


class QLinear(nn.Module):
    """Dequantising linear with learnable codes (STE), log-scale and log-col-scale; fixed mask and zero."""

    def __init__(self, q, scale, zero, col_scale, mask, nbits, learn):
        super().__init__()
        K, N = q.shape
        self.K, self.N = K, N
        self.ng = scale.shape[1]
        self.g = N // self.ng
        self.nlev = 2 ** nbits - 1
        self.register_buffer("mask", mask.to(torch.bool))
        self.register_buffer("zero", zero.float())
        self.register_buffer("scale0", scale.float())
        self.register_buffer("col0", col_scale.float())
        lat = q.float()
        self.lat = nn.Parameter(lat, requires_grad="codes" in learn)
        self.ls = nn.Parameter(torch.zeros_like(self.scale0), requires_grad="scale" in learn)
        self.lc = nn.Parameter(torch.zeros_like(self.col0), requires_grad="col" in learn)

    def scale(self):
        return self.scale0 * torch.exp(self.ls)

    def col(self):
        return self.col0 * torch.exp(self.lc)

    def codes(self):
        return torch.clamp(round_ste(self.lat), 0.0, float(self.nlev))

    def weight_hat(self):
        K, N, ng, g = self.K, self.N, self.ng, self.g
        W = (self.codes().view(K, ng, g) - self.zero.view(K, ng, 1)) * self.scale().view(K, ng, 1)
        return W.view(K, N) * self.col().view(1, -1) * self.mask

    def forward(self, x):
        return F.linear(x, self.weight_hat().to(x.dtype))

    @torch.no_grad()
    def export(self):
        q = torch.clamp(torch.round(self.lat), 0, self.nlev)
        q = torch.where(self.mask, q, torch.zeros_like(q))
        return (q.to(torch.uint8), self.scale().half(), self.zero.half(), self.col().half(), self.mask)


def load_art_layer(art, li, dev):
    p = os.path.join(art, f"layer_{li:02d}.safetensors")
    out = {}
    with safe_open(p, framework="pt") as h:
        for n in LINEARS:
            out[n] = dict(q=h.get_tensor(f"{n}.q").to(dev), scale=h.get_tensor(f"{n}.scale").to(dev),
                          zero=h.get_tensor(f"{n}.zero").to(dev), col_scale=h.get_tensor(f"{n}.col_scale").to(dev),
                          mask_packed=h.get_tensor(f"{n}.mask").to(dev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--art", required=True, help="finished CoBALT raw artifact (init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", choices=["ultrachat", "wikitext2"], default="ultrachat")
    ap.add_argument("--calib-file", default="results/accel4bit/calib_med_512x2048.txt")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--heldout-calib", type=int, default=16)
    ap.add_argument("--calib-offset", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr-code", type=float, default=3e-3, help="Adam lr on the code latent (units of one grid step)")
    ap.add_argument("--lr-scale", type=float, default=2e-4, help="Adam lr on log group-scale")
    ap.add_argument("--lr-col", type=float, default=2e-4, help="Adam lr on log col-scale")
    ap.add_argument("--learn", default="codes,scale,col", help="subset of codes,scale,col")
    ap.add_argument("--inputs", choices=["quant", "fp"], default="quant",
                    help="block inputs: outputs of the reconstructed quantized prefix (default) or bf16 prefix")
    ap.add_argument("--layers-limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam-eps", type=float, default=1e-16)
    ap.add_argument("--no-tf32", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    learn = set(x for x in a.learn.split(",") if x)
    t_all = time.time()

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    man = json.load(open(os.path.join(a.art, "manifest.json")))
    nbits_default = man["config"]["bits"]
    cfg = qc.load_config(a.model_path)
    family = qc.family_of(cfg)
    nlayers = cfg.num_hidden_layers if a.layers_limit <= 0 else min(a.layers_limit, cfg.num_hidden_layers)
    log(f"recon: art={a.art} family={family} layers={nlayers} learn={sorted(learn)} inputs={a.inputs} "
        f"epochs={a.epochs} bs={a.bs} lr code/scale/col={a.lr_code}/{a.lr_scale}/{a.lr_col}")

    ids, calib_src, n_tot = qc.build_calib(a.model_path, a.calib, a.n_calib + a.heldout_calib, a.seq_len,
                                           a.calib_file, a.calib_offset)
    n_tr = min(a.n_calib, n_tot)
    n_ho = n_tot - n_tr
    log(f"calib: {n_tr} train + {n_ho} held-out x {a.seq_len} from {calib_src}")

    reader = qc.ShardReader(a.model_path)
    skel = qc.build_skeleton(cfg, family)
    embed = qc.materialize(skel.embed_tokens, dev)
    embed.load_state_dict({"weight": reader.get("model.embed_tokens.weight")})
    rot_g = skel.rotary_emb.to(dev)
    rot_l = skel.rotary_emb_local.to(dev) if family == "gemma3" else None

    H0, T = cfg.hidden_size, a.seq_len
    Xq = torch.empty(n_tot, T, H0, dtype=torch.bfloat16, device="cpu", pin_memory=True)   # quantized-prefix inputs
    Xf = torch.empty_like(Xq, pin_memory=True)                                               # bf16-prefix inputs
    Yf = torch.empty_like(Xq, pin_memory=True)                                               # bf16 block outputs
    with torch.no_grad():
        for i in range(n_tot):
            e = embed(ids[i].to(dev))
            Xq[i].copy_(e); Xf[i].copy_(e)
    embed.to("cpu"); del embed; torch.cuda.empty_cache()

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
            sw = getattr(cfg, "sliding_window", None)
            masks = {"llama": (create_sliding_window_causal_mask(**mk) if sw else create_causal_mask(**mk))}
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

    def run_all(layer, src, dst, amask, bs):
        with torch.no_grad():
            for i0 in range(0, n_tot, bs):
                i1 = min(i0 + bs, n_tot)
                h = src[i0:i1].to(dev, non_blocking=True)
                o = call_layer(layer, h, amask)
                dst[i0:i1].copy_(o)
                del h, o
        torch.cuda.synchronize()

    def val_loss(layer, amask, idx, bs):
        tot = 0.0; den = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i0 in range(0, len(idx), bs):
                sel = idx[i0:i0 + bs]
                h = Xq[sel].to(dev, non_blocking=True)
                y = Yf[sel].to(dev, non_blocking=True).float()
                o = call_layer(layer, h, amask).float()
                tot += float(((o - y) ** 2).sum()); den += float((y ** 2).sum())
        return tot / max(den, 1e-30)

    man_out = copy.deepcopy(man)
    man_out["recon"] = dict(init_art=a.art, calib_source=calib_src, n_calib=n_tr, heldout=n_ho,
                            calib_offset=a.calib_offset, epochs=a.epochs, bs=a.bs, lr_code=a.lr_code,
                            lr_scale=a.lr_scale, lr_col=a.lr_col, learn=sorted(learn), inputs=a.inputs,
                            seed=a.seed, adam_eps=a.adam_eps, layers={})
    man_out["config"]["method"] = man["config"].get("method", "") + " + block-recon(" + ",".join(sorted(learn)) + ")"
    tr_idx = torch.arange(n_tr)
    ho_idx = torch.arange(n_tr, n_tot)
    peak = 0

    for li in range(nlayers):
        t0 = time.time()
        layer = qc.materialize(skel.layers[li], dev)
        sd = reader.prefix(f"model.layers.{li}.")
        layer.load_state_dict(sd, strict=True)
        del sd
        layer.eval()
        amask = mask_for(layer)
        # ---- bf16 targets from bf16-prefix inputs (and the next bf16-prefix inputs) ----
        run_all(layer, Xf, Yf, amask, a.bs)
        # ---- swap the 7 linears for dequantising modules initialised from the artifact ----
        art = load_art_layer(a.art, li, dev)
        qmods = {}
        for name in LINEARS:
            parent_name, _, child = name.rpartition(".")
            parent = layer.get_submodule(parent_name)
            lin = getattr(parent, child)
            t = art[name]
            nb = man["layers"][str(li)][name].get("bits", nbits_default)
            m = unpack_mask(t["mask_packed"], t["q"].shape[1])
            qm = QLinear(t["q"], t["scale"], t["zero"], t["col_scale"], m, nb, learn).to(dev)
            setattr(parent, child, qm)
            qmods[name] = qm
            del lin
        del art
        torch.cuda.empty_cache()
        # freeze everything that is not a QLinear learnable
        for n_, p in layer.named_parameters():
            if not any(n_.endswith(s) for s in (".lat", ".ls", ".lc")):
                p.requires_grad_(False)
        groups = []
        if "codes" in learn:
            groups.append(dict(params=[m_.lat for m_ in qmods.values()], lr=a.lr_code))
        if "scale" in learn:
            groups.append(dict(params=[m_.ls for m_ in qmods.values()], lr=a.lr_scale))
        if "col" in learn:
            groups.append(dict(params=[m_.lc for m_ in qmods.values()], lr=a.lr_col))
        opt = torch.optim.Adam(groups, betas=(0.9, 0.99), eps=a.adam_eps)   # grads on the code latent are ~1e-8: eps=1e-8 would mute them
        steps_per_epoch = (n_tr + a.bs - 1) // a.bs
        total_steps = steps_per_epoch * a.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / max(total_steps, 1)))

        v0 = val_loss(layer, amask, ho_idx, a.bs) if n_ho else float("nan")
        tr0 = val_loss(layer, amask, tr_idx[: min(n_tr, 16)], a.bs)
        best = (v0 if n_ho else tr0)
        best_state = {k: v.detach().clone() for k, v in layer.state_dict().items()
                      if k.endswith((".lat", ".ls", ".lc"))}
        best_ep = 0
        hist = [dict(epoch=0, train=tr0, val=v0)]
        g = torch.Generator().manual_seed(a.seed + li)
        for ep in range(1, a.epochs + 1):
            perm = tr_idx[torch.randperm(n_tr, generator=g)]
            run_loss = 0.0; nb_ = 0
            for i0 in range(0, n_tr, a.bs):
                sel = perm[i0:i0 + a.bs]
                h = Xq[sel].to(dev, non_blocking=True)
                y = Yf[sel].to(dev, non_blocking=True).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = call_layer(layer, h, amask)
                loss = F.mse_loss(o.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                run_loss += loss.item(); nb_ += 1
                del h, y, o, loss
            v = val_loss(layer, amask, ho_idx, a.bs) if n_ho else float("nan")
            trl = val_loss(layer, amask, tr_idx[: min(n_tr, 16)], a.bs)
            crit = v if n_ho else trl
            hist.append(dict(epoch=ep, train=trl, val=v, run_mse=run_loss / max(nb_, 1)))
            if crit < best:
                best, best_ep = crit, ep
                best_state = {k: v_.detach().clone() for k, v_ in layer.state_dict().items()
                              if k.endswith((".lat", ".ls", ".lc"))}
        # restore the best epoch (early stopping on the held-out blocks)
        with torch.no_grad():
            sd_ = layer.state_dict()
            for k, v_ in best_state.items():
                sd_[k].copy_(v_)
        vb = val_loss(layer, amask, ho_idx, a.bs) if n_ho else float("nan")
        # ---- export the layer + propagate the quantized-prefix inputs ----
        tensors, lstats = {}, {}
        W_fp = reader.prefix(f"model.layers.{li}.")
        for name, qm in qmods.items():
            q, sc, zr, cs, m = qm.export()
            tensors[f"{name}.mask"] = pack_mask(m).cpu()
            tensors[f"{name}.q"] = q.cpu()
            tensors[f"{name}.scale"] = sc.cpu()
            tensors[f"{name}.zero"] = zr.cpu()
            tensors[f"{name}.col_scale"] = cs.cpu()
            W = W_fp[f"{name}.weight"].to(dev).float()
            What = qm.weight_hat().detach().float()
            st = dict(man["layers"][str(li)][name])
            st["relerr_init"] = st.get("relerr")
            st["relerr"] = float((What - W).norm() / W.norm().clamp(min=1e-12))
            lstats[name] = st
            del W, What
        # fraction of changed codes (one pass, cheap)
        art0 = load_art_layer(a.art, li, dev)
        for name, qm in qmods.items():
            q0 = art0[name]["q"]
            q1 = tensors[f"{name}.q"].to(dev)
            msk = qm.mask
            lstats[name]["codes_changed_frac"] = float(((q0 != q1) & msk).float().sum() / msk.float().sum().clamp(min=1))
            lstats[name]["scale_logchange_rms"] = float(qm.ls.detach().pow(2).mean().sqrt())
            lstats[name]["col_logchange_rms"] = float(qm.lc.detach().pow(2).mean().sqrt())
        del art0, W_fp
        save_file(tensors, os.path.join(a.out, f"layer_{li:02d}.safetensors"))
        man_out["layers"][str(li)] = lstats
        man_out["recon"]["layers"][str(li)] = dict(val_init=v0, val_best=vb, best_epoch=best_ep, train_init=tr0, hist=hist)
        if a.inputs == "quant":
            run_all(layer, Xq, Xq, amask, a.bs)       # reconstructed block on quantized-prefix inputs
        else:
            Xq.copy_(Yf)
        Xf.copy_(Yf)                                   # bf16 prefix advances
        skel.layers[li] = nn.Identity()
        del layer, qmods, opt
        torch.cuda.empty_cache()
        peak = max(peak, torch.cuda.max_memory_allocated() // 2**20)
        ch = {n.split(".")[-1]: round(s["codes_changed_frac"], 3) for n, s in lstats.items()}
        log(f"layer {li:02d}/{nlayers} {time.time()-t0:.0f}s peak={peak}MiB val {v0:.5f} -> {vb:.5f} "
            f"({(1-vb/v0)*100 if v0 > 0 else 0:+.1f}%, best ep {best_ep}/{a.epochs}) train0 {tr0:.5f} codes_changed={ch}")
        man_out["recon"]["layers_done"] = li + 1
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump(man_out, f, indent=1)
    man_out["recon"]["wall_seconds"] = time.time() - t_all
    man_out["recon"]["peak_gpu_mib"] = peak
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man_out, f, indent=1)
    log(f"DONE {nlayers} layers in {time.time()-t_all:.0f}s peak={peak}MiB -> {a.out}")


if __name__ == "__main__":
    main()
