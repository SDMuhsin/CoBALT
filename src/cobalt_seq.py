#!/usr/bin/env python3
"""CoBALT-seq: the cross-layer v2 of CoBALT .

The submitted CoBALT compresses every block from DENSE-model calibration inputs and a single global
column-balance exponent beta, while its own Section 6 shows the failure mode is CROSS-layer: interior
error amplifying through depth past ratio 1. This module closes that loop with ONE sequential pass:

  for block l = 1..L (prefix 1..l-1 already compressed):
    1. run calibration through the COMPRESSED prefix -> the block's actual inputs X^_l (per linear)
       and, from a one-time dense pass, the dense block output h_l (the target).
    2. for each candidate beta in a small set: compress every linear of block l with the CoBALT recipe
       (balanced mask at beta -> OBS on X^_l -> col-norm group-RTN, identical to nosink's deployed
       tail), run the batch through the candidate block, measure the PROPAGATED block-output error
         r_l(beta) = ||h^_l(beta) - h_l||_F / ||h_l||_F
       (the quantity Table 9 of the paper uses as the collapse signature).
    3. commit the beta with the smallest r_l(beta); move on.

Per-block choice is closed-form (argmin over a finite set), there is no alternation or iteration to
convergence, the bit budget and sparsity are identical to CoBALT (beta does not change bpw), and the
only new ingredient is the sequential, propagated-error selection. `betas=(b,)` gives the
attribution arm cobalt-seqfix (sequential inputs, fixed beta) that isolates the compressed-prefix
compensation from the per-block beta selection.
"""
import copy
import gc
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa: E402
import nosink as ns  # noqa: E402
from sinq.sparse_quant import quantize_rtn  # noqa: E402


class _Stop(Exception):
    pass


def _get_linear(block, attr_path):
    parent = block
    parts = attr_path.split('.')
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None, None
        parent = getattr(parent, p)
    if not hasattr(parent, parts[-1]):
        return None, None, None
    lin = getattr(parent, parts[-1])
    return (parent, parts[-1], lin) if isinstance(lin, (nn.Linear, bs.SparseQuantLinear)) else (None, None, None)


def _block_out(o):
    return o[0] if isinstance(o, tuple) else o


def _run_to_block(model, blocks, li, batches, device, capture_paths=None):
    """Forward each batch through the model up to and including block li (prefix already compressed).
    Returns (list of block-li outputs [B,T,H] float cpu, dict attr_path -> concatenated inputs [tokens,N] cpu)."""
    outs, xin = [], {}
    hooks = []
    if capture_paths:
        for ap in capture_paths:
            _, _, lin = _get_linear(blocks[li], ap)
            if lin is None:
                continue

            def mk(ap_):
                def pre(mod, inp):
                    x = inp[0].detach
                    xin.setdefault(ap_, []).append(x.reshape(-1, x.shape[-1]).float.cpu)
                return pre
            hooks.append(lin.register_forward_pre_hook(mk(ap)))

    def out_hook(mod, inp, out):
        outs.append(_block_out(out).detach.float.cpu)
        raise _Stop
    hooks.append(blocks[li].register_forward_hook(out_hook))
    model.eval
    with torch.no_grad:
        for b in batches:
            try:
                model(b.to(device))
            except _Stop:
                pass
    for h in hooks:
        h.remove
    xin = {k: torch.cat(v, 0) for k, v in xin.items}
    return outs, xin


def _compress_block(block, dense_block, paths, xin, sp, beta, nbits, gsize, norm, device, quantizer='rtn'):
    """CoBALT recipe on every linear of `block` (restored from `dense_block` first), inputs from xin."""
    for ap in paths:
        parent, name, lin_d = _get_linear(dense_block, ap)
        if lin_d is None or not isinstance(lin_d, nn.Linear):
            continue
        X = xin.get(ap)
        if X is None:
            continue
        W = lin_d.weight.data.clone.float.to(device)
        bias = lin_d.bias.data.clone if lin_d.bias is not None else None
        Xd = X.to(device)
        if sp > 0.0:
            W_comp, mask = ns.balanced_mask_and_obs(W, Xd, sp, device, col_exp=beta)
        else:
            W_comp, mask = W, torch.ones_like(W)
        act_abs = Xd[:min(Xd.shape[0], 256)].abs.mean(0)
        K, N = W_comp.shape
        r, c = ns.compute_norm_scales(W_comp, mask, norm, device, act_abs=act_abs)
        W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
        q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** nbits - 1], group_size=gsize)
        scales = scales * (r.view(-1, 1, 1) if scales.dim == 3 else r.view(-1, 1))
        scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
        q = q * mask.to(q.dtype)
        if quantizer != 'rtn' and sp > 0.0:
            # SAME mask/OBS/norm/bits/group as the RTN path -- only the survivor SCALE selection
            # changes (awclip). bpw-identical. Mirrors nosink.apply_wanda_obs_rtn's quantizer branch
            # so the sequential arm gets exactly the lever the non-sequential awclip arm gets.
            import eout_quant as eq
            W_best, _info = eq.eout_requantize(W_comp.float, mask, Xd.float, nbits, gsize,
                                               r, c, q, scales.float, zeros.float, arm=quantizer)
            lin_new = nn.Linear(N, K, bias=bias is not None)
            lin_new.weight.data = W_best.half
            if bias is not None:
                lin_new.bias.data = bias.half
            lin_new = lin_new.to(device)
            tparent, tname, _ = _get_linear(block, ap)
            setattr(tparent, tname, lin_new)
            del W, W_comp, mask, q, scales, zeros, Xd, W_best
            continue
        meta = {'sparsity': sp, 'nbits': nbits, 'requested_nbits': nbits, 'group_size': gsize,
                'shape': (K, N), 'method': f'cobalt_seq_{norm}', 'beta': beta}
        new = bs.SparseQuantLinear(q.half, scales.half, zeros.half, mask.half, c.half,
                                   bias.to(device) if bias is not None else None, meta).to(device)
        tparent, tname, _ = _get_linear(block, ap)
        setattr(tparent, tname, new)
        del W, W_comp, mask, q, scales, zeros, Xd
    torch.cuda.empty_cache


def apply_cobalt_seq(model, calibration_data, nbits, sparsity, device='cuda', betas=(0.0, 0.3, 0.5, 0.7, 1.0),
                     group_size=128, norm='col', n_batches=None, log_path=None, tag='', quantizer='rtn'):
    """Sequential, propagated-error beta selection. Returns (model, chosen_betas[list], relerr[list])."""
    betas = tuple(float(b) for b in betas)
    blocks = bs.get_transformer_layers(model)
    paths = bs.get_layer_paths(model)
    model.config.use_cache = False
    nb = len(calibration_data) if n_batches is None else min(n_batches, len(calibration_data))
    batches = [calibration_data[i:i + 1] for i in range(nb)]
    L = len(blocks)
    # dense targets h_l for every block: one pass with the WHOLE model on device (embeddings, rotary,
    # norms, lm_head must share the device with the blocks), stopping after the last block; the dense
    # blocks are then offloaded and brought back one at a time (compressed blocks stay resident).
    model.to(device)
    caps = {}
    hooks = []
    for i, blk in enumerate(blocks):
        def mk(i_):
            def h(mod, inp, out):
                caps.setdefault(i_, []).append(_block_out(out).detach.float.cpu)
                if i_ == L - 1:
                    raise _Stop
            return h
        hooks.append(blk.register_forward_hook(mk(i)))
    model.eval
    with torch.no_grad:
        for b in batches:
            try:
                model(b.to(device))
            except _Stop:
                pass
    for h in hooks:
        h.remove
    h_dense = {i: caps[i] for i in range(L)}
    for blk in blocks:
        blk.to('cpu')
    torch.cuda.empty_cache

    chosen, rel = [], []
    for li in range(L):
        blk = blocks[li].to(device)
        dense_copy = copy.deepcopy(blk).to('cpu')          # restore source for each candidate
        # 1. inputs of every linear of block li from the COMPRESSED prefix
        _, xin = _run_to_block(model, blocks, li, batches, device, capture_paths=paths)
        hd = h_dense[li]
        den = sum(float(h.pow(2).sum) for h in hd)
        best = (None, float('inf')); cand = {}
        for beta in betas:
            _compress_block(blk, dense_copy, paths, xin, sparsity, beta, nbits, group_size, norm, device, quantizer=quantizer)
            if len(betas) == 1:
                best = (beta, float('nan')); break
            outs, _ = _run_to_block(model, blocks, li, batches, device)
            num = sum(float((o - h).pow(2).sum) for o, h in zip(outs, hd))
            r = (num / max(den, 1e-30)) ** 0.5
            cand[beta] = r
            if r < best[1]:
                best = (beta, r)
        if cand and log_path:   # DIAGNOSTIC: full r_l(beta) row per block (does beta have leverage?)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path.replace('.tsv', '_cand.tsv'), 'a') as f:
                f.write(f"{tag}\tbits={nbits}\tsp={sparsity}\tblock={li}\t" +
                        "\t".join(f"b{b:g}={cand[b]:.5f}" for b in betas) + "\n")
        if len(betas) > 1 and best[0] != betas[-1]:        # last candidate is still in place; re-commit best
            _compress_block(blk, dense_copy, paths, xin, sparsity, best[0], nbits, group_size, norm, device, quantizer=quantizer)
        chosen.append(best[0]); rel.append(best[1])
        print(f"[cobalt-seq] block {li:02d}: beta*={best[0]:g} relerr={best[1]:.4f}", flush=True)
        del dense_copy, xin
        gc.collect; torch.cuda.empty_cache
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(f"{tag}\tbits={nbits}\tsp={sparsity}\tbetas=" + ",".join(f"{b:g}" for b in chosen)
                    + "\trelerr=" + ",".join(f"{r:.4f}" for r in rel) + "\n")
    return model, chosen, rel


# ----------------------------------------------------------------------------------------------------
# SALVAGE v2 , after the diagnosis that per-block beta has ~no leverage on the PROPAGATED
# error: r_l(beta) differs by 0.1-0.5% across beta for blocks >= 3 on tinyllama sp0.6/3b, because the
# propagated error is dominated by the ACCUMULATED prefix error (0.12 -> 0.82 monotone), which beta
# cannot touch. The knob that CAN touch it is error CORRECTION: fit each block's weights to reproduce
# the DENSE output from the COMPRESSED input (W_ls = W (X^T X^)(X^T X^ + lam I)^-1, one ridge solve),
# then prune/quantize the corrected weights with the unchanged CoBALT recipe. The per-block knob becomes
# the correction strength alpha in {0, 0.5, 1}; because the ridge fit can overfit 8k calibration tokens
# (N up to 16k), alpha is selected by the propagated error on HELD-OUT text (ptb), not calibration.
# ----------------------------------------------------------------------------------------------------

def _capture_block_inputs(model, blocks, li, batches, device):
    """(args, kwargs) each batch presents to block li through the CURRENT (compressed-prefix) model."""
    caught = []

    def pre(mod, args, kwargs):
        caught.append((tuple(a.detach if torch.is_tensor(a) else a for a in args),
                       {k: (v.detach if torch.is_tensor(v) else v) for k, v in kwargs.items}))
        raise _Stop
    h = blocks[li].register_forward_pre_hook(pre, with_kwargs=True)
    model.eval
    with torch.no_grad:
        for b in batches:
            try:
                model(b.to(device))
            except _Stop:
                pass
    h.remove
    return caught


def _block_forward(block, inputs, capture_paths=None):
    """Run a block standalone on captured (args, kwargs); return (outputs list, dict linear inputs)."""
    xin, hooks = {}, []
    if capture_paths:
        for ap in capture_paths:
            _, _, lin = _get_linear(block, ap)
            if lin is None:
                continue

            def mk(ap_):
                def pre(mod, inp):
                    x = inp[0].detach
                    xin.setdefault(ap_, []).append(x.reshape(-1, x.shape[-1]).float)
                return pre
            hooks.append(lin.register_forward_pre_hook(mk(ap)))
    outs = []
    with torch.no_grad:
        for args, kwargs in inputs:
            outs.append(_block_out(block(*args, **kwargs)).detach.float)
    for h in hooks:
        h.remove
    return outs, {k: torch.cat(v, 0) for k, v in xin.items}


def _with_hidden(inputs, hiddens):
    """Same kwargs (mask / position embeddings) but a different hidden_states tensor per batch."""
    out = []
    for (args, kwargs), hnew in zip(inputs, hiddens):
        if args:
            out.append(((hnew.to(args[0].dtype),) + tuple(args[1:]), kwargs))
        else:
            kw = dict(kwargs); kw['hidden_states'] = hnew.to(kwargs['hidden_states'].dtype); out.append((kw))
    return out


def _corrected_weight(W, X, Xh, alpha, lam_frac):
    """W_alpha = W + alpha (W_ls - W), W_ls = argmin ||X^ W'^T - X W^T||_F (ridge, lam = lam_frac*mean diag)."""
    if alpha <= 0.0:
        return W
    # Ridge TOWARD W (not toward zero): argmin ||X^ W'^T - X W^T||^2 + lam ||W' - W||^2
    #   => W' = W (X^T X^ + lam I)(X^T X^ + lam I)^-1, which is exactly W when X^ = X (block 0) and tends
    # to the plain least-squares fit as lam -> 0. (First version omitted the lam I in the numerator and
    # shrank W in low-variance directions even with no incoming error -- monotone damage from block 0.)
    G = Xh.t @ Xh
    lam = lam_frac * G.diagonal.mean.clamp(min=1e-8)
    G.diagonal.add_(lam)
    C = X.t @ Xh                                   # [N,N]: X^T X^
    C.diagonal.add_(lam)
    M = torch.linalg.solve(G, C.t).t             # (C + lam I) (G + lam I)^-1
    W_ls = W @ M
    del G, C, M
    return W + alpha * (W_ls - W)


def _compress_block_ec(block, dense_block, paths, X_d, X_h, sp, beta, alpha, nbits, gsize, norm, device, lam_frac):
    for ap in paths:
        _, _, lin_d = _get_linear(dense_block, ap)
        if lin_d is None or not isinstance(lin_d, nn.Linear) or ap not in X_h:
            continue
        W = lin_d.weight.data.clone.float.to(device)
        bias = lin_d.bias.data.clone if lin_d.bias is not None else None
        Xh = X_h[ap]; Xd = X_d[ap]
        Wc = _corrected_weight(W, Xd, Xh, alpha, lam_frac)
        if sp > 0.0:
            W_comp, mask = ns.balanced_mask_and_obs(Wc, Xh, sp, device, col_exp=beta)
        else:
            W_comp, mask = Wc, torch.ones_like(Wc)
        act_abs = Xh[:min(Xh.shape[0], 256)].abs.mean(0)
        K, N = W_comp.shape
        r, c = ns.compute_norm_scales(W_comp, mask, norm, device, act_abs=act_abs)
        W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
        q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** nbits - 1], group_size=gsize)
        scales = scales * (r.view(-1, 1, 1) if scales.dim == 3 else r.view(-1, 1))
        scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
        q = q * mask.to(q.dtype)
        meta = {'sparsity': sp, 'nbits': nbits, 'requested_nbits': nbits, 'group_size': gsize,
                'shape': (K, N), 'method': f'cobalt_ec_{norm}', 'beta': beta, 'alpha': alpha}
        new = bs.SparseQuantLinear(q.half, scales.half, zeros.half, mask.half, c.half,
                                   bias.to(device) if bias is not None else None, meta).to(device)
        tparent, tname, _ = _get_linear(block, ap)
        setattr(tparent, tname, new)
        del W, Wc, W_comp, mask, q, scales, zeros
    torch.cuda.empty_cache


def apply_cobalt_ec(model, calibration_data, nbits, sparsity, device='cuda', alphas=(0.0, 0.5, 1.0),
                    betas=(0.5,), heldout_data=None, group_size=128, norm='col', lam_frac=0.01,
                    log_path=None, tag='', lams=None):
    """Error-correcting sequential CoBALT. Per block: ridge-fit W to map compressed inputs to the DENSE
    output (strength alpha), then the unchanged CoBALT mask/OBS/quant tail; (alpha, beta) chosen per block
    by the propagated block-output error on HELD-OUT text (calibration text if heldout_data is None).
    Returns (model, chosen [(alpha,beta)], relerr)."""
    blocks = bs.get_transformer_layers(model)
    paths = bs.get_layer_paths(model)
    model.config.use_cache = False
    cal_b = [calibration_data[i:i + 1] for i in range(len(calibration_data))]
    sel_b = ([heldout_data[i:i + 1] for i in range(len(heldout_data))] if heldout_data is not None else cal_b)
    L = len(blocks)
    lams = tuple(lams) if lams else (lam_frac,)
    # candidate = (alpha, beta, lam); alpha=0 needs no lam (dedupe)
    cands = []
    for a in alphas:
        for b in betas:
            for lm in (lams if a > 0.0 else (lams[0],)):
                cands.append((a, b, lm))
    # one dense pass on BOTH sets: block inputs (args/kwargs) and block outputs, per block, kept on CPU
    model.to(device)
    d_in, d_out = {}, {}
    for name, batches in (("cal", cal_b), ("sel", sel_b)):
        ins, outs, hooks = {}, {}, []
        for i, blk in enumerate(blocks):
            def mkp(i_):
                def pre(mod, args, kwargs):
                    ins.setdefault(i_, []).append((tuple(a.detach.cpu if torch.is_tensor(a) else a for a in args),
                                                   {k: (v.detach.cpu if torch.is_tensor(v) else v) for k, v in kwargs.items}))
                return pre

            def mko(i_):
                def h(mod, inp, out):
                    outs.setdefault(i_, []).append(_block_out(out).detach.float.cpu)
                    if i_ == L - 1:
                        raise _Stop
                return h
            hooks.append(blk.register_forward_pre_hook(mkp(i), with_kwargs=True))
            hooks.append(blk.register_forward_hook(mko(i)))
        model.eval
        with torch.no_grad:
            for b in batches:
                try:
                    model(b.to(device))
                except _Stop:
                    pass
        for h in hooks:
            h.remove
        d_in[name], d_out[name] = ins, outs
    for blk in blocks:
        blk.to('cpu')
    torch.cuda.empty_cache

    def to_dev(inputs):
        return [(tuple(a.to(device) if torch.is_tensor(a) else a for a in args),
                 {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items}) for args, kwargs in inputs]

    chosen, rel = [], []
    for li in range(L):
        blk = blocks[li].to(device)
        dense_copy = copy.deepcopy(blk)
        # compressed-prefix inputs to this block, for the calibration set (fit) and the selection set (score)
        c_in = _capture_block_inputs(model, blocks, li, cal_b, device)
        s_in = _capture_block_inputs(model, blocks, li, sel_b, device) if heldout_data is not None else c_in
        # dense-block linear inputs from DENSE hidden (X) and from COMPRESSED hidden (X^), same tokens
        _, X_d = _block_forward(dense_copy, to_dev(d_in["cal"][li]), capture_paths=paths)
        _, X_h = _block_forward(dense_copy, c_in, capture_paths=paths)
        h_ref = [h.to(device) for h in d_out["sel" if heldout_data is not None else "cal"][li]]
        den = sum(float(h.pow(2).sum) for h in h_ref)
        best, scores = (None, float('inf')), {}
        for (a, b, lm) in cands:
            _compress_block_ec(blk, dense_copy, paths, X_d, X_h, sparsity, b, a, nbits, group_size, norm, device, lm)
            outs, _ = _block_forward(blk, s_in)      # scored even for a single candidate (diagnostic trace)
            num = sum(float((o - h).pow(2).sum) for o, h in zip(outs, h_ref))
            r = (num / max(den, 1e-30)) ** 0.5
            scores[(a, b, lm)] = r
            if r < best[1]:
                best = ((a, b, lm), r)
        if len(cands) > 1 and best[0] != cands[-1]:
            a, b, lm = best[0]
            _compress_block_ec(blk, dense_copy, paths, X_d, X_h, sparsity, b, a, nbits, group_size, norm, device, lm)
        chosen.append(best[0]); rel.append(best[1])
        _lab = lambda c: f"a{c[0]:g}b{c[1]:g}l{c[2]:g}"
        print(f"[cobalt-ec] block {li:02d}: best={_lab(best[0])} relerr={best[1]:.4f} | "
              + " ".join(f"{_lab(c)}={s:.4f}" for c, s in scores.items), flush=True)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(f"{tag}\tbits={nbits}\tsp={sparsity}\tblock={li}\tbest={_lab(best[0])}\t"
                        + "\t".join(f"{_lab(c)}={s:.5f}" for c, s in scores.items) + "\n")
        del dense_copy, X_d, X_h, c_in, s_in, h_ref
        gc.collect; torch.cuda.empty_cache
    return model, chosen, rel
