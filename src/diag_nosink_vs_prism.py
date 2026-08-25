#!/usr/bin/env python3
"""Pinpoint WHY nosink(inverse_mu+sinkhorn_sa) != real PRISM at matched plan.
Feed the SAME W and X through both per-matrix pipelines, compare stage by stage.
Runs in seconds on a few real gemma-2b matrices (no eval)."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
from sinq.sparse_quant import quantize_rtn  # noqa
import nosink  # noqa  (src/ is on path via _ROOT)
from transformers import AutoModelForCausalLM  # noqa

DEV = "cuda"
SP = 0.7033492822966506
NB = 3


def dequant_like(q, scales, zeros, mask, scale2):
    meta = {'group_size': 64, 'shape': tuple(q.shape)}
    return bs.dequantize_sparse_sinq(q, scales, zeros, mask, scale2, meta)


def run_matrix(name, W, sp=SP, normB='sinkhorn_sa'):
    N = W.shape[1]
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    X = (torch.randn(256, N, device=DEV) * 0.7).float()

    # ---- Path A: real PRISM ----
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    Wq, sA, zA, mA, s2A, metaA = bs.sparse_quantize_sinq(
        W.clone(), X.clone(), sparsity=sp, nbits=NB, method='sinq_wanda_inverse',
        device=DEV, use_compensation=True if sp > 0 else False,
        compensation_mode='prism', is_prenorm=False)
    WdeqA = dequant_like(Wq, sA, zA, mA, s2A)

    # ---- Path B: nosink cell D (inverse_mu mask + sinkhorn_sa norm) ----
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    W_comp, mB = nosink.inverse_mu_mask_and_obs(W.clone().float().to(DEV), X.clone(), sp, DEV)
    r, c = nosink.compute_norm_scales(W_comp, mB, normB, DEV)
    W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
    q, sB, zB, _ = quantize_rtn(W_norm, [0, 2 ** NB - 1], group_size=64)
    sB = sB * r.view(-1, 1, 1)
    sB_clamped = torch.nan_to_num(sB, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    q = q * mB.to(q.dtype)
    WdeqB = dequant_like(q.half(), sB_clamped.half(), zB.half(), mB.half(), c.half())
    WdeqB_noclamp = dequant_like(q.half(), sB.half(), zB.half(), mB.half(), c.half())

    Wf = W.float().to(DEV)
    def rel(a, b): return ((a - b).norm() / (b.norm() + 1e-9)).item()
    mask_match = torch.equal(mA.to(DEV).float(), mB.float())
    print(f"\n=== {name}  shape={tuple(W.shape)}  n_pruned A={int((mA==0).sum())} B={int((mB==0).sum())} mask_equal={mask_match}")
    print(f"  scales:   A.shape={tuple(sA.shape)} B.shape={tuple(sB.shape)}  "
          f"A[max]={sA.float().max():.3g} B[max]={sB.float().max():.3g}  clampfired={int((sB>6e4).sum())}")
    print(f"  scale2:   relA-B={rel(s2A.to(DEV).float(), c.float()):.3e}  A[max]={s2A.float().max():.3g} B[max]={c.float().max():.3g}")
    print(f"  recon err vs W:   PRISM_A={rel(WdeqA.to(DEV), Wf):.4f}   nosink_B={rel(WdeqB.to(DEV), Wf):.4f}   nosink_noclamp={rel(WdeqB_noclamp.to(DEV), Wf):.4f}")
    print(f"  A-vs-B dequant rel diff = {rel(WdeqA.to(DEV), WdeqB.to(DEV)):.4f}")


def main():
    name = bs.MODELS["gemma-2b"]
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    layer = bs.get_transformer_layers(model)[5]
    targets = [
        ("v_proj@0.0 normB=sinkhorn_sa", layer.self_attn.v_proj.weight.data, 0.0, 'sinkhorn_sa'),
        ("v_proj@0.0 normB=sinkhorn(STD)", layer.self_attn.v_proj.weight.data, 0.0, 'sinkhorn'),
        ("v_proj@0.0 normB=col", layer.self_attn.v_proj.weight.data, 0.0, 'col'),
    ]
    for nm, W, sp, normB in targets:
        run_matrix(nm, W.to(DEV), sp, normB)


if __name__ == "__main__":
    main()
