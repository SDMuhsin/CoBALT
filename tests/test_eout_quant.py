"""Unit tests for src/eout_quant.py (math4 empirical arms).

Checks, on synthetic data mirroring the deployed pipeline:
  1. dequant_deployed == sinq.dequantize_sparse_sinq (bit-parity with deployment)
  2. E_out(repack) and E_out(eout) are finite; E_out(eout) <= E_out(rtn) (guarantee)
  3. code_descent is monotone (E never increases across calls)
  4. budget width b' >= b and matches hand accounting
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import eout_quant as eq  # noqa: E402
from sinq.sparse_quant import dequantize_sparse_sinq  # noqa: E402
from sinq.dual_shift import quantize_rtn  # noqa: E402

torch.manual_seed(0)


def make_layer(K=64, N=256, T=128, sparsity=0.6, gsize=64):
    W = torch.randn(K, N)
    X = torch.randn(T, N) * (0.5 + torch.rand(N))  # heteroscedastic columns
    imp = W.abs() * X.norm(dim=0)
    thresh = imp.flatten().kthvalue(int(K * N * sparsity)).values
    mask = (imp > thresh).float()
    W_comp = W * mask  # stand-in for OBS output (zeros off-support)
    return W_comp, mask, X


def deployed_rtn(W_comp, mask, nbits, gsize, r, c):
    W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
    q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** nbits - 1], group_size=gsize)
    if scales.dim() == 3:
        scales = scales * r.view(-1, 1, 1)
    else:
        scales = scales * r.view(-1, 1)
    scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    q = q * mask
    return q, scales, zeros


def test_dequant_parity():
    W_comp, mask, X = make_layer()
    K, N = W_comp.shape
    r, c = torch.ones(K), torch.ones(N)
    q, s, z = deployed_rtn(W_comp, mask, 3, 64, r, c)
    ours = eq.dequant_deployed(q, s, z, mask, c)
    meta = {"group_size": 64}
    ref = dequantize_sparse_sinq(q, s, z, mask, c.view(1, -1), meta)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()
    print("PASS dequant parity, max |diff| =", (ours - ref).abs().max().item())


def test_arms_and_guarantee():
    for sp in (0.4, 0.6, 0.8):
        W_comp, mask, X = make_layer(sparsity=sp)
        K, N = W_comp.shape
        r = torch.ones(K)
        c = (0.5 + torch.rand(N))  # nontrivial per-column scale2 (norm='col' analog)
        Wn = W_comp / c.view(1, -1)
        q, s, z = deployed_rtn(W_comp, mask, 3, 64, r, c * 0 + 1)  # grids on W_comp
        # redo grids in normalized space as deployed does with norm='col'
        q, s, z = None, None, None
        q2, s2, z2 = deployed_rtn(W_comp, mask, 3, 64, r, c)
        for arm in ("repack", "eout"):
            W_best, info = eq.eout_requantize(W_comp, mask, X, 3, 64, r, c,
                                              q2, s2, z2, arm=arm)
            assert torch.isfinite(W_best).all()
            assert info["e_arm"] == info["e_arm"]  # not NaN
            if arm == "eout":
                assert info["e_arm"] <= info["e_rtn"] * (1 + 1e-9), info
            print(f"PASS sp={sp} arm={arm}: b'={info['bprime']} "
                  f"e_rtn={info['e_rtn']:.4f} e_repack={info['e_repack']:.4f} "
                  f"e_arm={info['e_arm']:.4f} picked={info['picked']} "
                  f"rel={info['e_arm']/max(info['e_rtn'],1e-30):.4f}")


def test_descent_monotone():
    W_comp, mask, X = make_layer(sparsity=0.6)
    K, N = W_comp.shape
    r, c = torch.ones(K), torch.ones(N)
    q, s, z = deployed_rtn(W_comp, mask, 3, 64, r, c)
    H = X.t() @ X
    bprime, _ = eq.budget_width(mask, 3, 64)
    e_prev = eq.eout_sq(eq.dequant_deployed(q, s, z, mask, c), W_comp, H)
    qq = q.float()
    for p in range(3):
        qq = eq.code_descent(qq, s.float(), z.float(), mask, r, c, W_comp, H,
                             bprime, 64, passes=1)
        e_now = eq.eout_sq(eq.dequant_deployed(qq, s, z, mask, c), W_comp, H)
        assert e_now <= e_prev * (1 + 1e-9), (p, e_prev, e_now)
        print(f"PASS descent pass {p}: {e_prev:.4f} -> {e_now:.4f}")
        e_prev = e_now


def test_budget_width():
    mask = torch.zeros(64, 256)
    mask[:, :64] = 1  # first group of each row survives fully -> k = 64*64
    bprime, k = eq.budget_width(mask, 3, 64)
    K, N, g = 64, 256, 64
    g_ne = 64  # one nonempty group per row
    b_codes = 3 * K * N + 32 * (K * N // g - g_ne)
    assert k == 64 * 64 and bprime == min(b_codes // k, eq.WIDTH_CAP), (bprime, k)
    print(f"PASS budget width: b'={bprime} (codes={b_codes}, k={k})")


if __name__ == "__main__":
    test_dequant_parity()
    test_budget_width()
    test_descent_monotone()
    test_arms_and_guarantee()
    print("ALL PASS")
