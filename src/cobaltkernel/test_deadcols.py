#!/usr/bin/env python
"""Dead-input-column drop: is it free?

A "dead" input column j of a matrix has keep(k,j)=0 for EVERY row k, so its
dequantised column is exactly zero and its stored codes are never used.

Two distinct readings of "physically drop dead columns BEFORE grouping":

  (A) STORAGE-ONLY drop -- remove the codes, keep the group boundaries in the
      original column index space (groups become 128 minus their dead count).
      Claim: EXACTLY zero quality cost. Verified here on real artifacts:
        A1  W_hat[:, dead] == 0 exactly (so removing the codes changes nothing)
        A2  the survivor-hull scale/zero of every affected group is UNCHANGED
            when the dead columns are excluded from the group (a dead column
            contributes no survivors, so amin/amax over survivors is identical)

  (B) REGROUP drop -- physically compact the matrix to N' = N - n_dead columns
      and then form fresh 128-groups. This CHANGES group membership, hence the
      survivor hulls, hence scale/zero, hence W_hat. NOT free; measured here.

Usage: python test_deadcols.py --art <raw artifact dir> [--layers 3]
"""
import argparse, os, sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cobalt_math as cm  # noqa: E402


def unpack_mask(packed, N):
    K = packed.shape[0]
    bits = torch.arange(8, dtype=torch.uint8, device=packed.device)
    m = (packed.unsqueeze(-1) >> bits.view(1, 1, 8)) & 1
    return m.reshape(K, -1)[:, :N].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    tot_par = tot_dead_par = 0
    a1_fail = a2_fail = 0
    b_rows = []
    for li in range(a.layers):
        p = os.path.join(a.art, f"layer_{li:02d}.safetensors")
        if not os.path.exists(p):
            continue
        with safe_open(p, framework="pt") as h:
            names = sorted({k.rsplit(".", 1)[0] for k in h.keys()})
            for n in names:
                q = h.get_tensor(f"{n}.q").to(dev)
                scale = h.get_tensor(f"{n}.scale").to(dev).float()
                zero = h.get_tensor(f"{n}.zero").to(dev).float()
                c = h.get_tensor(f"{n}.col_scale").to(dev).float()
                K, N = q.shape
                g = N // scale.shape[1]
                mask = unpack_mask(h.get_tensor(f"{n}.mask").to(dev), N)
                dead = (mask.sum(0) == 0)
                nd = int(dead.sum())
                tot_par += K * N
                tot_dead_par += K * nd
                W_hat = cm.dequant(q.float(), scale, zero, c, mask, g)

                # ---- A1: dequantised dead columns are exactly zero ----
                if nd and float(W_hat[:, dead].abs().max()) != 0.0:
                    a1_fail += 1

                # ---- A2: survivor hull of each group is unchanged by excluding dead cols ----
                # W_norm reconstructed on the survivor grid: (q - zero)*scale
                Wn = (q.view(K, N // g, g).float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
                Mg = mask.view(K, N // g, g).bool()
                Dg = dead.view(1, N // g, g).expand(K, -1, -1)
                big = torch.finfo(torch.float32).max
                def hull(msk):
                    lo = torch.where(msk, Wn, torch.full_like(Wn, big)).amin(-1)
                    hi = torch.where(msk, Wn, torch.full_like(Wn, -big)).amax(-1)
                    return lo, hi
                lo0, hi0 = hull(Mg)                    # all columns present
                lo1, hi1 = hull(Mg & ~Dg)              # dead columns removed from the group
                if not (torch.equal(lo0, lo1) and torch.equal(hi0, hi1)):
                    a2_fail += 1

                # ---- B: regroup after compaction -> does scale/zero change? ----
                if nd:
                    keepcols = ~dead
                    Wn_c = (Wn.view(K, N)[:, keepcols])
                    M_c = mask[:, keepcols].bool()
                    Np = Wn_c.shape[1]
                    ng2 = Np // g                      # drop the ragged tail for the probe
                    if ng2:
                        Wc = Wn_c[:, :ng2 * g].view(K, ng2, g)
                        Mc = M_c[:, :ng2 * g].view(K, ng2, g)
                        loc = torch.where(Mc, Wc, torch.full_like(Wc, big)).amin(-1)
                        hic = torch.where(Mc, Wc, torch.full_like(Wc, -big)).amax(-1)
                        ref_lo = lo0[:, :ng2]; ref_hi = hi0[:, :ng2]
                        chg = float(((loc != ref_lo) | (hic != ref_hi)).float().mean())
                        b_rows.append((n, nd, N, chg))
                del q, scale, zero, c, mask, W_hat, Wn, Mg, Dg
                torch.cuda.empty_cache() if dev.type == "cuda" else None

    print(f"layers probed: {a.layers}")
    print(f"dead-column params: {tot_dead_par/1e6:.2f} M / {tot_par/1e6:.1f} M = {100*tot_dead_par/max(tot_par,1):.4f}%")
    print(f"[A1] W_hat[:,dead] == 0 exactly         : {'FAIL x%d' % a1_fail if a1_fail else 'PASS (all matrices)'}")
    print(f"[A2] survivor hull unchanged by dropping: {'FAIL x%d' % a2_fail if a2_fail else 'PASS (all matrices)'}")
    if b_rows:
        print("[B] REGROUP-after-compaction: fraction of groups whose survivor hull CHANGES")
        agg = {}
        for n, nd, N, chg in b_rows:
            agg.setdefault(n, []).append(chg)
        for n, v in agg.items():
            print(f"     {n:20s} {100*sum(v)/len(v):6.2f}% of groups change hull")


if __name__ == "__main__":
    main()
