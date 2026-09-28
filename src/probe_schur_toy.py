#!/usr/bin/env python3
"""Exhaustive small-instance analysis of the beta-family, for llmdocs/spectral_v2.

The mask M(beta) is piecewise constant in beta, so on a small instance the WHOLE family can be
enumerated exactly: compute every crossing beta, then evaluate Phi once per resulting interval.

Four modes.

  --mode search   Search random integer instances for one on which the Schur bound prefers
                  beta = 0 to the mid-range exponent, i.e. Phi(0) < Phi(0.5). NOTE: "Phi(0)
                  strictly below Phi(beta) for EVERY beta > 0" is impossible -- Phi is constant
                  on [0, beta_1) below the first breakpoint -- so that is not the criterion.
                  Prints the winning instance and its exact curve. This is a SELECTED instance:
                  it is a counterexample to a universal claim, and is reported as such.

  --mode sweep    Report the distribution of argmin_beta Phi over random instances, so the
                  frequency statement in the document has a committed artifact. The protocol
                  (shape, sparsity, integer range, draws, seed) is printed with the result.

Importances are integers and the row/column normalisations are exact rationals, so all
orderings and all C1/Cinf sums are exact; only the crossing betas involve logarithms, and each
interval is probed at its midpoint, which is robust to that.

Usage:
  python src/probe_schur_toy.py --mode search --K 4 --N 4 --s 0.5 --hi 30 --tries 60000 --seed 7
  python src/probe_schur_toy.py --mode sweep  --K 4 --N 4 --s 0.5 --hi 30 --tries 20000 --seed 1
"""
import argparse
import math
import random
from collections import Counter
from fractions import Fraction as F


def row_norm(I, K, N, kr):
    """Itil[i][j] = I[i][j] / (kr-th smallest of row i). Exact rationals."""
    out = []
    for i in range(K):
        q = sorted(I[i])[kr - 1]
        out.append([F(I[i][j], 1) / F(q, 1) for j in range(N)])
    return out


def col_quantiles(It, K, N, kc):
    return [sorted(It[i][j] for i in range(K))[kc - 1] for j in range(N)]


def mask_at(It, qc, K, N, n, beta):
    """Pruned set at exponent beta, replicating the deployed rule.

    _threshold_mask keeps iff score > tau, where tau is the n-th smallest score, so EVERY
    entry tied at tau is pruned and the pruned set can exceed n. We reproduce that here
    rather than taking a fixed n smallest with an arbitrary index tie-break.
    """
    vals = []
    for i in range(K):
        for j in range(N):
            v = math.log(float(It[i][j])) - beta * math.log(float(qc[j]))
            vals.append((v, i, j))
    vals.sort()
    tau = vals[n - 1][0]
    return set((i, j) for v, i, j in vals if v <= tau)


def schur(P, I, K, N):
    """C1 = max pruned column sum, Cinf = max pruned row sum, on the raw importance I."""
    cols, rows = [0] * N, [0] * K
    for (i, j) in P:
        cols[j] += I[i][j]
        rows[i] += I[i][j]
    C1, Cinf = max(cols), max(rows)
    return C1, Cinf, math.sqrt(C1 * Cinf)


def breakpoints(It, qc, K, N):
    """Crossings in the OPEN interval (1e-9, 1-1e-9) between entries in DIFFERENT columns.

    The guards exclude crossings within 1e-9 of an endpoint. That is deliberate: a crossing at
    beta = 0 exactly is generic (Theorem beta0 places K entries of Itilde at the value 1), and
    such a crossing bounds no interval to probe. Callers that need "every crossing in [0,1]"
    must treat the endpoints separately -- which the sweep does, by probing 0 and 1 themselves.
    """
    bs = set()
    ent = [(i, j) for i in range(K) for j in range(N)]
    for a in range(len(ent)):
        for b in range(a + 1, len(ent)):
            i, j = ent[a]
            p, q = ent[b]
            if j == q:
                continue                       # same column: order is beta-invariant
            lq = math.log(float(qc[j])) - math.log(float(qc[q]))
            if abs(lq) < 1e-15:
                continue                       # equal column quantiles: never cross
            beta = (math.log(float(It[i][j])) - math.log(float(It[p][q]))) / lq
            if 1e-9 < beta < 1 - 1e-9:
                bs.add(round(beta, 12))
    return sorted(bs)


def curve_of(I, K, N, s):
    """Exact Phi curve: one probe per inter-breakpoint interval, plus both endpoints."""
    kr, kc, n = int(N * s), int(K * s), int(K * N * s)
    It = row_norm(I, K, N, kr)
    qc = col_quantiles(It, K, N, kc)
    bps = breakpoints(It, qc, K, N)
    probes = [0.0, 1.0] + bps
    probes += [(a + b) / 2 for a, b in zip([0.0] + bps, bps + [1.0])]
    probes = sorted(set(round(p, 12) for p in probes))
    curve = []
    for beta in probes:
        P = mask_at(It, qc, K, N, n, beta)
        C1, Cinf, Phi = schur(P, I, K, N)
        curve.append((beta, C1, Cinf, Phi, sorted(P)))
    return curve, bps, It, qc, (kr, kc, n)


def _rank(I, K, N):
    """Exact integer rank of the importance matrix, by fraction-free Gaussian elimination."""
    A = [[F(I[i][j], 1) for j in range(N)] for i in range(K)]
    r, row = 0, 0
    for col in range(N):
        piv = next((i for i in range(row, K) if A[i][col] != 0), None)
        if piv is None:
            continue
        A[row], A[piv] = A[piv], A[row]
        for i in range(row + 1, K):
            if A[i][col] != 0:
                f = A[i][col] / A[row][col]
                for j in range(col, N):
                    A[i][j] -= f * A[row][j]
        row += 1
        r += 1
        if row == K:
            break
    return r


def draw(rng, K, N, hi):
    I = [[rng.randint(1, hi) for _ in range(N)] for _ in range(K)]
    flat = [I[i][j] for i in range(K) for j in range(N)]
    return I if len(set(flat)) == len(flat) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "sweep", "rank1", "dead", "nondeg", "closure", "closure-exact"],
                    default="search")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--hi", type=int, default=30)
    ap.add_argument("--tries", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    K, N, s = args.K, args.N, args.s
    rng = random.Random(args.seed)

    if args.mode in ("search", "sweep"):
        print(f"PROTOCOL: mode={args.mode} K={K} N={N} s={s} importances=iid uniform"
              f" int[1,{args.hi}] distinct tries={args.tries} seed={args.seed}")
    else:
        # rank1 and dead build their own instances; the search/sweep knobs do not apply.
        print(f"PROTOCOL: mode={args.mode} K={K} N={N} s={s} seed={args.seed} "
              f"(--hi/--tries do not apply to this mode; the dead mode fixes its own shapes)")

    if args.mode == "dead":
        # Proposition "dead input channels": the two branches of the dichotomy, run through the
        # PRODUCTION routine. Branch (ii) needs K*Ndead*(1-s) < Nlive, branch (i) the reverse.
        import sys as _sys
        _sys.path[:0] = ["/workspace/PTQResearch/benchmarks", "/workspace/PTQResearch",
                         "/workspace/PTQResearch/src"]
        import torch
        import nosink as _ns
        for (Kd, Nd, nd) in [(256, 2048, 1), (16384, 2048, 2)]:
            torch.manual_seed(args.seed)
            X = torch.rand(256, Nd).cuda() + 0.5
            X[:, :nd] = 0.0
            W = torch.rand(Kd, Nd).cuda() + 0.1
            _, mk = _ns.balanced_mask_and_obs(W, X, s, "cuda", col_exp=1.0, no_obs=True)
            pr = 1 - mk
            live = X.norm(dim=0) > 0
            kc, n = int(Kd * s), int(Kd * Nd * s)
            nlive = int(live.sum())
            # A = |{Ihat(1) < 1}|, computed directly rather than from the closed form, which
            # holds only when every column quantile is uniquely attained.
            imp = (W.abs() * X.norm(dim=0).view(1, -1))
            kr = int(Nd * s)
            imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
            qc_ = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
            A = int((imp / qc_ < 1.0).sum().item())
            A_closed = Kd * nd + nlive * (kc - 1)
            pc = pr.sum(0)
            print(f"  K={Kd} N={Nd} ndead={nd}: A={A} (closed form {A_closed}) n={n} -> branch "
                  f"{'(i)' if A >= n else '(ii)'};  live n_j min={int(pc[live].min())} "
                  f"max={int(pc[live].max())} (k_c={kc});  |P|-n={int(pr.sum())-n}")
        return

    if args.mode == "closure-exact":
        # The closure argument of the derivation disowns the floating-point artifacts (their
        # per-beta ordering resolves no breakpoint correctly), so the claim rests on an exact
        # re-derivation. This mode commits that re-derivation: exact rationals for the
        # normalisations, 60-digit decimals for the exponentials, and the DEPLOYED strict-keep
        # tie rule evaluated AT each breakpoint, not only at interval midpoints.
        from fractions import Fraction as _F
        from decimal import Decimal as _D, getcontext as _gc
        import itertools as _it, math as _m
        _gc().prec = 60
        INSTANCES = {
            "selected": [[8, 7, 19, 15], [22, 3, 26, 21], [23, 5, 11, 20], [1, 12, 16, 28]],
            "closure":  [[1971, 4805, 2671, 2125], [4460, 3433, 1074, 499],
                         [2899, 3754, 4779, 4234], [3446, 4110, 1072, 4357]],
        }
        excl = {}
        for label, Ix in INSTANCES.items():
            Kx = Nx = 4; krx = kcx = 2; nx = 8
            qr = [sorted(r)[krx - 1] for r in Ix]
            It = [[_F(Ix[i][j], qr[i]) for j in range(Nx)] for i in range(Kx)]
            qc = [sorted(It[i][j] for i in range(Kx))[kcx - 1] for j in range(Nx)]
            bps = set()
            for p_, q_ in _it.combinations([(i, j) for i in range(Kx) for j in range(Nx)], 2):
                j1, j2 = p_[1], q_[1]
                if qc[j1] == qc[j2]:
                    continue
                d = _m.log(float(qc[j1] / qc[j2]))
                if d == 0:
                    continue
                b = _m.log(float(It[p_[0]][j1] / It[q_[0]][j2])) / d
                if 0 < b < 1:
                    bps.add(b)
            bs = sorted(bps)
            def phi(bd):
                sc = [[(_D(It[i][j].numerator) / _D(It[i][j].denominator))
                       / ((_D(qc[j].numerator) / _D(qc[j].denominator)) ** bd)
                       for j in range(Nx)] for i in range(Kx)]
                tau = sorted(v for row in sc for v in row)[nx - 1]
                P = [[sc[i][j] <= tau for j in range(Nx)] for i in range(Kx)]
                c1 = max(sum(Ix[i][j] * P[i][j] for i in range(Kx)) for j in range(Nx))
                ci = max(sum(Ix[i][j] * P[i][j] for j in range(Nx)) for i in range(Kx))
                return (c1 * ci) ** 0.5, sum(sum(r) for r in P)
            edges = [0.0] + bs + [1.0]
            probes = [("mid", (a + b) / 2) for a, b in zip(edges, edges[1:])]
            probes += [("bp", b) for b in bs] + [("end", 0.0), ("end", 1.0)]
            vals = [(kind, b) + phi(_D(repr(b))) for kind, b in probes]
            mn = min(v[2] for v in vals)
            # The min-attaining set is a union of whole inter-breakpoint intervals plus,
            # possibly, breakpoints. Report the EXCLUDED set (the complement), by boundary,
            # not by probe location: a probe is a witness for its interval, not an endpoint.
            segs = []   # (lo, hi, attains_min) per interval, boundaries exact
            for (a, b), (kind, bb, v, _) in zip(zip(edges, edges[1:]),
                                                [x for x in vals if x[0] == "mid"]):
                segs.append((a, b, v <= mn + 1e-9))
            bp_min = {b: v for kind, b, v, _ in vals if kind == "bp"}
            excluded = [(a, b) for a, b, at in segs if not at]
            lo = min(a for a, b in excluded) if excluded else None
            hi = max(b for a, b in excluded) if excluded else None
            # tighten: an excluded set bounded by a breakpoint includes that breakpoint iff
            # Phi there exceeds the minimum, which clause (ii) guarantees when it is a jump
            note = ""
            for bd, nm in ((lo, "lower"), (hi, "upper")):
                if bd in bp_min:
                    note += (f"; Phi at the {nm} boundary (a breakpoint) is "
                             f"{bp_min[bd]:.4f} > {mn:.4f}, so it is excluded too")
            print(f"  {label}: {len(bs)} breakpoints in (0,1); min Phi={mn:.4f}; "
                  f"excluded set = [{lo:.8f}, {hi:.8f}]{note}")
            excl[label] = (lo, hi)
        lo_sel, hi_sel = excl["selected"]
        lo_clo, hi_clo = excl["closure"]
        covered = (lo_clo <= 0.0 + 1e-12) and (hi_sel >= 1.0 - 1e-12) and (lo_sel <= hi_clo)
        print(f"  selected excludes [{lo_sel:.8f}, {hi_sel:.8f}]; "
              f"closure excludes [{lo_clo:.8f}, {hi_clo:.8f}]")
        print(f"  union covers [0,1]: {covered}  -> no fixed exponent minimises Phi "
              f"on every instance")
        return

    if args.mode == "closure":
        # An instance excludes exactly those beta* with Phi(beta*) > min_beta Phi. This one's
        # minimum sits at the RIGHT end, so it excludes every interior exponent below 0.7008 --
        # including exponents below its own first breakpoint (0.2992). It is the counterexample
        # to the closure claim "an instance can only exclude exponents above its first
        # breakpoint". Exact rationals for the normalisation; breakpoints in floating point.
        from fractions import Fraction as _F
        import itertools as _it, math as _m
        Ic = [[1971, 4805, 2671, 2125], [4460, 3433, 1074, 499],
              [2899, 3754, 4779, 4234], [3446, 4110, 1072, 4357]]
        Kc, Nc, krc, kcc, nc = 4, 4, 2, 2, 8
        qr = [sorted(r)[krc - 1] for r in Ic]
        It = [[_F(Ic[i][j], qr[i]) for j in range(Nc)] for i in range(Kc)]
        qc = [sorted(It[i][j] for i in range(Kc))[kcc - 1] for j in range(Nc)]
        def _phi(beta):
            sc = [[float(It[i][j]) / (float(qc[j]) ** beta) for j in range(Nc)]
                  for i in range(Kc)]
            tau = sorted(v for row in sc for v in row)[nc - 1]
            P = [[sc[i][j] <= tau for j in range(Nc)] for i in range(Kc)]
            c1 = max(sum(Ic[i][j] * P[i][j] for i in range(Kc)) for j in range(Nc))
            ci = max(sum(Ic[i][j] * P[i][j] for j in range(Nc)) for i in range(Kc))
            return (c1 * ci) ** 0.5
        bps = set()
        for p_, q_ in _it.combinations([(i, j) for i in range(Kc) for j in range(Nc)], 2):
            j1, j2 = p_[1], q_[1]
            if qc[j1] == qc[j2]:
                continue
            d = float(qc[j1] / qc[j2])
            if d == 1.0:
                continue
            b = _m.log(float(It[p_[0]][j1] / It[q_[0]][j2])) / _m.log(d)
            if 0 < b < 1:
                bps.add(round(b, 10))
        bs = sorted(bps)
        print(f"  q^row={qr}  q^col={[str(x) for x in qc]}  breakpoints={bs}")
        probes = [0.0] + [(a + b) / 2 for a, b in zip([0.0] + bs, bs + [1.0])] + [1.0]
        for b in sorted(set(probes)):
            print(f"  beta={b:.5f}: Phi={_phi(b):.4f}")
        print(f"  -> min Phi is attained at the RIGHT end; every interior beta below "
              f"{bs[-1]:.4f} is excluded by this one instance, first breakpoint {bs[0]:.4f}")
        return

    if args.mode == "nondeg":
        # Corollary "no endpoint-derivative argument", clause (iii): mask constancy on an
        # initial interval needs (N2) and integrality. This instance satisfies integrality and
        # fails (N2)'s counting clause on one row; the tie at tau_0 is pruned wholesale, so
        # |P(0)| > n, and the tie breaks immediately for beta > 0. Production routine, not a
        # reimplementation.
        import sys as _sys
        _sys.path[:0] = ["/workspace/PTQResearch/benchmarks", "/workspace/PTQResearch",
                         "/workspace/PTQResearch/src"]
        import torch
        import nosink as _ns
        Wn = torch.tensor([[1., 1, 3, 6], [3, 6, 3, 2], [6, 7, 1, 1], [5, 3, 5, 2]])
        Xn = torch.eye(4)
        In = Wn.abs() * torch.norm(Xn, dim=0).view(1, -1)
        qr = torch.kthvalue(In, 2, dim=1, keepdim=True).values
        print(f"  q^row={qr.flatten().tolist()}  entries <= q^row per row="
              f"{(In <= qr).sum(1).tolist()}  (k_r=2, so row 2 fails (N2) counting)")
        for beta in (0.0, 1e-4, 0.01, 0.1, 0.5, 1.0):
            _, m = _ns.balanced_mask_and_obs(Wn.clone(), Xn.clone(), 0.5, "cpu",
                                             col_exp=beta, no_obs=True)
            pr = (m == 0)
            Ap = pr.float() * In
            c1, ci = float(Ap.sum(0).max()), float(Ap.sum(1).max())
            print(f"  beta={beta}: |P|={int(pr.sum())} (n=8)  C1={c1:g}  Cinf={ci:g}  "
                  f"Phi={(c1 * ci) ** 0.5:.4f}")
        return

    if args.mode == "rank1":
        # Proposition "the exponent can be wholly inactive": for |W| of rank one the column
        # step does nothing on [0,1), and at beta=1 every score coincides so the deployed
        # rule loses the budget. Run through the PRODUCTION routine, not a reimplementation.
        import sys as _sys
        _sys.path[:0] = ["/workspace/PTQResearch/benchmarks", "/workspace/PTQResearch",
                         "/workspace/PTQResearch/src"]
        import torch
        import nosink as _ns
        torch.manual_seed(args.seed)
        a = torch.rand(K) + 0.5
        b = torch.rand(N) + 0.5
        W = (a.view(-1, 1) * b.view(1, -1)).cuda()      # |W| rank one, entrywise positive
        Xr = torch.rand(8 * N, N).cuda() + 0.5          # all input channels live
        for beta in (0.0, 0.5, 0.9, 1.0):
            _, m = _ns.balanced_mask_and_obs(W, Xr, s, "cuda", col_exp=beta, no_obs=True)
            print(f"  beta={beta}: kept {int(m.sum().item())} of {K*N}"
                  f"  (budget keeps {K*N - int(K*N*s)})")
        return

    if args.mode == "sweep":
        kr, kc, n_ = int(N * s), int(K * s), int(K * N * s)
        tally, retained, beta0_min, beta0_strict = Counter(), 0, 0, 0
        n_nondistinct = n_inactive = n_inactive_fullrank = 0
        n_eqq = n_maskmoves = 0
        for _ in range(args.tries):
            I = draw(rng, K, N, args.hi)
            if I is None:
                n_nondistinct += 1
                continue
            curve, bps, *_ = curve_of(I, K, N, s)
            if not bps:
                # The family does not move with beta at all: nothing to rank. These are counted
                # and reported rather than silently dropped -- they are exactly the
                # beta-INACTIVE instances, and their rank is reported because the sufficient
                # condition for inactivity in the derivation is a rank-one one.
                n_inactive += 1
                r = _rank(I, K, N)
                n_inactive_fullrank += (r == min(K, N))
                # WHY is it inactive? Two distinct causes, counted separately rather than
                # asserted: all column quantiles equal (so the divisor is a global scalar), or
                # crossings exist but lie outside (0,1).
                It_ = row_norm(I, K, N, kr)
                qc_ = col_quantiles(It_, K, N, kc)
                n_eqq += (len(set(qc_)) == 1)
                # And "inactive" means the ORDER does not move; the deployed rule can still
                # prune a different number at beta=1 through its tie handling, so check the
                # realised masks at the two endpoints.
                n_maskmoves += (mask_at(It_, qc_, K, N, n_, 0.0)
                                != mask_at(It_, qc_, K, N, n_, 1.0))
                continue
            retained += 1
            phis = [c[3] for c in curve]
            mn = min(phis)
            am = curve[phis.index(mn)][0]
            tally["beta=0" if am < 1e-9 else
                  ("beta=1" if am > 1 - 1e-9 else "interior")] += 1
            if phis[0] <= mn * (1 + 1e-12):
                beta0_min += 1
            if all(p > phis[0] * (1 + 1e-9) for p in phis[1:]):
                beta0_strict += 1
        print(f"RESULT: retained={retained} (instances with >=1 breakpoint)")
        print(f"  discarded: {n_nondistinct} non-distinct draws, {n_inactive} with no "
              f"breakpoint in (0,1); of those, {n_inactive_fullrank} are full rank, "
              f"{n_eqq} have all column quantiles equal (the rest have crossings outside "
              f"(0,1)), and on {n_maskmoves} the realised mask still differs between beta=0 "
              f"and beta=1")
        for k in ("beta=0", "interior", "beta=1"):
            print(f"  argmin Phi at {k:9s}: {tally[k]:6d}"
                  f"  ({100.0*tally[k]/max(retained,1):.1f}%)")
        print(f"  beta=0 is A minimiser of Phi : {beta0_min} ({100.0*beta0_min/max(retained,1):.1f}%)")
        print(f"  beta=0 is a STRICT minimiser : {beta0_strict}  "
              f"(expected 0: Phi is constant on [0, beta_1))")
        return

    best = None
    for _ in range(args.tries):
        I = draw(rng, K, N, args.hi)
        if I is None:
            continue
        curve, bps, *_ = curve_of(I, K, N, s)
        if not bps:
            continue
        p0 = curve[0][3]                        # curve[0] is beta = 0 (probes are sorted)
        It = row_norm(I, K, N, int(N * s))
        qc = col_quantiles(It, K, N, int(K * s))
        p5 = schur(mask_at(It, qc, K, N, int(K * N * s), 0.5), I, K, N)[2]
        if p0 < p5 * (1 - 1e-9):
            margin = p5 / p0
            if best is None or margin > best[0]:
                best = (margin, I)
    if best is None:
        print("RESULT: NO_INSTANCE_FOUND")
        return
    margin, I = best
    print(f"RESULT: found instance with Phi(0.5)/Phi(0) = {margin:.6f}")
    print("I =")
    for r in I:
        print("   ", r)
    curve, bps, It, qc, (kr, kc, n) = curve_of(I, K, N, s)
    print(f"kr={kr} kc={kc} n={n} breakpoints={len(bps)}")
    print("q^row =", [sorted(I[i])[kr - 1] for i in range(K)])
    print("q^col =", [str(x) for x in qc])
    for beta in (0.0, 0.5, 1.0):
        P = mask_at(It, qc, K, N, n, beta)
        C1, Cinf, Phi = schur(P, I, K, N)
        rowc = [sum(1 for (i, j) in P if i == t) for t in range(K)]
        colc = [sum(1 for (i, j) in P if j == t) for t in range(N)]
        print(f"  beta={beta}: P={sorted(P)}")
        print(f"            C1={C1} Cinf={Cinf} Phi={Phi:.5f} rowcounts={rowc} colcounts={colc}")
    print("exact curve over all intervals:")
    for beta, C1, Cinf, Phi, _ in curve:
        print(f"    beta={beta:.9f} C1={C1:6d} Cinf={Cinf:6d} Phi={Phi:10.5f}")


if __name__ == "__main__":
    main()
