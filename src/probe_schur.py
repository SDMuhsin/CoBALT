#!/usr/bin/env python3
"""SCHUR/whitening probe for the spectral-bound analysis (llmdocs/spectral_v2).

Every quantitative claim in the revised derivation is produced here. For a sample of linear
layers of a model, at a given sparsity, over a beta grid, using the DEPLOYED mask object
(ns.balanced_mask_and_obs, no_obs=True) and the PURE-PRUNE error E = W - W*M that the
analysis models:

  I_ij        = |W_ij| * ||X_:,j||_2                      (Wanda importance, first CAP rows)
  C1(beta)    = max_j sum_i (1-M_ij) I_ij                 (largest pruned column sum, ||Ap||_1)
  Cinf(beta)  = max_i sum_j (1-M_ij) I_ij                 (largest pruned row sum,   ||Ap||_inf)
  Phi(beta)   = sqrt(C1*Cinf)                             (Schur factor of the bound)
  colmax/colmin/rowmax/rowmin = per-axis pruned COUNTS    (tests the exactness theorems)
  Rnorm       = lambda_max(D^-1 G D^-1) - 1               (whitening constant, G = X^T X)
  true        = ||E X^T||_2 = sqrt(lambda_max(E G E^T))   (the quantity being bounded)
  bound       = sqrt(1+Rnorm) * Phi
  ratio       = bound / true                              (tightness; >=1 by the theorems)
  r           = true / ||W X^T||_2                        (the per-layer relative spectral error)

Emits one CSV row per (model, layer, matrix, sparsity, beta). Everything is computed in
float64 on the GPU-resident float32 tensors.

Usage:
  python -u src/probe_schur.py --model gemma-2b --sparsity 0.5 --csv results/spectral_v2/schur.csv
"""
import argparse, csv, fcntl, os, sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from probe_heldout_gap import collect  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

CAP = 256  # matches balanced_mask_and_obs's activation cap exactly

FIELDS = ["model", "draw", "layer", "matrix", "K", "N", "sparsity", "beta",
          "C1", "Cinf", "Phi", "colmax", "colmin", "rowmax", "rowmin",
          "colmax_live", "colmin_live", "target_col", "target_row", "nlive", "ndead", "tau", "n_at_tau", "n_pruned", "n_budget", "n_dup_f64", "n_dup_f32", "n_dupcol_f64", "n_dupcol_f32", "n_multicol_f64", "n_multirow_f64", "n_distinct", "n_entries", "max_row_zeros", "max_col_zeros", "n_zeros", "Rnorm", "Rfloor", "true", "bound", "ratio", "r", "frob", "atil", "atil_conv", "sumI2", "true_h", "frob_h", "r_h", "atil_svd", "min_qrow", "min_qcol_live"]


def spectral_norm_gram(A, Xd):
    """||A G^{1/2}||_2 = ||A X^T||_2, with G = X^T X.

    Both squared equal lambda_max(A X^T X A^T), so we form Y = A X^T (K x S) and take the
    Gram on the S = 256 side: exact, and avoids any N x N or K x K eigendecomposition.
    """
    Y = A @ Xd.t()                                     # [K, S]
    M = Y.t() @ Y                                      # [S, S]
    M = 0.5 * (M + M.t())
    ev = torch.linalg.eigvalsh(M)
    return float(ev[-1].clamp(min=0).sqrt().item()), Y


def spectral_power(A, iters=200, tol=1e-12):
    """||A||_2 by power iteration on A^T A.

    Returns (value, relative change at the last step). NOTE: an earlier version assigned
    prev = cur before falling out of the loop, so a run that exhausted `iters` WITHOUT
    converging reported a last-step change of exactly 0.0 -- i.e. the diagnostic read as
    perfect convergence precisely when it had failed. The cross-check against
    torch.linalg.svdvals (--svd-check) caught this: the rows with the largest error were
    exactly the rows reporting 0.0. The relative change is now computed against the value
    from the step before the last one in both exit paths, so exhausting `iters` reports the
    genuine (non-zero) last-step change.
    """
    v = torch.randn(A.shape[1], dtype=A.dtype, device=A.device)
    v /= v.norm()
    prev = cur = 0.0
    for _ in range(iters):
        u = A @ v
        v = A.t() @ u
        nv = v.norm()
        if nv == 0:
            return 0.0, 0.0
        v = v / nv
        prev, cur = cur, float(nv.sqrt().item())
        if prev and abs(cur - prev) <= tol * cur:
            return cur, abs(cur - prev) / cur
    return cur, (abs(cur - prev) / cur if cur else 0.0)


def _prep(Xa, device):
    X = Xa.to(device).float()
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    return X[:min(X.shape[0], CAP)]


def run_matrix(W, Xa, sparsity, betas, device, Xh=None, svd_check=False):
    """Returns a list of per-beta dicts for one weight matrix.

    Xh, when given, is a DISJOINT activation draw used only to EVALUATE the injected error:
    the mask and every selection statistic still come from Xa, but true_h/frob_h/r_h report
    ||E Xh^T|| instead of ||E Xa^T||. This is the held-out version of the adjudication --
    scope item (S2) -- and it is the only quantity here that is not in-sample.
    """
    K, N = W.shape
    X = _prep(Xa, device)
    Xhd = _prep(Xh, device).double() if Xh is not None else None

    Xd = X.double()
    S = Xd.shape[0]
    colnorm = torch.sqrt((Xd * Xd).sum(0).clamp(min=0))    # ||X_:,j||_2
    Wd = W.double()
    I = Wd.abs() * colnorm.view(1, -1)                 # Wanda importance

    # Whitening constant. Gamma = D^-1 X^T X D^-1 = Z^T Z with Z = X D^-1 (S x N), so
    # lambda_max(Gamma) = lambda_max(Z Z^T) on the S side. Gamma has unit diagonal, hence
    # trace N and rank <= S, giving the structural floor lambda_max >= N/S.
    # Degenerate ("dead") input channels with ||X_:,j||_2 = 0 make D singular; they are excluded
    # from the whitening (they carry zero importance and contribute nothing to E X^T), and the
    # structural floor is taken over the live channel count.
    live = colnorm > 0
    nlive = int(live.sum().item())
    ndead = N - nlive
    dinv = torch.where(live, 1.0 / colnorm, torch.zeros_like(colnorm))
    Z = (Xd * dinv.view(1, -1))[:, live]
    ZZ = Z @ Z.t()
    ZZ = 0.5 * (ZZ + ZZ.t())
    lam_max = float(torch.linalg.eigvalsh(ZZ)[-1].item())
    Rnorm = max(lam_max - 1.0, 0.0)
    Rfloor = float(nlive) / float(S) - 1.0

    denom, _ = spectral_norm_gram(Wd, Xd)              # ||W X^T||_2

    kr, kc = int(N * sparsity), int(K * sparsity)
    out = []
    for beta in betas:
        _, mask = ns.balanced_mask_and_obs(W, Xa.to(device), sparsity, device,
                                           col_exp=beta, no_obs=True)
        pruned = (1.0 - mask.double())                 # 1 where pruned

        # Threshold accounting, recomputed here in float32 exactly as the deployed routine
        # does it, so tau, the tie multiplicity at tau, and the realised pruned count are
        # artifacts rather than prose. n_prune is int(K*N*sparsity), as in _threshold_mask.
        imp = (W.abs() * torch.norm(X, dim=0).view(1, -1))   # float32, as nosink builds it
        if kr > 0:
            imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        if beta > 0 and kc > 0:
            qc_ = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
            imp = imp / qc_.pow(beta)
        n_prune = int(K * N * sparsity)
        tau = float(torch.kthvalue(imp.reshape(-1), n_prune).values.item())
        n_at_tau = int((imp.reshape(-1) == tau).sum().item())
        n_pruned = int(pruned.sum().item())

        # Tie MECHANISM, separated. n_dup_f64 counts rows carrying an EXACT duplicate of their
        # own row quantile in the float64 importance I -- a genuine failure of distinctness,
        # not a rounding artifact. n_dup_f32 is the same count in the float32 the pipeline
        # actually uses; the difference between them is the rounding contribution.
        # Column-side analogue: live columns whose k_c-th order statistic of Itilde is NOT
        # uniquely attained. Measured in both precisions, because the row-side and column-side
        # deviations turn out to have different causes.
        n_dupcol_f64 = n_dupcol_f32 = n_multicol_f64 = n_multirow_f64 = 0
        if kc > 0 and kr > 0:
            qr_ = torch.kthvalue(I, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
            It64 = I / qr_
            qc64 = torch.kthvalue(It64, kc, dim=0, keepdim=True).values
            n_dupcol_f64 = int((((It64 <= qc64).sum(dim=0) > kc) & live).sum().item())
            # Unique attainment (the strictly stronger condition): does the k_c-th order
            # statistic of a live column appear more than once? Count-failure implies this,
            # but not conversely, so this count is the larger of the two.
            n_multicol_f64 = int(((((It64 == qc64).sum(dim=0)) > 1) & live).sum().item())
            I32_ = (W.abs() * torch.norm(X, dim=0).view(1, -1))
            It32 = I32_ / torch.kthvalue(I32_, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
            qc32 = torch.kthvalue(It32, kc, dim=0, keepdim=True).values
            n_dupcol_f32 = int((((It32 <= qc32).sum(dim=0) > kc) & live).sum().item())
        # How badly Hypothesis nondeg's distinctness clause fails on the raw importance.
        n_distinct = int(torch.unique(I).numel())
        # Degenerate cases for the two order-statistic hypotheses. These must be counted on the
        # IMPORTANCE I (equivalently on Itilde, which differs by a positive row factor), not on
        # W: a dead channel makes a whole column of I zero even where W is not.
        #   (N2) fails structurally if a row has k_r or more zeros of I.
        #   (N3) fails structurally if a LIVE column has k_c or more zeros of Itilde.
        # Dead columns are all-zero by construction and are Proposition prop:dead's business,
        # so the column statistic is taken over live columns only.
        Iz = (I == 0)
        max_row_zeros = int(Iz.sum(dim=1).max().item())
        max_col_zeros = int(Iz[:, live].sum(dim=0).max().item()) if nlive else 0
        n_zeros = int(Iz.sum().item())

        n_dup_f64 = n_dup_f32 = 0
        if kr > 0:
            qr64 = torch.kthvalue(I, kr, dim=1, keepdim=True).values
            n_dup_f64 = int(((I <= qr64).sum(dim=1) > kr).sum().item())
            n_multirow_f64 = int((((I == qr64).sum(dim=1)) > 1).sum().item())
            I32 = (W.abs() * torch.norm(X, dim=0).view(1, -1))
            qr32 = torch.kthvalue(I32, kr, dim=1, keepdim=True).values
            n_dup_f32 = int(((I32 <= qr32).sum(dim=1) > kr).sum().item())
        # Smallest quantile magnitudes actually seen. Hypothesis nondeg's positivity clause says
        # these are > 0; what the deployed clamp needs is that they clear 1e-30. Recording them
        # turns that from an assertion into a measurement.
        min_qrow = float(qr64.min().item()) if kr > 0 else float('nan')
        min_qcol_live = (float(qc64[0, live].min().item())
                         if (kc > 0 and kr > 0 and nlive) else float('nan'))
        Ap = pruned * I                                # |A~_p| entrywise
        # Scope item (S6'): the diagonal of ||E X^T||_F^2 = tr(E G E^T) is exactly
        # sum_{(i,j) in P(beta)} I_ij^2, and per-row Wanda (beta = 0) minimises that sum within
        # the per-row-count class. Measuring it says how much of the direction half of the
        # negative is structural rather than a finding about the bound.
        sumI2 = float((Ap * Ap).sum().item())
        C1 = float(Ap.sum(dim=0).max().item())
        Cinf = float(Ap.sum(dim=1).max().item())
        Phi = (C1 * Cinf) ** 0.5

        pc = pruned.sum(dim=0)                         # pruned count per column
        pr = pruned.sum(dim=1)                         # pruned count per row

        E = Wd * pruned                                # pure-prune error W - W*M
        true, Y = spectral_norm_gram(E, Xd)            # Y = E X^T
        # MIDDLE term of the chain: ||Atilde||_2 with Atilde = E D. Measuring it splits the
        # bound's slack into the whitened-reduction part and the Schur part, which decides
        # whether the argmin mismatch is a defect of the Schur test or of the whitening.
        atil, atil_conv = spectral_power(E * colnorm.view(1, -1))
        # ||Atilde||_2 carries the stage-attribution argument, so on request recompute it with a
        # direct (non-iterative) estimator -- the largest singular value -- and report both.
        atil_svd = float('nan')
        if svd_check:
            atil_svd = float(torch.linalg.svdvals(E * colnorm.view(1, -1))[0].item())
        if Xhd is not None:
            true_h, Yh = spectral_norm_gram(E, Xhd)
            denom_h, _ = spectral_norm_gram(Wd, Xhd)
            frob_h = float(torch.linalg.matrix_norm(Yh).item())
            r_h = (true_h / denom_h if denom_h > 0 else float('nan'))
        else:
            true_h, frob_h, r_h = float('nan'), float('nan'), float('nan')
        bound = (1.0 + Rnorm) ** 0.5 * Phi
        out.append(dict(K=K, N=N, sparsity=sparsity, beta=beta,
                        C1=C1, Cinf=Cinf, Phi=Phi,
                        colmax=int(pc.max().item()), colmin=int(pc.min().item()),
                        colmax_live=int(pc[live].max().item()) if nlive else -1,
                        colmin_live=int(pc[live].min().item()) if nlive else -1,
                        rowmax=int(pr.max().item()), rowmin=int(pr.min().item()),
                        target_col=kc, target_row=kr, nlive=nlive, ndead=ndead,
                        atil=atil, atil_conv=atil_conv,
                        tau=tau, n_at_tau=n_at_tau, n_pruned=n_pruned, n_budget=n_prune,
                        n_dup_f64=n_dup_f64, n_dup_f32=n_dup_f32,
                        n_dupcol_f64=n_dupcol_f64, n_dupcol_f32=n_dupcol_f32,
                        n_multicol_f64=n_multicol_f64, n_multirow_f64=n_multirow_f64,
                        n_distinct=n_distinct, n_entries=K * N,
                        max_row_zeros=max_row_zeros, max_col_zeros=max_col_zeros,
                        n_zeros=n_zeros,
                        Rnorm=Rnorm, Rfloor=Rfloor, true=true, bound=bound,
                        ratio=(bound / true if true > 0 else float('nan')),
                        r=(true / denom if denom > 0 else float('nan')),
                        frob=float(torch.linalg.matrix_norm(Y).item()),
                        sumI2=sumI2, true_h=true_h, frob_h=frob_h, r_h=r_h,
                        atil_svd=atil_svd, min_qrow=min_qrow,
                        min_qcol_live=min_qcol_live))
    return out


def append_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2b")
    ap.add_argument("--sparsity", type=float, nargs="+", default=[0.5])
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--take-slice", type=int, nargs=2, default=None,
                    help="disjoint calibration window (a b) for a second, independent draw")
    ap.add_argument("--draw", default="A", help="label for this calibration draw, written to the CSV")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layer indices to sample (default: 1, nl//2, nl-2)")
    ap.add_argument("--svd-check", action="store_true",
                    help="also compute ||Atilde||_2 by exact SVD, to validate the power iteration")
    ap.add_argument("--heldout-slice", type=int, nargs=2, default=None,
                    help="disjoint window (a b) whose activations EVALUATE the error while the "
                         "mask is still selected on the main draw (scope item S2)")
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "spectral_v2", "schur.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = collect(model, tok, device, "wikitext2", args.nsamples,
                take_slice=tuple(args.take_slice) if args.take_slice else None)
    H = None
    if args.heldout_slice:
        H = collect(model, tok, device, "wikitext2", args.nsamples,
                    take_slice=tuple(args.heldout_slice))
    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache()

    lp = bs.get_layer_paths(model)
    layers = ns.get_layers(model)
    nl = len(layers)
    sample = sorted(set(args.layers)) if args.layers else sorted(set([1, nl // 2, nl - 2]))
    print(f"######## {args.model} layers={sample}/{nl} sp={args.sparsity} ########", flush=True)

    for li in sample:
        layer = layers[li].to(device)
        for apth in lp:
            parts = apth.split('.')
            parent, ok = layer, True
            for p in parts[:-1]:
                if not hasattr(parent, p):
                    ok = False
                    break
                parent = getattr(parent, p)
            if not ok or not hasattr(parent, parts[-1]):
                continue
            lin = getattr(parent, parts[-1])
            if not isinstance(lin, torch.nn.Linear):
                continue
            key = f'layer_{li}.{apth}'
            if A.get(key) is None:
                continue
            W = lin.weight.data.clone().float().to(device)
            for sp in args.sparsity:
                rows = run_matrix(W, A[key], sp, args.betas, device,
                                  Xh=(H.get(key) if H is not None else None),
                                  svd_check=args.svd_check)
                for r in rows:
                    r["model"], r["layer"], r["matrix"] = args.model, li, apth
                    r["draw"] = args.draw
                append_rows(args.csv, rows)
                b0 = rows[0]
                b1 = rows[-1]
                print(f"  L{li:2d} {apth:22s} sp={sp} K={b0['K']} N={b0['N']} "
                      f"R={b0['Rnorm']:.1f} | b=0 colmax={b0['colmax']}/{b0['target_col']} "
                      f"rowmax={b0['rowmax']}/{b0['target_row']} ratio={b0['ratio']:.1f} "
                      f"| b=1 colmax={b1['colmax']} rowmax={b1['rowmax']} ratio={b1['ratio']:.1f}",
                      flush=True)
            del W
            torch.cuda.empty_cache()
        layer.to("cpu")
        torch.cuda.empty_cache()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
