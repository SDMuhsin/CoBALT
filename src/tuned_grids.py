"""Canonical bit-budget-preserving hyperparameter grids, shared by every suite.

One definition, used by the decoder, GLUE and ViT drivers and by all three
dispatchers, so an arm's grid and its `hp` labels are identical everywhere and a
result row means the same thing in every CSV.

The rule for what may appear here: a knob is sweepable iff changing it leaves the
stored artefact the same size.  Group size, bit-width, adaptive-nbits and SLiM's
rank ratio all move the bit budget, so they are pinned by the matched protocol and
are deliberately absent.

Every arm gets a grid at least as large as CoBALT's 11 betas, so no arm is ever
compared tuned against another arm's single default:

    cobalt      beta                                                     11
    sparsegpt   percdamp x blocksize                                     12
    jsq-wo      rho x clip_h                                             12
    awq         beta-grid x weight-scale x L1/L2       (also wanda-awq)  12
    sinq        Sinkhorn iterations x early stop       (also wanda-sinq) 12
    slim        adapter x cap bins x survivor-only cap                   12
    wanda       activation exponent x comparison group                   12
    fp16        -- dense and uncompressed, no knob exists                 1

Composite / ablation arms (CoBALT's mask crossed with another method's quantizer or
recipe). Each gets 12 configs too, split as (mask axis) x (recipe axis) so neither the
mask nor the borrowed recipe is left at a single default:

    cobalt-sinq      beta x Sinkhorn iterations                           4 x 3
    cobalt-awq       beta x AWQ activation exponent                       4 x 3
    cobaltmask-sgpt  beta x SparseGPT blocksize                           4 x 3
    sgptmask-cobalt  OBS damping x saliency threshold scope               6 x 2

The 4-point beta subset {0, 0.5, 0.7, 0.9} keeps beta=0 (column balance OFF, the internal
mask ablation) and the three values that win most often in the full 11-beta gemma-2b grid
(0.5 x4, 0.7 x3, 0.9 x1 of 10 cells). The borrowed-recipe axis is pinned at the value the
BASELINE's own tuned grid selects wherever that choice is unanimous, so the composite
isolates the mask instead of re-tuning a settled knob: SparseGPT picks percdamp=0.1 in
10/10 gemma-2b cells, so cobaltmask-sgpt pins it and sweeps blocksize (which does move,
32/64/128/256 across cells). SINQ's early stop splits 6/4 across cells, so cobalt-sinq
pins it at the default True and sweeps the iteration count, whose optima are 4/8/16/32.

sgptmask-cobalt is the mirror control: its mask is OBS saliency, so no beta exists at all.
Its two bit-budget-free axes are the OBS Hessian damping (the analog of SparseGPT's
percdamp, previously hard-coded at 1% of mean(diag H)) and the threshold scope (a global
top-k vs a per-output-row quota -- same total survivor count either way).
"""

BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PERCDAMPS = [1e-3, 1e-2, 1e-1]
BLOCKSIZES = [32, 64, 128, 256]

JSQ_RHOS = [0.0, 1.0, 2.1, 4.0]
JSQ_CLIPHS = [0.01, 0.05, 0.1]

AWQ_NUM_BETAS = [1, 4, 10]          # activation-STD exponent grid points (1 = search off)
AWQ_WEIGHTSCALE = [True, False]     # divide the searched scale by the weight scale
AWQ_L1 = [False, True]              # reconstruction-error norm

SINQ_ORDERS = [1, 2, 4, 8, 16, 32]  # Sinkhorn balancing iterations
SINQ_STOP = [True, False]           # freeze once the imbalance stops improving

SLIM_ADAPTER = [True, False]        # saliency-weighted SLiM-LoRA vs Naive-LoRA
SLIM_CAPBINS = [0, 2048, 8192]      # cap-search histogram bins (0 = upstream default)
SLIM_SPARSECAP = [False, True]      # MSE-optimal cap over surviving weights only

WANDA_ACTEXP = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # exponent on the activation norm
WANDA_SCOPE = ["row", "layer"]                    # comparison group for the threshold

# --- composite / ablation arms -------------------------------------------------
# Beta subset for the composites: beta=0 is the column-balance-OFF control, the rest are
# the modal winners of the full 11-point grid. See the module docstring.
COMPOSITE_BETAS = [0.0, 0.5, 0.7, 0.9]

COBSINQ_ORDERS = [4, 16, 32]        # Sinkhorn iterations (wanda-sinq's optima live here)
COBAWQ_ALPHAS = [0.25, 0.5, 0.75]   # exponent a in c = mu_w^(1-a) / mu_x^a (0.5 = AWQ default)
CMSGPT_BLOCKSIZES = [32, 128, 256]  # SparseGPT blocksize; percdamp pinned at its 10/10 winner
CMSGPT_PERCDAMP = 0.1

SGPTMASK_DAMPS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]   # OBS damping / mean(diag H)
SGPTMASK_SCOPES = ["global", "per_row"]                 # saliency threshold scope

# Arms that share another arm's knob set.
# NOTE: cobalt-sinq used to sit here (beta only, SINQ pinned at order=16). It now has its
# own beta x order grid, so its `hp` labels changed from "beta=B" to "beta=B,order=O" --
# rows written under the old labels (results/decoder_suite/) are not resumable into the new grid.
_COBALT_LIKE = {"cobalt", "cobalt-noobs", "cobalt-awclip",
                "cobalt-seqfix", "cobalt-seqfix-awclip"}


def variants(method):
    """[(hp_label, kwargs)] for one method. kwargs use the drivers' parameter names."""
    if method in _COBALT_LIKE:
        return [(f"beta={b:g}", {"col_balance_exp": b}) for b in BETAS]
    if method == "sparsegpt":
        return [(f"pd={pd:g},bs={bs}", {"percdamp": pd, "blocksize": bs})
                for pd in PERCDAMPS for bs in BLOCKSIZES]
    if method == "jsq-wo":
        return [(f"rho={r:g},clip={c:g}", {"jsq_rho": r, "jsq_clip_h": c})
                for r in JSQ_RHOS for c in JSQ_CLIPHS]
    if method in ("awq", "wanda-awq"):
        return [(f"nb={nb},ws={int(ws)},l1={int(l1)}",
                 {"awq_num_betas": nb, "awq_use_weightscale": ws, "awq_l1": l1})
                for nb in AWQ_NUM_BETAS for ws in AWQ_WEIGHTSCALE for l1 in AWQ_L1]
    if method in ("sinq", "wanda-sinq"):
        return [(f"order={o},stop={int(st)}", {"sinq_order": o, "sinq_stop": st})
                for o in SINQ_ORDERS for st in SINQ_STOP]
    if method == "slim":
        return [(f"lora={int(lo)},bins={bn},scap={int(sc)}",
                 {"slim_lora": lo, "slim_cap_bins": bn, "slim_sparse_cap": sc})
                for lo in SLIM_ADAPTER for bn in SLIM_CAPBINS for sc in SLIM_SPARSECAP]
    if method == "wanda":
        return [(f"aexp={e:g},scope={sc}", {"wanda_act_exp": e, "wanda_scope": sc})
                for e in WANDA_ACTEXP for sc in WANDA_SCOPE]
    if method == "cobalt-sinq":
        return [(f"beta={b:g},order={o}", {"col_balance_exp": b, "sinq_order": o})
                for b in COMPOSITE_BETAS for o in COBSINQ_ORDERS]
    if method == "cobalt-awq":
        return [(f"beta={b:g},alpha={a:g}", {"col_balance_exp": b, "awq_alpha": a})
                for b in COMPOSITE_BETAS for a in COBAWQ_ALPHAS]
    if method == "cobaltmask-sgpt":
        return [(f"beta={b:g},bs={bs}",
                 {"col_balance_exp": b, "percdamp": CMSGPT_PERCDAMP, "blocksize": bs})
                for b in COMPOSITE_BETAS for bs in CMSGPT_BLOCKSIZES]
    if method == "sgptmask-cobalt":
        return [(f"damp={d:g},scope={sc}", {"obs_damp": d, "mask_scope": sc})
                for d in SGPTMASK_DAMPS for sc in SGPTMASK_SCOPES]
    return [("default", {})]


def n_configs(method):
    return len(variants(method))


# kwargs -> src/camera_bench.py command line. The decoder driver names CoBALT's
# exponent --cobalt-beta; every other name matches one-to-one.
def cli_args(kw):
    """Translate one variants() kwargs dict into camera_bench.py CLI arguments."""
    out = []
    for k, v in kw.items():
        if k == "col_balance_exp":
            out += ["--cobalt-beta", f"{v:g}"]
        elif k == "percdamp":
            out += ["--sgpt-percdamp", f"{v:g}"]
        elif k == "blocksize":
            out += ["--sgpt-blocksize", str(int(v))]
        elif k == "jsq_rho":
            out += ["--jsq-rho", f"{v:g}"]
        elif k == "jsq_clip_h":
            out += ["--jsq-clip-h", f"{v:g}"]
        elif k == "awq_num_betas":
            out += ["--awq-num-betas", str(int(v))]
        elif k == "awq_use_weightscale":
            if not v:
                out += ["--awq-no-weightscale"]
        elif k == "awq_l1":
            if v:
                out += ["--awq-l1"]
        elif k == "sinq_order":
            out += ["--sinq-order", str(int(v))]
        elif k == "sinq_stop":
            if not v:
                out += ["--sinq-no-stop"]
        elif k == "slim_lora":
            if not v:
                out += ["--slim-naive"]
        elif k == "slim_cap_bins":
            if v:
                out += ["--slim-cap-bins", str(int(v))]
        elif k == "slim_sparse_cap":
            if v:
                out += ["--slim-sparse-cap"]
        elif k == "wanda_act_exp":
            out += ["--wanda-act-exp", f"{v:g}"]
        elif k == "wanda_scope":
            out += ["--wanda-scope", str(v)]
        elif k == "awq_alpha":
            out += ["--awq-alpha", f"{v:g}"]
        elif k == "obs_damp":
            out += ["--obs-damp", f"{v:g}"]
        elif k == "mask_scope":
            out += ["--mask-scope", str(v)]
        else:
            raise KeyError(f"no CLI mapping for tuned knob {k!r}")
    return out
