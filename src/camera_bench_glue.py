#!/usr/bin/env python3
"""CoBALT vs matched baselines on ENCODER language transformers (BERT / RoBERTa), 3-bit + prune.

Extends the joint quant+prune camera bench to sub-1B masked-LM encoders — a distinct
architecture class from the causal LMs (camera_bench.py) and image encoders
(camera_bench_vit.py). Motivation: the CoBALT collapse+rescue win is tied to LARGE-transformer
geometry (24x1024) + outlier pathology; BERT-large / RoBERTa-large share that geometry, so they
are the natural encoder-side test. Metric = bounded downstream (GLUE task accuracy), the governing
metric per the project charter (reconstruction != downstream).

Reuses ALL original compression code (ns.apply_wanda_obs_rtn, bs.apply_sparsegpt_pruning,
bs.apply_wanda_awq/sinq, ...) via a monkeypatch compat shim that teaches benchmark_suite's
structural accessors about BERT/RoBERTa (model.{bert,roberta}.encoder.layer + attention.self.*
Linear names). Same design as camera_bench_vit.install_vit_compat.

Encoder linears per block (6, same count as ViT): attention.self.{query,key,value},
attention.output.dense, intermediate.dense, output.dense.
"""
import os, sys, argparse, csv, fcntl, time
from datetime import datetime, timezone
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs   # noqa: E402
import nosink as ns            # noqa: E402
import tuned_grids as tg      # noqa: E402

DEV = "cuda"
# UNION of per-family block-linear paths (mirrors camera_bench_vit): absent paths are skipped
# per-block by the `except AttributeError` loops in benchmark_suite, so one list serves every
# encoder family. bert/roberta/electra share the classic BERT names; distilbert and deberta-v1
# use their own; deberta-v2/v3 (fused in_proj) is intentionally excluded (different qkv layout).
ENCODER_PATHS = [
    # bert / roberta / electra
    "attention.self.query", "attention.self.key", "attention.self.value",
    "attention.output.dense", "intermediate.dense", "output.dense",
    # deberta v1 (disentangled attn: q/k/v named *_proj; extra pos Linears left unpruned)
    "attention.self.query_proj", "attention.self.key_proj", "attention.self.value_proj",
    # distilbert
    "attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin",
    "ffn.lin1", "ffn.lin2",
]
# model_type -> (backbone attr name, layers path relative to backbone)
_ENC_ARCH = {
    "bert":        ("bert",       "encoder.layer"),
    "roberta":     ("roberta",    "encoder.layer"),
    "electra":     ("electra",    "encoder.layer"),
    "deberta":     ("deberta",    "encoder.layer"),
    # deberta-v2/v3: model_type="deberta-v2", HF's DisentangledSelfAttention exposes SEPARATE
    # query_proj/key_proj/value_proj (NOT a fused in_proj) — same 6 block-linear paths as v1, same
    # backbone attr `deberta` + encoder.layer, same fp32 requirement (handled by mt.startswith).
    "deberta-v2":  ("deberta",    "encoder.layer"),
    "distilbert":  ("distilbert", "transformer.layer"),
}
_ENC_TYPES = tuple(_ENC_ARCH.keys())


def _arch(model):
    """Return (backbone_module, layers_list) for a supported encoder, else (None, None)."""
    mt = getattr(getattr(model, "config", None), "model_type", "").lower()
    spec = _ENC_ARCH.get(mt)
    if spec is None:
        return None, None
    attr, lpath = spec
    if not hasattr(model, attr):
        return None, None
    bb = getattr(model, attr)
    layers = bb
    for p in lpath.split("."):
        if not hasattr(layers, p):
            return None, None
        layers = getattr(layers, p)
    return bb, layers


def _backbone_attr(model):
    bb, _ = _arch(model)
    return bb


def _is_encoder(model):
    bb, _ = _arch(model)
    return bb is not None


def install_encoder_compat():
    """Teach bs's accessors about BERT/RoBERTa. All bs.apply_* + ns.apply_wanda_obs_rtn route
    through these, so the original algorithms run unchanged on encoders."""
    o_paths, o_layers, o_setl = bs.get_layer_paths, bs.get_transformer_layers, bs.set_transformer_layer
    o_fln, o_emb, o_fin = bs.get_final_layernorm, bs.move_embed_to_device, bs.move_final_layers_to_device

    def get_layer_paths(model):
        return ENCODER_PATHS if _is_encoder(model) else o_paths(model)

    def get_transformer_layers(model):
        bb, layers = _arch(model)
        return layers if bb is not None else o_layers(model)

    def set_transformer_layer(model, i, layer):
        bb, layers = _arch(model)
        if bb is not None:
            layers[i] = layer
        else:
            o_setl(model, i, layer)

    def get_final_layernorm(model):
        return None if _is_encoder(model) else o_fln(model)   # encoders: no single final LN

    def move_embed_to_device(model, device):
        bb, _ = _arch(model)
        if bb is not None:
            bb.embeddings.to(device)
        else:
            o_emb(model, device)

    def move_final_layers_to_device(model, device):
        bb, _ = _arch(model)
        if bb is not None:
            # move everything that is NOT the transformer backbone (pooler lives in backbone;
            # heads pre_classifier/classifier/pooler live at the top level) to device.
            if getattr(bb, "pooler", None) is not None:
                bb.pooler.to(device)
            for name, child in model.named_children():
                if child is bb:
                    continue
                child.to(device)
        else:
            o_fin(model, device)

    for name, fn in [("get_layer_paths", get_layer_paths), ("get_transformer_layers", get_transformer_layers),
                     ("set_transformer_layer", set_transformer_layer), ("get_final_layernorm", get_final_layernorm),
                     ("move_embed_to_device", move_embed_to_device),
                     ("move_final_layers_to_device", move_final_layers_to_device)]:
        setattr(bs, name, fn)


# ------------------------------------------------------------- calibration + eval data
def _text_cols(ds):
    """The 1 or 2 text columns of a GLUE task, in dataset order (handles mnli premise/hypothesis,
    qnli question/sentence, qqp question1/question2, mrpc/rte/stsb sentence1/2, sst2/cola sentence)."""
    return [c for c in ds.column_names if c not in ("label", "idx")]


def build_calibration(tok, task, n_calib, seq_len):
    """Pack real task text into n_calib fixed-length token windows (no padding -> no attention-mask
    needed; matches collect_activations' model(input_ids) call)."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", task, split="train")
    keys = _text_cols(ds)
    ids = []
    for ex in ds:
        text = " ".join(str(ex[k]) for k in keys)
        ids.extend(tok(text, add_special_tokens=True)["input_ids"])
        if len(ids) >= (n_calib + 1) * seq_len:
            break
    t = torch.tensor(ids[:n_calib * seq_len], dtype=torch.long).view(n_calib, seq_len)
    return t


def load_eval(task):
    from datasets import load_dataset
    split = "validation_matched" if task == "mnli" else "validation"
    ds = load_dataset("nyu-mll/glue", task, split=split)
    return ds


@torch.no_grad()
def evaluate(model, tok, ds, task, device, batch_size=64, max_len=128):
    """GLUE accuracy. Label-aligned by comparing model.id2label[pred] to the dataset's
    ClassLabel name for the gold id (robust to model-vs-dataset label-order mismatch, e.g.
    roberta-large-mnli uses {0:CONTRADICTION,1:NEUTRAL,2:ENTAILMENT} != glue order)."""
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    gold_names = [n.lower() for n in ds.features["label"].names]  # dataset id -> name
    # If the model's head labels are semantic and overlap the dataset names, map by NAME (handles
    # order mismatch, e.g. roberta-large-mnli). Else (LABEL_0/1 heads) fall back to IDENTITY order.
    by_name = len(set(id2label.values()) & set(gold_names)) >= max(2, len(gold_names) - 1)
    tcols = _text_cols(ds)
    correct = total = 0
    model.eval()
    for i in range(0, len(ds), batch_size):
        chunk = ds[i:i + batch_size]
        a = chunk[tcols[0]]
        b = chunk[tcols[1]] if len(tcols) > 1 else None
        enc = tok(a, b, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds = logits.argmax(-1).tolist()
        for p, g in zip(preds, chunk["label"]):
            if g < 0:
                continue
            correct += int(id2label[p] == gold_names[g] if by_name else p == g)
            total += 1
    return 100.0 * correct / max(total, 1)


# ------------------------------------------------------------- build (compression)
def build_model(model_id, method, sparsity, bits, cal, group_size,
                col_balance_exp=0.5, percdamp=0.01, blocksize=128,
                jsq_rho=2.1, jsq_clip_h=0.01,
                awq_num_betas=1, awq_use_weightscale=True, awq_l1=False,
                sinq_order=16, sinq_stop=True,
                slim_lora=True, slim_cap_bins=0, slim_sparse_cap=False,
                wanda_act_exp=0.5, wanda_scope="row"):
    from transformers import AutoModelForSequenceClassification, AutoConfig
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    # DeBERTa v1's disentangled attention keeps float buffers and crashes under fp16
    # ("expected scalar type Float but found Half"); load it in fp32. All other encoders fp16.
    mt = getattr(AutoConfig.from_pretrained(model_id), "model_type", "").lower()
    dtype = torch.float32 if mt.startswith("deberta") else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=dtype).to(DEV)
    sbt = {ns.type_of(p): float(sparsity) for p in ENCODER_PATHS}

    if method == "fp16":
        pass
    elif method == "awq":
        model = bs.apply_awq_quantization(model, cal, bits, DEV,
                                          groupsize=group_size,
                                          awq_num_betas=awq_num_betas,
                                          awq_use_weightscale=awq_use_weightscale,
                                          awq_l1=awq_l1)
    elif method == "sinq":
        model = bs.apply_sinq_quantization(model, cal, bits, DEV,
                                           sinq_order=sinq_order, sinq_stop=sinq_stop,
                                           group_size=group_size)
    elif method == "wanda":
        model = bs.apply_wanda_pruning(model, cal, float(sparsity), DEV,
                                       act_exp=wanda_act_exp, scope=wanda_scope)
    elif method == "sparsegpt":
        model = bs.apply_sparsegpt_pruning(model, cal, float(sparsity), bits, DEV,
                                           percdamp=float(percdamp), blocksize=int(blocksize))
    elif method == "wanda-awq":
        model = bs.apply_wanda_awq_quantization(model, cal, bits, float(sparsity), DEV,
                                                groupsize=group_size,
                                                awq_num_betas=awq_num_betas,
                                                awq_use_weightscale=awq_use_weightscale,
                                                awq_l1=awq_l1)
    elif method == "wanda-sinq":
        model = bs.apply_wanda_sinq_quantization(model, cal, bits, float(sparsity), DEV,
                                                 sinq_order=sinq_order, sinq_stop=sinq_stop,
                                                 group_size=group_size)
    elif method == "jsq-wo":
        # JSQ weight-only (bit-matched). Its two live encoder knobs (rho, clip_h) are read from
        # env by _apply_jsq; set them per-cell here. (alpha/smoothing are inert on encoders — the
        # SmoothQuant LN<->fc pairs don't exist, so that step self-skips.)
        os.environ["JSQ_RHO"] = f"{float(jsq_rho):g}"
        os.environ["JSQ_CLIPH"] = f"{float(jsq_clip_h):g}"
        model = bs.apply_jsq_weightonly_quantization(model, cal, bits, float(sparsity), DEV)
    elif method == "cobalt":
        model, _ = ns.apply_wanda_obs_rtn(model, cal, bits, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=float(col_balance_exp), group_size=group_size)
    else:
        raise ValueError(method)
    bs.move_final_layers_to_device(model, DEV)
    model.eval()
    return model


# ------------------------------------------------------------- CSV (mirror camera_bench_vit)
CSV_FIELDS = ["timestamp", "model", "method", "bits", "sparsity", "hp", "task", "metric",
              "value", "correct", "total", "seconds", "error"]
PRUNE_METHODS = {"wanda", "sparsegpt", "wanda-awq", "wanda-sinq", "jsq-wo", "cobalt"}


def read_done(csv_path, model_id, method, bits, sparsity, task, hp):
    if not os.path.exists(csv_path):
        return False
    sp_s, b_s = f"{float(sparsity):.2f}", str(int(bits))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if (r.get("model") == model_id and r.get("method") == method and r.get("bits") == b_s
                    and r.get("sparsity") == sp_s and (r.get("hp") or "") == hp
                    and r.get("task") == task
                    and not (r.get("error") or "").strip()):
                return True
    return False


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            has_header = f.readline().startswith("timestamp")
            f.seek(0, 2)
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not has_header:
                w.writeheader()
            w.writerow(row)
            f.flush(); os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/roberta-large-mnli")
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--methods", default="cobalt,sparsegpt,wanda-awq,wanda-sinq")
    ap.add_argument("--hp-only", default="",
                    help="restrict every method's grid to this one canonical hp label "
                         "(from src/tuned_grids.py); used to transfer a tuned config here")
    ap.add_argument("--sparsities", default="0.5")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "benchmark_camera_glue", "glue.csv"))
    # ---- bit-MATCHED (bpw-preserving) hyperparameter grids (comma lists) ----
    # cobalt: column-balance exponent beta (0=per-row Wanda .. 1=full column balance).
    ap.add_argument("--col-balance-exps", default="0.5", help="cobalt beta grid, comma-sep")
    # sparsegpt: OBS Hessian damping + error-compensation block width (neither changes bpw).
    ap.add_argument("--sgpt-percdamps", default="0.01", help="sparsegpt percdamp grid, comma-sep")
    ap.add_argument("--sgpt-blocksizes", default="128", help="sparsegpt blocksize grid, comma-sep")
    # jsq-wo: rho = prune<->quant bridge weight; clip_h = activation-clip fraction (both bpw-neutral).
    ap.add_argument("--jsq-rhos", default="2.1", help="jsq-wo rho grid, comma-sep")
    ap.add_argument("--jsq-cliphs", default="0.01", help="jsq-wo clip_h grid, comma-sep")
    args = ap.parse_args()

    install_encoder_compat()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sparsities = [float(s) for s in args.sparsities.split(",") if s.strip()]
    betas = [float(x) for x in args.col_balance_exps.split(",") if x.strip()]
    percdamps = [float(x) for x in args.sgpt_percdamps.split(",") if x.strip()]
    blocksizes = [int(x) for x in args.sgpt_blocksizes.split(",") if x.strip()]
    jsq_rhos = [float(x) for x in args.jsq_rhos.split(",") if x.strip()]
    jsq_cliphs = [float(x) for x in args.jsq_cliphs.split(",") if x.strip()]

    def hp_variants(method):
        """Expand a method into its (hp_label, build_model_kwargs) grid.

        The canonical grids come from src/tuned_grids.py, so every arm carries a grid at
        least as large as CoBALT's and the labels match the decoder and ViT suites. A grid
        overridden on the command line replaces that arm's list."""
        if args.hp_only:
            v = [(lab, kw) for lab, kw in tg.variants(method) if lab == args.hp_only]
            if not v:
                raise SystemExit(f"hp label {args.hp_only!r} is not in the {method} grid")
            return v
        if method == "cobalt" and list(betas) != list(tg.BETAS):
            return [(f"beta={b:g}", {"col_balance_exp": b}) for b in betas]
        if method == "sparsegpt" and (list(percdamps) != list(tg.PERCDAMPS)
                                      or list(blocksizes) != list(tg.BLOCKSIZES)):
            return [(f"pd={pd:g},bs={bs}", {"percdamp": pd, "blocksize": bs})
                    for pd in percdamps for bs in blocksizes]
        if method == "jsq-wo" and (jsq_rhos != list(tg.JSQ_RHOS)
                                   or jsq_cliphs != list(tg.JSQ_CLIPHS)):
            return [(f"rho={r:g},clip={c:g}", {"jsq_rho": r, "jsq_clip_h": c})
                    for r in jsq_rhos for c in jsq_cliphs]
        return tg.variants(method)

    cal = build_calibration(tok, args.task, args.n_calib, args.seq_len)
    eval_ds = load_eval(args.task)
    print(f"# model={args.model} task={args.task} methods={methods} sparsities={sparsities} "
          f"bits={args.bits} gsize={args.group_size} n_calib={cal.shape[0]} n_eval={len(eval_ds)}", flush=True)

    cells = []  # (method, sparsity, bits, hp_label, hp_kwargs)
    for m in methods:
        sps = sparsities if m in PRUNE_METHODS else [0.0]
        mbits = args.bits if m != "fp16" else 16
        for sp in sps:
            for hp_label, hp_kw in hp_variants(m):
                cells.append((m, sp, mbits, hp_label, hp_kw))

    for method, sp, bits, hp_label, hp_kw in cells:
        if read_done(args.csv, args.model, method, bits, sp, args.task, hp_label):
            print(f"SKIP {method} sp={sp:.2f} bits={bits} hp={hp_label} (cached)", flush=True)
            continue
        t0 = time.time()
        row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model": args.model, "method": method, "bits": int(bits), "sparsity": f"{float(sp):.2f}",
               "hp": hp_label, "task": args.task, "metric": "accuracy"}
        try:
            model = build_model(args.model, method, sp, bits, cal, args.group_size, **hp_kw)
            acc = evaluate(model, tok, eval_ds, args.task, DEV)
            row["value"] = f"{acc:.4f}"; row["total"] = len(eval_ds)
            row["seconds"] = f"{time.time() - t0:.1f}"
            append_row(args.csv, row)
            print(f"RESULT {method} sp={sp:.2f} bits={bits} hp={hp_label} acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            row["value"] = ""; row["error"] = str(e)[:200]
            append_row(args.csv, row)
            print(f"FAIL {method} sp={sp:.2f} bits={bits} hp={hp_label}: {e}", flush=True)


if __name__ == "__main__":
    main()
