#!/usr/bin/env python3
"""Unified apples-to-apples comparison of the column-BALANCED mask vs PUBLISHED baselines
(Wanda+AWQ, Wanda+SINQ, SparseGPT) and PRISM (internal ref), at 3-bit, BOTH uniform-0.70 and
v-dense — one harness, identical PPL + full downstream eval. Answers: does the balanced mask beat
the STANDARD published baselines, and does the v-dense mixed-sparsity lever (given to ALL arms)
change the ranking?  Every arm gets the same calibration, grid, evaluator, and (per --mode) v-dense.
"""
import argparse, os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

MODEL, NBITS, DEV = "gemma-2b", 3, "cuda"
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


def build_sp_by_type(model, mode, target=0.70):
    counts, tot = bs_param_fractions(model)   # MUST be pre-quant (quant replaces nn.Linear modules)
    v_frac = counts["v"] / tot
    if mode == "vdense":
        other = target / (1.0 - v_frac)
        sbt = {t: other for t in TYPES}; sbt["v"] = 0.0
    else:  # uniform
        sbt = {t: target for t in TYPES}
    gsp = sum(counts[t] * sbt.get(t, 0.70) for t in TYPES) / tot
    return sbt, v_frac, gsp


def bs_param_fractions(model):
    paths = bs.get_layer_paths(model)
    counts = {t: 0 for t in TYPES}
    for layer in bs.get_transformer_layers(model):
        for ap in paths:
            m = layer; ok = True
            for p in ap.split('.'):
                if not hasattr(m, p): ok = False; break
                m = getattr(m, p)
            if ok and isinstance(m, torch.nn.Linear):
                suf = ap.split('.')[-1]
                counts[suf[:-5] if suf.endswith('_proj') else suf] += m.weight.numel()
    return counts, sum(counts.values())


def global_sparsity(model, sbt):
    counts, tot = bs_param_fractions(model)
    return sum(counts[t] * sbt.get(t, 0.70) for t in TYPES) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--technique", required=True,
                    choices=["balanced", "wanda-awq", "wanda-sinq", "sparsegpt", "prism"])
    ap.add_argument("--mode", default="uniform", choices=["uniform", "vdense"])
    ap.add_argument("--target-global", type=float, default=0.70)
    ap.add_argument("--col-balance-exp", type=float, default=0.5)   # for balanced
    ap.add_argument("--ntest", type=int, default=40)
    ap.add_argument("--downstream", action="store_true")
    ap.add_argument("--tasks", default="arc_easy,hellaswag",
                    help="comma-sep downstream tasks (arc_easy,arc_challenge,hellaswag,mmlu). ARC runs "
                         "once for both easy+challenge. MMLU=5-shot full test unless --mmlu-limit.")
    ap.add_argument("--mmlu-limit", type=int, default=None, help="subsample MMLU test to N (None=full 14042)")
    ap.add_argument("--downstream-csv-dir", default=os.path.join(_ROOT, "results", "baseline_compare"))
    ap.add_argument("--technique-tag", default=None)
    args = ap.parse_args()

    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"], n_samples=args.ntest, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)

    sbt, v_frac, gsp = build_sp_by_type(model, args.mode, args.target_global)  # gsp from PRE-quant params
    print(f"[plan] technique={args.technique} mode={args.mode} v_frac={v_frac:.5f} "
          f"global_sparsity={gsp:.4f} sp_by_type={sbt}", flush=True)

    tq = args.technique
    if tq == "balanced":
        model, _ = ns.apply_wanda_obs_rtn(model, cal, NBITS, sbt, DEV, norm="col",
                                          mask_mode="balanced", dense_norm="col",
                                          col_balance_exp=args.col_balance_exp)
    elif tq == "wanda-awq":
        model = bs.apply_wanda_awq_quantization(model, cal, NBITS, args.target_global, DEV,
                                                sparsity_by_type=sbt)
    elif tq == "wanda-sinq":
        model = bs.apply_wanda_sinq_quantization(model, cal, NBITS, args.target_global, DEV,
                                                 sparsity_by_type=sbt)
    elif tq == "sparsegpt":
        model = bs.apply_sparsegpt_pruning(model, cal, args.target_global, NBITS, DEV,
                                           sparsity_by_type=sbt)
    elif tq == "prism":
        import diag_prism_vdense_downstream as dp  # noqa
        model, _ = dp.apply_prism_pertype(model, cal, NBITS, sbt, DEV)

    bs.move_final_layers_to_device(model, DEV)
    model.eval()
    ppl = bs.evaluate_perplexity(model, test, DEV)  # gsp computed pre-quant (above)
    tag = args.technique_tag or f"{tq}_{args.mode}"
    print(f"RESULT technique={tq} mode={args.mode} global_sparsity={gsp:.4f} ntest={args.ntest} "
          f"nbits={NBITS} ppl={ppl:.4f}", flush=True)

    if args.downstream:
        from downstream import run_downstream_suite  # noqa
        model.seqlen = bs.EVAL_CONFIG["seq_len"]
        ds_cfg = {"timestamp": None, "model": MODEL, "model_name": name, "technique": tag,
                  "precision": NBITS, "sparsity": round(gsp, 4), "dataset": "wikitext2"}
        ds_tasks = [t.strip() for t in args.tasks.split(",")]
        run_downstream_suite(model, tok, DEV, tasks=ds_tasks, limit=args.mmlu_limit,
                             config=ds_cfg, results_dir=args.downstream_csv_dir,
                             seqlen=bs.EVAL_CONFIG["seq_len"], verbose=True)
        print(f"DOWNSTREAM_DONE dir={args.downstream_csv_dir}", flush=True)


if __name__ == "__main__":
    main()
