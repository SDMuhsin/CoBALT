#!/usr/bin/env python3
"""Config B for the PPL<->downstream divergence 2x2: PRISM (Sinkhorn) with an
optional GQA v-dense reallocation (v_proj sparsity=0, other types raised so the
GLOBAL sparsity stays == target), 3-bit, then run the SAME downstream suite
(identical adapters/splits/shots) as VALOR/nosink and the baselines.

Single activation-collection pass (matches nosink methodology). PRISM
quantization is bs.sparse_quantize_sinq with the canonical
method='sinq_wanda_inverse' + compensation_mode='prism' -- byte-identical call to
apply_prism_quantization_filtered, only the per-type sparsity differs.

  --vdense off (default): uniform PRISM at --target-global (== the deliverable PRISM).
  --vdense on:            v=0, others=target/(1-v_frac) so global==target.
"""
import argparse
import gc
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from tqdm import tqdm  # noqa: E402

MODEL = "gemma-2b"
NBITS = 3
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


def type_of(attr_path):
    s = attr_path.split('.')[-1]
    return s[:-len('_proj')] if s.endswith('_proj') else s


def param_fractions(model):
    paths = bs.get_layer_paths(model)
    counts = {t: 0 for t in TYPES}
    for layer in bs.get_transformer_layers(model):
        for attr_path in paths:
            parts = attr_path.split('.')
            m = layer
            ok = True
            for p in parts:
                if not hasattr(m, p):
                    ok = False
                    break
                m = getattr(m, p)
            if ok and isinstance(m, nn.Linear):
                counts[type_of(attr_path)] += m.weight.numel()
    tot = sum(counts.values())
    return counts, tot


def apply_prism_pertype(model, cal, nbits, sp_by_type, device='cuda'):
    is_prenorm = bs.is_prenorm_architecture(model)
    layer_activations = bs.collect_activations(model, cal, device)
    layer_paths = bs.get_layer_paths(model)
    stats = []
    for layer_idx, layer in enumerate(tqdm(bs.get_transformer_layers(model), desc=f"PRISM-pertype {nbits}b")):
        layer = layer.to(device)
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if not isinstance(linear, nn.Linear):
                continue
            t = type_of(attr_path)
            sp = sp_by_type.get(t, 0.70)
            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None
            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)
            # IDENTICAL call to apply_prism_quantization_filtered, only sparsity varies.
            W_q, scales, zeros, mask, scale2, meta = bs.sparse_quantize_sinq(
                W, activations,
                sparsity=sp,
                nbits=nbits,
                method='sinq_wanda_inverse',
                device=device,
                use_compensation=True if activations is not None and sp > 0 else False,
                compensation_mode='prism',
                is_prenorm=is_prenorm,
            )
            new_layer = bs.SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta).to(device)
            setattr(parent, parts[-1], new_layer)
            stats.append((t, W.numel(), sp))
            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()
        bs.set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()
    return model, stats


def global_sparsity(stats):
    tot = sum(n for _, n, _ in stats)
    pruned = sum(n * sp for _, n, sp in stats)
    return pruned / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-global", type=float, default=0.70)
    ap.add_argument("--vdense", action="store_true",
                    help="v=0, others=target/(1-v_frac) so GLOBAL stays == target")
    ap.add_argument("--ntest", type=int, default=20)
    ap.add_argument("--downstream-tasks", default="arc_easy,hellaswag")
    ap.add_argument("--downstream-limit", type=int, default=None)
    ap.add_argument("--downstream-csv-dir", default="/workspace/PTQResearch/results/diag_divergence")
    ap.add_argument("--technique-tag", default="prism-vdense")
    args = ap.parse_args()

    device = "cuda"
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"], n_samples=args.ntest, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu",
                                                 trust_remote_code=True, low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)

    counts, tot = param_fractions(model)
    v_frac = counts["v"] / tot
    if args.vdense:
        other = args.target_global / (1.0 - v_frac)
        sp_by_type = {t: other for t in TYPES}
        sp_by_type["v"] = 0.0
    else:
        sp_by_type = {t: args.target_global for t in TYPES}
    print(f"[plan] vdense={args.vdense} v_frac={v_frac:.5f} sp_by_type={sp_by_type}", flush=True)

    model, stats = apply_prism_pertype(model, cal, NBITS, sp_by_type, device)
    bs.move_final_layers_to_device(model, device)
    model.eval()
    gsp = global_sparsity(stats)
    ppl = bs.evaluate_perplexity(model, test, device)
    print(f"RESULT prism vdense={args.vdense} global_sparsity={gsp:.4f} ntest={args.ntest} "
          f"nbits={NBITS} ppl={ppl:.4f}", flush=True)

    from downstream import run_downstream_suite  # noqa: E402
    model.seqlen = bs.EVAL_CONFIG["seq_len"]
    ds_tasks = [t.strip() for t in args.downstream_tasks.split(",")]
    ds_cfg = {
        "timestamp": None,
        "model": MODEL,
        "model_name": name,
        "technique": args.technique_tag,
        "precision": NBITS,
        "sparsity": round(gsp, 4),
        "dataset": "wikitext2",
    }
    run_downstream_suite(model, tok, device, tasks=ds_tasks, limit=args.downstream_limit,
                         config=ds_cfg, results_dir=args.downstream_csv_dir,
                         seqlen=bs.EVAL_CONFIG["seq_len"], verbose=True)
    print(f"DOWNSTREAM_DONE dir={args.downstream_csv_dir}", flush=True)


if __name__ == "__main__":
    main()
