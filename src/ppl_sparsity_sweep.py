#!/usr/bin/env python3
"""Sweep wikitext2 perplexity vs sparsity for a prune+quant technique.

Reuses benchmark_suite's exact quantization + perplexity code so the numbers
match `--dataset wikitext2` runs. Reloads a fresh model for every sparsity
(quantization is destructive). Prints one machine-readable line per point:

    RESULT technique=<t> nbits=<n> sparsity=<s> ppl=<float>
    FAIL   technique=<t> nbits=<n> sparsity=<s> err=<msg>

Usage:
    python src/ppl_sparsity_sweep.py --technique wanda-sinq --nbits 3 \
        --sparsities 0.55,0.60,0.65 --n-test 40
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
import benchmark_suite as bs  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

APPLY = {
    "prism":       lambda m, cal, nb, sp, dev: bs.apply_prism_quantization(m, cal, nb, sp, dev),
    "wanda-sinq":  lambda m, cal, nb, sp, dev: bs.apply_wanda_sinq_quantization(m, cal, nb, sp, dev),
    "wanda-awq":   lambda m, cal, nb, sp, dev: bs.apply_wanda_awq_quantization(m, cal, nb, sp, dev),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2b")
    ap.add_argument("--technique", required=True, choices=list(APPLY))
    ap.add_argument("--nbits", type=int, default=3)
    ap.add_argument("--sparsities", required=True,
                    help="comma-separated, e.g. 0.55,0.60,0.65")
    ap.add_argument("--n-test", type=int, default=40, help="wikitext2 test samples (x2048)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sparsities = [float(x) for x in args.sparsities.split(",") if x.strip()]
    name = bs.MODELS[args.model]
    dev = args.device

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"],
                            n_samples=args.n_test, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                                  dataset_key="wikitext2")

    for sp in sparsities:
        try:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16, device_map="cpu",
                trust_remote_code=True, low_cpu_mem_usage=True)
            bs.move_embed_to_device(model, dev)
            model = APPLY[args.technique](model, cal, args.nbits, sp, dev)
            bs.move_final_layers_to_device(model, dev)
            model.eval()
            ppl = bs.evaluate_perplexity(model, test, dev)
            print(f"RESULT technique={args.technique} nbits={args.nbits} "
                  f"sparsity={sp:.4f} ppl={ppl:.4f}", flush=True)
            del model
        except Exception as e:  # noqa: BLE001
            print(f"FAIL technique={args.technique} nbits={args.nbits} "
                  f"sparsity={sp:.4f} err={type(e).__name__}: {e}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
