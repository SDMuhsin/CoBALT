"""
Reproduce SINQ Paper Results: 3-Bit Weight-Only PTQ on Qwen3-1.7B
Target perplexity: ~22.39 WikiText2

From Table 1 in the SINQ paper:
- Model: Qwen3-1.7B
- Method: SINQ (calibration-free, uniform quantization)
- Bits: 3-bit
- Group size: 64
- Tiling: 1D
- WikiText2 PPL: 22.39
- C4 PPL: 24.88
- Memory: 1.28 GB

Run from repository root:
    python src/reproduce_3bit_sinq.py
"""

import os
import sys
import json
from timeit import default_timer as timer
from datetime import datetime

import torch
import numpy as np
import random

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig

# Add tests directory for evaluation utilities
sys.path.insert(0, os.path.join(project_root, 'tests'))
from eval_my.evaluate_ import evaluate_model


def main():
    # Configuration - matches paper's Table 1
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    NBITS = 3
    GROUP_SIZE = 64
    TILING_MODE = "1D"
    METHOD = "sinq"  # calibration-free
    AXIS = 1
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Paper's target results
    TARGET_WIKITEXT2_PPL = 22.39
    TARGET_C4_PPL = 24.88
    TARGET_MEMORY_GB = 1.28

    print("=" * 60)
    print("SINQ Paper Reproduction: 3-Bit Qwen3-1.7B")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Bits: {NBITS}")
    print(f"  Group Size: {GROUP_SIZE}")
    print(f"  Tiling Mode: {TILING_MODE}")
    print(f"  Method: {METHOD}")
    print(f"  Device: {DEVICE}")
    print(f"\nTarget Results (from paper):")
    print(f"  WikiText2 PPL: {TARGET_WIKITEXT2_PPL}")
    print(f"  C4 PPL: {TARGET_C4_PPL}")
    print(f"  Memory: {TARGET_MEMORY_GB} GB")
    print("-" * 60)

    # Step 1: Load the model and tokenizer
    print(f"\n[1/4] Loading model: {MODEL_NAME}...")
    load_start = timer()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = timer() - load_start
    print(f"    Model loaded in {load_time:.2f} seconds")

    # Step 2: Configure quantization
    print(f"\n[2/4] Configuring {NBITS}-bit SINQ quantization...")

    quant_config = BaseQuantizeConfig(
        nbits=NBITS,
        group_size=GROUP_SIZE,
        axis=AXIS,
        tiling_mode=TILING_MODE,
        method=METHOD + "_nogemlite"  # Disable gemlite for evaluation compatibility
    )
    print(f"    Quant config: {quant_config}")

    # Step 3: Quantize the model
    print(f"\n[3/4] Quantizing model with SINQ...")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    quant_start = timer()
    AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=DEVICE
    )
    quant_time = timer() - quant_start

    # Get memory usage
    if torch.cuda.is_available():
        memory_allocated_gb = torch.cuda.memory_allocated() / 1e9
        memory_reserved_gb = torch.cuda.memory_reserved() / 1e9
    else:
        memory_allocated_gb = 0
        memory_reserved_gb = 0

    print(f"    Quantization completed in {quant_time:.2f} seconds")
    print(f"    CUDA Memory Allocated: {memory_allocated_gb:.2f} GB")
    print(f"    CUDA Memory Reserved: {memory_reserved_gb:.2f} GB")

    # Step 4: Evaluate perplexity
    print(f"\n[4/4] Evaluating perplexity...")

    # Compile the model for faster inference
    model = torch.compile(model)

    eval_start = timer()
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks="",
        eval_ppl="wikitext2",
        batch_size=8
    )
    eval_time = timer() - eval_start

    wikitext2_ppl = results.get('wikitext2', None)

    print(f"    Evaluation completed in {eval_time:.2f} seconds")

    # Print final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nWikiText2 Perplexity: {wikitext2_ppl:.2f}")
    print(f"Memory Allocated: {memory_allocated_gb:.2f} GB")
    print(f"Quantization Time: {quant_time:.2f} seconds")
    print(f"Evaluation Time: {eval_time:.2f} seconds")

    # Compare with paper results
    print("\n" + "-" * 60)
    print("COMPARISON WITH PAPER")
    print("-" * 60)
    ppl_diff = wikitext2_ppl - TARGET_WIKITEXT2_PPL if wikitext2_ppl else float('inf')
    mem_diff = memory_allocated_gb - TARGET_MEMORY_GB
    print(f"WikiText2 PPL: {wikitext2_ppl:.2f} (paper: {TARGET_WIKITEXT2_PPL}, diff: {ppl_diff:+.2f})")
    print(f"Memory: {memory_allocated_gb:.2f} GB (paper: {TARGET_MEMORY_GB} GB, diff: {mem_diff:+.2f} GB)")

    # Determine if reproduction was successful (within 5% of paper's PPL)
    if wikitext2_ppl and abs(ppl_diff) / TARGET_WIKITEXT2_PPL < 0.05:
        print("\n[SUCCESS] Results match paper within 5% tolerance!")
    elif wikitext2_ppl and abs(ppl_diff) / TARGET_WIKITEXT2_PPL < 0.10:
        print("\n[CLOSE] Results within 10% of paper's values.")
    else:
        print("\n[MISMATCH] Results differ significantly from paper.")

    # Save results
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "nbits": NBITS,
        "group_size": GROUP_SIZE,
        "tiling_mode": TILING_MODE,
        "method": METHOD,
        "device": DEVICE,
        "wikitext2_ppl": wikitext2_ppl,
        "memory_allocated_gb": memory_allocated_gb,
        "memory_reserved_gb": memory_reserved_gb,
        "quant_time_seconds": quant_time,
        "eval_time_seconds": eval_time,
        "target_wikitext2_ppl": TARGET_WIKITEXT2_PPL,
        "ppl_difference": ppl_diff if wikitext2_ppl else None,
    }

    results_path = os.path.join(project_root, "results", "3bit_sinq_qwen3_1.7b_results.json")
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results_dict


if __name__ == "__main__":
    results = main()
