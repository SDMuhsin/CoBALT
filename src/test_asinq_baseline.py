"""
Test A-SINQ (Calibrated SINQ) Baseline
This will establish what A-SINQ achieves on 3-bit Qwen3-1.7B WikiText2
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
    # Configuration - matches paper's Table 1 but with A-SINQ (calibrated)
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    NBITS = 3
    GROUP_SIZE = 64
    TILING_MODE = "1D"
    METHOD = "asinq"  # CALIBRATED version with AWQ scales
    AXIS = 1
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Baselines
    SINQ_PPL = 22.39

    print("=" * 60)
    print("A-SINQ (Calibrated) Baseline Test: 3-Bit Qwen3-1.7B")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Bits: {NBITS}")
    print(f"  Group Size: {GROUP_SIZE}")
    print(f"  Tiling Mode: {TILING_MODE}")
    print(f"  Method: {METHOD}")
    print(f"  Device: {DEVICE}")
    print(f"\nBaseline to beat:")
    print(f"  SINQ (calibration-free) PPL: {SINQ_PPL}")
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
    print(f"\n[2/4] Configuring {NBITS}-bit A-SINQ quantization...")

    quant_config = BaseQuantizeConfig(
        nbits=NBITS,
        group_size=GROUP_SIZE,
        axis=AXIS,
        tiling_mode=TILING_MODE,
        method=METHOD + "_nogemlite"  # A-SINQ without gemlite
    )
    print(f"    Quant config: {quant_config}")

    # Step 3: Quantize the model
    print(f"\n[3/4] Quantizing model with A-SINQ (calibrated)...")
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
    print("A-SINQ RESULTS")
    print("=" * 60)
    print(f"\nWikiText2 Perplexity: {wikitext2_ppl:.2f}")
    print(f"Memory Allocated: {memory_allocated_gb:.2f} GB")
    print(f"Quantization Time: {quant_time:.2f} seconds")

    # Compare with baselines
    print("\n" + "-" * 60)
    print("COMPARISON")
    print("-" * 60)
    improvement = SINQ_PPL - wikitext2_ppl if wikitext2_ppl else 0
    print(f"A-SINQ PPL: {wikitext2_ppl:.2f}")
    print(f"SINQ PPL:   {SINQ_PPL:.2f}")
    print(f"Improvement: {improvement:.2f} PPL ({100*improvement/SINQ_PPL:.1f}%)")

    if wikitext2_ppl and wikitext2_ppl < SINQ_PPL:
        print(f"\n[SUCCESS] A-SINQ beats SINQ by {improvement:.2f} PPL!")
    else:
        print(f"\n[NOTE] A-SINQ did not improve over SINQ.")

    return wikitext2_ppl


if __name__ == "__main__":
    results = main()
