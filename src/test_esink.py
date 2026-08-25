"""
Test ESINK (Early-Stop Sinkhorn) vs standard SINQ on Qwen3-1.7B

This script:
1. Quantizes Qwen3-1.7B with standard SINQ (order=16)
2. Quantizes Qwen3-1.7B with ESINK (order=2)
3. Compares perplexity on WikiText-2

Target: Beat 22.39 perplexity with ESINK
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig
from sinq import sinkhorn
from sinq import dual_shift

# Add tests directory for evaluation utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
from eval_my.evaluate_ import evaluate_model

# Store original sinkhorn function
_original_sinkhorn_log = sinkhorn.sinkhorn_log


def patch_sinkhorn_order(order):
    """Patch sinkhorn_log to use specified order in all relevant modules."""
    def patched_sinkhorn_log(matrix, order_arg=16, **kwargs):
        # Override the order argument with our target order
        return _original_sinkhorn_log(matrix, order=order, **kwargs)

    # Patch in both modules
    sinkhorn.sinkhorn_log = patched_sinkhorn_log
    dual_shift.sinkhorn_log = patched_sinkhorn_log


def restore_sinkhorn():
    """Restore original sinkhorn function."""
    sinkhorn.sinkhorn_log = _original_sinkhorn_log
    dual_shift.sinkhorn_log = _original_sinkhorn_log


def evaluate_perplexity(model, tokenizer, device='cuda'):
    """Evaluate perplexity on WikiText-2 test set."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')

    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    total_tokens = 0

    for begin_loc in range(0, seq_len - max_length, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * (end_loc - begin_loc - 1)

        nlls.append(neg_log_likelihood)
        total_tokens += (end_loc - begin_loc - 1)

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()


def quantize_and_evaluate(model_name, order, device='cuda'):
    """Quantize model with specified Sinkhorn order and evaluate."""
    print(f"\n{'='*60}")
    print(f"Testing order={order}")
    print(f"{'='*60}")

    # Patch sinkhorn to use specified order
    patch_sinkhorn_order(order)

    # Load model (following reproduce_3bit_sinq.py pattern)
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create quantization config
    quant_config = BaseQuantizeConfig(
        nbits=3,
        group_size=64,
        axis=1,
        tiling_mode="1D",
        method="sinq" + "_nogemlite"  # Use slow inference (no gemlite)
    )

    # Quantize
    print("Quantizing model...")
    AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=device
    )

    # Model should already be on device after quantize_model
    model.eval()

    # Compile model for faster inference (like reproduce script)
    model = torch.compile(model)

    # Evaluate using the same method as reproduce script
    print("Evaluating perplexity on WikiText-2...")
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks="",
        eval_ppl="wikitext2",
        batch_size=8
    )
    ppl = results.get('wikitext2', float('inf'))

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Restore sinkhorn
    restore_sinkhorn()

    return ppl


def main():
    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("ESINK (Early-Stop Sinkhorn) Evaluation")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Target: Beat 22.39 perplexity")

    # Test different orders
    results = {}

    # Standard SINQ (order=16)
    results[16] = quantize_and_evaluate(model_name, order=16, device=device)
    print(f"\n>>> Order 16 (Standard SINQ): PPL = {results[16]:.2f}")

    # ESINK (order=2)
    results[2] = quantize_and_evaluate(model_name, order=2, device=device)
    print(f"\n>>> Order 2 (ESINK): PPL = {results[2]:.2f}")

    # Also test order=4 as backup
    results[4] = quantize_and_evaluate(model_name, order=4, device=device)
    print(f"\n>>> Order 4: PPL = {results[4]:.2f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for order, ppl in sorted(results.items()):
        marker = " <-- BEST" if ppl == min(results.values()) else ""
        print(f"Order {order:2d}: Perplexity = {ppl:.2f}{marker}")

    best_order = min(results, key=results.get)
    best_ppl = results[best_order]

    print(f"\nBest configuration: order={best_order} with PPL={best_ppl:.2f}")
    if best_ppl < 22.39:
        print(f"✓ SUCCESS: Achieved {22.39 - best_ppl:.2f} improvement over target!")
    else:
        print(f"✗ FAILED: {best_ppl - 22.39:.2f} worse than target")


if __name__ == "__main__":
    main()
