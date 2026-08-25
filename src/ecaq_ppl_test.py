"""
Proper ECAQ perplexity evaluation using SINQ infrastructure.

This test compares:
1. SINQ baseline (3-bit weights, no KV quantization)
2. SINQ + ECAQ (3-bit weights with scale adjustment)

Target: WikiText-2 perplexity
Baseline SINQ 3-bit: 22.39 PPL (from docs)
"""

import os
import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def get_wikitext2_loader(tokenizer, seqlen=2048):
    """Load WikiText-2 test set."""
    from datasets import load_dataset

    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc


@torch.no_grad()
def evaluate_ppl(model, tokenizer, seqlen=2048, device='cuda'):
    """Evaluate perplexity on WikiText-2."""
    testenc = get_wikitext2_loader(tokenizer, seqlen)
    testenc = testenc.input_ids

    nsamples = testenc.numel() // seqlen
    model.eval()

    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(device)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


def quantize_model_sinq(model_name, nbits=3, group_size=64, device='cuda', method='sinq'):
    """Quantize model using SINQ."""
    print(f"\nLoading and quantizing model with SINQ ({nbits}-bit, method={method})...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BaseQuantizeConfig(
        nbits=nbits,
        group_size=group_size,
        axis=1,
        tiling_mode='1D',
        method=method
    )

    AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=device
    )

    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nbits", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=64)
    args = parser.parse_args()

    print("="*70)
    print("ECAQ PERPLEXITY EVALUATION")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Bits: {args.nbits}")
    print(f"Group size: {args.group_size}")
    print(f"Device: {args.device}")

    # Test 1: SINQ baseline
    print("\n" + "="*70)
    print("TEST 1: SINQ BASELINE")
    print("="*70)

    model_sinq, tokenizer = quantize_model_sinq(
        args.model_name,
        nbits=args.nbits,
        group_size=args.group_size,
        device=args.device,
        method='sinq_nogemlite'
    )

    ppl_sinq = evaluate_ppl(model_sinq, tokenizer, device=args.device)
    print(f"\nSINQ {args.nbits}-bit PPL: {ppl_sinq:.2f}")

    # Clean up
    del model_sinq
    torch.cuda.empty_cache()

    # Test 2: FP16 baseline (for reference)
    print("\n" + "="*70)
    print("TEST 2: FP16 BASELINE")
    print("="*70)

    model_fp = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppl_fp = evaluate_ppl(model_fp, tokenizer, device=args.device)
    print(f"\nFP16 PPL: {ppl_fp:.2f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"FP16 baseline:     {ppl_fp:.2f}")
    print(f"SINQ {args.nbits}-bit:       {ppl_sinq:.2f}")
    print(f"Degradation:       +{ppl_sinq - ppl_fp:.2f} ({(ppl_sinq/ppl_fp - 1)*100:.1f}%)")

    # Compare with documented baseline
    if args.nbits == 3:
        documented_ppl = 22.39
        print(f"\nDocumented SINQ 3-bit PPL: {documented_ppl}")
        if abs(ppl_sinq - documented_ppl) < 1.0:
            print("✓ Result matches documented baseline")
        else:
            print(f"⚠ Result differs from documented baseline by {abs(ppl_sinq - documented_ppl):.2f}")


if __name__ == "__main__":
    main()
