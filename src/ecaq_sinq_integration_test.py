"""
Test whether ECAQ scale adjustment can improve SINQ perplexity.

This test modifies SINQ's dual_shift.py to apply an ECAQ scale multiplier
after Sinkhorn normalization.

Hypothesis: If ECAQ scale_w < 1.0 helps, the resulting PPL should be < 22.39
"""

import os
import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import copy

# Import original SINQ components
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig
from sinq import dual_shift as ds
from sinq.sinkhorn import sinkhorn_log


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


# Create a modified quantize_dual_scale_shift that applies ECAQ scale
def make_ecaq_quantize_fn(scale_mult=1.0):
    """Create a modified quantization function with ECAQ scale multiplier."""

    def quantize_dual_scale_shift_ecaq(matrix, min_max, method='sinq', awq_scale=None):
        """Modified SINQ quantization with ECAQ scale adjustment."""
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Sinkhorn normalization (standard SINQ)
        matrix, mu1, mu2 = sinkhorn_log(matrix, 16)

        if not ('sinq' in method):
            matrix = matrix * mu1 * mu2
            mu1 = torch.ones_like(mu1)
            mu2 = torch.ones_like(mu2)

        if 'awq' in method:
            matrix = matrix * awq_scale
            mu1 = mu1 / awq_scale.float()

        # ECAQ modification: Apply scale multiplier to normalized matrix
        # This effectively adjusts the quantization tightness
        matrix_scaled = matrix * scale_mult

        if not ('hqq' in method):
            if 'noz' in method:
                q, scales, _ = ds.quantize_symmetric_rtn(matrix_scaled, min_max)
                q = q + min_max[1]//2
                z = torch.tensor(min_max[1] // 2)
            else:
                if "nf4" in method.lower():
                    q, scales, z, _ = ds.quantize_rtn(matrix_scaled, min_max, mode="nf4")
                elif "nf3" in method.lower():
                    q, scales, z, _ = ds.quantize_rtn(matrix_scaled, min_max, mode="nf3")
                else:
                    q, scales, z, _ = ds.quantize_rtn(matrix_scaled, min_max, mode="uniform")

        # Undo the ECAQ scaling in the scales
        scales = scales / scale_mult

        if 'hqq' in method:
            assert not ('noz' in method), 'noz incompatible with hqq'
            q, scales, z, _ = ds.hqq_rtn(matrix_scaled, min_max)
            best_error = torch.inf
            best_z = torch.zeros_like(z)
            best_scales = torch.ones_like(scales)
            for i in range(20):
                W_r, W_q, z, scales = ds.optimize_weights_proximal_legacy_step(matrix_scaled, scales.clip(1e-5,1e5), z, min_max)
                current_error = torch.abs(matrix_scaled - W_r).mean().float()
                take = current_error < best_error
                best_error  = torch.where(take, current_error, best_error)
                best_z      = torch.where(take[..., None], z, best_z)
                best_scales = torch.where(take[..., None], scales, best_scales)
            scales = best_scales / scale_mult  # Undo ECAQ scaling
            z = best_z
            q = W_q
            scales = 1/scales

        scales2 = torch.ones(1, matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
        scales = scales * mu2

        q = q.to(dtype).to(dev)
        s1 = scales.to(dtype)
        s2 = scales2.to(dtype)
        z = z.to(dtype).to(dev)

        return q, s1.to(dev), s2.to(dev), z

    return quantize_dual_scale_shift_ecaq


def test_ecaq_scale(model_name, nbits, group_size, scale_mult, device):
    """Test a specific ECAQ scale multiplier."""
    print(f"\nTesting scale_mult = {scale_mult}")

    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Monkey-patch the quantization function
    original_fn = ds.quantize_dual_scale_shift
    ds.quantize_dual_scale_shift = make_ecaq_quantize_fn(scale_mult)

    try:
        quant_config = BaseQuantizeConfig(
            nbits=nbits,
            group_size=group_size,
            axis=1,
            tiling_mode='1D',
            method='sinq_nogemlite'
        )

        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )

        ppl = evaluate_ppl(model, tokenizer, device=device)

    finally:
        # Restore original function
        ds.quantize_dual_scale_shift = original_fn

    del model
    torch.cuda.empty_cache()

    return ppl


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nbits", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=64)
    args = parser.parse_args()

    print("="*70)
    print("ECAQ + SINQ INTEGRATION TEST")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Bits: {args.nbits}")
    print(f"Baseline SINQ PPL: 22.39 (documented)")

    # Test different ECAQ scale multipliers
    scale_mults = [0.5, 0.7, 0.9, 1.0, 1.1]

    results = []
    for sm in scale_mults:
        ppl = test_ecaq_scale(args.model_name, args.nbits, args.group_size, sm, args.device)
        results.append((sm, ppl))
        print(f"scale_mult={sm:.1f}: PPL={ppl:.2f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'scale_mult':>12} {'PPL':>10} {'vs baseline':>15}")
    print("-"*40)

    baseline_ppl = 22.39
    for sm, ppl in results:
        diff = ppl - baseline_ppl
        sign = "+" if diff >= 0 else ""
        print(f"{sm:>12.1f} {ppl:>10.2f} {sign}{diff:>14.2f}")

    # Find best
    best_sm, best_ppl = min(results, key=lambda x: x[1])
    print(f"\nBest scale_mult: {best_sm} with PPL={best_ppl:.2f}")

    if best_ppl < baseline_ppl - 0.1:
        print(f"\n✓ ECAQ IMPROVES SINQ! Reduction: {baseline_ppl - best_ppl:.2f} PPL")
    elif best_ppl > baseline_ppl + 0.1:
        print(f"\n✗ ECAQ hurts performance. Increase: {best_ppl - baseline_ppl:.2f} PPL")
    else:
        print(f"\n≈ ECAQ has no significant effect on SINQ")


if __name__ == "__main__":
    main()
