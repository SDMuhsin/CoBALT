"""
Test: Is MWC compensation essential, or is it the importance criterion?

Previous tests showed:
- SCAB (inverse-μ importance + MWC compensation) = works
- OBS-SINQ (OBS importance + OBS compensation) = fails
- μ-Weighted OBS (OBS+inv-μ importance + OBS compensation) = fails

New hypothesis: The MWC compensation formula is essential for SINQ.

Test:
1. Inverse-μ importance + OBS compensation = ?
2. Inverse-μ importance + MWC compensation = works (SCAB)

If (1) fails, it proves MWC is essential, not the importance criterion.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sinq.sparse_quant import (
    sparse_quantize_sinq,
    dequantize_sparse_sinq
)


def evaluate_ppl(model, tokenizer, dataset_text, seq_len=2048, n_samples=16):
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer("\n\n".join(dataset_text), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    nlls = []
    for i in range(min(n_samples, len(input_ids) // seq_len)):
        begin = i * seq_len
        end = begin + seq_len
        batch = input_ids[begin:end].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            nlls.append(outputs.loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


def run_test():
    print("="*70)
    print("TEST: Is MWC Compensation Essential?")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]

    calib_text = "\n\n".join(dataset_text[:16])
    calib_tokens = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []
    sparsity = 0.50  # Fixed sparsity for comparison

    # Test different configurations
    configs = [
        # (importance_method, compensation_mode, label)
        ('sinq_wanda_inverse', 'bit_adaptive_mwc', 'SCAB (inv-μ + MWC)'),
        ('sinq_wanda_inverse', 'fast', 'Inv-μ + OBS comp'),
        ('sinq_wanda', 'bit_adaptive_mwc', 'Standard-μ + MWC'),
        ('sinq_wanda', 'fast', 'Standard-μ + OBS comp'),
    ]

    for method, comp_mode, label in configs:
        print(f"\nTesting: {label}")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True
        )

        n_layers = len(model.model.layers)

        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]

            activations_cache = {}
            def make_hook(name):
                def hook(m, inp, out):
                    activations_cache[name] = inp[0].detach()
                return hook

            handles = []
            for name, module in [
                ('q_proj', layer.self_attn.q_proj),
                ('k_proj', layer.self_attn.k_proj),
                ('v_proj', layer.self_attn.v_proj),
                ('o_proj', layer.self_attn.o_proj),
                ('up_proj', layer.mlp.up_proj),
                ('gate_proj', layer.mlp.gate_proj),
                ('down_proj', layer.mlp.down_proj),
            ]:
                handles.append(module.register_forward_hook(make_hook(name)))

            with torch.no_grad():
                _ = model(calib_ids)

            for h in handles:
                h.remove()

            for name, module in [
                ('q_proj', layer.self_attn.q_proj),
                ('k_proj', layer.self_attn.k_proj),
                ('v_proj', layer.self_attn.v_proj),
                ('o_proj', layer.self_attn.o_proj),
                ('up_proj', layer.mlp.up_proj),
                ('gate_proj', layer.mlp.gate_proj),
                ('down_proj', layer.mlp.down_proj),
            ]:
                W = module.weight.data.clone()
                X = activations_cache[name]

                W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                    W, X, sparsity=sparsity, nbits=4, group_size=64,
                    method=method,
                    use_compensation=True,
                    compensation_mode=comp_mode,
                    device=device
                )
                W_deq = dequantize_sparse_sinq(W_q, scales, zeros, mask, scale2, meta)
                module.weight.data = W_deq.to(module.weight.dtype)

        ppl = evaluate_ppl(model, tokenizer, dataset_text)
        print(f"  PPL = {ppl:.2f}")

        results.append({
            'config': label,
            'method': method,
            'comp_mode': comp_mode,
            'ppl': ppl
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Compensation Mode Impact at 50% Sparsity")
    print("="*70)
    print(f"{'Configuration':<30} {'PPL':<15}")
    print("-"*50)
    for r in results:
        print(f"{r['config']:<30} {r['ppl']:<15.2f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Analysis
    inv_mu_mwc = next(r['ppl'] for r in results if 'inv-μ' in r['config'] and 'MWC' in r['config'])
    inv_mu_obs = next(r['ppl'] for r in results if 'inv-μ' in r['config'] and 'OBS' in r['config'])
    std_mu_mwc = next(r['ppl'] for r in results if 'Standard' in r['config'] and 'MWC' in r['config'])
    std_mu_obs = next(r['ppl'] for r in results if 'Standard' in r['config'] and 'OBS' in r['config'])

    print(f"\n1. Effect of MWC vs OBS compensation (same importance):")
    print(f"   Inv-μ importance: MWC={inv_mu_mwc:.2f}, OBS={inv_mu_obs:.2f}")
    print(f"   Standard-μ importance: MWC={std_mu_mwc:.2f}, OBS={std_mu_obs:.2f}")

    print(f"\n2. Effect of importance criterion (same compensation):")
    print(f"   MWC comp: Inv-μ={inv_mu_mwc:.2f}, Standard-μ={std_mu_mwc:.2f}")
    print(f"   OBS comp: Inv-μ={inv_mu_obs:.2f}, Standard-μ={std_mu_obs:.2f}")

    if inv_mu_obs > 1000 and inv_mu_mwc < 100:
        print("\n>>> CONCLUSION: MWC compensation is ESSENTIAL for SINQ-sparse!")
        print("    Without MWC, even good importance criteria fail.")

    return results


if __name__ == "__main__":
    run_test()
