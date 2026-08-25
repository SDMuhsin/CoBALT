"""
Test: Iterative (SparseGPT-style) vs Batch OBS compensation with Inv-μ importance

Results so far at 50% sparsity:
- SCAB (Inv-μ + MWC): 44.49
- Inv-μ + batch OBS (fast): 26.57
- SparseGPT: 24.76

Can iterative compensation close the gap at high sparsity?
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq


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
    print("ITERATIVE vs BATCH OBS COMPENSATION")
    print("="*70)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]
    calib_tokens = tokenizer("\n\n".join(dataset_text[:16]), return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []

    configs = [
        ('sinq_wanda_inverse', 'fast', 'Inv-μ + Batch OBS'),
        ('sinq_wanda_inverse', 'batched_row_obs', 'Inv-μ + Row OBS'),
        ('sinq_wanda_inverse', 'sparsegpt', 'Inv-μ + SparseGPT-style'),
    ]

    for sparsity in [0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        for method, comp_mode, label in configs:
            print(f"\nTesting: {label}...")

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
            print(f"  {label}: PPL = {ppl:.2f}")

            results.append({
                'sparsity': sparsity,
                'label': label,
                'ppl': ppl
            })

            del model
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Sparsity':<10} {'Batch OBS':<15} {'Row OBS':<15} {'SparseGPT-style':<18} {'SparseGPT ref'}")
    print("-"*75)

    sparsegpt_ref = {0.50: 24.76, 0.60: 40.42}

    for sp in [0.50, 0.60]:
        batch = next((r['ppl'] for r in results if r['sparsity'] == sp and 'Batch' in r['label']), None)
        row = next((r['ppl'] for r in results if r['sparsity'] == sp and 'Row' in r['label']), None)
        sgpt_style = next((r['ppl'] for r in results if r['sparsity'] == sp and 'SparseGPT-style' in r['label']), None)
        ref = sparsegpt_ref[sp]

        print(f"{sp*100:>5.0f}%     {batch:<15.2f} {row:<15.2f} {sgpt_style:<18.2f} {ref}")

    return results


if __name__ == "__main__":
    run_test()
