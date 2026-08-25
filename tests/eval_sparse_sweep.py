"""
Sweep sparsity levels to find feasible operating point.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import gc
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq


def get_wikitext2(tokenizer, seq_len=2048, n_samples=32):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]
    n_tokens = len(input_ids)
    samples = []
    for i in range(0, min(n_tokens - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])
    return torch.stack(samples[:n_samples])


def get_calibration_data(tokenizer, n_samples=32, seq_len=512):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt', max_length=seq_len * n_samples, truncation=True)
    input_ids = encodings.input_ids[0]
    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])
    return torch.stack(samples[:n_samples])


@torch.no_grad()
def evaluate_perplexity(model, test_data, device='cuda', batch_size=1):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size].to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        n_tokens = batch.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


class SparseQuantLinear(nn.Module):
    def __init__(self, W_q, scales, zeros, mask, scale2, bias, meta):
        super().__init__()
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('scale2', scale2)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.meta = meta
        self._W_cached = None

    def forward(self, x):
        if self._W_cached is None:
            self._W_cached = dequantize_sparse_sinq(
                self.W_q, self.scales, self.zeros, self.mask, self.scale2, self.meta
            ).to(x.dtype)
        out = torch.matmul(x, self._W_cached.t())
        if self.bias is not None:
            out = out + self.bias
        return out


def sparse_quantize_model(model, calibration_data, sparsity=0.5, nbits=4,
                          use_compensation=False, device='cuda'):
    layers_quantized = 0
    layer_activations = {}

    if use_compensation:
        hooks = []
        activation_cache = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if name not in activation_cache:
                    activation_cache[name] = []
                activation_cache[name].append(input[0].detach().cpu())
            return hook_fn

        for layer_idx, layer in enumerate(model.model.layers):
            layer = layer.to(device)
            for attr_path in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                             'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
                parts = attr_path.split('.')
                module = layer
                for p in parts:
                    module = getattr(module, p)
                if isinstance(module, nn.Linear):
                    name = f'layer_{layer_idx}.{attr_path}'
                    hooks.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        with torch.no_grad():
            for i in range(min(8, len(calibration_data))):
                batch = calibration_data[i:i+1].to(device)
                try:
                    model(batch)
                except:
                    pass

        for h in hooks:
            h.remove()

        for name, acts in activation_cache.items():
            layer_activations[name] = torch.cat(acts, dim=0)

    for layer_idx, layer in enumerate(model.model.layers):
        layer = layer.to(device)

        for attr_path in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                         'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            parts = attr_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            linear = getattr(parent, parts[-1])

            if not isinstance(linear, nn.Linear):
                continue

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method='sinq_wanda' if activations is not None else 'sinq',
                device=device,
                use_compensation=use_compensation and activations is not None,
                compensation_mode='fast'
            )

            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            layers_quantized += 1
            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        model.model.layers[layer_idx] = layer
        gc.collect()
        torch.cuda.empty_cache()

    return model


def main():
    print("="*70)
    print("SINQ-Sparse Sparsity Sweep")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=32)
    calibration_data = get_calibration_data(tokenizer, n_samples=32, seq_len=512)

    results = {}

    # FP16 Baseline
    print("\n" + "="*70)
    print("FP16 BASELINE")
    print("="*70)

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True,
    )
    model_fp16.eval()
    ppl_fp16 = evaluate_perplexity(model_fp16, test_data, device)
    print(f"FP16 Perplexity: {ppl_fp16:.2f}")
    results['FP16'] = ppl_fp16
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # SINQ Dense
    print("\n" + "="*70)
    print("SINQ DENSE (0% sparsity)")
    print("="*70)

    model_sinq = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu',
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model_sinq.model.embed_tokens = model_sinq.model.embed_tokens.to(device)
    model_sinq = sparse_quantize_model(model_sinq, calibration_data, sparsity=0.0, use_compensation=False, device=device)
    model_sinq.model.norm = model_sinq.model.norm.to(device)
    model_sinq.lm_head = model_sinq.lm_head.to(device)
    model_sinq.eval()
    ppl_sinq = evaluate_perplexity(model_sinq, test_data, device)
    print(f"SINQ Dense Perplexity: {ppl_sinq:.2f}")
    results['SINQ_Dense'] = ppl_sinq
    del model_sinq
    gc.collect()
    torch.cuda.empty_cache()

    # Sweep sparsity levels with compensation
    sparsity_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for sparsity in sparsity_levels:
        print(f"\n{'='*70}")
        print(f"SINQ-Sparse {int(sparsity*100)}% (with compensation)")
        print("="*70)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='cpu',
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model = sparse_quantize_model(model, calibration_data, sparsity=sparsity,
                                      use_compensation=True, device=device)
        model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)
        model.eval()

        ppl = evaluate_perplexity(model, test_data, device)
        ratio = ppl / ppl_sinq * 100
        print(f"PPL: {ppl:.2f} ({ratio:.1f}% of SINQ baseline)")
        results[f'Sparse_{int(sparsity*100)}'] = ppl

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Method':<20} | {'PPL':>10} | {'vs SINQ':>12} | {'Status':>10}")
    print("-"*60)

    for name, ppl in results.items():
        ratio = ppl / ppl_sinq * 100
        status = "PASS" if ratio <= 110 else "FAIL"
        print(f"{name:<20} | {ppl:>10.2f} | {ratio:>11.1f}% | {status:>10}")

    # Find max sparsity that meets target
    print("\n" + "="*70)
    print("MAXIMUM FEASIBLE SPARSITY")
    print("="*70)

    max_sparsity = 0
    for sparsity in sparsity_levels:
        key = f'Sparse_{int(sparsity*100)}'
        if key in results and results[key] / ppl_sinq <= 1.10:
            max_sparsity = sparsity

    if max_sparsity > 0:
        print(f"\nMax sparsity meeting 110% target: {int(max_sparsity*100)}%")
    else:
        print("\nNo sparsity level meets the 110% target.")
        # Find closest
        best = min([(k, v/ppl_sinq) for k, v in results.items() if 'Sparse' in k], key=lambda x: x[1])
        print(f"Closest: {best[0]} at {best[1]*100:.1f}% of SINQ baseline")


if __name__ == '__main__':
    main()
