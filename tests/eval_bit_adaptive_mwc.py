"""
Evaluate Bit-Adaptive MWC at 3-bit, 35% sparsity.

Key innovation: Scale MWC correction strength based on bit-width.
At 3-bit: 50% MWC + 50% standard OBS, tighter ratio cap (2.5), lower CV threshold (0.10)
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
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating PPL"):
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


def sparse_quantize_model(model, calibration_data, sparsity=0.35, nbits=3,
                          compensation_mode='bit_adaptive_mwc', device='cuda'):
    print(f"\nSparse quantization: {nbits}-bit, {sparsity*100:.0f}% sparsity, mode={compensation_mode}")

    layers_quantized = 0
    layer_activations = {}

    print("Collecting layer activations...")
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

    print(f"Collected activations for {len(layer_activations)} layers")

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Quantizing")):
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
                method='sinq_wanda_inverse' if activations is not None else 'sinq',
                device=device,
                use_compensation=activations is not None,
                compensation_mode=compensation_mode
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

    print(f"Quantized {layers_quantized} layers")
    return model


def test_model(model_name, nbits=3, sparsity=0.35):
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Config: {nbits}-bit, {sparsity*100:.0f}% sparsity")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=32)
    calibration_data = get_calibration_data(tokenizer, n_samples=32, seq_len=512)

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    ppl_fp16 = evaluate_perplexity(model, test_data, device)
    print(f"FP16 PPL: {ppl_fp16:.2f}")
    results['fp16'] = ppl_fp16
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Standard OBS baseline
    print("\n--- Standard OBS (Baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    model = sparse_quantize_model(model, calibration_data, sparsity=sparsity, nbits=nbits,
                                   compensation_mode='fast', device=device)
    ppl_std = evaluate_perplexity(model, test_data, device)
    print(f"Standard OBS PPL: {ppl_std:.2f}")
    results['std_obs'] = ppl_std
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Bit-Adaptive MWC
    print("\n--- Bit-Adaptive MWC ---")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    model = sparse_quantize_model(model, calibration_data, sparsity=sparsity, nbits=nbits,
                                   compensation_mode='bit_adaptive_mwc', device=device)
    ppl_bit_adaptive = evaluate_perplexity(model, test_data, device)
    print(f"Bit-Adaptive MWC PPL: {ppl_bit_adaptive:.2f}")
    results['bit_adaptive_mwc'] = ppl_bit_adaptive
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    print("="*70)
    print("BIT-ADAPTIVE MWC EVALUATION - 3-BIT, 35% SPARSITY")
    print("="*70)
    print("\nKey parameters at 3-bit:")
    print("  - correction_strength: 0.5 (50% MWC + 50% standard OBS)")
    print("  - ratio_cap: 3.75 (tighter than 4-bit's 5.0)")
    print("  - cv_threshold: 0.125 (lower than 4-bit's 0.15)")

    all_results = {}

    # Test 0.5B
    results_05b = test_model("Qwen/Qwen2.5-0.5B", nbits=3, sparsity=0.35)
    all_results['0.5B'] = results_05b

    torch.cuda.empty_cache()
    gc.collect()

    # Test 1.5B
    results_15b = test_model("Qwen/Qwen2.5-1.5B", nbits=3, sparsity=0.35)
    all_results['1.5B'] = results_15b

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS - 3-BIT, 35% SPARSITY")
    print("="*70)
    print(f"\n{'Model':<10} | {'FP16':>8} | {'Std OBS':>10} | {'Bit-Adapt MWC':>14} | {'vs Baseline':>12}")
    print("-"*70)

    for model_name, results in all_results.items():
        improvement = (results['std_obs'] - results['bit_adaptive_mwc']) / results['std_obs'] * 100
        print(f"{model_name:<10} | {results['fp16']:>8.2f} | {results['std_obs']:>10.2f} | {results['bit_adaptive_mwc']:>14.2f} | {improvement:>+11.2f}%")

    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS METHODS AT 3-BIT")
    print("="*70)
    print("\nPrevious Adaptive MWC at 3-bit (failed):")
    print("  0.5B: Std OBS 28.93 → Adaptive MWC 29.91 (-3.4% WORSE)")
    print("  1.5B: Std OBS 13.22 → Adaptive MWC 13.23 (neutral)")
    print("\nBit-Adaptive MWC at 3-bit:")
    for model_name, results in all_results.items():
        improvement = (results['std_obs'] - results['bit_adaptive_mwc']) / results['std_obs'] * 100
        status = "✓ BETTER" if improvement > 0 else "✗ WORSE" if improvement < -1 else "~ SAME"
        print(f"  {model_name}: Std OBS {results['std_obs']:.2f} → Bit-Adapt MWC {results['bit_adaptive_mwc']:.2f} ({improvement:+.2f}%) {status}")


if __name__ == '__main__':
    main()
