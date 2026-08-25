"""
MWC PPL Evaluation on Qwen2.5-1.5B for scalability test.
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


def sparse_quantize_model(model, calibration_data, sparsity=0.5, nbits=4,
                          use_compensation=False, compensation_mode='fast', device='cuda'):
    print(f"\nQuantizing: sparsity={sparsity}, compensation={use_compensation}, mode={compensation_mode}")
    layers_quantized = 0
    layer_activations = {}

    print("Collecting activations...")
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
                W, activations, sparsity=sparsity, nbits=nbits,
                method='sinq_wanda' if activations is not None else 'sinq',
                device=device, use_compensation=use_compensation and activations is not None,
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


def main():
    print("="*70)
    print("MWC PPL Evaluation on Qwen2.5-1.5B")
    print("="*70)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=32)
    calibration_data = get_calibration_data(tokenizer, n_samples=32, seq_len=512)
    results = {}
    sparsity = 0.35

    # FP16 Baseline
    print("\n" + "="*70 + "\nFP16 BASELINE\n" + "="*70)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)
    ppl_fp16 = evaluate_perplexity(model_fp16, test_data, device)
    print(f"FP16 Perplexity: {ppl_fp16:.2f}")
    results['FP16'] = ppl_fp16
    del model_fp16; gc.collect(); torch.cuda.empty_cache()

    # Standard OBS
    print("\n" + "="*70 + f"\nSINQ-SPARSE {sparsity*100:.0f}% (Standard OBS)\n" + "="*70)
    model_fast = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True)
    model_fast.model.embed_tokens = model_fast.model.embed_tokens.to(device)
    model_fast = sparse_quantize_model(model_fast, calibration_data, sparsity=sparsity, nbits=4,
                                        use_compensation=True, compensation_mode='fast', device=device)
    model_fast.model.norm = model_fast.model.norm.to(device)
    model_fast.lm_head = model_fast.lm_head.to(device)
    ppl_fast = evaluate_perplexity(model_fast, test_data, device)
    print(f"Standard OBS PPL: {ppl_fast:.2f}")
    results['Standard_OBS'] = ppl_fast
    del model_fast; gc.collect(); torch.cuda.empty_cache()

    # MWC
    print("\n" + "="*70 + f"\nSINQ-SPARSE {sparsity*100:.0f}% (MWC)\n" + "="*70)
    model_mwc = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True)
    model_mwc.model.embed_tokens = model_mwc.model.embed_tokens.to(device)
    model_mwc = sparse_quantize_model(model_mwc, calibration_data, sparsity=sparsity, nbits=4,
                                       use_compensation=True, compensation_mode='mwc', device=device)
    model_mwc.model.norm = model_mwc.model.norm.to(device)
    model_mwc.lm_head = model_mwc.lm_head.to(device)
    ppl_mwc = evaluate_perplexity(model_mwc, test_data, device)
    print(f"MWC PPL: {ppl_mwc:.2f}")
    results['MWC'] = ppl_mwc
    del model_mwc; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    for name, ppl in results.items():
        print(f"{name:<20}: {ppl:.2f}")

    std_ppl, mwc_ppl = results['Standard_OBS'], results['MWC']
    improvement = (std_ppl - mwc_ppl) / std_ppl * 100
    print(f"\nMWC Improvement: {improvement:+.2f}%")
    print("[SUCCESS]" if improvement > 0 else "[FAIL]")


if __name__ == '__main__':
    main()
