"""
Test: Layer-Wise Adaptive Sparsity

Key insight from GCLI analysis: Global ranking reveals natural per-layer sparsity:
- V_proj layers: 95-100% (can be heavily pruned)
- K_proj, gate_proj: 18-25% (should keep more)

Instead of global ranking, use LEARNED per-layer sparsity ratios
from the global analysis, applied with proper SINQ framework.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq


# Per-layer sparsity from GCLI analysis (capped at 80% for safety)
LAYER_SPARSITY = {
    'v_proj': 0.80,  # Originally 95-100%, cap at 80%
    'o_proj': 0.50,  # Originally 40-65%
    'q_proj': 0.50,  # Originally 30-80%
    'k_proj': 0.30,  # Originally 18-40%
    'gate_proj': 0.25,  # Originally 18-35%
    'up_proj': 0.40,  # Originally 20-70%
    'down_proj': 0.30,  # Originally 20-40%
}


def get_layer_sparsity(name):
    """Get adaptive sparsity for a layer based on type."""
    for key, sparsity in LAYER_SPARSITY.items():
        if key in name:
            return sparsity
    return 0.35  # Default


def eval_ppl(model, tokenizer, device='cuda', max_samples=32):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)

    model.eval()
    seq_len = 2048
    n_samples = min(max_samples, (input_ids.shape[1] - 1) // seq_len)
    total_loss = 0.0

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="PPL"):
            start = i * seq_len
            inputs = input_ids[:, start:start+seq_len]
            targets = input_ids[:, start+1:start+seq_len+1]
            outputs = model(inputs)
            logits = outputs.logits[:, :-1, :]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets[:, :-1].reshape(-1)
            )
            total_loss += loss.item()

    return torch.exp(torch.tensor(total_loss / n_samples)).item()


class AdaptiveQuantizedLinear(nn.Module):
    def __init__(self, weight, bias, activations, sparsity, nbits):
        super().__init__()
        W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
            weight.data, activations, sparsity=sparsity, nbits=nbits, group_size=64,
            method='sinq_wanda_inverse', structured='unstructured',
            device=weight.device, use_compensation=True, compensation_mode='fast'
        )
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('scale2', scale2)
        self.meta = meta
        self.register_buffer('bias', bias.data if bias is not None else None)

    def forward(self, x):
        input_dtype = x.dtype
        W = dequantize_sparse_sinq(self.W_q, self.scales, self.zeros, self.mask, self.scale2, self.meta)
        W = W.to(input_dtype)
        return nn.functional.linear(x, W, self.bias)


def get_activations(model, tokenizer, n_samples=64):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset['text'][:1000])
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512*n_samples)
    input_ids = encodings.input_ids[0]

    activations = {}
    def hook_fn(name):
        def fn(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(input[0].detach().cpu())
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            batch = input_ids[i*512:(i+1)*512].unsqueeze(0).to(model.device)
            if batch.shape[1] == 0:
                break
            model(batch)

    for hook in hooks:
        hook.remove()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).view(-1, activations[name][0].shape[-1])

    return activations


def quantize_adaptive(model, activations, nbits=4):
    """Replace layers with adaptive sparsity."""
    replaced = 0
    total_weights = 0
    total_pruned = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name and name in activations:
            sparsity = get_layer_sparsity(name)
            n_weights = module.weight.numel()
            total_weights += n_weights
            total_pruned += n_weights * sparsity

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            act = activations[name].to(module.weight.device)
            new_module = AdaptiveQuantizedLinear(
                module.weight, module.bias, act, sparsity, nbits
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    actual_sparsity = total_pruned / total_weights
    print(f"Replaced {replaced} layers with adaptive sparsity")
    print(f"Effective total sparsity: {actual_sparsity:.1%}")


def quantize_uniform(model, activations, sparsity=0.35, nbits=4):
    """Replace layers with uniform sparsity."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name and name in activations:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            act = activations[name].to(module.weight.device)
            new_module = AdaptiveQuantizedLinear(
                module.weight, module.bias, act, sparsity, nbits
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    print(f"Replaced {replaced} layers with uniform {sparsity:.0%} sparsity")


def main():
    print("="*70)
    print("LAYER-WISE ADAPTIVE SPARSITY TEST")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Test 1: Uniform sparsity (baseline)
    print("\n--- Uniform Sparsity (35%) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    quantize_uniform(model, activations, sparsity=0.35, nbits=4)
    ppl_uniform = eval_ppl(model, tokenizer, max_samples=32)
    print(f"Uniform PPL: {ppl_uniform:.2f}")
    del model, activations
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: Adaptive sparsity
    print("\n--- Adaptive Sparsity (layer-dependent) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    quantize_adaptive(model, activations, nbits=4)
    ppl_adaptive = eval_ppl(model, tokenizer, max_samples=32)
    print(f"Adaptive PPL: {ppl_adaptive:.2f}")
    del model, activations
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Uniform (35%):   {ppl_uniform:.2f}")
    print(f"Adaptive:        {ppl_adaptive:.2f}")
    improvement = (ppl_uniform - ppl_adaptive) / ppl_uniform * 100
    print(f"Improvement: {improvement:+.2f}%")


if __name__ == '__main__':
    main()
