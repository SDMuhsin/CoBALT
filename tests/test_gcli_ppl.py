"""
PPL Test: Global Cross-Layer Importance (GCLI)

Tests whether global importance ranking (with per-layer constraints)
outperforms uniform per-layer sparsity.

Constraint: At least 10% of weights kept in each layer (max 90% sparse)
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from sinq.sinkhorn import sinkhorn_log
from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq


def get_wikitext2(tokenizer, seq_len=2048):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    return encodings.input_ids


def eval_ppl(model, tokenizer, device='cuda', max_samples=32):
    model.eval()
    input_ids = get_wikitext2(tokenizer).to(device)
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


class GCLIQuantizedLinear(nn.Module):
    """Quantized linear with pre-computed global sparsity mask."""

    def __init__(self, weight, bias, mask, nbits=4, group_size=64):
        super().__init__()
        W = weight.data.float()
        K, N = W.shape

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Apply pre-computed mask and quantize
        W_sparse = W_norm * mask

        min_max = [0, 2**nbits - 1]
        from sinq.dual_shift import quantize_rtn
        Q, scales, zeros, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)

        self.register_buffer('W_q', Q.to(torch.int8))
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('mu1', mu1)
        self.register_buffer('mu2', mu2)

        if bias is not None:
            self.register_buffer('bias', bias.data)
        else:
            self.bias = None

        self.K, self.N = K, N
        self.group_size = group_size

    def forward(self, x):
        input_dtype = x.dtype
        n_groups = self.scales.shape[1]
        Q_grouped = self.W_q.float().view(self.K, n_groups, self.group_size)
        W_deq_norm = (Q_grouped - self.zeros) * self.scales
        W_deq = W_deq_norm.view(self.K, self.N) * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq.to(input_dtype)
        return nn.functional.linear(x, W_deq, self.bias)


def compute_global_masks(model, total_sparsity=0.35, max_layer_sparsity=0.90):
    """Compute global importance and create per-layer masks with constraints."""
    print("Computing global importance...")

    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            W = module.weight.data.float()
            K, N = W.shape

            W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
            mu2 = mu2.squeeze()

            # Inverse μ importance
            importance = W.abs() / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

            layer_info.append({
                'name': name,
                'module': module,
                'importance': importance.flatten().cpu(),
                'shape': (K, N),
                'n_weights': K * N
            })

    total_weights = sum(d['n_weights'] for d in layer_info)
    n_prune_total = int(total_weights * total_sparsity)

    # Concatenate all importance
    all_importance = torch.cat([d['importance'] for d in layer_info])

    # Global threshold
    global_threshold = all_importance.sort().values[n_prune_total].item()

    # Create masks with constraints
    masks = {}
    actual_pruned = 0

    for d in layer_info:
        layer_imp = d['importance']
        n_layer = d['n_weights']

        # Base mask from global threshold
        base_mask = (layer_imp > global_threshold).float()
        base_sparsity = 1 - base_mask.mean().item()

        # Apply constraint: max sparsity
        if base_sparsity > max_layer_sparsity:
            # Keep at least (1 - max_layer_sparsity) of weights
            n_keep = int(n_layer * (1 - max_layer_sparsity))
            threshold = layer_imp.sort(descending=True).values[n_keep - 1].item()
            mask = (layer_imp >= threshold).float()
        else:
            mask = base_mask

        masks[d['name']] = mask.view(d['shape']).to(d['module'].weight.device)
        actual_pruned += (1 - mask.mean().item()) * n_layer

    actual_sparsity = actual_pruned / total_weights
    print(f"Target sparsity: {total_sparsity:.1%}, Actual: {actual_sparsity:.1%}")

    return masks


def quantize_with_gcli(model, masks, nbits=4):
    """Replace layers with GCLI quantized versions."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if name in masks and isinstance(module, nn.Linear):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            new_module = GCLIQuantizedLinear(
                module.weight, module.bias, masks[name], nbits=nbits
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    print(f"Replaced {replaced} layers")


def quantize_uniform(model, tokenizer, sparsity=0.35, nbits=4):
    """Standard uniform per-layer sparsity using official SINQ."""
    # Get activations
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset['text'][:1000])
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512*64)
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
        for i in range(64):
            batch = input_ids[i*512:(i+1)*512].unsqueeze(0).to(model.device)
            if batch.shape[1] == 0:
                break
            model(batch)

    for hook in hooks:
        hook.remove()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).view(-1, activations[name][0].shape[-1])

    # Replace layers
    class UniformQuantizedLinear(nn.Module):
        def __init__(self, weight, bias, act, sparsity, nbits):
            super().__init__()
            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                weight.data, act, sparsity=sparsity, nbits=nbits, group_size=64,
                method='sinq_wanda_inverse', structured='unstructured',
                device=weight.device, use_compensation=True, compensation_mode='fast'
            )
            self.register_buffer('W_q', W_q)
            self.register_buffer('scales', scales)
            self.register_buffer('zeros', zeros)
            self.register_buffer('mask', mask)
            self.register_buffer('scale2', scale2)
            self.meta = meta
            if bias is not None:
                self.register_buffer('bias', bias.data)
            else:
                self.bias = None

        def forward(self, x):
            W = dequantize_sparse_sinq(self.W_q, self.scales, self.zeros, self.mask, self.scale2, self.meta)
            return nn.functional.linear(x.to(W.dtype), W, self.bias)

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name and name in activations:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            act = activations[name].to(module.weight.device)
            new_module = UniformQuantizedLinear(module.weight, module.bias, act, sparsity, nbits)
            setattr(parent, parts[-1], new_module)
            replaced += 1

    print(f"Replaced {replaced} layers (uniform)")
    del activations
    gc.collect()


def main():
    print("="*70)
    print("GCLI PPL TEST")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Test 1: GCLI (global ranking with 90% max constraint)
    print("\n--- GCLI (Global Cross-Layer Importance) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    masks = compute_global_masks(model, total_sparsity=0.35, max_layer_sparsity=0.90)
    quantize_with_gcli(model, masks, nbits=4)
    ppl_gcli = eval_ppl(model, tokenizer, max_samples=32)
    print(f"GCLI PPL: {ppl_gcli:.2f}")
    del model, masks
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: Uniform sparsity (baseline)
    print("\n--- Uniform Sparsity (baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    quantize_uniform(model, tokenizer, sparsity=0.35, nbits=4)
    ppl_uniform = eval_ppl(model, tokenizer, max_samples=32)
    print(f"Uniform PPL: {ppl_uniform:.2f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"GCLI (global ranking):  {ppl_gcli:.2f}")
    print(f"Uniform (per-layer):    {ppl_uniform:.2f}")
    improvement = (ppl_uniform - ppl_gcli) / ppl_uniform * 100
    print(f"Improvement: {improvement:+.2f}%")


if __name__ == '__main__':
    main()
