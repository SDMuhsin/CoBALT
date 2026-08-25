"""
PPL Evaluation: Inverse SGSS

Key finding: Low-μ rows should be pruned MORE, high-μ rows pruned LESS.
This is the OPPOSITE of initial intuition.

Hypothesis: Low-μ rows have concentrated variance → individual weights matter less.
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
from sinq.dual_shift import quantize_rtn


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


class InvSGSSQuantizedLinear(nn.Module):
    """Quantized linear with inverse SGSS mask."""

    def __init__(self, weight, bias, activations, sparsity, nbits, group_size, scale=1.0):
        super().__init__()
        W = weight.data.float()
        K, N = W.shape
        n_prune = int(K * N * sparsity)

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2_flat = mu2.squeeze()

        # Importance
        act_norms = torch.norm(activations, dim=0).to(W.device)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2_flat.unsqueeze(1) + 1e-6)

        # Inverse SGSS: Low μ → more pruning
        mu2_norm = (mu2_flat - mu2_flat.min()) / (mu2_flat.max() - mu2_flat.min() + 1e-8)
        prune_weight = (1 - mu2_norm) ** scale
        prune_weight = prune_weight / prune_weight.sum() * n_prune

        n_prune_per_row = prune_weight.round().int()
        diff = n_prune - n_prune_per_row.sum().item()
        if diff > 0:
            indices = (1-mu2_norm).argsort(descending=True)[:abs(diff)]
            n_prune_per_row[indices] += 1
        elif diff < 0:
            indices = (1-mu2_norm).argsort()[:abs(diff)]
            n_prune_per_row[indices] -= 1
        n_prune_per_row = n_prune_per_row.clamp(0, N-1)

        # Create mask
        mask = torch.ones_like(W)
        for i in range(K):
            n_to_prune = n_prune_per_row[i].item()
            if n_to_prune > 0:
                _, prune_idx = importance[i].topk(n_to_prune, largest=False)
                mask[i, prune_idx] = 0

        # Quantize
        min_max = [0, 2**nbits - 1]
        W_sparse = W_norm * mask
        Q, scales, zeros, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)

        n_groups = N // group_size
        self.register_buffer('W_q', Q.to(torch.int8))
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mu1', mu1)
        self.register_buffer('mu2', mu2_flat)
        self.K, self.N = K, N
        self.group_size = group_size

        if bias is not None:
            self.register_buffer('bias', bias.data)
        else:
            self.bias = None

    def forward(self, x):
        input_dtype = x.dtype
        n_groups = self.scales.shape[1]
        Q_g = self.W_q.float().view(self.K, n_groups, self.group_size)
        W_deq_norm = (Q_g - self.zeros) * self.scales
        W_deq = W_deq_norm.view(self.K, self.N) * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq.to(input_dtype)
        return nn.functional.linear(x, W_deq, self.bias)


class UniformQuantizedLinear(nn.Module):
    """Standard uniform sparsity baseline."""

    def __init__(self, weight, bias, activations, sparsity, nbits, group_size):
        super().__init__()
        W = weight.data.float()
        K, N = W.shape
        n_prune = int(K * N * sparsity)

        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2_flat = mu2.squeeze()

        act_norms = torch.norm(activations, dim=0).to(W.device)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2_flat.unsqueeze(1) + 1e-6)

        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        min_max = [0, 2**nbits - 1]
        W_sparse = W_norm * mask
        Q, scales, zeros, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)

        n_groups = N // group_size
        self.register_buffer('W_q', Q.to(torch.int8))
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mu1', mu1)
        self.register_buffer('mu2', mu2_flat)
        self.K, self.N = K, N
        self.group_size = group_size

        if bias is not None:
            self.register_buffer('bias', bias.data)
        else:
            self.bias = None

    def forward(self, x):
        input_dtype = x.dtype
        n_groups = self.scales.shape[1]
        Q_g = self.W_q.float().view(self.K, n_groups, self.group_size)
        W_deq_norm = (Q_g - self.zeros) * self.scales
        W_deq = W_deq_norm.view(self.K, self.N) * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq.to(input_dtype)
        return nn.functional.linear(x, W_deq, self.bias)


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


def replace_layers(model, activations, sparsity, nbits, group_size, method='uniform', scale=1.0):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name and name in activations:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            act = activations[name].to(module.weight.device)
            if method == 'uniform':
                new_module = UniformQuantizedLinear(
                    module.weight, module.bias, act, sparsity, nbits, group_size
                )
            else:
                new_module = InvSGSSQuantizedLinear(
                    module.weight, module.bias, act, sparsity, nbits, group_size, scale=scale
                )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    return replaced


def main():
    print("="*70)
    print("INVERSE SGSS PPL EVALUATION")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    sparsity = 0.35
    nbits = 4
    group_size = 64

    # Test 1: Uniform baseline
    print("\n--- Uniform Sparsity (baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    replaced = replace_layers(model, activations, sparsity, nbits, group_size, method='uniform')
    print(f"Replaced {replaced} layers")
    ppl_uniform = eval_ppl(model, tokenizer, max_samples=32)
    print(f"Uniform PPL: {ppl_uniform:.2f}")
    del model, activations
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: Inverse SGSS (scale=0.5)
    print("\n--- Inverse SGSS (scale=0.5) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    replaced = replace_layers(model, activations, sparsity, nbits, group_size, method='inv_sgss', scale=0.5)
    print(f"Replaced {replaced} layers")
    ppl_inv05 = eval_ppl(model, tokenizer, max_samples=32)
    print(f"Inv-SGSS 0.5 PPL: {ppl_inv05:.2f}")
    del model, activations
    gc.collect()
    torch.cuda.empty_cache()

    # Test 3: Inverse SGSS (scale=1.0)
    print("\n--- Inverse SGSS (scale=1.0) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    replaced = replace_layers(model, activations, sparsity, nbits, group_size, method='inv_sgss', scale=1.0)
    print(f"Replaced {replaced} layers")
    ppl_inv10 = eval_ppl(model, tokenizer, max_samples=32)
    print(f"Inv-SGSS 1.0 PPL: {ppl_inv10:.2f}")
    del model, activations
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Uniform:        {ppl_uniform:.2f}")
    print(f"Inv-SGSS 0.5:   {ppl_inv05:.2f} ({(ppl_uniform-ppl_inv05)/ppl_uniform*100:+.2f}%)")
    print(f"Inv-SGSS 1.0:   {ppl_inv10:.2f} ({(ppl_uniform-ppl_inv10)/ppl_uniform*100:+.2f}%)")


if __name__ == '__main__':
    main()
