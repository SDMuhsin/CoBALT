"""
PPL Evaluation: Layer-Adaptive α

Key insight: Different layer types might benefit from different μ power scaling.
Single-layer MSE shows:
- q_proj, k_proj: α=0 is 50% better than α=1
- gate_proj: α=0 is 9% better than α=1

But global α=0 hurts PPL. Maybe some layers need α=1 while others need α=0.

Test: Use α=0 for attention layers (q,k,v) and α=1 for MLP layers.
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


class AdaptiveAlphaQuantizedLinear(nn.Module):
    def __init__(self, weight, bias, activations, sparsity, nbits, group_size, alpha):
        super().__init__()
        W = weight.data.float()
        K, N = W.shape
        n_prune = int(K * N * sparsity)

        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2_flat = mu2.squeeze()

        act_norms = torch.norm(activations, dim=0).to(W.device)

        if alpha == 0:
            importance = W.abs() * act_norms.unsqueeze(0)
        else:
            mu_factor = (mu1.unsqueeze(0) * mu2_flat.unsqueeze(1)) ** alpha
            importance = W.abs() * act_norms.unsqueeze(0) / (mu_factor + 1e-6)

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


def get_alpha_for_layer(name, strategy):
    """Get α based on layer name and strategy."""
    if strategy == 'uniform_0':
        return 0.0
    elif strategy == 'uniform_1':
        return 1.0
    elif strategy == 'attn_0':
        # α=0 for attention layers, α=1 for MLP
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 0.0
        return 1.0
    elif strategy == 'attn_05':
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 0.5
        return 1.0
    elif strategy == 'mlp_0':
        # α=0 for MLP, α=1 for attention
        if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            return 0.0
        return 1.0
    elif strategy == 'qkv_0':
        # Only α=0 for q,k,v (not o_proj)
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj']):
            return 0.0
        return 1.0
    return 1.0


def replace_layers(model, activations, sparsity, nbits, group_size, strategy):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name and name in activations:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            alpha = get_alpha_for_layer(name, strategy)
            act = activations[name].to(module.weight.device)
            new_module = AdaptiveAlphaQuantizedLinear(
                module.weight, module.bias, act, sparsity, nbits, group_size, alpha=alpha
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    return replaced


def main():
    print("="*70)
    print("LAYER-ADAPTIVE α PPL EVALUATION")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    sparsity = 0.35
    nbits = 4
    group_size = 64

    strategies = [
        ('uniform_1', 'All layers α=1 (baseline)'),
        ('attn_0', 'Attention α=0, MLP α=1'),
        ('attn_05', 'Attention α=0.5, MLP α=1'),
        ('mlp_0', 'MLP α=0, Attention α=1'),
        ('qkv_0', 'QKV α=0, rest α=1'),
    ]

    results = {}

    for strategy, desc in strategies:
        print(f"\n--- {desc} ---")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
        )
        activations = get_activations(model, tokenizer, n_samples=64)
        replaced = replace_layers(model, activations, sparsity, nbits, group_size, strategy=strategy)
        print(f"Replaced {replaced} layers")
        ppl = eval_ppl(model, tokenizer, max_samples=32)
        print(f"{strategy} PPL: {ppl:.2f}")
        results[strategy] = ppl
        del model, activations
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    baseline = results['uniform_1']
    for strategy, desc in strategies:
        change = (baseline - results[strategy]) / baseline * 100
        print(f"{strategy:15s}: {results[strategy]:.2f} ({change:+.2f}%) - {desc}")


if __name__ == '__main__':
    main()
