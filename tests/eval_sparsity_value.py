"""
Evaluate the VALUE of sparsity in SINQ.

Questions:
1. Does 35% sparsity + 4-bit beat 0% sparsity + 3-bit? (same memory footprint)
2. At what sparsity level does SINQ start hurting?
3. Is sparsity actually helping or just adding overhead?
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


class SINQQuantizedLinear(nn.Module):
    def __init__(self, weight, bias, activations, sparsity, nbits, group_size):
        super().__init__()
        W = weight.data.float()
        K, N = W.shape

        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2_flat = mu2.squeeze()

        if sparsity > 0 and activations is not None:
            n_prune = int(K * N * sparsity)
            act_norms = torch.norm(activations, dim=0).to(W.device)
            importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2_flat.unsqueeze(1) + 1e-6)
            threshold = importance.view(-1).sort().values[n_prune]
            mask = (importance > threshold).float()
            W_sparse = W_norm * mask
        else:
            W_sparse = W_norm

        min_max = [0, 2**nbits - 1]
        Q, scales, zeros, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)

        n_groups = N // group_size
        self.register_buffer('W_q', Q.to(torch.int8))
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mu1', mu1)
        self.register_buffer('mu2', mu2_flat)
        self.K, self.N = K, N
        self.group_size = group_size
        self.sparsity = sparsity

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


def replace_layers(model, activations, sparsity, nbits, group_size):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            act = activations.get(name)
            if act is not None:
                act = act.to(module.weight.device)

            new_module = SINQQuantizedLinear(
                module.weight, module.bias, act, sparsity, nbits, group_size
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    return replaced


def main():
    print("="*70)
    print("EVALUATING THE VALUE OF SPARSITY IN SINQ")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    group_size = 64

    configs = [
        # (sparsity, nbits, description)
        (0.0, 4, "Dense 4-bit"),
        (0.35, 4, "35% sparse 4-bit"),
        (0.50, 4, "50% sparse 4-bit"),
        (0.0, 3, "Dense 3-bit"),
        (0.35, 3, "35% sparse 3-bit"),
    ]

    results = {}

    for sparsity, nbits, desc in configs:
        print(f"\n--- {desc} ---")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
        )

        if sparsity > 0:
            activations = get_activations(model, tokenizer, n_samples=64)
        else:
            activations = {}

        replaced = replace_layers(model, activations, sparsity, nbits, group_size)
        print(f"Replaced {replaced} layers")

        ppl = eval_ppl(model, tokenizer, max_samples=32)
        print(f"{desc} PPL: {ppl:.2f}")
        results[(sparsity, nbits)] = ppl

        del model
        if activations:
            del activations
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<25} {'PPL':>10}")
    print("-" * 35)
    for (sparsity, nbits), ppl in results.items():
        desc = f"{sparsity:.0%} sparse {nbits}-bit"
        print(f"{desc:<25} {ppl:>10.2f}")

    # Memory analysis
    print("\n--- Memory Analysis (relative) ---")
    # Dense 4-bit: 4 bits per weight = 1.0x baseline
    # 35% sparse 4-bit: 0.65 * 4 bits + overhead ≈ 2.6 bits effective
    # Dense 3-bit: 3 bits per weight = 0.75x baseline
    # 35% sparse 3-bit: 0.65 * 3 bits + overhead ≈ 1.95 bits effective

    print(f"Dense 4-bit:      4.00 bits/weight, PPL = {results.get((0.0, 4), 'N/A'):.2f}")
    print(f"35% sparse 4-bit: 2.60 bits/weight*, PPL = {results.get((0.35, 4), 'N/A'):.2f}")
    print(f"Dense 3-bit:      3.00 bits/weight, PPL = {results.get((0.0, 3), 'N/A'):.2f}")
    print(f"50% sparse 4-bit: 2.00 bits/weight*, PPL = {results.get((0.50, 4), 'N/A'):.2f}")
    print("* Assumes sparse overhead is negligible")


if __name__ == '__main__':
    main()
