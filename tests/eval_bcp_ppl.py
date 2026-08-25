"""
PPL Evaluation: Benefit/Cost Pruning (BCP)

BCP adds per-group quantization scale to importance:
importance_bcp = |W| × ||X|| / (μ₁ × μ₂ × scale)

This penalizes weights in groups with high quantization scales (higher error).
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_activation_norms


def get_wikitext2(tokenizer, seq_len=2048, split='test'):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    return encodings.input_ids


def eval_ppl(model, tokenizer, device='cuda', seq_len=2048, max_samples=64):
    model.eval()
    input_ids = get_wikitext2(tokenizer, seq_len=seq_len).to(device)
    n_samples = min(max_samples, (input_ids.shape[1] - 1) // seq_len)
    total_loss = 0.0

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="PPL"):
            start = i * seq_len
            end = start + seq_len
            inputs = input_ids[:, start:end]
            targets = input_ids[:, start+1:end+1]
            outputs = model(inputs)
            logits = outputs.logits[:, :-1, :]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets[:, :-1].reshape(-1)
            )
            total_loss += loss.item()

    ppl = torch.exp(torch.tensor(total_loss / n_samples)).item()
    return ppl


class BCPQuantizedLinear(nn.Module):
    """Quantized linear with BCP importance."""

    def __init__(self, weight, bias, activations, sparsity=0.35, nbits=4, group_size=64, use_bcp=True):
        super().__init__()
        W = weight.data.float()
        K, N = W.shape
        device = W.device

        # Sinkhorn normalization
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Compute activation norms
        act_norms = compute_activation_norms(activations).to(device)

        # First quantize to get scales (needed for BCP)
        min_max = [0, 2**nbits - 1]
        Q_init, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

        # Expand scales to per-weight [K, N]
        n_groups = scales.shape[1]
        scales_expanded = scales.squeeze(-1).repeat_interleave(group_size, dim=1)

        # Compute importance
        mu_product = mu1.unsqueeze(0) * mu2.unsqueeze(1)

        if use_bcp:
            # BCP: importance / (μ × scale)
            cost = mu_product * scales_expanded
            importance = W.abs() * act_norms.unsqueeze(0) / (cost + 1e-6)
        else:
            # Standard inverse μ
            importance = W.abs() * act_norms.unsqueeze(0) / (mu_product + 1e-6)

        # Create mask
        n_prune = int(K * N * sparsity)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        # Apply mask and re-quantize
        W_sparse_norm = W_norm * mask
        Q, scales, zeros, _ = quantize_rtn(W_sparse_norm, min_max, group_size=group_size)

        # Apply OBS-style compensation (simplified fast version)
        # For proper comparison, we skip compensation here to isolate BCP effect
        # Or we can add it - let me add simple compensation

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

        self.K = K
        self.N = N
        self.group_size = group_size

    def forward(self, x):
        input_dtype = x.dtype

        # Dequantize
        n_groups = self.scales.shape[1]
        Q_grouped = self.W_q.float().view(self.K, n_groups, self.group_size)
        W_deq_norm = (Q_grouped - self.zeros) * self.scales
        W_deq = W_deq_norm.view(self.K, self.N) * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq * self.mask
        W_deq = W_deq.to(input_dtype)

        return nn.functional.linear(x, W_deq, self.bias)


def get_activations(model, tokenizer, n_samples=64, seq_len=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"][:1000])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len * n_samples)
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
        for i in range(min(n_samples, len(input_ids) // seq_len)):
            batch = input_ids[i*seq_len:(i+1)*seq_len].unsqueeze(0).to(model.device)
            model(batch)

    for hook in hooks:
        hook.remove()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).view(-1, activations[name][0].shape[-1])

    return activations


def quantize_model(model, tokenizer, use_bcp=True, sparsity=0.35, nbits=4):
    activations = get_activations(model, tokenizer, n_samples=64)
    replaced = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            act = activations[name].to(module.weight.device)
            new_module = BCPQuantizedLinear(
                module.weight, module.bias, act,
                sparsity=sparsity, nbits=nbits, use_bcp=use_bcp
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    del activations
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Replaced {replaced} layers (use_bcp={use_bcp})")


def main():
    print("="*70)
    print("BCP (BENEFIT/COST PRUNING) PPL EVALUATION")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Test 1: Standard inverse μ (baseline)
    print("\n--- Standard Inverse-μ (baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    quantize_model(model, tokenizer, use_bcp=False, sparsity=0.35, nbits=4)
    ppl_std = eval_ppl(model, tokenizer, max_samples=64)
    print(f"Standard PPL: {ppl_std:.2f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: BCP
    print("\n--- BCP (with scale in importance) ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    quantize_model(model, tokenizer, use_bcp=True, sparsity=0.35, nbits=4)
    ppl_bcp = eval_ppl(model, tokenizer, max_samples=64)
    print(f"BCP PPL: {ppl_bcp:.2f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Standard Inverse-μ: {ppl_std:.2f}")
    print(f"BCP (μ × scale):    {ppl_bcp:.2f}")
    improvement = (ppl_std - ppl_bcp) / ppl_std * 100
    print(f"Improvement: {improvement:+.2f}%")

    if ppl_bcp < ppl_std:
        print(f"\n✓ BCP improves over baseline!")
    else:
        print(f"\n✗ BCP does not improve over baseline")


if __name__ == '__main__':
    main()
