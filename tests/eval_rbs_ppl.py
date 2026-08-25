"""
PPL Evaluation: Row-Balanced Sparsity (RBS)

Compares standard inverse-μ with RBS (μ₂-guided row sparsity allocation).
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


class RBSQuantizedLinear(nn.Module):
    """Linear layer with RBS sparsity and SINQ quantization."""

    def __init__(self, W, activations, nbits=4, group_size=64, sparsity=0.35, use_rbs=True):
        super().__init__()
        K, N = W.shape
        device = W.device

        # Sinkhorn normalization
        W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)
        mu2 = mu2.squeeze()

        # Compute importance
        act_norms = torch.norm(activations, dim=0).to(device)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Create mask
        target_pruned = int(K * N * sparsity)

        if use_rbs:
            # RBS: Row-balanced sparsity using μ₂
            row_weights = 1 / (mu2 + 1e-6)
            row_weights = row_weights / row_weights.sum() * K
            sparsity_per_row = sparsity * row_weights
            sparsity_per_row = sparsity_per_row.clamp(0.15, 0.60)
            total_from_clip = (sparsity_per_row * N).sum()
            sparsity_per_row = sparsity_per_row * (target_pruned / total_from_clip)
            sparsity_per_row = sparsity_per_row.clamp(0.15, 0.60)

            mask = torch.zeros_like(W)
            for i in range(K):
                row_imp = importance[i, :]
                n_prune = int(N * sparsity_per_row[i].item())
                n_prune = max(1, min(n_prune, N - 1))
                thresh = row_imp.sort().values[n_prune]
                mask[i, :] = (row_imp > thresh).float()
        else:
            # Standard: Global threshold
            threshold = importance.view(-1).sort().values[target_pruned]
            mask = (importance > threshold).float()

        # Apply mask and quantize
        W_sparse_norm = W_norm * mask

        min_max = [0, 2**nbits - 1]
        actual_gs = min(group_size, N)
        Q, scales, zeros, _ = quantize_rtn(W_sparse_norm, min_max, group_size=actual_gs)

        # Store quantized parameters
        self.register_buffer('W_q', Q.to(torch.int8))
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('mu1', mu1)
        self.register_buffer('mu2', mu2)
        self.K = K
        self.N = N
        self.group_size = actual_gs

    def forward(self, x):
        # Dequantize
        input_dtype = x.dtype
        n_groups = self.scales.shape[1]
        Q_grouped = self.W_q.float().view(self.K, n_groups, self.group_size)
        W_deq_norm = (Q_grouped - self.zeros) * self.scales
        W_deq = W_deq_norm.view(self.K, self.N) * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq * self.mask
        W_deq = W_deq.to(input_dtype)

        return nn.functional.linear(x, W_deq)


def get_activations(model, tokenizer, n_samples=128, seq_len=512):
    """Collect activations for calibration."""
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

    # Concatenate activations
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).view(-1, activations[name][0].shape[-1])

    return activations


def quantize_model(model, activations, nbits=4, group_size=64, sparsity=0.35, use_rbs=True):
    """Replace Linear layers with RBS quantized versions."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            W = module.weight.data
            act = activations.get(name, torch.randn(128, W.shape[1]) * 0.1)
            act = act.to(W.device)

            # Create quantized layer
            quant_layer = RBSQuantizedLinear(
                W, act, nbits=nbits, group_size=group_size,
                sparsity=sparsity, use_rbs=use_rbs
            )

            # Replace
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, quant_layer)

    return model


def evaluate_ppl(model, tokenizer, max_length=2048):
    """Evaluate perplexity on WikiText-2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length * 100)

    model.eval()
    nlls = []
    seq_len = max_length

    for i in tqdm(range(0, min(10000, encodings.input_ids.size(1) - 1), seq_len), desc="Evaluating"):
        begin = i
        end = min(i + seq_len, encodings.input_ids.size(1) - 1)
        input_ids = encodings.input_ids[:, begin:end].to(model.device)
        target_ids = encodings.input_ids[:, begin+1:end+1].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='sum'
        )
        nlls.append(loss.item())

    ppl = torch.exp(torch.tensor(sum(nlls) / (len(nlls) * seq_len)))
    return ppl.item()


def main():
    print("="*70)
    print("RBS PPL EVALUATION")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("\nLoading Qwen-0.5B...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test 1: Standard (no RBS)
    print("\n--- Standard Inverse-μ (baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    model = quantize_model(model, activations, nbits=4, sparsity=0.35, use_rbs=False)
    ppl_std = evaluate_ppl(model, tokenizer)
    print(f"Standard PPL: {ppl_std:.2f}")
    del model
    torch.cuda.empty_cache()

    # Test 2: RBS
    print("\n--- RBS (Row-Balanced Sparsity) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    activations = get_activations(model, tokenizer, n_samples=64)
    model = quantize_model(model, activations, nbits=4, sparsity=0.35, use_rbs=True)
    ppl_rbs = evaluate_ppl(model, tokenizer)
    print(f"RBS PPL: {ppl_rbs:.2f}")
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Standard Inverse-μ: {ppl_std:.2f}")
    print(f"RBS (μ₂-guided):    {ppl_rbs:.2f}")
    improvement = (ppl_std - ppl_rbs) / ppl_std * 100
    print(f"Improvement: {improvement:+.2f}%")

    if ppl_rbs < 18.15:
        print(f"\n✓ BEATS BASELINE (18.15)!")
    else:
        print(f"\n✗ Does not beat baseline (18.15)")


if __name__ == '__main__':
    main()
