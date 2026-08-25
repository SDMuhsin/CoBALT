"""
Full PPL evaluation of inverse μ importance.

CRITICAL TEST: MSE improvement doesn't guarantee PPL improvement.
We learned this from SFS (MSE improved but PPL degraded 148%).

This test will determine if inverse μ is a viable hypothesis.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
import torch.nn as nn


class InverseImportanceLinear(nn.Module):
    """Linear layer with inverse μ importance-based pruning and SINQ quantization."""

    def __init__(self, weight, bias, X_sample, sparsity=0.35, bits=3, group_size=64,
                 use_inverse=True):
        super().__init__()

        W = weight.data.float()
        K, N = W.shape
        device = W.device

        # Compute Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Compute activation norms
        act_norms = torch.norm(X_sample.float(), dim=0)

        # Compute importance
        if use_inverse:
            # INVERSE: |W| × ||X|| / (μ₁ × μ₂)
            importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
        else:
            # STANDARD: |W| × ||X|| × μ₁ × μ₂
            importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

        # Create mask
        n_prune = int(K * N * sparsity)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        # Apply sparsity and quantize
        W_sparse_norm = W_norm * mask

        min_max = [0, 2**bits - 1]
        Q, scales, zeros, _ = quantize_rtn(W_sparse_norm, min_max, group_size=group_size)

        # Store parameters
        self.register_buffer('Q', Q.to(torch.uint8))
        self.register_buffer('scales', scales.half())
        self.register_buffer('zeros', zeros.half())
        self.register_buffer('mu1', mu1.half())
        self.register_buffer('mu2', mu2.half())
        self.register_buffer('mask', mask.half())

        self.K = K
        self.N = N
        self.group_size = group_size

        if bias is not None:
            self.register_buffer('bias', bias.data.half())
        else:
            self.bias = None

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()

        # Dequantize
        Q = self.Q.float()
        n_groups = self.scales.shape[1]
        Q_grouped = Q.view(self.K, n_groups, self.group_size)
        W_deq_norm = (Q_grouped - self.zeros.float()) * self.scales.float()
        W_deq_norm = W_deq_norm.view(self.K, self.N)

        # Apply Sinkhorn scales
        W_deq = W_deq_norm * self.mu2.float().unsqueeze(1) * self.mu1.float().unsqueeze(0)

        # Convert back to input dtype for matmul
        W_deq = W_deq.to(input_dtype)
        x = x.to(input_dtype)

        out = x @ W_deq.T

        if self.bias is not None:
            out = out + self.bias

        return out


def evaluate_ppl(model, tokenizer, dataset, seq_len=2048, max_samples=128):
    """Evaluate perplexity on WikiText-2."""
    model.eval()
    device = next(model.parameters()).device

    # Prepare data
    test_data = dataset['test']
    text = "\n\n".join(test_data['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)

    n_samples = min(max_samples, (input_ids.shape[1] - 1) // seq_len)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Evaluating PPL"):
            start = i * seq_len
            end = start + seq_len

            input_chunk = input_ids[:, start:end+1]
            target = input_chunk[:, 1:].clone()
            input_chunk = input_chunk[:, :-1]

            outputs = model(input_chunk)
            logits = outputs.logits

            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = target.view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return ppl


def apply_sparse_quant(model, tokenizer, use_inverse=True, sparsity=0.35, bits=3):
    """Apply sparse quantization to all linear layers."""

    # Get calibration samples
    calib_text = "The quick brown fox jumps over the lazy dog. " * 50
    calib_input = tokenizer(calib_text, return_tensors='pt', truncation=True, max_length=512)
    calib_input = {k: v.to(model.device) for k, v in calib_input.items()}

    # Hook to capture activations
    activations = {}

    def make_hook(name, expected_features):
        def hook(module, input, output):
            if name not in activations:
                inp = input[0].detach()
                # Reshape to 2D if needed
                if inp.dim() > 2:
                    inp = inp.view(-1, inp.size(-1))
                # Only store if dimension matches
                if inp.size(-1) == expected_features:
                    activations[name] = inp
        return hook

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            expected_features = module.in_features
            handles.append(module.register_forward_hook(make_hook(name, expected_features)))

    # Run forward pass
    with torch.no_grad():
        model(**calib_input)

    # Remove hooks
    for h in handles:
        h.remove()

    # Replace linear layers
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            if name in activations:
                X_sample = activations[name]

                # Get parent module and attribute name
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1]

                # Create new module
                new_module = InverseImportanceLinear(
                    weight=module.weight,
                    bias=module.bias,
                    X_sample=X_sample,
                    sparsity=sparsity,
                    bits=bits,
                    use_inverse=use_inverse
                )

                setattr(parent, attr_name, new_module)
                replaced += 1

    print(f"Replaced {replaced} linear layers")
    del activations
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print("="*70)
    print("INVERSE μ IMPORTANCE - FULL PPL EVALUATION")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer and dataset
    print("\nLoading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Test configurations
    configs = [
        ("Standard (×μ)", False),
        ("Inverse (/μ)", True),
    ]

    results = {}

    for name, use_inverse in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Apply sparse quantization
        print(f"Applying sparse quantization (use_inverse={use_inverse})...")
        apply_sparse_quant(model, tokenizer, use_inverse=use_inverse, sparsity=0.35, bits=3)

        # Evaluate
        ppl = evaluate_ppl(model, tokenizer, dataset, seq_len=2048, max_samples=64)
        results[name] = ppl
        print(f"PPL: {ppl:.2f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nBaseline (SINQ-Sparse): PPL = 28.10 (from benchmark)")
    for name, ppl in results.items():
        print(f"{name}: PPL = {ppl:.2f}")

    ppl_std = results["Standard (×μ)"]
    ppl_inv = results["Inverse (/μ)"]

    improvement = (ppl_std - ppl_inv) / ppl_std * 100

    print(f"\nImprovement (Inverse vs Standard): {improvement:+.2f}%")

    if ppl_inv < ppl_std:
        print("\n>>> INVERSE μ IMPORTANCE IS BETTER!")
        print(">>> This validates the hypothesis.")
    else:
        print("\n>>> INVERSE μ IMPORTANCE IS WORSE.")
        print(">>> MSE improvement did not translate to PPL improvement.")


if __name__ == '__main__':
    main()
