"""
Benchmark inverse μ importance using the official SINQ-Sparse framework.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn

# Import from SINQ
from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq


def get_wikitext2(tokenizer, seq_len=2048, split='test'):
    """Load WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    return encodings.input_ids


def eval_ppl(model, tokenizer, device='cuda', seq_len=2048, max_samples=64):
    """Evaluate perplexity."""
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


class QuantizedLinear(nn.Module):
    """Quantized linear layer using SINQ-Sparse."""

    def __init__(self, weight, bias, activations, sparsity, nbits, method, use_compensation=True):
        super().__init__()

        # Apply sparse quantization
        W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
            weight.data,
            activations,
            sparsity=sparsity,
            nbits=nbits,
            group_size=64,
            method=method,
            structured='unstructured',
            device=weight.device,
            use_compensation=use_compensation,
            compensation_mode='fast'
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

        self.K, self.N = weight.shape
        self.group_size = 64
        self.nbits = nbits

    def forward(self, x):
        input_dtype = x.dtype

        # Dequantize
        W = dequantize_sparse_sinq(
            self.W_q, self.scales, self.zeros, self.mask, self.scale2, self.meta
        )

        W = W.to(input_dtype)
        out = x @ W.T

        if self.bias is not None:
            out = out + self.bias

        return out


def quantize_model(model, tokenizer, method, sparsity=0.35, nbits=4, use_compensation=True):
    """Apply sparse quantization to all linear layers."""
    device = next(model.parameters()).device

    # Get calibration data
    calib_text = "The quick brown fox jumps over the lazy dog. " * 100
    calib_input = tokenizer(calib_text, return_tensors='pt', max_length=512, truncation=True)
    calib_input = {k: v.to(device) for k, v in calib_input.items()}

    # Collect activations
    activations = {}

    def make_hook(name, in_features):
        def hook(module, inp, out):
            if name not in activations:
                x = inp[0].detach()
                if x.dim() > 2:
                    x = x.view(-1, x.size(-1))
                if x.size(-1) == in_features:
                    activations[name] = x
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            handles.append(module.register_forward_hook(make_hook(name, module.in_features)))

    with torch.no_grad():
        model(**calib_input)

    for h in handles:
        h.remove()

    # Replace layers
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name and name in activations:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            new_module = QuantizedLinear(
                module.weight, module.bias, activations[name],
                sparsity=sparsity, nbits=nbits, method=method,
                use_compensation=use_compensation
            )
            setattr(parent, parts[-1], new_module)
            replaced += 1

    print(f"Replaced {replaced} layers with method={method}")
    del activations
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print("="*70)
    print("INVERSE μ BENCHMARK - OFFICIAL SINQ-SPARSE FRAMEWORK")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    methods = [
        ('sinq_wanda', "Standard (×μ)"),
        ('sinq_wanda_inverse', "Inverse (/μ)"),
        ('wanda', "Pure Wanda (no μ)"),
    ]

    results = {}

    for method, name in methods:
        print(f"\n{'='*70}")
        print(f"Testing: {name} (method={method})")
        print(f"{'='*70}")

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        quantize_model(model, tokenizer, method=method, sparsity=0.35, nbits=4, use_compensation=True)

        ppl = eval_ppl(model, tokenizer, device=device, max_samples=64)
        results[name] = ppl
        print(f"PPL: {ppl:.2f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nBaseline SINQ-Sparse: PPL = 28.10 (from official benchmark)")
    for name, ppl in results.items():
        print(f"{name}: PPL = {ppl:.2f}")

    std_ppl = results["Standard (×μ)"]
    inv_ppl = results["Inverse (/μ)"]

    if inv_ppl < std_ppl:
        improvement = (std_ppl - inv_ppl) / std_ppl * 100
        print(f"\n>>> INVERSE μ IS BETTER by {improvement:.1f}%")
    else:
        print("\n>>> Standard is better or equal")


if __name__ == '__main__':
    main()
