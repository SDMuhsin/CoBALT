"""
Evaluate Adaptive MWC on both 0.5B and 1.5B models.

Hypothesis 043: Adaptive MWC should:
- Match or beat MWC on 0.5B (already works there)
- Avoid the regression on 1.5B by not applying MWC to low-CV layers
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
    """Load WikiText-2 test set for perplexity evaluation."""
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
    """Get calibration data from wikitext for activation collection."""
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
    """Evaluate perplexity on test data."""
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
    """Linear layer with sparse quantization applied."""

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


def sparse_quantize_model(model, calibration_data, sparsity=0.35, nbits=4,
                          compensation_mode='adaptive_mwc', device='cuda'):
    """Apply sparse quantization with proper activation collection."""
    print(f"\nSparse quantization: sparsity={sparsity}, compensation_mode={compensation_mode}")

    layers_quantized = 0
    layer_activations = {}

    print("Collecting layer activations...")

    hooks = []
    activation_cache = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_cache:
                activation_cache[name] = []
            activation_cache[name].append(input[0].detach().cpu())
        return hook_fn

    # Register hooks
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

    # Run calibration
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

    # Quantize
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
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method='sinq_wanda_inverse' if activations is not None else 'sinq',
                device=device,
                use_compensation=activations is not None,
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


def test_model(model_name, nbits=4, sparsity=0.35):
    """Test Adaptive MWC on a specific model."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Config: {nbits}-bit, {sparsity*100:.0f}% sparsity")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get test and calibration data
    print("\nLoading data...")
    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=32)
    calibration_data = get_calibration_data(tokenizer, n_samples=32, seq_len=512)

    # FP16 baseline
    print("\nEvaluating FP16 baseline...")
    ppl_fp16 = evaluate_perplexity(model, test_data, device)
    print(f"FP16 PPL: {ppl_fp16:.2f}")

    # Quantize with Adaptive MWC
    print("\nQuantizing with Adaptive MWC...")
    model = sparse_quantize_model(model, calibration_data, sparsity=sparsity, nbits=nbits,
                                   compensation_mode='adaptive_mwc', device=device)

    # Evaluate
    print("\nEvaluating Adaptive MWC...")
    ppl_adaptive_mwc = evaluate_perplexity(model, test_data, device)
    print(f"Adaptive MWC PPL: {ppl_adaptive_mwc:.2f}")

    # Results
    degradation = (ppl_adaptive_mwc - ppl_fp16) / ppl_fp16 * 100
    print(f"\nResults for {model_name}:")
    print(f"  FP16: {ppl_fp16:.2f}")
    print(f"  Adaptive MWC: {ppl_adaptive_mwc:.2f}")
    print(f"  Degradation: {degradation:.2f}%")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'model': model_name,
        'fp16': ppl_fp16,
        'adaptive_mwc': ppl_adaptive_mwc,
        'degradation': degradation
    }


def main():
    print("="*70)
    print("ADAPTIVE MWC EVALUATION (Hypothesis 043)")
    print("="*70)

    results = []

    # Test 0.5B
    result_05b = test_model("Qwen/Qwen2.5-0.5B")
    results.append(result_05b)

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Test 1.5B
    result_15b = test_model("Qwen/Qwen2.5-1.5B")
    results.append(result_15b)

    # Summary
    print("\n" + "="*70)
    print("ADAPTIVE MWC SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} | {'FP16':>8} | {'Adaptive MWC':>12} | {'Degradation':>12}")
    print("-"*60)

    for r in results:
        print(f"{r['model'].split('/')[-1]:<20} | {r['fp16']:>8.2f} | {r['adaptive_mwc']:>12.2f} | {r['degradation']:>11.2f}%")

    # Compare with previous MWC results
    print("\n" + "="*70)
    print("COMPARISON WITH VANILLA MWC")
    print("="*70)
    print("\nPrevious MWC results (PPL):")
    print("  0.5B: Standard OBS 19.54 → MWC 19.37 (+0.86%)")
    print("  1.5B: Standard OBS 11.79 → MWC 12.27 (-4.10% WORSE)")
    print("\nAdaptive MWC results:")
    for r in results:
        model_short = r['model'].split('/')[-1]
        print(f"  {model_short}: {r['adaptive_mwc']:.2f} ({r['degradation']:.2f}% vs FP16)")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    # Check if 1.5B passes
    if len(results) >= 2:
        r_15b = results[1]
        # Compare with vanilla MWC (12.27)
        improvement_over_mwc = 12.27 - r_15b['adaptive_mwc']
        if improvement_over_mwc > 0:
            print(f"✓ Adaptive MWC FIXES the 1.5B regression!")
            print(f"  Vanilla MWC: 12.27")
            print(f"  Adaptive MWC: {r_15b['adaptive_mwc']:.2f}")
            print(f"  Improvement: {improvement_over_mwc:.2f} PPL points")
        else:
            print(f"✗ Adaptive MWC does not fix the regression")
            print(f"  Vanilla MWC: 12.27")
            print(f"  Adaptive MWC: {r_15b['adaptive_mwc']:.2f}")


if __name__ == '__main__':
    main()
