"""
Evaluation script for SINQ-Sparse joint sparse-quantization.

Measures perplexity on WikiText-2 for:
1. FP16 baseline
2. SINQ baseline (dense quantized)
3. SINQ-Sparse at various sparsity levels
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import gc
from tqdm import tqdm
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq, compute_activation_norms


def get_wikitext2(tokenizer, seq_len=2048, n_samples=128):
    """Load WikiText-2 test set for perplexity evaluation."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])

    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]

    # Split into chunks
    n_tokens = len(input_ids)
    samples = []
    for i in range(0, min(n_tokens - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])

    return torch.stack(samples[:n_samples])


def get_calibration_data(tokenizer, n_samples=128, seq_len=2048):
    """Get calibration data from C4 for activation collection."""
    try:
        dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    except:
        # Fallback to smaller dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text = '\n\n'.join(dataset['text'])
        encodings = tokenizer(text, return_tensors='pt', max_length=seq_len * n_samples, truncation=True)
        input_ids = encodings.input_ids[0]
        samples = []
        for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
            samples.append(input_ids[i:i + seq_len])
        return torch.stack(samples[:n_samples])

    samples = []
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
        text = sample['text']
        if len(text) < 100:
            continue
        encodings = tokenizer(text, return_tensors='pt', max_length=seq_len, truncation=True)
        if encodings.input_ids.shape[1] >= seq_len // 2:
            samples.append(encodings.input_ids[0][:seq_len])
            if len(samples) >= n_samples:
                break

    # Pad to same length
    max_len = max(s.shape[0] for s in samples)
    padded = []
    for s in samples:
        if s.shape[0] < max_len:
            s = torch.cat([s, torch.zeros(max_len - s.shape[0], dtype=s.dtype)])
        padded.append(s)

    return torch.stack(padded)


@torch.no_grad()
def evaluate_perplexity(model, test_data, device='cuda', batch_size=1):
    """Evaluate perplexity on test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating PPL"):
        batch = test_data[i:i + batch_size].to(device)

        # Forward pass
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        # Accumulate
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

        # Cache dequantized weights for faster inference
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


def collect_layer_activations(model, calibration_data, layer_name, device='cuda'):
    """Collect activations for a specific layer."""
    activations = []

    def hook_fn(module, input, output):
        activations.append(input[0].detach().cpu())

    # Find the layer
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)

    handle = module.register_forward_hook(hook_fn)

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(calibration_data))):  # Use subset for speed
            batch = calibration_data[i:i+1].to(device)
            try:
                model(batch)
            except:
                pass

    handle.remove()

    if activations:
        return torch.cat(activations, dim=0)
    return None


def sparse_quantize_model(model, calibration_data, sparsity=0.5, nbits=4, method='sinq_wanda',
                          device='cuda', use_compensation=False, compensation_mode='fast'):
    """Apply sparse quantization to all linear layers with optional error compensation."""
    print(f"\nApplying sparse quantization: sparsity={sparsity}, bits={nbits}, method={method}")
    print(f"Error compensation: {use_compensation}, mode: {compensation_mode}")

    layers_quantized = 0

    # First pass: collect activations for each layer
    layer_activations = {}

    if use_compensation:
        print("Collecting layer activations for compensation...")

        # Hook to collect activations
        hooks = []
        activation_cache = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if name not in activation_cache:
                    activation_cache[name] = []
                # Store input activations (before linear transform)
                activation_cache[name].append(input[0].detach().cpu())
            return hook_fn

        # Register hooks for all linear layers
        for layer_idx, layer in enumerate(model.model.layers):
            layer = layer.to(device)
            linear_layers = [
                (f'layer_{layer_idx}.self_attn.q_proj', layer.self_attn.q_proj),
                (f'layer_{layer_idx}.self_attn.k_proj', layer.self_attn.k_proj),
                (f'layer_{layer_idx}.self_attn.v_proj', layer.self_attn.v_proj),
                (f'layer_{layer_idx}.self_attn.o_proj', layer.self_attn.o_proj),
                (f'layer_{layer_idx}.mlp.gate_proj', layer.mlp.gate_proj),
                (f'layer_{layer_idx}.mlp.up_proj', layer.mlp.up_proj),
                (f'layer_{layer_idx}.mlp.down_proj', layer.mlp.down_proj),
            ]
            for name, linear in linear_layers:
                if isinstance(linear, nn.Linear):
                    hooks.append(linear.register_forward_hook(make_hook(name)))

        # Run calibration data
        model.eval()
        with torch.no_grad():
            for i in range(min(8, len(calibration_data))):
                batch = calibration_data[i:i+1].to(device)
                try:
                    model(batch)
                except:
                    pass

        # Remove hooks
        for h in hooks:
            h.remove()

        # Concatenate activations
        for name, acts in activation_cache.items():
            layer_activations[name] = torch.cat(acts, dim=0)

        print(f"Collected activations for {len(layer_activations)} layers")

    # Second pass: quantize with compensation
    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Quantizing layers")):
        layer = layer.to(device)

        linear_layers = [
            ('self_attn.q_proj', layer.self_attn.q_proj),
            ('self_attn.k_proj', layer.self_attn.k_proj),
            ('self_attn.v_proj', layer.self_attn.v_proj),
            ('self_attn.o_proj', layer.self_attn.o_proj),
            ('mlp.gate_proj', layer.mlp.gate_proj),
            ('mlp.up_proj', layer.mlp.up_proj),
            ('mlp.down_proj', layer.mlp.down_proj),
        ]

        for name, linear in linear_layers:
            if not isinstance(linear, nn.Linear):
                continue

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            # Get activations if we collected them
            act_key = f'layer_{layer_idx}.{name}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            # Sparse quantize
            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method=method if activations is not None else 'sinq',
                device=device,
                use_compensation=use_compensation and activations is not None,
                compensation_mode=compensation_mode
            )

            # Replace layer
            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)

            # Set the new layer
            parent = layer
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_layer)

            layers_quantized += 1
            del W, linear, activations
            torch.cuda.empty_cache()

        model.model.layers[layer_idx] = layer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Quantized {layers_quantized} layers")
    return model


def main():
    print("="*70)
    print("SINQ-Sparse Evaluation")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Model to evaluate
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    print("\nLoading WikiText-2 test data...")
    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=64)
    print(f"Test samples: {len(test_data)}, seq_len: {test_data.shape[1]}")

    # Load calibration data
    print("Loading calibration data...")
    calibration_data = get_calibration_data(tokenizer, n_samples=32, seq_len=512)
    print(f"Calibration samples: {len(calibration_data)}")

    results = {}

    # =========================================================================
    # FP16 Baseline
    # =========================================================================
    print("\n" + "="*70)
    print("FP16 BASELINE")
    print("="*70)

    gc.collect()
    torch.cuda.empty_cache()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model_fp16.eval()

    ppl_fp16 = evaluate_perplexity(model_fp16, test_data, device)
    print(f"FP16 Perplexity: {ppl_fp16:.2f}")
    results['FP16'] = ppl_fp16

    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # SINQ Baseline (Dense, 0% sparsity)
    # =========================================================================
    print("\n" + "="*70)
    print("SINQ BASELINE (Dense Quantized, 4-bit)")
    print("="*70)

    model_sinq = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Move embeddings to GPU
    model_sinq.model.embed_tokens = model_sinq.model.embed_tokens.to(device)

    model_sinq = sparse_quantize_model(
        model_sinq, calibration_data,
        sparsity=0.0,  # No sparsity = dense SINQ
        nbits=4,
        method='sinq',
        device=device
    )

    model_sinq.model.norm = model_sinq.model.norm.to(device)
    model_sinq.lm_head = model_sinq.lm_head.to(device)
    model_sinq.eval()

    ppl_sinq = evaluate_perplexity(model_sinq, test_data, device)
    print(f"SINQ (Dense) Perplexity: {ppl_sinq:.2f}")
    print(f"vs FP16: {ppl_sinq/ppl_fp16*100:.1f}%")
    results['SINQ_Dense'] = ppl_sinq

    del model_sinq
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # SINQ-Sparse at various sparsity levels
    # =========================================================================
    # Start with lower sparsity levels - 50% uncompensated sparsity is too aggressive
    sparsity_levels = [0.10, 0.20, 0.30, 0.40, 0.50]

    for sparsity in sparsity_levels:
        print("\n" + "="*70)
        print(f"SINQ-SPARSE ({int(sparsity*100)}% sparsity)")
        print("="*70)

        model_sparse = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='cpu',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model_sparse.model.embed_tokens = model_sparse.model.embed_tokens.to(device)

        model_sparse = sparse_quantize_model(
            model_sparse, calibration_data,
            sparsity=sparsity,
            nbits=4,
            method='sinq',  # Use 'sinq' since we don't have per-layer activations
            device=device
        )

        model_sparse.model.norm = model_sparse.model.norm.to(device)
        model_sparse.lm_head = model_sparse.lm_head.to(device)
        model_sparse.eval()

        ppl_sparse = evaluate_perplexity(model_sparse, test_data, device)
        print(f"SINQ-Sparse ({int(sparsity*100)}%) Perplexity: {ppl_sparse:.2f}")
        print(f"vs FP16: {ppl_sparse/ppl_fp16*100:.1f}%")
        print(f"vs SINQ Dense: {ppl_sparse/ppl_sinq*100:.1f}%")
        results[f'Sparse_{int(sparsity*100)}'] = ppl_sparse

        del model_sparse
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Method':<25} | {'PPL':>10} | {'vs FP16':>10} | {'vs SINQ':>10}")
    print("-"*60)

    sinq_ppl = results['SINQ_Dense']
    for name, ppl in results.items():
        vs_fp16 = ppl / results['FP16'] * 100
        vs_sinq = ppl / sinq_ppl * 100 if sinq_ppl > 0 else 0
        print(f"{name:<25} | {ppl:>10.2f} | {vs_fp16:>9.1f}% | {vs_sinq:>9.1f}%")

    # Check success criteria
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK")
    print("="*70)

    target_sparsity = 0.50
    target_ppl_ratio = 1.10  # PPL ≤ 110% of SINQ baseline

    sparse_50_ppl = results.get('Sparse_50', float('inf'))
    actual_ratio = sparse_50_ppl / sinq_ppl

    print(f"\nTarget: 50% sparsity with PPL ≤ 110% of SINQ baseline")
    print(f"Result: 50% sparsity with PPL = {actual_ratio*100:.1f}% of SINQ baseline")

    if actual_ratio <= target_ppl_ratio:
        print(f"\n[SUCCESS] Target met! ({actual_ratio*100:.1f}% ≤ {target_ppl_ratio*100:.0f}%)")
    else:
        print(f"\n[FAIL] Target not met ({actual_ratio*100:.1f}% > {target_ppl_ratio*100:.0f}%)")
        print(f"Need to improve by: {(actual_ratio - target_ppl_ratio)*100:.1f}%")


if __name__ == '__main__':
    main()
