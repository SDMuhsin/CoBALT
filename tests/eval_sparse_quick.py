"""
Quick evaluation of SINQ-Sparse with error compensation.

Tests at 50% sparsity with and without compensation.
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


def sparse_quantize_model_with_activations(model, calibration_data, sparsity=0.5, nbits=4,
                                            use_compensation=False, compensation_mode='fast',
                                            device='cuda'):
    """Apply sparse quantization with proper activation collection."""
    print(f"\nSparse quantization: sparsity={sparsity}, compensation={use_compensation}")

    layers_quantized = 0

    # Collect activations by layer
    layer_activations = {}

    if use_compensation or True:  # Always collect for sinq_wanda method
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
                method='sinq_wanda' if activations is not None else 'sinq',
                device=device,
                use_compensation=use_compensation and activations is not None,
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


def main():
    print("="*70)
    print("SINQ-Sparse Quick Evaluation (with Error Compensation)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading test data...")
    test_data = get_wikitext2(tokenizer, seq_len=2048, n_samples=32)
    print(f"Test samples: {len(test_data)}, seq_len: {test_data.shape[1]}")

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
    # SINQ Dense Baseline
    # =========================================================================
    print("\n" + "="*70)
    print("SINQ DENSE (0% sparsity, 4-bit)")
    print("="*70)

    model_sinq = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model_sinq.model.embed_tokens = model_sinq.model.embed_tokens.to(device)

    model_sinq = sparse_quantize_model_with_activations(
        model_sinq, calibration_data,
        sparsity=0.0,
        nbits=4,
        use_compensation=False,
        device=device
    )

    model_sinq.model.norm = model_sinq.model.norm.to(device)
    model_sinq.lm_head = model_sinq.lm_head.to(device)
    model_sinq.eval()

    ppl_sinq = evaluate_perplexity(model_sinq, test_data, device)
    print(f"SINQ Dense Perplexity: {ppl_sinq:.2f}")
    print(f"vs FP16: {ppl_sinq/ppl_fp16*100:.1f}%")
    results['SINQ_Dense'] = ppl_sinq

    del model_sinq
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # SINQ-Sparse 50% WITHOUT compensation
    # =========================================================================
    print("\n" + "="*70)
    print("SINQ-SPARSE 50% (NO compensation)")
    print("="*70)

    model_no_comp = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model_no_comp.model.embed_tokens = model_no_comp.model.embed_tokens.to(device)

    model_no_comp = sparse_quantize_model_with_activations(
        model_no_comp, calibration_data,
        sparsity=0.5,
        nbits=4,
        use_compensation=False,
        device=device
    )

    model_no_comp.model.norm = model_no_comp.model.norm.to(device)
    model_no_comp.lm_head = model_no_comp.lm_head.to(device)
    model_no_comp.eval()

    ppl_no_comp = evaluate_perplexity(model_no_comp, test_data, device)
    print(f"SINQ-Sparse 50% (no comp) Perplexity: {ppl_no_comp:.2f}")
    print(f"vs FP16: {ppl_no_comp/ppl_fp16*100:.1f}%")
    print(f"vs SINQ Dense: {ppl_no_comp/ppl_sinq*100:.1f}%")
    results['Sparse_50_NoComp'] = ppl_no_comp

    del model_no_comp
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # SINQ-Sparse 50% WITH fast compensation
    # =========================================================================
    print("\n" + "="*70)
    print("SINQ-SPARSE 50% (with FAST compensation)")
    print("="*70)

    model_fast = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model_fast.model.embed_tokens = model_fast.model.embed_tokens.to(device)

    model_fast = sparse_quantize_model_with_activations(
        model_fast, calibration_data,
        sparsity=0.5,
        nbits=4,
        use_compensation=True,
        compensation_mode='fast',
        device=device
    )

    model_fast.model.norm = model_fast.model.norm.to(device)
    model_fast.lm_head = model_fast.lm_head.to(device)
    model_fast.eval()

    ppl_fast = evaluate_perplexity(model_fast, test_data, device)
    print(f"SINQ-Sparse 50% (fast comp) Perplexity: {ppl_fast:.2f}")
    print(f"vs FP16: {ppl_fast/ppl_fp16*100:.1f}%")
    print(f"vs SINQ Dense: {ppl_fast/ppl_sinq*100:.1f}%")
    results['Sparse_50_FastComp'] = ppl_fast

    del model_fast
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Method':<35} | {'PPL':>10} | {'vs FP16':>10} | {'vs SINQ':>10}")
    print("-"*70)

    sinq_ppl = results['SINQ_Dense']
    for name, ppl in results.items():
        vs_fp16 = ppl / results['FP16'] * 100
        vs_sinq = ppl / sinq_ppl * 100
        print(f"{name:<35} | {ppl:>10.2f} | {vs_fp16:>9.1f}% | {vs_sinq:>9.1f}%")

    # Success check
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK")
    print("="*70)

    target_ratio = 1.10  # PPL <= 110% of SINQ baseline
    best_sparse = min(results.get('Sparse_50_FastComp', float('inf')),
                      results.get('Sparse_50_NoComp', float('inf')))
    actual_ratio = best_sparse / sinq_ppl

    print(f"\nTarget: 50% sparsity with PPL <= 110% of SINQ baseline")
    print(f"Result: 50% sparsity with PPL = {actual_ratio*100:.1f}% of SINQ baseline")

    if actual_ratio <= target_ratio:
        print(f"\n[SUCCESS] Target met! ({actual_ratio*100:.1f}% <= {target_ratio*100:.0f}%)")
    else:
        print(f"\n[FAIL] Target not met ({actual_ratio*100:.1f}% > {target_ratio*100:.0f}%)")
        print(f"Need to improve by: {(actual_ratio - target_ratio)*100:.1f}%")


if __name__ == '__main__':
    main()
