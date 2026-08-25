"""
Comprehensive Bit-Adaptive MWC Benchmark.

Benchmark Bit-Adaptive MWC at 35% sparsity across bit widths 3-8
and compare with SINQ-Sparse, SparseGPT, Wanda, and FP16 baselines.

Uses same evaluation parameters as benchmark_suite.py for fair comparison:
- 16 test samples @ 2048 seq length
- 16 calibration samples @ 512 seq length
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import gc
import json
from datetime import datetime
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq


# Evaluation parameters (matching benchmark_suite.py)
EVAL_CONFIG = {
    'seq_len': 2048,
    'n_test_samples': 16,
    'n_calibration_samples': 16,
    'calibration_seq_len': 512,
    'sparsity': 0.35,
}

MODELS = {
    'qwen-0.5b': 'Qwen/Qwen2.5-0.5B',
    'qwen-1.5b': 'Qwen/Qwen2.5-1.5B',
}

PRECISIONS = [3, 4, 5, 6, 8]


def get_test_data(tokenizer, seq_len=2048, n_samples=16):
    """Load WikiText-2 test data for perplexity evaluation."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])

    return torch.stack(samples[:n_samples])


def get_calibration_data(tokenizer, n_samples=16, seq_len=512):
    """Load WikiText-2 training data for calibration."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt', max_length=seq_len * n_samples, truncation=True)
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])

    return torch.stack(samples[:n_samples])


@torch.no_grad()
def evaluate_perplexity(model, test_data, device='cuda'):
    """Evaluate perplexity on test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(len(test_data)):
        batch = test_data[i:i + 1].to(device)
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


def collect_activations(model, calibration_data, device='cuda'):
    """Collect activations for quantization calibration."""
    layer_activations = {}
    hooks = []
    activation_cache = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_cache:
                activation_cache[name] = []
            activation_cache[name].append(input[0].detach().cpu())
        return hook_fn

    layer_paths = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

    for layer_idx, layer in enumerate(model.model.layers):
        layer = layer.to(device)
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            module = layer
            try:
                for p in parts:
                    module = getattr(module, p)
                if isinstance(module, nn.Linear):
                    name = f'layer_{layer_idx}.{attr_path}'
                    hooks.append(module.register_forward_hook(make_hook(name)))
            except AttributeError:
                continue

    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(calibration_data))):
            batch = calibration_data[i:i+1].to(device)
            try:
                model(batch)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    for name, acts in activation_cache.items():
        layer_activations[name] = torch.cat(acts, dim=0)

    return layer_activations


def apply_bit_adaptive_mwc(model, calibration_data, nbits, sparsity, device='cuda'):
    """Apply Bit-Adaptive MWC quantization to model."""
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc=f"Bit-Adaptive MWC {nbits}b/{int(sparsity*100)}%")):
        layer = layer.to(device)

        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue

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
                compensation_mode='bit_adaptive_mwc'
            )

            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        model.model.layers[layer_idx] = layer
        gc.collect()
        torch.cuda.empty_cache()

    return model


def run_benchmark(model_key, nbits, device='cuda'):
    """Run Bit-Adaptive MWC benchmark for a single configuration."""
    model_name = MODELS[model_key]
    sparsity = EVAL_CONFIG['sparsity']

    result = {
        'model': model_key,
        'model_name': model_name,
        'technique': 'bit_adaptive_mwc',
        'precision': nbits,
        'sparsity': sparsity,
        'timestamp': datetime.now().isoformat(),
        'ppl': None,
        'error': None,
    }

    try:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Technique: Bit-Adaptive MWC")
        print(f"Precision: {nbits}-bit")
        print(f"Sparsity: {sparsity*100:.0f}%")
        print('='*60)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading test data...")
        test_data = get_test_data(
            tokenizer,
            seq_len=EVAL_CONFIG['seq_len'],
            n_samples=EVAL_CONFIG['n_test_samples']
        )

        print("Loading calibration data...")
        calibration_data = get_calibration_data(
            tokenizer,
            n_samples=EVAL_CONFIG['n_calibration_samples'],
            seq_len=EVAL_CONFIG['calibration_seq_len']
        )

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='cpu',
            low_cpu_mem_usage=True,
        )
        model.model.embed_tokens = model.model.embed_tokens.to(device)

        # Apply Bit-Adaptive MWC
        model = apply_bit_adaptive_mwc(model, calibration_data, nbits, sparsity, device)

        # Move remaining components
        model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)
        model.eval()

        print("Evaluating perplexity...")
        ppl = evaluate_perplexity(model, test_data, device)
        result['ppl'] = ppl

        print(f"\nPerplexity: {ppl:.2f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result['error'] = str(e)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return result


def load_baseline_results():
    """Load existing benchmark results for comparison."""
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'all_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []


def main():
    print("="*70)
    print("BIT-ADAPTIVE MWC COMPREHENSIVE BENCHMARK")
    print("="*70)
    print(f"\nConfig: 35% sparsity, {EVAL_CONFIG['n_test_samples']} test samples @ {EVAL_CONFIG['seq_len']} seq len")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Precisions: {PRECISIONS}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    # Run benchmarks
    for model_key in MODELS.keys():
        for nbits in PRECISIONS:
            result = run_benchmark(model_key, nbits, device)
            all_results.append(result)

            # Clear memory between runs
            gc.collect()
            torch.cuda.empty_cache()

    # Load baseline results
    baseline_results = load_baseline_results()

    # Create comparison tables
    print("\n" + "="*100)
    print("COMPARISON TABLES")
    print("="*100)

    for model_key in MODELS.keys():
        model_name = MODELS[model_key].split('/')[-1]
        print(f"\n### {model_name} - 35% Sparsity")
        print("-"*100)

        # Get FP16 baseline
        fp16_ppl = None
        for r in baseline_results:
            if r['model'] == model_key and r['technique'] == 'fp16':
                fp16_ppl = r['ppl']
                break

        # Get Wanda baseline
        wanda_ppl = None
        for r in baseline_results:
            if r['model'] == model_key and r['technique'] == 'wanda':
                wanda_ppl = r['ppl']
                break

        # Header
        print(f"{'Method':<20} | {'3-bit':>10} | {'4-bit':>10} | {'5-bit':>10} | {'6-bit':>10} | {'8-bit':>10}")
        print("-"*100)

        # FP16 row
        if fp16_ppl:
            print(f"{'FP16 (baseline)':<20} | {fp16_ppl:>10.2f} | {fp16_ppl:>10.2f} | {fp16_ppl:>10.2f} | {fp16_ppl:>10.2f} | {fp16_ppl:>10.2f}")

        # Wanda row
        if wanda_ppl:
            print(f"{'Wanda (35%)':<20} | {wanda_ppl:>10.2f} | {wanda_ppl:>10.2f} | {wanda_ppl:>10.2f} | {wanda_ppl:>10.2f} | {wanda_ppl:>10.2f}")

        # SparseGPT row
        sparsegpt_row = f"{'SparseGPT (35%)':<20} |"
        for nbits in PRECISIONS:
            ppl = None
            for r in baseline_results:
                if r['model'] == model_key and r['technique'] == 'sparsegpt' and r['precision'] == nbits:
                    ppl = r['ppl']
                    break
            if ppl:
                sparsegpt_row += f" {ppl:>10.2f} |"
            else:
                sparsegpt_row += f" {'N/A':>10} |"
        print(sparsegpt_row.rstrip('|'))

        # SINQ-Sparse row
        sinq_sparse_row = f"{'SINQ-Sparse (35%)':<20} |"
        for nbits in PRECISIONS:
            ppl = None
            for r in baseline_results:
                if r['model'] == model_key and r['technique'] == 'sinq-sparse' and r['precision'] == nbits:
                    ppl = r['ppl']
                    break
            if ppl:
                sinq_sparse_row += f" {ppl:>10.2f} |"
            else:
                sinq_sparse_row += f" {'N/A':>10} |"
        print(sinq_sparse_row.rstrip('|'))

        # Bit-Adaptive MWC row
        mwc_row = f"{'Bit-Adaptive MWC':<20} |"
        for nbits in PRECISIONS:
            ppl = None
            for r in all_results:
                if r['model'] == model_key and r['precision'] == nbits and r['ppl']:
                    ppl = r['ppl']
                    break
            if ppl:
                mwc_row += f" {ppl:>10.2f} |"
            else:
                mwc_row += f" {'N/A':>10} |"
        print(mwc_row.rstrip('|'))

        # Delta vs SINQ-Sparse
        print("-"*100)
        delta_row = f"{'Delta vs SINQ-Sparse':<20} |"
        for nbits in PRECISIONS:
            sinq_ppl = None
            mwc_ppl = None
            for r in baseline_results:
                if r['model'] == model_key and r['technique'] == 'sinq-sparse' and r['precision'] == nbits:
                    sinq_ppl = r['ppl']
                    break
            for r in all_results:
                if r['model'] == model_key and r['precision'] == nbits and r['ppl']:
                    mwc_ppl = r['ppl']
                    break
            if sinq_ppl and mwc_ppl:
                delta = sinq_ppl - mwc_ppl
                pct = delta / sinq_ppl * 100
                delta_row += f" {pct:>+9.2f}% |"
            else:
                delta_row += f" {'N/A':>10} |"
        print(delta_row.rstrip('|'))

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    for r in all_results:
        filename = f"{r['model']}_bit_adaptive_mwc_{r['precision']}bit.json"
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(r, f, indent=2)

    # Save summary
    summary_path = os.path.join(results_dir, 'bit_adaptive_mwc_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_dir}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for model_key in MODELS.keys():
        model_name = MODELS[model_key].split('/')[-1]
        print(f"\n{model_name}:")

        for nbits in PRECISIONS:
            sinq_ppl = None
            mwc_ppl = None
            for r in baseline_results:
                if r['model'] == model_key and r['technique'] == 'sinq-sparse' and r['precision'] == nbits:
                    sinq_ppl = r['ppl']
                    break
            for r in all_results:
                if r['model'] == model_key and r['precision'] == nbits and r['ppl']:
                    mwc_ppl = r['ppl']
                    break

            if sinq_ppl and mwc_ppl:
                delta = sinq_ppl - mwc_ppl
                status = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
                print(f"  {nbits}-bit: SINQ-Sparse {sinq_ppl:.2f} -> Bit-Adaptive MWC {mwc_ppl:.2f} ({delta:+.2f} PPL, {status})")


if __name__ == '__main__':
    main()
