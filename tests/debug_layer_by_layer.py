"""
Debug: Layer-by-layer PPL to find where OBS-SINQ fails.

Test PPL after quantizing N layers to isolate the failure point.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_hessian_inverse,
    compute_activation_norms,
    sparse_quantize_sinq,
    dequantize_sparse_sinq
)


def evaluate_ppl(model, tokenizer, dataset_text, seq_len=2048, n_samples=16):
    """Evaluate perplexity."""
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer("\n\n".join(dataset_text), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    nlls = []
    for i in range(min(n_samples, len(input_ids) // seq_len)):
        begin = i * seq_len
        end = begin + seq_len
        batch = input_ids[begin:end].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            nlls.append(outputs.loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


def obs_quantize_layer(W, X, sparsity, nbits=4, group_size=64, device='cuda'):
    """OBS-based pruning + SINQ quantization."""
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Sinkhorn normalization
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)

    # OBS importance
    X_flat = X.float().view(-1, X.shape[-1])[:256].to(device)
    H_inv = compute_hessian_inverse(X_flat)
    H_inv_diag = H_inv.diag()
    obs_importance = (W ** 2) * H_inv_diag.view(1, -1)

    # Create mask
    n_prune = int(K * N * sparsity)
    if n_prune > 0:
        flat_imp = obs_importance.view(-1)
        threshold = torch.kthvalue(flat_imp, n_prune).values
        mask = (flat_imp > threshold).view(K, N).float()
    else:
        mask = torch.ones(K, N, device=device)

    # OBS compensation
    W_comp = W.clone()
    pruned_mask = (1 - mask)
    for i in range(K):
        pruned_weights = W[i] * pruned_mask[i]
        compensation = -H_inv @ (pruned_weights / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + compensation * mask[i]

    # Re-normalize
    W_norm_comp = W_comp / (mu2.view(-1, 1) * mu1.view(1, -1))

    # Quantize
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm_comp, min_max, group_size=group_size)

    # Apply scales
    if len(scales.shape) == 3:
        scales = scales * mu2.unsqueeze(1)
    else:
        scales = scales * mu2
    scale2 = mu1
    q = q * mask.to(q.dtype)

    return q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), scale2.to(orig_dtype)


def dequantize(W_q, scales, zeros, mask, scale2):
    """Dequantize SINQ-style."""
    Q = W_q.float()
    s1 = scales.float()
    s2 = scale2.float()
    z = zeros.float()

    if len(s1.shape) == 3:
        K, N = Q.shape
        n_groups = s1.shape[1]
        group_size = N // n_groups
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = (Q_grouped - z) * s1
        W_deq = W_deq.view(K, N)
    else:
        W_deq = (Q - z) * s1
    W_deq = W_deq * s2 * mask.float()
    return W_deq


def run_layer_by_layer_test():
    """Test PPL after quantizing each layer."""
    print("="*70)
    print("LAYER-BY-LAYER PPL ANALYSIS")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"
    sparsity = 0.5

    print(f"\nLoading model: {model_name}")
    print(f"Sparsity: {sparsity*100:.0f}%")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]

    calib_text = "\n\n".join(dataset_text[:16])
    calib_tokens = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []

    for method in ['SCAB', 'OBS-SINQ']:
        print(f"\n{'='*70}")
        print(f"Method: {method}")
        print("="*70)

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True
        )

        # Baseline PPL
        base_ppl = evaluate_ppl(model, tokenizer, dataset_text)
        print(f"Baseline PPL: {base_ppl:.2f}")

        n_layers = len(model.model.layers)

        # Quantize one layer at a time and measure PPL
        for layer_idx in range(min(n_layers, 8)):  # First 8 layers
            layer = model.model.layers[layer_idx]

            # Capture activations
            activations_cache = {}
            def make_hook(name):
                def hook(module, input, output):
                    activations_cache[name] = input[0].detach()
                return hook

            handles = []
            for name, module in [
                ('q_proj', layer.self_attn.q_proj),
                ('k_proj', layer.self_attn.k_proj),
                ('v_proj', layer.self_attn.v_proj),
                ('o_proj', layer.self_attn.o_proj),
                ('up_proj', layer.mlp.up_proj),
                ('gate_proj', layer.mlp.gate_proj),
                ('down_proj', layer.mlp.down_proj),
            ]:
                handles.append(module.register_forward_hook(make_hook(name)))

            with torch.no_grad():
                _ = model(calib_ids)

            for h in handles:
                h.remove()

            # Quantize each sub-layer
            for name, module in [
                ('q_proj', layer.self_attn.q_proj),
                ('k_proj', layer.self_attn.k_proj),
                ('v_proj', layer.self_attn.v_proj),
                ('o_proj', layer.self_attn.o_proj),
                ('up_proj', layer.mlp.up_proj),
                ('gate_proj', layer.mlp.gate_proj),
                ('down_proj', layer.mlp.down_proj),
            ]:
                W = module.weight.data.clone()
                X = activations_cache[name]

                if method == 'SCAB':
                    W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                        W, X, sparsity=sparsity, nbits=4, group_size=64,
                        method='sinq_wanda_inverse',
                        use_compensation=True,
                        compensation_mode='bit_adaptive_mwc',
                        device=device
                    )
                    W_deq = dequantize_sparse_sinq(W_q, scales, zeros, mask, scale2, meta)
                else:  # OBS-SINQ
                    W_q, scales, zeros, mask, scale2 = obs_quantize_layer(
                        W, X, sparsity, nbits=4, group_size=64, device=device
                    )
                    W_deq = dequantize(W_q, scales, zeros, mask, scale2)

                module.weight.data = W_deq.to(module.weight.dtype)

            # Measure PPL after this layer
            ppl = evaluate_ppl(model, tokenizer, dataset_text)
            print(f"Layer {layer_idx}: PPL = {ppl:.2f}")

            results.append({
                'method': method,
                'layer': layer_idx,
                'ppl': ppl
            })

            if ppl > 100000:
                print("  -> CATASTROPHIC FAILURE, stopping")
                break

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Layer':<8} {'SCAB PPL':<15} {'OBS-SINQ PPL':<15}")
    print("-"*40)
    for layer in range(min(8, max(r['layer'] for r in results) + 1)):
        scab = next((r['ppl'] for r in results if r['method'] == 'SCAB' and r['layer'] == layer), None)
        obs = next((r['ppl'] for r in results if r['method'] == 'OBS-SINQ' and r['layer'] == layer), None)
        scab_str = f"{scab:.2f}" if scab else "N/A"
        obs_str = f"{obs:.2f}" if obs else "FAILED"
        print(f"{layer:<8} {scab_str:<15} {obs_str:<15}")

    return results


if __name__ == "__main__":
    run_layer_by_layer_test()
