"""
Test: OBS Importance + SINQ Quantization Hybrid

Hypothesis: SCAB's degradation at high sparsity is caused by the inverse-μ
importance criterion making poor pruning decisions, NOT by the SINQ quantization
or MWC compensation.

Test Plan:
1. SCAB Original: inverse-μ importance + SINQ quant + MWC compensation
2. OBS-SINQ Hybrid: OBS importance + SINQ quant + standard OBS compensation
3. SparseGPT Baseline: OBS importance + RTN quant + OBS compensation

If OBS-SINQ outperforms SCAB at high sparsity, the importance criterion is the issue.
If OBS-SINQ still fails, the issue is elsewhere (e.g., Sinkhorn factor interaction).
"""

import torch
import torch.nn as nn
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


def compute_obs_importance(W, X, damping=None):
    """
    Compute OBS importance scores: W² / diag(H_inv)²

    This is what SparseGPT uses for pruning decisions.
    Lower score = better to prune (less error when removed).
    We return the inverse so higher score = more important (consistent with SCAB).
    """
    K, N = W.shape
    device = W.device

    X = X.float().to(device)
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    n_samples = min(X.shape[0], 256)
    X = X[:n_samples]

    H_inv = compute_hessian_inverse(X, damping=damping)
    H_inv_diag = H_inv.diag()

    # OBS criterion: error from pruning w_j = w_j² / H_inv[j,j]
    # Lower = better to prune, so importance = H_inv_diag / |W| (inverse)
    # To keep consistent with "higher = more important", we use H_inv_diag * |W|
    # Actually, for importance: we want to KEEP high-importance weights
    # OBS says prune weights with LOW w²/H_inv_diag
    # So importance = w² * H_inv_diag (higher = harder to prune = more important)

    H_inv_diag_expanded = H_inv_diag.view(1, -1).expand(K, -1)
    importance = (W.float() ** 2) * H_inv_diag_expanded

    return importance.float(), H_inv


def obs_sinq_hybrid_quantize(
    W,
    activations,
    sparsity,
    nbits=4,
    group_size=64,
    device='cuda'
):
    """
    OBS-SINQ Hybrid: Use OBS importance for pruning, SINQ for quantization.

    This tests whether SparseGPT's OBS importance criterion combined with
    SINQ's Sinkhorn-based quantization can outperform SCAB at high sparsity.
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Step 1: Sinkhorn normalization (for quantization scales, not importance)
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)

    # Step 2: Compute OBS importance (NOT inverse-μ)
    obs_importance, H_inv = compute_obs_importance(W, activations)
    H_inv_diag = H_inv.diag()

    # Step 3: Create mask using OBS importance
    n_weights = K * N
    n_prune = int(n_weights * sparsity)

    if n_prune > 0:
        flat_importance = obs_importance.view(-1)
        threshold = torch.kthvalue(flat_importance, n_prune).values
        mask = (flat_importance > threshold).view(K, N).float()
    else:
        mask = torch.ones(K, N, device=device)

    # Step 4: Standard OBS compensation (not MWC)
    W_compensated = W.clone()
    pruned_mask = (1 - mask)

    for i in range(K):
        pruned_weights = W[i] * pruned_mask[i]
        compensation = -H_inv @ (pruned_weights / H_inv_diag)
        W_compensated[i] = W[i] * mask[i] + compensation * mask[i]

    # Step 5: Re-normalize compensated weights with Sinkhorn
    W_norm_comp = W_compensated / (mu2.view(-1, 1) * mu1.view(1, -1))

    # Step 6: Quantize using SINQ approach
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm_comp, min_max, group_size=group_size)

    # Apply Sinkhorn scales
    if len(scales.shape) == 3:
        scales = scales * mu2.unsqueeze(1)
    else:
        scales = scales * mu2

    scale2 = mu1

    # Apply sparsity mask
    q = q * mask.to(q.dtype)

    meta = {
        'sparsity': sparsity,
        'actual_sparsity': 1.0 - mask.sum().item() / mask.numel(),
        'nbits': nbits,
        'method': 'obs_sinq_hybrid'
    }

    return q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), scale2.to(orig_dtype), meta


def dequantize_obs_sinq(W_q, scales, zeros, mask, scale2, meta):
    """Dequantize OBS-SINQ hybrid (same as SINQ dequant)."""
    Q = W_q.float()
    z = zeros.float()
    s1 = scales.float()
    s2 = scale2.float()

    if len(s1.shape) == 3:
        K, N = Q.shape
        n_groups = s1.shape[1]
        group_size = N // n_groups
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = (Q_grouped - z) * s1
        W_deq = W_deq.view(K, N)
    else:
        W_deq = (Q - z) * s1

    W_deq = W_deq * s2
    W_deq = W_deq * mask.float()
    return W_deq


def evaluate_ppl(model, tokenizer, dataset_text, seq_len=2048, n_samples=16):
    """Evaluate perplexity on dataset."""
    model.eval()
    device = next(model.parameters()).device

    # Tokenize
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


def patch_linear_with_sparse(linear, W_q, scales, zeros, mask, scale2, meta):
    """Patch linear layer with sparse-quantized weights."""
    W_deq = dequantize_obs_sinq(W_q, scales, zeros, mask, scale2, meta)
    linear.weight.data = W_deq.to(linear.weight.dtype).to(linear.weight.device)


def run_importance_comparison():
    """Compare importance criteria at various sparsity levels."""
    print("="*70)
    print("IMPORTANCE CRITERION COMPARISON: OBS vs Inverse-μ")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"

    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]

    # Get calibration data
    calib_text = "\n\n".join(dataset_text[:16])
    calib_tokens = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    # Measure FP16 baseline
    print("\nMeasuring FP16 baseline...")
    fp16_ppl = evaluate_ppl(model, tokenizer, dataset_text)
    print(f"FP16 Baseline PPL: {fp16_ppl:.2f}")

    results = []

    for sparsity in [0.35, 0.45, 0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        # Reload fresh model for each test
        for method_name in ['SCAB (inverse-μ)', 'OBS-SINQ Hybrid']:
            print(f"\nTesting {method_name}...")

            # Reload model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )

            # Get activations for first layer
            activations_cache = {}

            def capture_hook(name):
                def hook(module, input, output):
                    activations_cache[name] = input[0].detach()
                return hook

            # Process all decoder layers
            n_layers = len(model.model.layers)

            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]

                # Capture activations
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
                    handles.append(module.register_forward_hook(capture_hook(name)))

                # Forward pass to capture activations
                with torch.no_grad():
                    _ = model(calib_ids)

                # Remove hooks
                for h in handles:
                    h.remove()

                # Quantize each linear layer
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
                    X = activations_cache[name].float()

                    if method_name == 'SCAB (inverse-μ)':
                        # Original SCAB with inverse-μ importance
                        W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                            W, X, sparsity=sparsity, nbits=4, group_size=64,
                            method='sinq_wanda_inverse',
                            use_compensation=True,
                            compensation_mode='bit_adaptive_mwc',
                            device=device
                        )
                        W_deq = dequantize_sparse_sinq(W_q, scales, zeros, mask, scale2, meta)
                    else:
                        # OBS-SINQ Hybrid
                        W_q, scales, zeros, mask, scale2, meta = obs_sinq_hybrid_quantize(
                            W, X, sparsity=sparsity, nbits=4, group_size=64, device=device
                        )
                        W_deq = dequantize_obs_sinq(W_q, scales, zeros, mask, scale2, meta)

                    module.weight.data = W_deq.to(module.weight.dtype)

            # Evaluate PPL
            ppl = evaluate_ppl(model, tokenizer, dataset_text)
            print(f"  {method_name}: PPL = {ppl:.2f}")

            results.append({
                'sparsity': sparsity,
                'method': method_name,
                'ppl': ppl
            })

            del model
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Importance Criterion Impact on PPL")
    print("="*70)
    print(f"{'Sparsity':<12} {'SCAB (inverse-μ)':<20} {'OBS-SINQ Hybrid':<20} {'Winner'}")
    print("-"*70)

    sparsities = sorted(set(r['sparsity'] for r in results))
    for sp in sparsities:
        scab = next(r['ppl'] for r in results if r['sparsity'] == sp and 'SCAB' in r['method'])
        obs = next(r['ppl'] for r in results if r['sparsity'] == sp and 'OBS' in r['method'])
        winner = "SCAB" if scab < obs else "OBS-SINQ"
        diff = abs(scab - obs) / min(scab, obs) * 100
        print(f"{sp*100:>5.0f}%       {scab:<20.2f} {obs:<20.2f} {winner} ({diff:.1f}% diff)")

    print("\nFP16 Baseline:", fp16_ppl)

    return results


if __name__ == "__main__":
    results = run_importance_comparison()
