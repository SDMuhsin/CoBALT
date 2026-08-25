"""
Test: μ-Weighted OBS Importance

Hypothesis: The OBS-SINQ failure occurs because OBS keeps weights with HIGH μ
factors, causing error amplification during Sinkhorn dequantization.

Solution: Combine OBS importance with inverse-μ weighting:
  μ-Weighted OBS = (W² × H_inv_diag) / (μ₁ × μ₂)

This preserves:
1. OBS's Hessian-based output sensitivity analysis
2. Inverse-μ's penalization of error-amplifying weights

This is theoretically principled: we're computing output sensitivity weighted by
how much quantization error will be amplified in the final output.
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


def mu_weighted_obs_quantize(W, X, sparsity, nbits=4, group_size=64, device='cuda'):
    """
    μ-Weighted OBS: Combines OBS importance with inverse-μ weighting.

    importance = (W² × H_inv_diag) / (μ₁ × μ₂)

    This accounts for:
    - OBS: How much the output changes when pruning weight w_j
    - Inverse-μ: How much quantization error gets amplified

    Uses standard OBS compensation (not MWC) since we've already accounted
    for μ factors in the importance criterion.
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Step 1: Sinkhorn normalization
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)

    # Step 2: Compute Hessian inverse
    X_flat = X.float().view(-1, X.shape[-1])[:256].to(device)
    H_inv = compute_hessian_inverse(X_flat)
    H_inv_diag = H_inv.diag()

    # Step 3: Compute μ-Weighted OBS importance
    # OBS part: W² × H_inv_diag
    obs_importance = (W ** 2) * H_inv_diag.view(1, -1)

    # μ weighting: divide by (μ₁ × μ₂) to penalize error-amplifying weights
    mu_product = mu1.view(1, -1) * mu2.view(-1, 1)
    mu_weighted_obs = obs_importance / (mu_product + 1e-6)

    # Step 4: Create mask
    n_prune = int(K * N * sparsity)
    if n_prune > 0:
        flat_imp = mu_weighted_obs.view(-1)
        threshold = torch.kthvalue(flat_imp, n_prune).values
        mask = (flat_imp > threshold).view(K, N).float()
    else:
        mask = torch.ones(K, N, device=device)

    # Step 5: Standard OBS compensation
    # (not MWC since μ factors are already accounted in importance)
    W_comp = W.clone()
    pruned_mask = (1 - mask)
    for i in range(K):
        pruned_w = W[i] * pruned_mask[i]
        compensation = -H_inv @ (pruned_w / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + compensation * mask[i]

    # Step 6: Normalize and quantize
    W_norm_comp = W_comp / (mu2.view(-1, 1) * mu1.view(1, -1))

    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm_comp, min_max, group_size=group_size)

    if len(scales.shape) == 3:
        scales = scales * mu2.unsqueeze(1)
    else:
        scales = scales * mu2
    scale2 = mu1
    q = q * mask.to(q.dtype)

    meta = {
        'sparsity': sparsity,
        'actual_sparsity': 1.0 - mask.sum().item() / mask.numel(),
        'nbits': nbits,
        'method': 'mu_weighted_obs'
    }

    return q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), scale2.to(orig_dtype), meta


def dequantize(W_q, scales, zeros, mask, scale2, meta):
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


def run_comparison():
    """Compare SCAB vs μ-Weighted OBS at multiple sparsity levels."""
    print("="*70)
    print("COMPARISON: SCAB vs μ-Weighted OBS")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]

    calib_text = "\n\n".join(dataset_text[:16])
    calib_tokens = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []

    for sparsity in [0.35, 0.45, 0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        for method_name in ['SCAB (bit-adaptive MWC)', 'μ-Weighted OBS']:
            print(f"\nTesting {method_name}...")

            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map=device, trust_remote_code=True
            )

            n_layers = len(model.model.layers)

            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]

                # Capture activations
                activations_cache = {}
                def make_hook(name):
                    def hook(m, inp, out):
                        activations_cache[name] = inp[0].detach()
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

                # Quantize each linear
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

                    if 'SCAB' in method_name:
                        W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                            W, X, sparsity=sparsity, nbits=4, group_size=64,
                            method='sinq_wanda_inverse',
                            use_compensation=True,
                            compensation_mode='bit_adaptive_mwc',
                            device=device
                        )
                        W_deq = dequantize_sparse_sinq(W_q, scales, zeros, mask, scale2, meta)
                    else:  # μ-Weighted OBS
                        W_q, scales, zeros, mask, scale2, meta = mu_weighted_obs_quantize(
                            W, X, sparsity, nbits=4, group_size=64, device=device
                        )
                        W_deq = dequantize(W_q, scales, zeros, mask, scale2, meta)

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

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SCAB vs μ-Weighted OBS")
    print("="*70)
    print(f"{'Sparsity':<12} {'SCAB':<15} {'μ-Weighted OBS':<15} {'Winner'}")
    print("-"*60)

    for sp in sorted(set(r['sparsity'] for r in results)):
        scab = next(r['ppl'] for r in results if r['sparsity'] == sp and 'SCAB' in r['method'])
        mwobs = next(r['ppl'] for r in results if r['sparsity'] == sp and 'μ-Weighted' in r['method'])
        winner = "SCAB" if scab < mwobs else "μ-Weighted OBS"
        diff = abs(scab - mwobs) / min(scab, mwobs) * 100
        print(f"{sp*100:>5.0f}%       {scab:<15.2f} {mwobs:<15.2f} {winner} ({diff:.1f}%)")

    # Also compare to SparseGPT results from earlier
    print("\n" + "="*70)
    print("CONTEXT: Earlier SparseGPT PPL results")
    print("="*70)
    print("35%: SparseGPT = 19.22")
    print("50%: SparseGPT = 24.76")
    print("60%: SparseGPT = 40.42")

    return results


if __name__ == "__main__":
    results = run_comparison()
