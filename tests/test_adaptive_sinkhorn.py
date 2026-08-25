"""
Test: Adaptive Sinkhorn with Post-Compensation Re-normalization

Current best: Inv-μ + OBS gives 26.57 at 50%, 108.88 at 60%
SparseGPT: 24.76 at 50%, 40.42 at 60%

Hypothesis: After OBS compensation changes the weight distribution significantly,
the original Sinkhorn factors (μ₁, μ₂) are no longer optimal.

Adaptive Sinkhorn approach:
1. Compute Sinkhorn on original weights → μ₁, μ₂ (for importance calculation)
2. Prune using inverse-μ importance
3. Compensate using OBS
4. Re-compute Sinkhorn on compensated weights → μ₁', μ₂' (for quantization)
5. Quantize using the new, adapted factors

This is "sparse-aware Sinkhorn quantization" - it adapts to the post-pruning structure.
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
    compute_importance_scores,
    sparse_quantize_sinq,
    dequantize_sparse_sinq
)


def evaluate_ppl(model, tokenizer, dataset_text, seq_len=2048, n_samples=16):
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


def adaptive_sinkhorn_quantize(W, X, sparsity, nbits=4, group_size=64, device='cuda'):
    """
    Adaptive Sinkhorn: Re-normalize after pruning+compensation.

    Steps:
    1. Sinkhorn on original W → μ₁, μ₂ (for inverse-μ importance)
    2. Prune based on inverse-μ importance
    3. OBS compensation
    4. Re-Sinkhorn on W_compensated → μ₁', μ₂' (adapted to sparse structure)
    5. Quantize using adapted factors
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Step 1: Original Sinkhorn for importance calculation
    W_norm_orig, mu1_orig, mu2_orig = sinkhorn_log(W, order=16)

    # Step 2: Compute inverse-μ importance
    X_flat = X.float().view(-1, X.shape[-1])[:256].to(device)
    act_norms = compute_activation_norms(X)
    importance = compute_importance_scores(W, mu1_orig, mu2_orig, act_norms, method='sinq_wanda_inverse')

    # Step 3: Create mask
    n_prune = int(K * N * sparsity)
    if n_prune > 0:
        flat_imp = importance.view(-1)
        threshold = torch.kthvalue(flat_imp, n_prune).values
        mask = (flat_imp > threshold).view(K, N).float()
    else:
        mask = torch.ones(K, N, device=device)

    # Step 4: OBS compensation
    H_inv = compute_hessian_inverse(X_flat)
    H_inv_diag = H_inv.diag()

    W_comp = W.clone()
    pruned_mask = (1 - mask)
    for i in range(K):
        pruned_w = W[i] * pruned_mask[i]
        compensation = -H_inv @ (pruned_w / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + compensation * mask[i]

    # Step 5: Apply mask to get sparse compensated weights
    W_sparse_comp = W_comp * mask

    # Step 6: RE-COMPUTE Sinkhorn on the sparse compensated weights
    # Add small epsilon to avoid division by zero for pruned weights
    W_for_sinkhorn = W_sparse_comp.clone()
    # For completely zero rows/cols, add tiny noise to get valid Sinkhorn factors
    row_zeros = (W_for_sinkhorn.abs().sum(dim=1) < 1e-10)
    col_zeros = (W_for_sinkhorn.abs().sum(dim=0) < 1e-10)
    if row_zeros.any() or col_zeros.any():
        W_for_sinkhorn = W_for_sinkhorn + torch.randn_like(W_for_sinkhorn) * 1e-8

    W_norm_adapted, mu1_adapted, mu2_adapted = sinkhorn_log(W_for_sinkhorn, order=16)

    # Step 7: Normalize compensated weights using ADAPTED factors
    W_norm_final = W_sparse_comp / (mu2_adapted.view(-1, 1) * mu1_adapted.view(1, -1))

    # Step 8: Quantize
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm_final, min_max, group_size=group_size)

    # Step 9: Apply adapted scales
    if len(scales.shape) == 3:
        scales = scales * mu2_adapted.unsqueeze(1)
    else:
        scales = scales * mu2_adapted
    scale2 = mu1_adapted

    q = q * mask.to(q.dtype)

    meta = {
        'sparsity': sparsity,
        'actual_sparsity': 1.0 - mask.sum().item() / mask.numel(),
        'nbits': nbits,
        'method': 'adaptive_sinkhorn'
    }

    return q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), scale2.to(orig_dtype), meta


def dequantize(W_q, scales, zeros, mask, scale2, meta):
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


def run_test():
    print("="*70)
    print("ADAPTIVE SINKHORN TEST")
    print("="*70)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]
    calib_tokens = tokenizer("\n\n".join(dataset_text[:16]), return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []

    for sparsity in [0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        for method_name, quantize_fn in [
            ('Inv-μ + OBS (baseline)', None),
            ('Adaptive Sinkhorn', adaptive_sinkhorn_quantize),
        ]:
            print(f"\nTesting: {method_name}...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map=device, trust_remote_code=True
            )

            n_layers = len(model.model.layers)

            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]

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

                    if quantize_fn is None:
                        # Baseline: Inv-μ + OBS
                        W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                            W, X, sparsity=sparsity, nbits=4, group_size=64,
                            method='sinq_wanda_inverse',
                            use_compensation=True,
                            compensation_mode='fast',
                            device=device
                        )
                        W_deq = dequantize_sparse_sinq(W_q, scales, zeros, mask, scale2, meta)
                    else:
                        W_q, scales, zeros, mask, scale2, meta = quantize_fn(
                            W, X, sparsity, nbits=4, group_size=64, device=device
                        )
                        W_deq = dequantize(W_q, scales, zeros, mask, scale2, meta)

                    module.weight.data = W_deq.to(module.weight.dtype)

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
    print("SUMMARY")
    print("="*70)
    print(f"{'Sparsity':<12} {'Baseline':<15} {'Adaptive Sink.':<15} {'SparseGPT'}")
    print("-"*55)

    sparsegpt_ref = {0.50: 24.76, 0.60: 40.42}

    for sp in [0.50, 0.60]:
        baseline = next(r['ppl'] for r in results if r['sparsity'] == sp and 'baseline' in r['method'])
        adaptive = next(r['ppl'] for r in results if r['sparsity'] == sp and 'Adaptive' in r['method'])
        ref = sparsegpt_ref[sp]
        winner = min(baseline, adaptive)
        print(f"{sp*100:>5.0f}%       {baseline:<15.2f} {adaptive:<15.2f} {ref}")

    return results


if __name__ == "__main__":
    run_test()
