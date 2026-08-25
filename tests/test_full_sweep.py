"""
Full Validation Sweep: SCAB-OPT (Iterative n=2) vs SparseGPT

Final method: Inverse-μ importance + OBS compensation + Iterative Refinement (n=2)

This combines:
1. Inverse-μ importance: |W| × act_norms / (μ₁ × μ₂)
2. Standard OBS compensation (not MWC - which was overcorrecting)
3. Iterative refinement: re-compute Sinkhorn factors on masked weights 2x
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


def scab_opt_quantize(W, X, sparsity, nbits=4, group_size=64, device='cuda', n_iter=2):
    """
    SCAB-OPT: Optimized SCAB with iterative importance refinement.

    Key improvements over original SCAB:
    1. Standard OBS compensation instead of MWC (which overcorrects)
    2. Iterative importance refinement (n=2) to adapt to sparse structure
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    X_flat = X.float().view(-1, X.shape[-1])[:256].to(device)
    act_norms = compute_activation_norms(X)

    # Iteratively refine the mask
    current_W = W.clone()
    mask = torch.ones(K, N, device=device)
    n_prune = int(K * N * sparsity)

    for iteration in range(n_iter):
        # Add small noise to zero regions to enable Sinkhorn
        W_for_sinkhorn = current_W.clone()
        zero_mask = current_W.abs() < 1e-10
        W_for_sinkhorn[zero_mask] = torch.randn(zero_mask.sum().item(), device=device) * 1e-8

        # Compute Sinkhorn on current weights
        _, mu1, mu2 = sinkhorn_log(W_for_sinkhorn, order=16)

        # Compute inverse-μ importance
        importance = compute_importance_scores(W, mu1, mu2, act_norms, method='sinq_wanda_inverse')

        # Create new mask
        if n_prune > 0:
            flat_imp = importance.view(-1)
            threshold = torch.kthvalue(flat_imp, n_prune).values
            new_mask = (flat_imp > threshold).view(K, N).float()
        else:
            new_mask = torch.ones(K, N, device=device)

        current_W = W * new_mask
        mask = new_mask

    # Final Sinkhorn factors
    W_final = W * mask
    W_final[W_final.abs() < 1e-10] = 1e-8
    _, mu1_final, mu2_final = sinkhorn_log(W_final + torch.randn_like(W_final) * 1e-8, order=16)

    # OBS compensation (not MWC)
    H_inv = compute_hessian_inverse(X_flat)
    H_inv_diag = H_inv.diag()

    W_comp = W.clone()
    pruned_mask = (1 - mask)
    for i in range(K):
        pruned_w = W[i] * pruned_mask[i]
        compensation = -H_inv @ (pruned_w / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + compensation * mask[i]

    W_sparse_comp = W_comp * mask

    # Normalize and quantize
    W_norm = W_sparse_comp / (mu2_final.view(-1, 1) * mu1_final.view(1, -1))
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    if len(scales.shape) == 3:
        scales = scales * mu2_final.unsqueeze(1)
    else:
        scales = scales * mu2_final
    scale2 = mu1_final
    q = q * mask.to(q.dtype)

    meta = {'sparsity': sparsity, 'actual_sparsity': 1.0 - mask.sum().item() / mask.numel(),
            'nbits': nbits, 'method': 'scab_opt'}

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


def run_sweep():
    print("="*70)
    print("FULL VALIDATION SWEEP: SCAB-OPT vs SparseGPT")
    print("="*70)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]
    calib_tokens = tokenizer("\n\n".join(dataset_text[:16]), return_tensors="pt", truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []
    sparsegpt_ref = {0.35: 19.22, 0.40: 20.87, 0.45: 21.65, 0.50: 24.76, 0.60: 40.42}

    for sparsity in [0.35, 0.40, 0.45, 0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        print(f"\nTesting SCAB-OPT (Iterative n=2)...")

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

                W_q, scales, zeros, mask, scale2, meta = scab_opt_quantize(
                    W, X, sparsity, nbits=4, group_size=64, device=device, n_iter=2
                )
                W_deq = dequantize(W_q, scales, zeros, mask, scale2, meta)
                module.weight.data = W_deq.to(module.weight.dtype)

        ppl = evaluate_ppl(model, tokenizer, dataset_text)
        sgpt = sparsegpt_ref[sparsity]
        winner = "SCAB-OPT" if ppl < sgpt else "SparseGPT"
        diff = (ppl - sgpt) / sgpt * 100

        print(f"  SCAB-OPT: PPL = {ppl:.2f}")
        print(f"  SparseGPT: PPL = {sgpt:.2f}")
        print(f"  Winner: {winner} ({diff:+.1f}%)")

        results.append({
            'sparsity': sparsity,
            'scab_opt': ppl,
            'sparsegpt': sgpt,
            'winner': winner,
            'diff_pct': diff
        })

        del model
        torch.cuda.empty_cache()

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS: SCAB-OPT vs SparseGPT")
    print("="*70)
    print(f"{'Sparsity':<12} {'SCAB-OPT':<12} {'SparseGPT':<12} {'Diff':<12} {'Winner'}")
    print("-"*60)

    wins = 0
    for r in results:
        print(f"{r['sparsity']*100:>5.0f}%       {r['scab_opt']:<12.2f} {r['sparsegpt']:<12.2f} {r['diff_pct']:>+8.1f}%    {r['winner']}")
        if r['winner'] == 'SCAB-OPT':
            wins += 1

    print("-"*60)
    print(f"SCAB-OPT wins {wins}/{len(results)} sparsity levels")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    crossover = None
    for i in range(len(results)-1):
        if results[i]['winner'] == 'SCAB-OPT' and results[i+1]['winner'] == 'SparseGPT':
            crossover = (results[i]['sparsity'] + results[i+1]['sparsity']) / 2
            break

    if crossover:
        print(f"Crossover point: ~{crossover*100:.0f}% sparsity")
        print(f"SCAB-OPT dominates at sparsity < {crossover*100:.0f}%")
        print(f"SparseGPT dominates at sparsity > {crossover*100:.0f}%")

    print("\nKey improvements over original SCAB:")
    print("1. Replaced MWC compensation with standard OBS (MWC was overcorrecting)")
    print("2. Added iterative importance refinement (n=2) to adapt to sparse structure")
    print("3. Inverse-μ importance remains key for accounting for Sinkhorn error amplification")

    return results


if __name__ == "__main__":
    run_sweep()
