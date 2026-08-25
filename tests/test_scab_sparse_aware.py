"""
SCAB-OPT + Sparse-Aware Sinkhorn

Test if combining:
1. Inverse-μ importance (for pruning decisions)
2. OBS compensation (error correction)
3. Iterative refinement (n=2)
4. Sparse-Aware Sinkhorn (for final quantization)

provides better results than standard SCAB-OPT.
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
)


def sinkhorn_log_sparse_aware(W, mask, order=16, eps=1e-4):
    """Sparse-Aware Sinkhorn: Computes μ factors only on non-zero weights."""
    W = W.float()
    mask = mask.float()
    K, N = W.shape
    device = W.device

    log_mu1 = torch.zeros(N, device=device)
    log_mu2 = torch.zeros(K, device=device)

    row_count = mask.sum(dim=1)
    col_count = mask.sum(dim=0)
    empty_rows = (row_count == 0)
    empty_cols = (col_count == 0)

    CLAMP_MIN, CLAMP_MAX = -10, 10

    for _ in range(order):
        log_scale = (-log_mu2.view(-1, 1) - log_mu1.view(1, -1)).clamp(CLAMP_MIN, CLAMP_MAX)
        W_scaled = W * torch.exp(log_scale)

        row_sum_sq = (mask * W_scaled ** 2).sum(dim=1)
        row_count_safe = row_count.clamp(min=1)
        row_sum = (mask * W_scaled).sum(dim=1)
        row_mean = row_sum / row_count_safe
        row_var = row_sum_sq / row_count_safe - row_mean ** 2
        row_std = (row_var.clamp(min=eps ** 2)).sqrt()

        log_update = torch.log(row_std.clamp(min=eps))
        log_update[empty_rows] = 0
        log_mu2 = (log_mu2 + log_update).clamp(CLAMP_MIN, CLAMP_MAX)

        log_scale = (-log_mu2.view(-1, 1) - log_mu1.view(1, -1)).clamp(CLAMP_MIN, CLAMP_MAX)
        W_scaled = W * torch.exp(log_scale)

        col_sum_sq = (mask * W_scaled ** 2).sum(dim=0)
        col_count_safe = col_count.clamp(min=1)
        col_sum = (mask * W_scaled).sum(dim=0)
        col_mean = col_sum / col_count_safe
        col_var = col_sum_sq / col_count_safe - col_mean ** 2
        col_std = (col_var.clamp(min=eps ** 2)).sqrt()

        log_update = torch.log(col_std.clamp(min=eps))
        log_update[empty_cols] = 0
        log_mu1 = (log_mu1 + log_update).clamp(CLAMP_MIN, CLAMP_MAX)

    mu1 = torch.exp(log_mu1)
    mu2 = torch.exp(log_mu2)
    W_norm = W / (mu2.view(-1, 1) * mu1.view(1, -1))

    return W_norm, mu1, mu2


def scab_opt_quantize(W, X, sparsity, nbits=4, group_size=64, device='cuda',
                       n_iter=2, use_sparse_aware_sinkhorn=False):
    """
    SCAB-OPT with optional Sparse-Aware Sinkhorn.

    Steps:
    1. Iterative importance refinement (n=2)
    2. OBS compensation
    3. Either standard or sparse-aware Sinkhorn for quantization
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    X_flat = X.float().view(-1, X.shape[-1])[:256].to(device)
    act_norms = compute_activation_norms(X)

    # Iterative importance refinement
    current_W = W.clone()
    mask = torch.ones(K, N, device=device)
    n_prune = int(K * N * sparsity)

    for iteration in range(n_iter):
        W_for_sinkhorn = current_W.clone()
        zero_mask = current_W.abs() < 1e-10
        W_for_sinkhorn[zero_mask] = torch.randn(zero_mask.sum().item(), device=device) * 1e-8

        _, mu1, mu2 = sinkhorn_log(W_for_sinkhorn, order=16)
        importance = compute_importance_scores(W, mu1, mu2, act_norms, method='sinq_wanda_inverse')

        if n_prune > 0:
            flat_imp = importance.view(-1)
            threshold = torch.kthvalue(flat_imp, n_prune).values
            new_mask = (flat_imp > threshold).view(K, N).float()
        else:
            new_mask = torch.ones(K, N, device=device)

        current_W = W * new_mask
        mask = new_mask

    # OBS compensation
    H_inv = compute_hessian_inverse(X_flat)
    H_inv_diag = H_inv.diag()

    W_comp = W.clone()
    pruned_mask = (1 - mask)
    for i in range(K):
        pruned_w = W[i] * pruned_mask[i]
        compensation = -H_inv @ (pruned_w / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + compensation * mask[i]

    W_sparse_comp = W_comp * mask

    # Sinkhorn normalization (sparse-aware or standard)
    if use_sparse_aware_sinkhorn:
        W_norm, mu1_final, mu2_final = sinkhorn_log_sparse_aware(W_sparse_comp, mask, order=16)
    else:
        W_for_sink = W_sparse_comp.clone()
        W_for_sink[W_for_sink.abs() < 1e-10] = 1e-8
        _, mu1_final, mu2_final = sinkhorn_log(W_for_sink + torch.randn_like(W_for_sink) * 1e-8, order=16)
        W_norm = W_sparse_comp / (mu2_final.view(-1, 1) * mu1_final.view(1, -1))

    # Quantize
    min_max = [0, 2**nbits - 1]
    q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    if len(scales.shape) == 3:
        scales = scales * mu2_final.view(-1, 1, 1)
    else:
        scales = scales * mu2_final.view(-1, 1)
    scale2 = mu1_final
    q = q * mask.to(q.dtype)

    method = 'scab_opt_sparse_aware' if use_sparse_aware_sinkhorn else 'scab_opt_standard'
    meta = {'sparsity': sparsity, 'method': method}

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
    W_deq = W_deq * s2.view(1, -1) * mask.float()
    return W_deq


def evaluate_ppl(model, tokenizer, dataset_text, seq_len=2048, n_samples=16):
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer("\n\n".join(dataset_text), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    nlls = []
    for i in range(min(n_samples, len(input_ids) // seq_len)):
        begin, end = i * seq_len, (i + 1) * seq_len
        batch = input_ids[begin:end].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            nlls.append(outputs.loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


def run_comparison():
    print("="*70)
    print("SCAB-OPT: Standard vs Sparse-Aware Sinkhorn")
    print("="*70)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = [t for t in dataset["text"] if len(t) > 100][:100]
    calib_tokens = tokenizer("\n\n".join(dataset_text[:16]), return_tensors="pt",
                             truncation=True, max_length=512)
    calib_ids = calib_tokens.input_ids.to(device)

    results = []
    sparsegpt_ref = {0.35: 19.22, 0.50: 24.76, 0.60: 40.42}

    for sparsity in [0.35, 0.50, 0.60]:
        print(f"\n{'='*70}")
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("="*70)

        for method_name, use_sparse_aware in [
            ('SCAB-OPT (Standard)', False),
            ('SCAB-OPT (Sparse-Aware)', True),
        ]:
            print(f"\nTesting: {method_name}...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map=device, trust_remote_code=True
            )

            for layer_idx in range(len(model.model.layers)):
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
                        W, X, sparsity, nbits=4, group_size=64,
                        device=device, n_iter=2, use_sparse_aware_sinkhorn=use_sparse_aware
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
    print("SUMMARY: SCAB-OPT Standard vs Sparse-Aware")
    print("="*70)
    print(f"{'Sparsity':<12} {'Standard':<15} {'Sparse-Aware':<15} {'SparseGPT':<12} {'Best'}")
    print("-"*65)

    for sp in [0.35, 0.50, 0.60]:
        std = next(r['ppl'] for r in results if r['sparsity'] == sp and 'Standard' in r['method'])
        spa = next(r['ppl'] for r in results if r['sparsity'] == sp and 'Sparse-Aware' in r['method'])
        ref = sparsegpt_ref[sp]
        best = min(std, spa)
        winner = "Sparse-Aware" if spa < std else "Standard"
        vs_sgpt = "BEATS" if best < ref else "loses to"
        print(f"{sp*100:>5.0f}%       {std:<15.2f} {spa:<15.2f} {ref:<12.2f} {winner} ({vs_sgpt} SparseGPT)")

    return results


if __name__ == "__main__":
    run_comparison()
