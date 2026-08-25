"""
MWC Multi-Layer Test - Validate improvement scales with μ₁ CV
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_hessian_inverse


def test_mwc_multi():
    print("="*70)
    print("MWC Multi-Layer Validation")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    # Test multiple layer types
    test_layers = []
    for layer_idx in [0, 5, 10, 15, 20]:
        layer = model.model.layers[layer_idx]
        test_layers.extend([
            (f'L{layer_idx}.q_proj', layer.self_attn.q_proj.weight.data),
            (f'L{layer_idx}.k_proj', layer.self_attn.k_proj.weight.data),
            (f'L{layer_idx}.gate_proj', layer.mlp.gate_proj.weight.data),
        ])

    sparsity = 0.35
    nbits = 4
    group_size = 64

    results = []

    for name, W_orig in test_layers:
        W = W_orig.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]
        n_prune = int(K * N * sparsity)

        batch = 256
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()
        mu1_cv = (mu1.std() / mu1.mean()).item()

        # Importance and mask
        act_norms = torch.norm(X, dim=0)
        importance = W_norm.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()
        pruned_mask = 1 - mask

        # Hessian
        H_inv = compute_hessian_inverse(X.float(), damping=None)
        H_inv_diag = H_inv.diag()

        W_sparse = W_norm * mask

        def evaluate(W_comp):
            Q, s, z, _ = quantize_rtn(W_comp, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            return ((Y_ref - Y_approx) ** 2).mean().item()

        # No compensation
        mse_sparse = evaluate(W_sparse)

        # Standard OBS
        W_std = W_sparse.clone()
        for i in range(K):
            pruned_weights = W_norm[i] * pruned_mask[i]
            compensation = -H_inv @ (pruned_weights / H_inv_diag)
            W_std[i] = W_sparse[i] + compensation * mask[i]
        mse_std = evaluate(W_std)

        # MWC
        W_mwc = W_sparse.clone()
        for i in range(K):
            pruned_weights = W_norm[i] * pruned_mask[i]
            weighted_pruned = pruned_weights * mu1 / H_inv_diag
            compensation = -H_inv @ weighted_pruned / mu1
            W_mwc[i] = W_sparse[i] + compensation * mask[i]
        mse_mwc = evaluate(W_mwc)

        improvement_std = (mse_sparse - mse_std) / mse_sparse * 100
        improvement_mwc_vs_std = (mse_std - mse_mwc) / mse_std * 100

        results.append({
            'name': name,
            'mu1_cv': mu1_cv,
            'mse_sparse': mse_sparse,
            'mse_std': mse_std,
            'mse_mwc': mse_mwc,
            'std_improvement': improvement_std,
            'mwc_vs_std': improvement_mwc_vs_std
        })

        print(f"{name}: μ₁ CV={mu1_cv:.3f}, OBS={improvement_std:+.1f}%, MWC vs OBS={improvement_mwc_vs_std:+.2f}%")

    # Analyze correlation between μ₁ CV and MWC improvement
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    mu1_cvs = torch.tensor([r['mu1_cv'] for r in results])
    mwc_improvements = torch.tensor([r['mwc_vs_std'] for r in results])

    correlation = torch.corrcoef(torch.stack([mu1_cvs, mwc_improvements]))[0, 1].item()
    print(f"Correlation(μ₁ CV, MWC improvement): {correlation:.3f}")

    # Average improvement
    avg_mwc = mwc_improvements.mean().item()
    print(f"Average MWC improvement over OBS: {avg_mwc:+.2f}%")

    # High μ₁ CV layers
    high_cv_mask = mu1_cvs > 0.15
    if high_cv_mask.any():
        avg_high_cv = mwc_improvements[high_cv_mask].mean().item()
        print(f"Average MWC improvement on high-CV layers (>0.15): {avg_high_cv:+.2f}%")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mwc_multi()
