"""
Multi-layer test: Row-Balanced Sparsity (RBS)

Tests RBS across multiple layers to verify consistent improvement.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_rbs_multilayer():
    print("="*70)
    print("ROW-BALANCED SPARSITY (RBS) - MULTI-LAYER TEST")
    print("="*70)

    torch.manual_seed(42)

    # Load real weights
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Test multiple layers and projection types
    layers_to_test = [
        ("layer0_gate", model.model.layers[0].mlp.gate_proj.weight),
        ("layer0_up", model.model.layers[0].mlp.up_proj.weight),
        ("layer0_down", model.model.layers[0].mlp.down_proj.weight),
        ("layer0_qkv", model.model.layers[0].self_attn.q_proj.weight),
        ("layer5_gate", model.model.layers[5].mlp.gate_proj.weight),
        ("layer10_gate", model.model.layers[10].mlp.gate_proj.weight),
        ("layer15_gate", model.model.layers[15].mlp.gate_proj.weight),
    ]

    results = []

    for name, weight in layers_to_test:
        W = weight.data.float()
        K, N = W.shape

        # Create activations
        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Norms
        act_norms = torch.norm(X, dim=0)

        # Target sparsity
        total_sparsity = 0.35
        target_pruned = int(K * N * total_sparsity)

        # Standard importance
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Standard mask
        threshold_std = importance.view(-1).sort().values[target_pruned]
        mask_std = (importance > threshold_std).float()

        # RBS mask using μ₂
        row_weights = 1 / (mu2 + 1e-6)
        row_weights = row_weights / row_weights.sum() * K
        sparsity_per_row = total_sparsity * row_weights
        sparsity_per_row = sparsity_per_row.clamp(0.1, 0.7)
        total_from_clip = (sparsity_per_row * N).sum()
        sparsity_per_row = sparsity_per_row * (target_pruned / total_from_clip)
        sparsity_per_row = sparsity_per_row.clamp(0.1, 0.7)

        mask_rbs = torch.zeros_like(W)
        for i in range(K):
            row_imp = importance[i, :]
            n_prune = int(N * sparsity_per_row[i].item())
            n_prune = max(1, min(n_prune, N - 1))
            thresh = row_imp.sort().values[n_prune]
            mask_rbs[i, :] = (row_imp > thresh).float()

        # Evaluate
        nbits = 4
        group_size = 64
        min_max = [0, 2**nbits - 1]
        n_groups = N // group_size if N >= group_size else 1
        actual_gs = group_size if N >= group_size else N
        Y_ref = X @ W.T

        def eval_mask(mask):
            W_sparse = W_norm * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=actual_gs)
            Q_grouped = Q.view(K, -1, actual_gs)
            W_deq_norm = (Q_grouped - z) * s
            W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            out_mse = ((Y_ref - Y_approx) ** 2).mean().item()
            return out_mse

        out_std = eval_mask(mask_std)
        out_rbs = eval_mask(mask_rbs)
        improvement = (out_std - out_rbs) / out_std * 100

        results.append({
            'name': name,
            'shape': f"{K}x{N}",
            'out_std': out_std,
            'out_rbs': out_rbs,
            'improvement': improvement
        })

        print(f"{name}: {out_std:.6f} → {out_rbs:.6f} ({improvement:+.2f}%)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    improvements = [r['improvement'] for r in results]
    print(f"Mean improvement: {sum(improvements)/len(improvements):+.2f}%")
    print(f"Layers improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
    print(f"Best: {max(improvements):+.2f}%, Worst: {min(improvements):+.2f}%")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_rbs_multilayer()
