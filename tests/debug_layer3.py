"""
Debug: What makes layer 3 catastrophically fail for OBS-SINQ?

Layer 0-2: Both methods work (PPL ~12-22)
Layer 3: SCAB = 17.54, OBS-SINQ = 7759 (catastrophic failure)

Let's investigate the difference.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_hessian_inverse,
    compute_activation_norms,
    compute_importance_scores
)


def debug_layer3():
    print("="*70)
    print("DEBUGGING LAYER 3 CATASTROPHIC FAILURE")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"
    sparsity = 0.5

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    text = "The quick brown fox jumps over the lazy dog. " * 50
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    calib_ids = tokens.input_ids.to(device)

    # Analyze layers 2, 3, 4 to see the transition
    for layer_idx in [2, 3, 4]:
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print("="*70)

        layer = model.model.layers[layer_idx]

        # Capture activations
        activations = {}
        def make_hook(name):
            def hook(m, inp, out):
                activations[name] = inp[0].detach()
            return hook

        handles = []
        for name, module in [
            ('q_proj', layer.self_attn.q_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = model(calib_ids)

        for h in handles:
            h.remove()

        # Analyze q_proj (typically problematic)
        for proj_name in ['q_proj', 'down_proj']:
            print(f"\n--- {proj_name} ---")

            if proj_name == 'q_proj':
                module = layer.self_attn.q_proj
            else:
                module = layer.mlp.down_proj

            W = module.weight.data.clone().float()
            X = activations[proj_name].float()
            K, N = W.shape

            print(f"Weight shape: {K} x {N}")

            # 1. Analyze weight distribution
            print(f"\nWeight statistics:")
            print(f"  mean={W.mean():.6f}, std={W.std():.6f}")
            print(f"  min={W.min():.6f}, max={W.max():.6f}")

            # 2. Sinkhorn factors
            W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
            print(f"\nSinkhorn factors:")
            print(f"  μ₁: min={mu1.min():.4f}, max={mu1.max():.4f}, mean={mu1.mean():.4f}, std={mu1.std():.4f}")
            print(f"  μ₂: min={mu2.min():.4f}, max={mu2.max():.4f}, mean={mu2.mean():.4f}, std={mu2.std():.4f}")
            print(f"  μ₁ CV: {(mu1.std()/mu1.mean()).item():.4f}")
            print(f"  μ₂ CV: {(mu2.std()/mu2.mean()).item():.4f}")

            # 3. Activation statistics
            X_flat = X.view(-1, X.shape[-1])[:256]
            print(f"\nActivation statistics:")
            print(f"  mean={X_flat.mean():.6f}, std={X_flat.std():.6f}")
            print(f"  min={X_flat.min():.6f}, max={X_flat.max():.6f}")

            # 4. Hessian properties
            H_inv = compute_hessian_inverse(X_flat.to(device))
            H_inv_diag = H_inv.diag()
            print(f"\nHessian inverse diagonal:")
            print(f"  min={H_inv_diag.min():.6f}, max={H_inv_diag.max():.6f}")
            print(f"  mean={H_inv_diag.mean():.6f}, std={H_inv_diag.std():.6f}")

            # Check for near-zero or negative values
            near_zero = (H_inv_diag.abs() < 1e-6).sum()
            print(f"  Near-zero entries: {near_zero}")

            # 5. Compare importance scores
            act_norms = compute_activation_norms(X)

            # Inverse-μ importance
            inv_mu_imp = compute_importance_scores(W, mu1, mu2, act_norms, method='sinq_wanda_inverse')
            # OBS importance
            obs_imp = (W ** 2) * H_inv_diag.view(1, -1)

            print(f"\nImportance score statistics:")
            print(f"  Inverse-μ: min={inv_mu_imp.min():.6f}, max={inv_mu_imp.max():.6f}, mean={inv_mu_imp.mean():.6f}")
            print(f"  OBS:       min={obs_imp.min():.6f}, max={obs_imp.max():.6f}, mean={obs_imp.mean():.6f}")

            # Correlation
            corr = torch.corrcoef(torch.stack([inv_mu_imp.view(-1), obs_imp.view(-1)]))[0, 1]
            print(f"  Correlation: {corr:.4f}")

            # 6. Compare pruning decisions
            n_prune = int(K * N * sparsity)

            inv_threshold = torch.kthvalue(inv_mu_imp.view(-1), n_prune).values
            inv_mask = (inv_mu_imp.view(-1) > inv_threshold).view(K, N)

            obs_threshold = torch.kthvalue(obs_imp.view(-1), n_prune).values
            obs_mask = (obs_imp.view(-1) > obs_threshold).view(K, N)

            agreement = (inv_mask == obs_mask).float().mean()
            print(f"\nPruning decisions:")
            print(f"  Mask agreement: {agreement*100:.1f}%")

            # 7. Key diagnostic: What weights do they disagree on?
            disagree_mask = inv_mask != obs_mask
            n_disagree = disagree_mask.sum().item()
            if n_disagree > 0:
                # Weights that OBS keeps but inverse-μ prunes
                obs_keeps_inv_prunes = obs_mask & ~inv_mask
                obs_prunes_inv_keeps = ~obs_mask & inv_mask

                print(f"  OBS keeps, inv-μ prunes: {obs_keeps_inv_prunes.sum()}")
                print(f"  OBS prunes, inv-μ keeps: {obs_prunes_inv_keeps.sum()}")

                # Statistics of disagreement weights
                if obs_keeps_inv_prunes.any():
                    obs_keep_weights = W[obs_keeps_inv_prunes]
                    obs_keep_mu1 = mu1.view(1, -1).expand(K, -1)[obs_keeps_inv_prunes]
                    obs_keep_mu2 = mu2.view(-1, 1).expand(K, N)[obs_keeps_inv_prunes]
                    print(f"\n  OBS-kept weights (that inv-μ prunes):")
                    print(f"    W magnitude: mean={obs_keep_weights.abs().mean():.6f}")
                    print(f"    μ₁: mean={obs_keep_mu1.mean():.4f}")
                    print(f"    μ₂: mean={obs_keep_mu2.mean():.4f}")

            # 8. Simulate compensation effect
            print(f"\nCompensation analysis:")

            # OBS compensation
            W_obs_comp = W.clone()
            obs_pruned_mask = (1 - obs_mask.float())
            for i in range(K):
                pruned_w = W[i] * obs_pruned_mask[i]
                comp = -H_inv @ (pruned_w / H_inv_diag)
                W_obs_comp[i] = W[i] * obs_mask[i].float() + comp * obs_mask[i].float()

            # Check for extreme compensation values
            comp_ratio = (W_obs_comp.abs() / (W.abs() + 1e-8))
            extreme_comp = (comp_ratio > 10).sum()
            print(f"  Extreme compensation (>10x): {extreme_comp}")

            # Check the Sinkhorn-normalized compensated weights
            W_obs_norm = W_obs_comp / (mu2.view(-1, 1) * mu1.view(1, -1))
            print(f"  Normalized comp weights: min={W_obs_norm.min():.4f}, max={W_obs_norm.max():.4f}")

            # Check for outliers
            outliers = (W_obs_norm.abs() > 10).sum()
            print(f"  Outliers (|W_norm| > 10): {outliers}")

            if outliers > 0:
                # Where are the outliers?
                outlier_mask = W_obs_norm.abs() > 10
                outlier_mu1 = mu1.view(1, -1).expand(K, -1)[outlier_mask]
                outlier_mu2 = mu2.view(-1, 1).expand(K, N)[outlier_mask]
                print(f"  Outlier μ₁: min={outlier_mu1.min():.4f}, max={outlier_mu1.max():.4f}")
                print(f"  Outlier μ₂: min={outlier_mu2.min():.4f}, max={outlier_mu2.max():.4f}")


if __name__ == "__main__":
    debug_layer3()
