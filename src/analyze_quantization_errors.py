"""
Phase 1.2: Quantization Error Analysis Script

Analyzes SINQ quantization to identify potential improvement opportunities:
- Per-layer error distributions
- Weight statistics before/after Sinkhorn
- Outlier patterns
- Correlation with quantization error
"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log


def imbalance(mat):
    """Compute matrix imbalance: max std / min std across rows and columns."""
    s1, s2 = torch.std(mat, 1), torch.std(mat, 0)
    s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
    s_max = torch.maximum(s1.max(), s2.max())
    return s_max / s_min


def analyze_weight_matrix(name, W, nbits=3, group_size=64):
    """Analyze a single weight matrix for quantization properties."""
    W = W.float()
    rows, cols = W.shape

    # Original statistics
    orig_std_rows = W.std(dim=1)
    orig_std_cols = W.std(dim=0)
    orig_imbalance = imbalance(W).item()
    orig_kurtosis_rows = ((W - W.mean(dim=1, keepdim=True))**4).mean(dim=1) / (W.std(dim=1)**4 + 1e-8)
    orig_kurtosis_cols = ((W - W.mean(dim=0, keepdim=True))**4).mean(dim=0) / (W.std(dim=0)**4 + 1e-8)

    # Apply Sinkhorn normalization
    # sinkhorn_log returns: (scaled_matrix, mu1_col_scales, mu2_row_scales)
    # where scaled_matrix = W / mu1 / mu2
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    # mu1 has shape (cols,), mu2 has shape (rows, 1)

    # Post-Sinkhorn statistics
    sink_std_rows = W_norm.std(dim=1)
    sink_std_cols = W_norm.std(dim=0)
    sink_imbalance = imbalance(W_norm).item()

    # Quantization error analysis
    # Simple RTN quantization of normalized matrix
    W_norm_flat = W_norm.reshape(-1, group_size)
    w_max = W_norm_flat.max(dim=1, keepdim=True).values
    w_min = W_norm_flat.min(dim=1, keepdim=True).values
    scale = (w_max - w_min) / (2**nbits - 1)
    zero = w_min
    Q_flat = torch.round((W_norm_flat - zero) / (scale + 1e-10))
    Q_flat = torch.clamp(Q_flat, 0, 2**nbits - 1)
    W_recon_flat = Q_flat * scale + zero

    # Per-group quantization error
    group_errors = (W_norm_flat - W_recon_flat).abs().mean(dim=1)
    group_errors_max = (W_norm_flat - W_recon_flat).abs().max(dim=1).values

    # Reconstruct and compute full matrix error
    W_recon = W_recon_flat.reshape(W_norm.shape)

    # Reconstruct with dual scales: W_orig ≈ W_recon * mu1 * mu2
    # mu1 is (cols,), mu2 is (rows, 1)
    W_recon_orig_scale = W_recon * mu1.view(1, -1) * mu2.view(-1, 1)

    full_mse = ((W - W_recon_orig_scale)**2).mean().item()
    full_mae = (W - W_recon_orig_scale).abs().mean().item()

    # Analyze which rows/cols have highest error
    row_errors = ((W - W_recon_orig_scale)**2).mean(dim=1)
    col_errors = ((W - W_recon_orig_scale)**2).mean(dim=0)

    # Correlation analysis: does high original std correlate with high error?
    row_std_error_corr = torch.corrcoef(torch.stack([orig_std_rows, row_errors]))[0, 1].item()
    col_std_error_corr = torch.corrcoef(torch.stack([orig_std_cols, col_errors]))[0, 1].item()

    # Analyze residual imbalance after Sinkhorn
    # Check if certain rows/cols are still problematic
    residual_row_imbalance = sink_std_rows.max() / sink_std_rows.min()
    residual_col_imbalance = sink_std_cols.max() / sink_std_cols.min()

    return {
        'name': name,
        'shape': tuple(W.shape),
        'orig_imbalance': orig_imbalance,
        'sink_imbalance': sink_imbalance,
        'imbalance_reduction': orig_imbalance / (sink_imbalance + 1e-8),
        'full_mse': full_mse,
        'full_mae': full_mae,
        'mean_group_error': group_errors.mean().item(),
        'max_group_error': group_errors_max.max().item(),
        'std_group_error': group_errors.std().item(),
        'row_std_error_corr': row_std_error_corr if not np.isnan(row_std_error_corr) else 0,
        'col_std_error_corr': col_std_error_corr if not np.isnan(col_std_error_corr) else 0,
        'residual_row_imbalance': residual_row_imbalance.item(),
        'residual_col_imbalance': residual_col_imbalance.item(),
        'orig_kurtosis_mean': orig_kurtosis_rows.mean().item(),
        'sink_std_mean': sink_std_rows.mean().item(),
        'sink_std_std': sink_std_rows.std().item(),
        'mu1_range': (mu1.min().item(), mu1.max().item()),
        'mu2_range': (mu2.min().item(), mu2.max().item()),
    }


def main():
    print("=" * 70)
    print("PHASE 1.2: Quantization Error Analysis")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3-1.7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,  # Use float32 for analysis precision
        trust_remote_code=True,
    )

    # Analyze all linear layers
    results = []
    layer_types = defaultdict(list)

    print("\nAnalyzing weight matrices...")
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            # Skip small matrices
            if param.shape[0] < 64 or param.shape[1] < 64:
                continue

            W = param.data.clone()
            # Ensure dimensions are divisible by group_size
            if W.shape[1] % 64 != 0:
                pad_size = 64 - (W.shape[1] % 64)
                W = torch.nn.functional.pad(W, (0, pad_size))

            try:
                result = analyze_weight_matrix(name, W)
                results.append(result)

                # Categorize by layer type
                if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    layer_types['attention_qkv'].append(result)
                elif 'o_proj' in name:
                    layer_types['attention_o'].append(result)
                elif 'gate_proj' in name or 'up_proj' in name:
                    layer_types['mlp_gate_up'].append(result)
                elif 'down_proj' in name:
                    layer_types['mlp_down'].append(result)
                else:
                    layer_types['other'].append(result)
            except Exception as e:
                print(f"  Skipping {name}: {e}")
                continue

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    all_mse = [r['full_mse'] for r in results]
    all_imb_red = [r['imbalance_reduction'] for r in results]
    all_residual_row = [r['residual_row_imbalance'] for r in results]
    all_residual_col = [r['residual_col_imbalance'] for r in results]
    all_row_corr = [r['row_std_error_corr'] for r in results]
    all_col_corr = [r['col_std_error_corr'] for r in results]

    print(f"\nTotal layers analyzed: {len(results)}")
    print(f"\nMSE Statistics:")
    print(f"  Mean MSE: {np.mean(all_mse):.6f}")
    print(f"  Max MSE:  {np.max(all_mse):.6f}")
    print(f"  Std MSE:  {np.std(all_mse):.6f}")

    print(f"\nImbalance Reduction (orig/sink):")
    print(f"  Mean: {np.mean(all_imb_red):.2f}x")
    print(f"  Min:  {np.min(all_imb_red):.2f}x")
    print(f"  Max:  {np.max(all_imb_red):.2f}x")

    print(f"\nResidual Imbalance After Sinkhorn:")
    print(f"  Row - Mean: {np.mean(all_residual_row):.4f}, Max: {np.max(all_residual_row):.4f}")
    print(f"  Col - Mean: {np.mean(all_residual_col):.4f}, Max: {np.max(all_residual_col):.4f}")

    print(f"\nCorrelation: Original Std vs Quantization Error:")
    print(f"  Row correlation - Mean: {np.mean(all_row_corr):.4f}")
    print(f"  Col correlation - Mean: {np.mean(all_col_corr):.4f}")

    # Per layer-type analysis
    print("\n" + "=" * 70)
    print("PER LAYER-TYPE ANALYSIS")
    print("=" * 70)

    for ltype, lresults in layer_types.items():
        if not lresults:
            continue
        mses = [r['full_mse'] for r in lresults]
        imbs = [r['sink_imbalance'] for r in lresults]
        print(f"\n{ltype} ({len(lresults)} layers):")
        print(f"  MSE - Mean: {np.mean(mses):.6f}, Max: {np.max(mses):.6f}")
        print(f"  Post-Sink Imbalance - Mean: {np.mean(imbs):.4f}, Max: {np.max(imbs):.4f}")

    # Find worst layers
    print("\n" + "=" * 70)
    print("WORST LAYERS (by MSE)")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: x['full_mse'], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"\n{i+1}. {r['name']}")
        print(f"   Shape: {r['shape']}")
        print(f"   MSE: {r['full_mse']:.6f}")
        print(f"   Orig Imbalance: {r['orig_imbalance']:.2f} → Sink: {r['sink_imbalance']:.4f}")
        print(f"   Residual Imb - Row: {r['residual_row_imbalance']:.4f}, Col: {r['residual_col_imbalance']:.4f}")
        print(f"   Std-Error Corr - Row: {r['row_std_error_corr']:.4f}, Col: {r['col_std_error_corr']:.4f}")

    # KEY OBSERVATIONS for hypothesis generation
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS FOR HYPOTHESIS GENERATION")
    print("=" * 70)

    # Check if residual imbalance correlates with error
    residual_imb = [max(r['residual_row_imbalance'], r['residual_col_imbalance']) for r in results]
    mse_residual_corr = np.corrcoef(residual_imb, all_mse)[0, 1]
    print(f"\n1. Residual imbalance vs MSE correlation: {mse_residual_corr:.4f}")

    # Check group error variance
    group_error_stds = [r['std_group_error'] for r in results]
    print(f"\n2. Group error std - Mean: {np.mean(group_error_stds):.6f}, Max: {np.max(group_error_stds):.6f}")
    print("   (High variance suggests some groups are much harder to quantize)")

    # Check if certain layer depths are worse
    depths = []
    for r in results:
        try:
            depth = int(r['name'].split('.')[2])
            depths.append((depth, r['full_mse']))
        except:
            pass

    if depths:
        depths.sort(key=lambda x: x[0])
        early_mse = np.mean([d[1] for d in depths[:len(depths)//3]])
        mid_mse = np.mean([d[1] for d in depths[len(depths)//3:2*len(depths)//3]])
        late_mse = np.mean([d[1] for d in depths[2*len(depths)//3:]])
        print(f"\n3. MSE by depth - Early: {early_mse:.6f}, Mid: {mid_mse:.6f}, Late: {late_mse:.6f}")

    # Analyze scale factor distributions
    mu1_ranges = [r['mu1_range'] for r in results]
    mu2_ranges = [r['mu2_range'] for r in results]
    mu1_ratios = [m[1]/m[0] for m in mu1_ranges]
    mu2_ratios = [m[1]/m[0] for m in mu2_ranges]
    print(f"\n4. Scale factor (mu1) max/min ratios - Mean: {np.mean(mu1_ratios):.2f}, Max: {np.max(mu1_ratios):.2f}")
    print(f"   Scale factor (mu2) max/min ratios - Mean: {np.mean(mu2_ratios):.2f}, Max: {np.max(mu2_ratios):.2f}")

    # Analyze post-Sinkhorn standard deviation uniformity
    sink_std_stds = [r['sink_std_std'] for r in results]
    print(f"\n5. Post-Sinkhorn std variation - Mean: {np.mean(sink_std_stds):.6f}, Max: {np.max(sink_std_stds):.6f}")
    print("   (Lower = more uniform distribution, better for quantization)")

    # Save results for future reference
    import json
    results_path = os.path.join(project_root, "results", "quantization_analysis.json")

    # Convert to serializable format
    serializable = []
    for r in results:
        sr = dict(r)
        sr['mu1_range'] = list(sr['mu1_range'])
        sr['mu2_range'] = list(sr['mu2_range'])
        sr['shape'] = list(sr['shape'])
        serializable.append(sr)

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
