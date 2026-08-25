"""
Analyze Sinkhorn Convergence Dynamics.

Novel angle: Instead of using just the FINAL μ factors, analyze the
CONVERGENCE BEHAVIOR during Sinkhorn iteration.

Key questions:
1. Do some rows/columns converge faster than others?
2. Is convergence rate correlated with importance?
3. Is convergence rate correlated with quantization error?
4. Can convergence rate provide NEW information not in final μ values?
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import numpy as np
from typing import Tuple, List


def sinkhorn_with_dynamics(W: torch.Tensor, order: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """
    Run Sinkhorn normalization and record μ values at each iteration.

    Returns:
        W_norm: Normalized matrix
        mu1: Final column factors
        mu2: Final row factors
        history: List of (mu1_t, mu2_t) at each iteration
    """
    K, N = W.shape
    device = W.device
    dtype = W.dtype

    # Initialize
    log_mu1 = torch.zeros(N, device=device, dtype=dtype)
    log_mu2 = torch.zeros(K, device=device, dtype=dtype)

    tgt_small = 0.9
    history = []

    W2 = W ** 2

    for t in range(order):
        # Store current values
        mu1_t = torch.exp(log_mu1).clone()
        mu2_t = torch.exp(log_mu2).clone()
        history.append((mu1_t, mu2_t))

        # Row normalization
        row_var = (W2 * (mu1_t ** 2).unsqueeze(0)).sum(dim=1)
        std_row = torch.sqrt(row_var + 1e-8)
        log_mu2 = log_mu2 + torch.log(std_row / tgt_small + 1e-8)
        log_mu2 = log_mu2.clamp(-0.3, 10)

        mu2_t = torch.exp(log_mu2)

        # Column normalization
        col_var = (W2 * (mu2_t ** 2).unsqueeze(1)).sum(dim=0)
        std_col = torch.sqrt(col_var + 1e-8)
        log_mu1 = log_mu1 + torch.log(std_col / tgt_small + 1e-8)
        log_mu1 = log_mu1.clamp(-0.3, 10)

    # Final values
    mu1 = torch.exp(log_mu1)
    mu2 = torch.exp(log_mu2)
    history.append((mu1.clone(), mu2.clone()))

    # Normalize
    W_norm = W / (mu2.unsqueeze(1) * mu1.unsqueeze(0))

    return W_norm, mu1, mu2, history


def compute_convergence_residuals(history: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute convergence dynamics for rows and columns.

    Instead of final residual (which is ~0 after convergence), measure:
    1. Total path length: how much did μ change across all iterations
    2. Early vs late ratio: did this row/col stabilize early or late
    """
    T = len(history)

    if T < 3:
        mu1_T, mu2_T = history[-1]
        return torch.zeros_like(mu1_T), torch.zeros_like(mu2_T)

    # Compute total change (path length) for each row/column
    total_change_mu1 = torch.zeros_like(history[0][0])
    total_change_mu2 = torch.zeros_like(history[0][1])

    for t in range(1, T):
        mu1_t, mu2_t = history[t]
        mu1_tm1, mu2_tm1 = history[t-1]
        total_change_mu1 += torch.abs(mu1_t - mu1_tm1)
        total_change_mu2 += torch.abs(mu2_t - mu2_tm1)

    # Normalize by final value (relative change)
    mu1_final = history[-1][0]
    mu2_final = history[-1][1]

    rel_change_mu1 = total_change_mu1 / (mu1_final.abs() + 1e-8)
    rel_change_mu2 = total_change_mu2 / (mu2_final.abs() + 1e-8)

    return rel_change_mu1, rel_change_mu2


def compute_convergence_speed(history: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute when each row/col reached 90% of its final value.
    Earlier = faster convergence.
    """
    T = len(history)
    mu1_final, mu2_final = history[-1]

    # Find iteration where each row/col first reached 90% of final
    threshold = 0.9
    speed_mu1 = torch.full_like(mu1_final, T)  # Default: last iteration
    speed_mu2 = torch.full_like(mu2_final, T)

    for t in range(T):
        mu1_t, mu2_t = history[t]

        # Check which columns reached 90% of final
        reached_mu1 = (mu1_t / (mu1_final + 1e-8)).abs() > threshold
        speed_mu1 = torch.where((speed_mu1 == T) & reached_mu1,
                                torch.full_like(speed_mu1, t),
                                speed_mu1)

        reached_mu2 = (mu2_t / (mu2_final + 1e-8)).abs() > threshold
        speed_mu2 = torch.where((speed_mu2 == T) & reached_mu2,
                                torch.full_like(speed_mu2, t),
                                speed_mu2)

    return speed_mu1, speed_mu2


def analyze_dynamics():
    """Main analysis of Sinkhorn dynamics."""
    print("="*70)
    print("SINKHORN DYNAMICS ANALYSIS")
    print("="*70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test on synthetic data
    print("\n--- Synthetic Data (256x512) ---")
    K, N = 256, 512
    W = torch.randn(K, N, device=device, dtype=torch.float32)

    # Run Sinkhorn with dynamics tracking
    W_norm, mu1, mu2, history = sinkhorn_with_dynamics(W, order=16)

    # Compute path length (total change across iterations)
    r_mu1, r_mu2 = compute_convergence_residuals(history)

    # Compute convergence speed
    speed_mu1, speed_mu2 = compute_convergence_speed(history)

    print(f"\nPath length statistics (total relative change across iterations):")
    print(f"  Column (μ₁): mean={r_mu1.mean():.4f}, std={r_mu1.std():.4f}, CV={r_mu1.std()/r_mu1.mean():.4f}")
    print(f"  Row (μ₂):    mean={r_mu2.mean():.4f}, std={r_mu2.std():.4f}, CV={r_mu2.std()/r_mu2.mean():.4f}")

    print(f"\nConvergence speed (iteration to reach 90% of final):")
    print(f"  Column (μ₁): mean={speed_mu1.mean():.2f}, std={speed_mu1.std():.2f}")
    print(f"  Row (μ₂):    mean={speed_mu2.mean():.2f}, std={speed_mu2.std():.2f}")

    # Correlation between residual and final μ value
    corr_mu1_r1 = np.corrcoef(mu1.cpu().numpy(), r_mu1.cpu().numpy())[0, 1]
    corr_mu2_r2 = np.corrcoef(mu2.cpu().numpy(), r_mu2.cpu().numpy())[0, 1]

    print(f"\nCorrelation (final μ vs residual):")
    print(f"  μ₁ vs r₁: {corr_mu1_r1:.4f}")
    print(f"  μ₂ vs r₂: {corr_mu2_r2:.4f}")

    # Compute importance and quantization error
    batch = 64
    X = torch.randn(batch, N, device=device, dtype=torch.float32)
    act_norms = torch.norm(X, dim=0)

    # sinq_wanda importance (per-weight)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

    # Per-row average importance
    row_importance = importance.mean(dim=1)
    # Per-column average importance
    col_importance = importance.mean(dim=0)

    # Correlation between importance and residual
    corr_imp_r2 = np.corrcoef(row_importance.cpu().numpy(), r_mu2.cpu().numpy())[0, 1]
    corr_imp_r1 = np.corrcoef(col_importance.cpu().numpy(), r_mu1.cpu().numpy())[0, 1]

    print(f"\nCorrelation (importance vs residual):")
    print(f"  Row importance vs r₂: {corr_imp_r2:.4f}")
    print(f"  Col importance vs r₁: {corr_imp_r1:.4f}")

    # Quantize and compute per-row/col error
    from sinq.dual_shift import quantize_rtn

    min_max = [0, 7]  # 3-bit
    group_size = 64
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Dequantize
    n_groups = scales.shape[1]
    Q_grouped = Q.view(K, n_groups, group_size)
    W_deq_norm = (Q_grouped - zeros) * scales
    W_deq_norm = W_deq_norm.view(K, N)
    W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)

    # Per-row and per-column quantization error
    quant_error = (W - W_deq).abs()
    row_error = quant_error.mean(dim=1)
    col_error = quant_error.mean(dim=0)

    # Correlation between error and residual
    corr_err_r2 = np.corrcoef(row_error.cpu().numpy(), r_mu2.cpu().numpy())[0, 1]
    corr_err_r1 = np.corrcoef(col_error.cpu().numpy(), r_mu1.cpu().numpy())[0, 1]

    print(f"\nCorrelation (quant error vs residual):")
    print(f"  Row error vs r₂: {corr_err_r2:.4f}")
    print(f"  Col error vs r₁: {corr_err_r1:.4f}")

    # Key question: Does path length provide NEW info beyond final μ?
    # Check if path length has variance independent of final μ
    print(f"\n=== KEY QUESTION: Is path length informative? ===")

    # If path length is highly correlated with final μ, it's redundant
    if abs(corr_mu2_r2) > 0.9:
        print("  Path length is highly correlated with final μ - REDUNDANT")
    elif abs(corr_err_r2) > abs(corr_mu2_r2):
        print("  Path length correlates MORE with error than with μ - NOVEL INFO!")
    else:
        print("  Path length provides SOME independent information")

    # Check on real weights
    print("\n" + "="*70)
    print("REAL WEIGHTS (Qwen-0.5B layer 0 gate_proj)")
    print("="*70)

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        W_real = model.model.layers[0].mlp.gate_proj.weight.data.float()
        K_r, N_r = W_real.shape
        print(f"Weight shape: [{K_r}x{N_r}]")

        # Run Sinkhorn with dynamics
        W_norm_r, mu1_r, mu2_r, history_r = sinkhorn_with_dynamics(W_real, order=16)
        r_mu1_r, r_mu2_r = compute_convergence_residuals(history_r)

        print(f"\nConvergence residual statistics (real weights):")
        print(f"  Column (μ₁) residual: mean={r_mu1_r.mean():.6f}, std={r_mu1_r.std():.6f}")
        print(f"  Row (μ₂) residual:    mean={r_mu2_r.mean():.6f}, std={r_mu2_r.std():.6f}")

        # Check variance of residuals - high variance = potential for differentiation
        print(f"\nResidual coefficient of variation:")
        print(f"  μ₁ residual CV: {(r_mu1_r.std()/r_mu1_r.mean()).item():.4f}")
        print(f"  μ₂ residual CV: {(r_mu2_r.std()/r_mu2_r.mean()).item():.4f}")

        # Correlation analysis
        corr_mu1_r1_r = np.corrcoef(mu1_r.cpu().numpy(), r_mu1_r.cpu().numpy())[0, 1]
        corr_mu2_r2_r = np.corrcoef(mu2_r.cpu().numpy(), r_mu2_r.cpu().numpy())[0, 1]

        print(f"\nCorrelation (final μ vs residual):")
        print(f"  μ₁ vs r₁: {corr_mu1_r1_r:.4f}")
        print(f"  μ₂ vs r₂: {corr_mu2_r2_r:.4f}")

        # Check if high-residual rows/cols have different properties
        high_r2_mask = r_mu2_r > r_mu2_r.median()
        low_r2_mask = ~high_r2_mask

        print(f"\nHigh vs Low residual row comparison:")
        print(f"  High-residual rows: avg |W|={W_real[high_r2_mask].abs().mean():.6f}, avg μ₂={mu2_r[high_r2_mask].mean():.4f}")
        print(f"  Low-residual rows:  avg |W|={W_real[low_r2_mask].abs().mean():.6f}, avg μ₂={mu2_r[low_r2_mask].mean():.4f}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
Key findings to determine if convergence dynamics provide novel information:
1. If residual is highly correlated with final μ: No new info
2. If residual is uncorrelated with μ but correlated with error: NEW info!
3. If residual has high variance: Can differentiate weights
""")


if __name__ == '__main__':
    analyze_dynamics()
