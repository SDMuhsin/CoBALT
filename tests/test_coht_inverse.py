"""
Deep dive into COHT inverse transform
"""
import torch
import numpy as np
from sinq.sinkhorn import sinkhorn_log

def generate_hadamard(n):
    """Generate Hadamard matrix of size n (must be power of 2)"""
    if n == 1:
        return torch.tensor([[1.0]])

    H_half = generate_hadamard(n // 2)
    H = torch.zeros(n, n, dtype=torch.float32)
    H[:n//2, :n//2] = H_half
    H[:n//2, n//2:] = H_half
    H[n//2:, :n//2] = H_half
    H[n//2:, n//2:] = -H_half

    return H / np.sqrt(2)

print("="*80)
print("COHT FORWARD AND INVERSE TRANSFORM ANALYSIS")
print("="*80)

W_original = torch.randn(256, 128)
H128 = generate_hadamard(128)

print("\n1. FORWARD TRANSFORM")
print("-" * 80)
print("W_original shape:", W_original.shape)

# Step 1: Apply Hadamard on columns
W_rotated = W_original @ H128.T
print("After Hadamard (W @ H.T):", W_rotated.shape)

# Step 2: Apply Sinkhorn
W_normalized, mu1, mu2 = sinkhorn_log(W_rotated, 16)
print("After Sinkhorn normalization:", W_normalized.shape)
print("  mu1 shape:", mu1.shape, "range:", [f"{mu1.min():.3f}", f"{mu1.max():.3f}"])
print("  mu2 shape:", mu2.shape, "range:", [f"{mu2.min():.3f}", f"{mu2.max():.3f}"])

# Verify Sinkhorn relationship
check = W_rotated / mu1 / mu2
print("  Sinkhorn check (W_rotated / mu1 / mu2 == W_normalized):",
      torch.allclose(check, W_normalized))

print("\n2. WHAT GETS QUANTIZED")
print("-" * 80)
print("The matrix that gets quantized is W_normalized")
print("  min:", f"{W_normalized.min():.3f}", "max:", f"{W_normalized.max():.3f}")

print("\n3. INVERSE TRANSFORM OPTIONS")
print("-" * 80)

print("\nOPTION A (PROPOSAL'S VERSION):")
print("  Step 1: dequantize to get W_normalized")
print("  Step 2: rescale -> W_rescaled = W_normalized * mu1 * mu2")
print("  Step 3: apply Hadamard inverse -> W_final = W_rescaled @ H128")

W_rescaled_A = W_normalized * mu1 * mu2
W_final_A = W_rescaled_A @ H128
error_A = (W_original - W_final_A).abs().max().item()
print(f"  Error: {error_A:.6f}")
print(f"  Expected to recover: W_rotated (intermediate)")
print(f"  Actually equals W_rotated? {torch.allclose(W_rescaled_A, W_rotated, atol=1e-5)}")

print("\nOPTION B (HADAMARD FIRST):")
print("  Step 1: dequantize to get W_normalized")
print("  Step 2: apply Hadamard inverse -> W_rotated_back = W_normalized @ H128")
print("  Step 3: rescale -> W_final = W_rotated_back * mu1 * mu2")

W_rotated_back_B = W_normalized @ H128
W_final_B = W_rotated_back_B * mu1 * mu2
error_B = (W_original - W_final_B).abs().max().item()
print(f"  Error: {error_B:.6f}")

print("\n4. UNDERSTANDING THE MATH")
print("-" * 80)
print("Forward:")
print("  W_rotated = W_original @ H.T")
print("  W_normalized = W_rotated / mu1 / mu2")
print("")
print("To invert Sinkhorn:")
print("  W_rotated = W_normalized * mu1 * mu2  ✓")
print("")
print("To invert Hadamard (since H is orthogonal, H.T @ H = I):")
print("  W_original = W_rotated @ H")
print("  because: (W_original @ H.T) @ H = W_original @ (H.T @ H) = W_original @ I = W_original")
print("")
print("Combined inverse (Option A):")
print("  W_final = (W_normalized * mu1 * mu2) @ H")
print("         = W_rotated @ H")
print("         = W_original  ✓ CORRECT")
print("")
print("Option B would give:")
print("  W_final = (W_normalized @ H) * mu1 * mu2")
print("  This is NOT the same as (W_normalized * mu1 * mu2) @ H")
print("  because mu1, mu2 have different shapes and broadcasting!")
print("")

print("\n5. VERIFYING THE MATH WITH SHAPES")
print("-" * 80)
print(f"W_normalized: {W_normalized.shape}")
print(f"mu1: {mu1.shape}  (broadcasts along columns)")
print(f"mu2: {mu2.shape}  (broadcasts along rows)")
print(f"H128: {H128.shape}")
print("")
print("Option A: (W_normalized * mu1 * mu2) @ H128")
print(f"  Step 1: W_normalized * mu1 -> element-wise, broadcasts to {W_normalized.shape}")
print(f"  Step 2: result * mu2 -> element-wise, broadcasts to {W_normalized.shape}")
print(f"  Step 3: result @ H128 -> matmul {W_normalized.shape} @ {H128.shape} = {W_original.shape}")
print("")
print("Option B: (W_normalized @ H128) * mu1 * mu2")
print(f"  Step 1: W_normalized @ H128 -> {W_normalized.shape} @ {H128.shape} = {W_original.shape}")
print(f"  Step 2: result * mu1 -> WRONG! mu1 is for W_rotated columns, not W_original!")
print(f"  Step 3: result * mu2 -> WRONG! mu2 is for W_rotated rows, not W_original!")
print("")

print("\n6. THE KEY INSIGHT")
print("-" * 80)
print("The Sinkhorn scales (mu1, mu2) are TIED TO W_rotated:")
print("  - mu1 scales the COLUMNS of W_rotated")
print("  - mu2 scales the ROWS of W_rotated")
print("")
print("Therefore:")
print("  1. We must FIRST undo Sinkhorn: W_normalized * mu1 * mu2 = W_rotated")
print("  2. THEN undo Hadamard: W_rotated @ H = W_original")
print("")
print("The proposal's inverse is CORRECT!")
print("")

print("\n7. FINAL VERIFICATION")
print("-" * 80)
print(f"Option A error: {error_A:.9f}")
print(f"Option B error: {error_B:.9f}")
print("")
if error_A < 1e-5:
    print("✓ PROPOSAL'S INVERSE IS CORRECT")
else:
    print("✗ ERROR TOO HIGH")
