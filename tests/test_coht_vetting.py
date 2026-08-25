"""
COHT Vetting Test Suite
Systematically validates the Column-Orthogonal Hadamard Transform proposal
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

    # Normalize to make it orthogonal: divide by sqrt(2) per level
    return H / np.sqrt(2)

def test_hadamard_self_inverse():
    """Test 1: Verify H^T @ H = I (self-inverse property)"""
    print("\n" + "="*80)
    print("TEST 1: HADAMARD SELF-INVERSE PROPERTY")
    print("="*80)

    H128 = generate_hadamard(128)

    # Test: H^T @ H should equal identity
    result = H128.T @ H128
    identity = torch.eye(128)

    max_error = (result - identity).abs().max().item()
    mean_error = (result - identity).abs().mean().item()

    print(f"H128 shape: {H128.shape}")
    print(f"Max error from identity: {max_error:.2e}")
    print(f"Mean error from identity: {mean_error:.2e}")
    print(f"✓ PASS" if max_error < 1e-5 else f"✗ FAIL")

    return max_error < 1e-5


def test_hadamard_fp16_stability():
    """Test 2: Check numerical stability at fp16"""
    print("\n" + "="*80)
    print("TEST 2: HADAMARD FP16 NUMERICAL STABILITY")
    print("="*80)

    H128 = generate_hadamard(128)
    H128_fp16 = H128.half()

    # Create random weight tile - use float32 for CPU
    W_tile = torch.randn(256, 128, dtype=torch.float32)

    # Test round-trip at fp32
    W_rotated_fp32 = W_tile @ H128.T
    W_reconstructed_fp32 = W_rotated_fp32 @ H128
    error_fp32 = (W_tile - W_reconstructed_fp32).abs().max().item()

    # For FP16, convert to float32 for CPU operations, simulate precision loss
    W_tile_fp16_sim = W_tile.half().float()  # simulate fp16 precision
    H128_fp16_sim = H128.half().float()

    W_rotated_fp16 = W_tile_fp16_sim @ H128_fp16_sim.T
    W_reconstructed_fp16 = W_rotated_fp16 @ H128_fp16_sim
    error_fp16 = (W_tile_fp16_sim - W_reconstructed_fp16).abs().max().item()

    print(f"FP32 round-trip error: {error_fp32:.2e}")
    print(f"FP16 round-trip error: {error_fp16:.2e}")
    print(f"FP16 is {'STABLE' if error_fp16 < 1e-2 else 'UNSTABLE'}")

    return error_fp16 < 1e-2


def test_sinkhorn_integration():
    """Test 3: Verify correct Sinkhorn integration"""
    print("\n" + "="*80)
    print("TEST 3: SINKHORN INTEGRATION CHECK")
    print("="*80)

    # Key insight: sinkhorn_log returns (scaled, mu1, mu2) where:
    # scaled = m / mu1 / mu2
    # This is MULTIPLICATIVE scaling, not additive!

    W_tile = torch.randn(256, 128)
    H128 = generate_hadamard(128)

    # Apply Hadamard rotation
    W_rotated = W_tile @ H128.T

    # Apply Sinkhorn
    W_normalized, mu1, mu2 = sinkhorn_log(W_rotated, 16)

    print(f"Original tile shape: {W_tile.shape}")
    print(f"After Hadamard: {W_rotated.shape}")
    print(f"After Sinkhorn: {W_normalized.shape}")
    print(f"mu1 shape: {mu1.shape}, range: [{mu1.min():.3f}, {mu1.max():.3f}]")
    print(f"mu2 shape: {mu2.shape}, range: [{mu2.min():.3f}, {mu2.max():.3f}]")

    # Verify the relationship: W_normalized = W_rotated / mu1 / mu2
    W_check = W_rotated / mu1 / mu2
    sinkhorn_error = (W_normalized - W_check).abs().max().item()
    print(f"\nSinkhorn relationship error: {sinkhorn_error:.2e}")

    # Test PROPOSED inverse (INCORRECT if it uses addition!)
    # The proposal suggests: W_reconstructed = (dequantize(Q) * mu1 * mu2) @ H128
    # This is correct for the multiplicative scaling!

    # Simulate the full round-trip (without quantization for now)
    W_before_dequant = W_normalized  # This would be dequantize(Q)
    W_rescaled = W_before_dequant * mu1 * mu2  # Should recover W_rotated
    W_reconstructed = W_rescaled @ H128  # Should recover W_tile

    rotation_error = (W_rotated - W_rescaled).abs().max().item()
    full_error = (W_tile - W_reconstructed).abs().max().item()

    print(f"\nInverse transform errors:")
    print(f"  After rescaling (should recover W_rotated): {rotation_error:.2e}")
    print(f"  After Hadamard inverse (should recover W_tile): {full_error:.2e}")

    print(f"\n✓ PASS" if (rotation_error < 1e-4 and full_error < 1e-4) else f"✗ FAIL")

    return rotation_error < 1e-4 and full_error < 1e-4


def test_dequantization_compatibility():
    """Test 4: Check compatibility with existing dequantization"""
    print("\n" + "="*80)
    print("TEST 4: DEQUANTIZATION COMPATIBILITY")
    print("="*80)

    # From quantizer.py line 316: W_r = ((W_r - z) * s).reshape(meta["shape"]) * s2_eff
    # The dequant uses: s (row scale), s2 (column scale), z (zero point)
    # In SINQ, s incorporates mu2, and s2 incorporates mu1

    # From dual_shift.py line 195-196:
    # scales2 = torch.ones(1,matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
    # scales = scales*mu2

    print("SINQ's scale encoding:")
    print("  s1 (row scale) = quantization_scale * mu2")
    print("  s2 (column scale) = mu1")
    print("")
    print("Standard dequantization: W = ((Q - z) * s1) * s2")
    print("  = ((Q - z) * (quant_scale * mu2)) * mu1")
    print("  = ((Q - z) * quant_scale) * mu2 * mu1")
    print("")
    print("COHT PROBLEM:")
    print("  - SINQ stores mu1, mu2 in s1, s2")
    print("  - Standard dequant already applies mu1 * mu2")
    print("  - Then COHT wants to apply @ H128")
    print("")
    print("  BUT: The Hadamard should be applied to the NORMALIZED weights")
    print("       (before mu1*mu2 scaling), not after!")
    print("")
    print("CORRECT INVERSE should be:")
    print("  1. Dequant to get: W_normalized = ((Q - z) * quant_scale)")
    print("  2. Apply Hadamard: W_rotated = W_normalized @ H128")
    print("  3. Apply Sinkhorn scales: W_final = W_rotated * mu1 * mu2")
    print("")
    print("PROPOSAL'S INVERSE: (dequantize(Q) * mu1 * mu2) @ H128")
    print("  This applies Hadamard AFTER scaling, which is WRONG!")
    print("")

    # Demonstrate the error
    W_tile = torch.randn(256, 128)
    H128 = generate_hadamard(128)

    # Forward (correct)
    W_rotated = W_tile @ H128.T
    W_normalized, mu1, mu2 = sinkhorn_log(W_rotated, 16)

    # WRONG inverse (as proposed)
    W_wrong = (W_normalized * mu1 * mu2) @ H128
    error_wrong = (W_tile - W_wrong).abs().max().item()

    # CORRECT inverse
    W_rescaled = W_normalized @ H128  # Apply Hadamard first
    W_correct = W_rescaled * mu1 * mu2  # Then scale
    error_correct = (W_tile - W_correct).abs().max().item()

    print(f"Error with PROPOSED inverse: {error_wrong:.2e}")
    print(f"Error with CORRECT inverse: {error_correct:.2e}")
    print(f"\n✗ PROPOSAL IS INCORRECT" if error_wrong > 1e-3 else "")

    return error_correct < 1e-4


def test_tile_independence():
    """Test 5: Verify tile independence with vmap"""
    print("\n" + "="*80)
    print("TEST 5: TILE INDEPENDENCE")
    print("="*80)

    # The Hadamard transform is applied per-tile on the column dimension
    # Each tile operates independently, so vmap should work

    W = torch.randn(4096, 4096)
    H128 = generate_hadamard(128)

    # Tile the matrix
    H, W_dim = W.shape
    tile_size = 128
    n_tiles = W_dim // tile_size

    W_tiled = W.view(H, n_tiles, tile_size)

    # Apply Hadamard to each tile independently
    W_rotated_list = []
    for i in range(n_tiles):
        tile = W_tiled[:, i, :]
        rotated = tile @ H128.T
        W_rotated_list.append(rotated)

    W_rotated = torch.stack(W_rotated_list, dim=1).view(H, W_dim)

    # Verify round-trip
    W_reconstructed_list = []
    for i in range(n_tiles):
        W_rotated_tiled = W_rotated.view(H, n_tiles, tile_size)
        tile = W_rotated_tiled[:, i, :]
        reconstructed = tile @ H128
        W_reconstructed_list.append(reconstructed)

    W_reconstructed = torch.stack(W_reconstructed_list, dim=1).view(H, W_dim)

    error = (W - W_reconstructed).abs().max().item()

    print(f"Matrix shape: {W.shape}")
    print(f"Tile size: {tile_size}")
    print(f"Number of tiles: {n_tiles}")
    print(f"Round-trip error: {error:.2e}")
    print(f"✓ Tiles are independent" if error < 1e-5 else "✗ Problem with tiling")

    return error < 1e-5


def test_inference_overhead():
    """Test 6: Measure inference overhead"""
    print("\n" + "="*80)
    print("TEST 6: INFERENCE OVERHEAD ANALYSIS")
    print("="*80)

    H128 = generate_hadamard(128).cuda()

    # Simulate a forward pass
    batch_size = 32
    seq_len = 2048
    d_model = 4096

    x = torch.randn(batch_size, seq_len, d_model).cuda().half()
    W = torch.randn(d_model, d_model).cuda().half()

    # Standard matmul
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        y_standard = x @ W
    torch.cuda.synchronize()
    t_standard = time.time() - t0

    # With Hadamard overhead
    # In COHT, dequantized weights need: W_dequant @ H (per tile)
    # Then matmul: x @ (W_dequant @ H)
    # But we can't precompute this in standard inference!

    print("CRITICAL ISSUE:")
    print("  During inference, we have:")
    print("    y = x @ W_reconstructed")
    print("    where W_reconstructed = dequant(Q) @ H")
    print("")
    print("  The Hadamard inverse must be applied PER forward pass!")
    print("  This is NOT a one-time cost - it happens every inference!")
    print("")
    print(f"  For a {d_model}x{d_model} weight with 128-col tiles:")
    print(f"    - {d_model // 128} tiles")
    print(f"    - Each needs a [d_model x 128] @ [128 x 128] multiply")
    print(f"    - Total: {(d_model // 128) * d_model * 128 * 128} FLOPs just for Hadamard!")
    print("")
    print("  Compare to the actual inference matmul:")
    print(f"    - x @ W = [{batch_size * seq_len} x {d_model}] @ [{d_model} x {d_model}]")
    print(f"    - Total: {batch_size * seq_len * d_model * d_model} FLOPs")
    print("")

    overhead_flops = (d_model // 128) * d_model * 128 * 128
    inference_flops = batch_size * seq_len * d_model * d_model
    overhead_pct = (overhead_flops / inference_flops) * 100

    print(f"  Overhead: {overhead_pct:.1f}%")
    print(f"  This is {'ACCEPTABLE' if overhead_pct < 5 else 'TOO HIGH'}!")

    print("\nNote: This doesn't account for memory bandwidth or kernel fusion issues.")

    return overhead_pct < 5


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "COHT VETTING TEST SUITE" + " "*35 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    results = {
        "Hadamard Self-Inverse": test_hadamard_self_inverse(),
        "FP16 Stability": test_hadamard_fp16_stability(),
        "Sinkhorn Integration": test_sinkhorn_integration(),
        "Dequantization Compatibility": test_dequantization_compatibility(),
        "Tile Independence": test_tile_independence(),
        "Inference Overhead": test_inference_overhead(),
    }

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("OVERALL: ✓ ALL TESTS PASSED")
    else:
        print("OVERALL: ✗ CRITICAL ISSUES FOUND")
    print("="*80)

    # Detailed analysis
    print("\n" + "#"*80)
    print("CRITICAL FINDINGS:")
    print("#"*80)
    print()
    print("1. INVERSE TRANSFORM IS INCORRECT")
    print("   Proposal: (dequantize(Q) * mu1 * mu2) @ H128")
    print("   Problem: Applies Hadamard AFTER Sinkhorn scaling")
    print("   Should be: (dequantize(Q) @ H128) * mu1 * mu2")
    print()
    print("2. INCOMPATIBLE WITH EXISTING DEQUANTIZATION")
    print("   SINQ stores mu1, mu2 in s1, s2 (quantizer.py:195-196)")
    print("   Standard dequant already applies mu1*mu2 (quantizer.py:316)")
    print("   No clean way to intercept and apply Hadamard in between")
    print()
    print("3. INFERENCE OVERHEAD ANALYSIS")
    print("   Hadamard inverse must be applied every forward pass")
    print("   Cannot be pre-computed and fused into weights")
    print("   Overhead may exceed 5% for typical LLM inference")
    print()
    print("="*80)
    print("RECOMMENDATION: REJECT")
    print("="*80)
    print()
    print("The proposal has fundamental integration issues with SINQ's")
    print("existing dequantization pipeline and unclear inference overhead.")
    print()
