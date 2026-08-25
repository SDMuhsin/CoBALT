"""
Precise COHT inference overhead calculation
"""
import torch

print("="*80)
print("COHT INFERENCE OVERHEAD - PRECISE CALCULATION")
print("="*80)

print("\n1. THE ACTUAL COMPUTATION FLOW")
print("-" * 80)
print("WITHOUT COHT:")
print("  y = x @ W")
print("  where x: [batch, seq, d_in], W: [d_in, d_out]")
print("  Cost: batch * seq * d_in * d_out FLOPs")
print("")
print("WITH COHT (naive):")
print("  Step 1: W_reconstructed = W_dequant @ H  (per tile)")
print("  Step 2: y = x @ W_reconstructed")
print("  Cost_step1: (d_in // tile) * d_in * tile * tile FLOPs")
print("  Cost_step2: batch * seq * d_in * d_out FLOPs")
print("")

print("\n2. BUT WAIT - CAN WE FUSE?")
print("-" * 80)
print("The key insight: (x @ W) @ H can be computed as x @ (W @ H)")
print("")
print("Option A (naive): Compute W @ H first, then x @ result")
print("  Cost: (d_in * d_out * 128) + (batch * seq * d_in * d_out)")
print("")
print("Option B (smarter): Tile both operations")
print("  For each tile W_tile:")
print("    1. W_tile_reconstructed = W_tile_dequant @ H")
print("    2. y_partial = x @ W_tile_reconstructed")
print("")
print("But this still requires computing W @ H!")
print("")

print("\n3. ACTUAL OVERHEAD CALCULATION")
print("-" * 80)

# Example: Llama-7B dimensions
d_model = 4096
batch = 1
seq_len = 1  # Token generation
tile_size = 128

print(f"Model dimensions:")
print(f"  d_model = {d_model}")
print(f"  batch = {batch}")
print(f"  seq_len = {seq_len}")
print(f"  tile_size = {tile_size}")
print("")

# Self-attention projection (d_model -> d_model)
print("Self-attention projection: W is [4096 x 4096]")
n_tiles = d_model // tile_size
print(f"  Number of tiles: {n_tiles}")
print("")

# Cost of applying Hadamard to all tiles
hadamard_flops = n_tiles * d_model * tile_size * tile_size
print(f"  Hadamard cost (W @ H for all tiles):")
print(f"    {n_tiles} tiles * {d_model} * {tile_size} * {tile_size} = {hadamard_flops:,} FLOPs")
print("")

# Cost of actual inference matmul
inference_flops = batch * seq_len * d_model * d_model
print(f"  Inference matmul (x @ W):")
print(f"    {batch} * {seq_len} * {d_model} * {d_model} = {inference_flops:,} FLOPs")
print("")

overhead_pct = (hadamard_flops / inference_flops) * 100
print(f"  Overhead: {overhead_pct:.1f}%")
print("")

print("\n4. SCALING WITH BATCH AND SEQ_LEN")
print("-" * 80)

for batch, seq_len in [(1, 1), (1, 128), (8, 128), (32, 2048)]:
    inference_flops = batch * seq_len * d_model * d_model
    overhead_pct = (hadamard_flops / inference_flops) * 100
    print(f"  batch={batch:3}, seq={seq_len:4}: overhead = {overhead_pct:6.2f}%")

print("")
print("Key observation:")
print("  - For token generation (batch=1, seq=1): MASSIVE overhead")
print("  - For training/large batches: overhead becomes negligible")
print("")

print("\n5. WAIT - DO WE NEED TO RECOMPUTE EVERY TIME?")
print("-" * 80)
print("CRITICAL REALIZATION:")
print("")
print("Current SINQ flow:")
print("  forward():")
print("    1. Dequantize on-the-fly: W = dequant(Q, scales, zeros)")
print("    2. Compute: y = x @ W")
print("")
print("With COHT:")
print("  forward():")
print("    1. Dequantize on-the-fly: W = dequant(Q, scales, zeros)")
print("    2. Apply Hadamard: W = W @ H  # <-- NEW")
print("    3. Compute: y = x @ W")
print("")
print("The question: Can we cache W after step 2?")
print("")
print("Answer: NO for standard quantization!")
print("  - The whole point of quantization is to save memory")
print("  - Storing W defeats the purpose")
print("")
print("BUT: What if we have a hybrid approach?")
print("")

print("\n6. HYBRID APPROACH: LAZY DEQUANTIZATION")
print("-" * 80)
print("Idea: Cache the dequantized + Hadamard-transformed weights")
print("")
print("Memory comparison:")
n_layers = 32
n_weights_per_layer = 4  # Q, K, V, O projections

total_weights = n_layers * n_weights_per_layer
quantized_size_per_weight = (d_model * d_model * 4) // 8  # 4-bit
fp16_size_per_weight = d_model * d_model * 2  # fp16

print(f"  Per weight matrix: {d_model} x {d_model}")
print(f"  Quantized (4-bit): {quantized_size_per_weight / 1e6:.1f} MB")
print(f"  FP16 (cached): {fp16_size_per_weight / 1e6:.1f} MB")
print("")
print(f"  Full model ({total_weights} weight matrices):")
print(f"    Quantized: {total_weights * quantized_size_per_weight / 1e9:.2f} GB")
print(f"    If cached FP16: {total_weights * fp16_size_per_weight / 1e9:.2f} GB")
print("")
print("Conclusion:")
print("  Caching defeats the 4x memory savings of 4-bit quantization!")
print("")

print("\n7. ALTERNATIVE: FUSED KERNEL")
print("-" * 80)
print("Could we fuse the operations?")
print("")
print("Standard: y = x @ W")
print("COHT: y = x @ (dequant(Q) @ H)")
print("")
print("Challenges:")
print("  1. Hadamard is dense [128 x 128] matmul")
print("  2. Can't skip it or factor it out")
print("  3. Fused kernel would be complex: unpack + dequant + Hadamard + matmul")
print("")
print("Even with fusion, still need to compute:")
print(f"  {hadamard_flops:,} additional FLOPs per forward pass")
print("")

print("\n8. REAL-WORLD IMPACT")
print("-" * 80)
print("Token generation (batch=1, seq=1):")
print(f"  Overhead per layer: {overhead_pct:.1f}%")
print(f"  For {total_weights} attention layers: still {overhead_pct:.1f}% per layer")
print("")
print("This is CATASTROPHIC!")
print("")
print("The Hadamard inverse costs 128x MORE than the actual inference matmul!")
print("")

print("\n" + "="*80)
print("FINAL OVERHEAD VERDICT")
print("="*80)
print("")
print(f"For token generation (batch=1, seq=1):")
print(f"  COHT overhead: {overhead_pct:.0f}%")
print(f"  In other words: {overhead_pct/100:.0f}x slower")
print("")
print("Assessment:")
if overhead_pct > 10:
    print("  ✗✗✗ COMPLETELY UNACCEPTABLE for production inference")
    print("  ✗ The 4x memory savings are DESTROYED by {:.0f}x longer inference".format(1 + overhead_pct/100))
    print("  ✗ This makes the quantized model SLOWER than full precision!")
else:
    print("  ✓ Acceptable overhead")
print("")
print("For training (batch=32, seq=2048):")
batch, seq_len = 32, 2048
inference_flops = batch * seq_len * d_model * d_model
overhead_pct_training = (hadamard_flops / inference_flops) * 100
print(f"  COHT overhead: {overhead_pct_training:.2f}%")
if overhead_pct_training < 1:
    print("  ✓ Acceptable for training/batch inference")
else:
    print("  ✗ Still problematic")
print("")
print("="*80)
print("")
print("CONCLUSION:")
print("  COHT is fundamentally incompatible with token-by-token generation,")
print("  which is the PRIMARY use case for LLM quantization.")
print("  ")
print("  The proposal's claimed '~5% overhead' is completely wrong.")
print("  Actual overhead for generation: 12,800%")
print("="*80)
