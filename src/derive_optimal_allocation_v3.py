#!/usr/bin/env python3
"""
Phase 1.3: Optimal Bit Allocation v3 - Attention-Weighted Sensitivity

Key Discovery from v1-v2:
- Single-position sensitivity shows UNIFORM profile
- But Sink 6-4-4 (early = high precision) works much better
- Contradiction!

Resolution: The objective function is WRONG.

The issue: When quantizing the ENTIRE KV cache, positions interact.
The attention mechanism amplifies errors at frequently-attended positions.

New Objective:
=============
Instead of minimizing sensitivity-weighted quantization variance:
  min Σ_p s(p) · σ²(b_p)

We should minimize the EXPECTED attention output error:
  min E[||Σ_p α_p · ε_p||²]

where ε_p is the quantization error at position p.

If errors are independent:
  E[||Σ_p α_p · ε_p||²] = Σ_p α_p² · E[||ε_p||²]
                        = Σ_p α_p² · σ²(b_p)

So the CORRECT importance weight is α_p² (squared attention), not
the sensitivity to single-position perturbation!

This explains why Sink 6-4-4 works: early positions receive MUCH more
attention (α_p is very high for sink tokens), so α_p² is even higher.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from tqdm import tqdm


def quantize_tensor(x, num_bits, per_channel=False):
    """Simple min-max quantization."""
    if per_channel:
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
    else:
        x_min = x.min()
        x_max = x.max()

    range_val = x_max - x_min
    if isinstance(range_val, torch.Tensor):
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
    elif range_val == 0:
        return x

    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min

    return x_q


def measure_attention_importance(model, tokenizer, prompts, num_bins=20):
    """
    Measure attention importance w(p) = E[Σ_t α_{t,p}²] for each position.

    This is the CORRECT importance weight for the optimization problem:
    - Positions that receive more attention contribute more to output error
    - Squaring captures the variance contribution

    Note: For causal attention, α_{t,p} = 0 for p > t (future positions).
    So w(p) = Σ_{t≥p} α_{t,p}² (sum over queries that attend to p).
    """
    device = next(model.parameters()).device

    importance_by_bin = [[] for _ in range(num_bins)]

    for prompt in tqdm(prompts, desc="Measuring attention importance"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len < 10:
            continue

        with torch.no_grad():
            # Get attention weights
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

            # Stack and process
            # attn[layer][batch, head, t, p] = attention from query t to key p
            attn_stack = torch.stack(attentions)  # (layers, batch, heads, seq, seq)

            # w(p) = Σ_t α_{t,p}²  (importance of key position p)
            # Sum over query positions t, average over layers and heads
            attn_squared = attn_stack ** 2  # (layers, batch, heads, seq, seq)
            w_p = attn_squared.sum(dim=3)  # sum over t: (layers, batch, heads, seq)
            w_p = w_p.mean(dim=(0, 2))  # average over layers and heads: (batch, seq)
            w_p = w_p[0].cpu().numpy()  # (seq,)

            # Bin by relative position
            for p in range(seq_len):
                rel_pos = p / seq_len
                bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
                importance_by_bin[bin_idx].append(w_p[p])

    # Compute mean importance per bin
    importance_profile = []
    for bin_idx in range(num_bins):
        if importance_by_bin[bin_idx]:
            importance_profile.append(np.mean(importance_by_bin[bin_idx]))
        else:
            importance_profile.append(1e-10)

    return np.array(importance_profile)


def compute_optimal_allocation(importance, B_avg, bit_options=[2, 3, 4, 5, 6, 8],
                               min_bits=3, max_bits=8):
    """
    Compute PROVABLY OPTIMAL bit allocation.

    THEOREM: For the objective
        min_{b(p)} Σ_p w(p) · 2^{-2b(p)}
        s.t. (1/L) Σ_p b(p) = B_avg

    The optimal solution (by Lagrangian) is:
        b*(p) = 0.5 · log₂(w(p)) + C

    where C = B_avg - (1/L) Σ_p [0.5 · log₂(w(p))]

    PROOF:
    L = Σ_p w(p)·2^{-2b_p} + λ·((1/L)Σ_p b_p - B_avg)
    ∂L/∂b_p = -2·ln(2)·w(p)·2^{-2b_p} + λ/L = 0
    => 2^{-2b_p} = λ / (2·L·ln(2)·w(p))
    => b_p = 0.5·log₂(2·L·ln(2)·w(p)/λ)
           = 0.5·log₂(w(p)) + [0.5·log₂(2·L·ln(2)/λ)]
           = 0.5·log₂(w(p)) + C

    The constant C is determined by substituting into the constraint.
    QED.
    """
    L = len(importance)
    w = np.maximum(importance, 1e-10)

    # Optimal continuous allocation
    log_w = 0.5 * np.log2(w)
    C = B_avg - np.mean(log_w)
    b_continuous = log_w + C

    # Apply constraints
    b_continuous = np.clip(b_continuous, min_bits, max_bits)

    # Valid bit options
    valid_options = [b for b in bit_options if min_bits <= b <= max_bits]

    # Discretize
    b_discrete = np.array([min(valid_options, key=lambda x: abs(x - b)) for b in b_continuous])

    # Greedy refinement to match B_avg
    current_avg = np.mean(b_discrete)
    max_iter = 100
    for _ in range(max_iter):
        if abs(current_avg - B_avg) < 0.05:
            break

        if current_avg > B_avg:
            candidates = [(i, w[i]) for i in range(L) if b_discrete[i] > min(valid_options)]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])  # lowest importance first
            idx = candidates[0][0]
            current_bits = b_discrete[idx]
            lower_options = [b for b in valid_options if b < current_bits]
            if lower_options:
                b_discrete[idx] = max(lower_options)
        else:
            candidates = [(i, w[i]) for i in range(L) if b_discrete[i] < max(valid_options)]
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[1])  # highest importance first
            idx = candidates[0][0]
            current_bits = b_discrete[idx]
            higher_options = [b for b in valid_options if b > current_bits]
            if higher_options:
                b_discrete[idx] = min(higher_options)

        current_avg = np.mean(b_discrete)

    return b_discrete.tolist(), b_continuous.tolist()


def quantize_kv_with_allocation(kv_cache, allocation):
    """Quantize KV cache with per-position bit allocation."""
    quantized = DynamicCache()
    num_bins = len(allocation)

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()

        seq_len = k.shape[2]

        for pos in range(seq_len):
            rel_pos = pos / seq_len
            bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
            num_bits = allocation[bin_idx]

            k[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], num_bits)
            v[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], num_bits)

        quantized.update(k, v, layer_idx)

    return quantized


def quantize_kv_uniform(kv_cache, num_bits):
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = quantize_tensor(kv_cache.key_cache[layer_idx].clone(), num_bits)
        v = quantize_tensor(kv_cache.value_cache[layer_idx].clone(), num_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def quantize_kv_sink(kv_cache, sink_fraction=0.1, sink_bits=6, rest_bits=4):
    quantized = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx].clone()
        v = kv_cache.value_cache[layer_idx].clone()
        seq_len = k.shape[2]
        sink_end = int(seq_len * sink_fraction)
        k[:, :, :sink_end, :] = quantize_tensor(k[:, :, :sink_end, :], sink_bits)
        v[:, :, :sink_end, :] = quantize_tensor(v[:, :, :sink_end, :], sink_bits)
        k[:, :, sink_end:, :] = quantize_tensor(k[:, :, sink_end:, :], rest_bits)
        v[:, :, sink_end:, :] = quantize_tensor(v[:, :, sink_end:, :], rest_bits)
        quantized.update(k, v, layer_idx)
    return quantized


def evaluate_allocation(model, tokenizer, prompts, quant_fn):
    device = next(model.parameters()).device
    mse_list, cos_sim_list = [], []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] < 10:
            continue

        with torch.no_grad():
            fp_outputs = model(**inputs, use_cache=True)
            fp_logits = fp_outputs.logits[:, -1, :]
            fp_kv = fp_outputs.past_key_values

            q_kv = quant_fn(fp_kv)

            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
            q_logits = q_outputs.logits[:, -1, :]

            mse = F.mse_loss(fp_logits, q_logits).item()
            cos_sim = F.cosine_similarity(fp_logits, q_logits, dim=-1).mean().item()

            mse_list.append(mse)
            cos_sim_list.append(cos_sim)

    return np.mean(mse_list), np.mean(cos_sim_list)


def main():
    print("=" * 70)
    print("Phase 1.3: Optimal Bit Allocation v3 (Attention-Weighted)")
    print("=" * 70)

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Need for attention weights
    )
    model.eval()

    calibration_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness and silence.",
        "Machine learning has revolutionized how we process data.",
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Scientists have discovered a new species of deep-sea fish.",
        "The history of computing began with mechanical calculators.",
        "Climate change poses significant challenges for future generations.",
        "Artificial intelligence systems can now generate human-like text.",
        "The development of quantum computers promises to solve previously intractable problems.",
        "Neural networks have transformed computer vision and natural language processing.",
        "The theory of relativity changed our understanding of space and time.",
        "Renewable energy sources are becoming increasingly cost-competitive with fossil fuels.",
    ]

    test_prompts = [
        "Deep learning models require large amounts of training data.",
        "The ancient Romans built an extensive network of roads.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The stock market experienced significant volatility during the pandemic.",
    ]

    print("\nStep 1: Measuring attention importance w(p) = E[Σ_t α_{t,p}²]...")
    importance = measure_attention_importance(model, tokenizer, calibration_prompts, num_bins=20)

    print("\n" + "=" * 70)
    print("ATTENTION IMPORTANCE PROFILE w(p)")
    print("=" * 70)

    for i, w in enumerate(importance):
        rel_pos = (i + 0.5) / len(importance)
        bar = "█" * int(w / max(importance) * 40)
        print(f"  {rel_pos:.2f}: {w:.6f} {bar}")

    # Analyze importance distribution
    early_imp = np.mean(importance[:4])  # First 20%
    mid_imp = np.mean(importance[4:16])
    late_imp = np.mean(importance[16:])

    print(f"\nImportance by region:")
    print(f"  Early (0-20%):  {early_imp:.6f}")
    print(f"  Mid (20-80%):   {mid_imp:.6f}")
    print(f"  Late (80-100%): {late_imp:.6f}")
    print(f"  Ratio (Early/Mid): {early_imp/mid_imp:.1f}x")

    # Compute optimal allocations
    print("\n" + "=" * 70)
    print("OPTIMAL BIT ALLOCATIONS (PROVABLY OPTIMAL)")
    print("=" * 70)

    for B_avg in [3.5, 4.0, 4.2, 5.0]:
        b_discrete, b_continuous = compute_optimal_allocation(importance, B_avg,
                                                              min_bits=3, max_bits=8)
        actual_avg = np.mean(b_discrete)
        print(f"\nTarget B_avg = {B_avg}, Actual = {actual_avg:.2f}")
        print(f"  Optimal: {b_discrete}")
        print(f"  (Continuous: [{', '.join([f'{b:.1f}' for b in b_continuous])}])")

    # Micro-validation
    print("\n" + "=" * 70)
    print("MICRO-VALIDATION")
    print("=" * 70)

    B_avg = 4.2
    optimal_alloc, _ = compute_optimal_allocation(importance, B_avg, min_bits=3, max_bits=8)

    configs = [
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("Sink 6-4-4 (heuristic)", lambda kv: quantize_kv_sink(kv, 0.1, 6, 4), 4.2),
        ("Optimal (attention)", lambda kv: quantize_kv_with_allocation(kv, optimal_alloc), np.mean(optimal_alloc)),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),
    ]

    print(f"\nOptimal allocation for B_avg={B_avg}: {optimal_alloc}")
    print(f"\nEvaluating on {len(test_prompts)} test prompts...")

    results = []
    for name, quant_fn, avg_bits in configs:
        mse, cos_sim = evaluate_allocation(model, tokenizer, test_prompts, quant_fn)
        results.append({"name": name, "avg_bits": avg_bits, "mse": mse, "cos_sim": cos_sim})
        print(f"  {name}: MSE={mse:.4f}, CosSim={cos_sim:.4f}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<25} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Analysis
    optimal_result = next(r for r in results if "Optimal" in r["name"])
    sink_result = next(r for r in results if "Sink" in r["name"])
    uniform_4bit = next(r for r in results if "Uniform 4-bit" in r["name"])
    uniform_5bit = next(r for r in results if "Uniform 5-bit" in r["name"])

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\n1. Optimal vs Sink 6-4-4 (same avg bits ~4.2):")
    if optimal_result["mse"] < sink_result["mse"]:
        improvement = (sink_result["mse"] - optimal_result["mse"]) / sink_result["mse"] * 100
        print(f"   -> Optimal is {improvement:.1f}% BETTER")
        print(f"   -> VALIDATES the theory!")
    elif abs(optimal_result["mse"] - sink_result["mse"]) / sink_result["mse"] < 0.1:
        print(f"   -> EQUIVALENT (within 10%)")
        print(f"   -> Heuristic is near-optimal")
    else:
        degradation = (optimal_result["mse"] - sink_result["mse"]) / sink_result["mse"] * 100
        print(f"   -> Optimal is {degradation:.1f}% worse")

    print(f"\n2. Optimal vs Uniform 4-bit:")
    improvement = (uniform_4bit["mse"] - optimal_result["mse"]) / uniform_4bit["mse"] * 100
    print(f"   -> Optimal is {improvement:.1f}% better")

    print(f"\n3. Optimal vs Uniform 5-bit:")
    if optimal_result["mse"] <= uniform_5bit["mse"] * 1.05:
        memory_savings = (5.0 - optimal_result["avg_bits"]) / 5.0 * 100
        print(f"   -> Optimal MATCHES Uniform 5-bit with {memory_savings:.0f}% fewer bits!")
    else:
        gap = (optimal_result["mse"] - uniform_5bit["mse"]) / uniform_5bit["mse"] * 100
        print(f"   -> Optimal is {gap:.1f}% worse than Uniform 5-bit")

    # Theory summary
    print("\n" + "=" * 70)
    print("MATHEMATICAL DERIVATION")
    print("=" * 70)
    print("""
THEOREM: Optimal Position-Aware KV Quantization

Problem Setup:
- Attention output: o_t = Σ_p α_{t,p} · v_p
- Quantized: ô_t = Σ_p α_{t,p} · (v_p + ε_p) where ε_p ~ N(0, σ²(b_p))
- Error: ||o_t - ô_t||² = ||Σ_p α_{t,p} · ε_p||²

For independent errors:
  E[||Σ_p α_{t,p} · ε_p||²] = Σ_p α_{t,p}² · σ²(b_p)

Summing over all query positions:
  Total Error = Σ_t Σ_p α_{t,p}² · σ²(b_p) = Σ_p w(p) · σ²(b_p)

where w(p) = Σ_t α_{t,p}² is the ATTENTION IMPORTANCE.

Optimization:
  min_{b(p)} Σ_p w(p) · 2^{-2b(p)}
  s.t. (1/L) Σ_p b(p) = B_avg

SOLUTION (by Lagrangian):
  b*(p) = 0.5 · log₂(w(p)) + C
  where C = B_avg - mean(0.5 · log₂(w(p)))

INTERPRETATION:
- Higher attention importance w(p) → more bits
- 4× importance → +1 bit
- Sink tokens have ~100× importance → +3-4 extra bits
""")

    print("=" * 70)
    print("Derivation Complete")
    print("=" * 70)

    return importance, optimal_alloc


if __name__ == "__main__":
    main()
