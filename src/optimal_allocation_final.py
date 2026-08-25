#!/usr/bin/env python3
"""
Final Optimal Bit Allocation Test (Corrected Methodology)

Key fix: Use proper KV cache evaluation by:
1. Encoding prompt to get KV cache
2. Quantizing the prefix KV (all but last position)
3. Running forward with last token + quantized prefix KV
4. Comparing logits vs FP reference

This gives deterministic, monotonic error measurements.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from tqdm import tqdm


def quantize_tensor(x, num_bits):
    """Simple min-max quantization."""
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min
    if range_val == 0:
        return x
    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    return x_q


def evaluate_kv_quantization(model, tokenizer, prompts, quant_fn):
    """
    Proper evaluation of KV cache quantization.

    For each prompt:
    1. Encode full sequence to get KV cache
    2. Quantize prefix KV (all but last position)
    3. Run forward with last token + quantized prefix KV
    4. Compare logits vs FP reference

    Returns: mean MSE and cosine similarity
    """
    device = next(model.parameters()).device

    mse_list = []
    cos_list = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len < 10:
            continue

        with torch.no_grad():
            # Get FP KV cache
            torch.manual_seed(42)
            outputs = model(**inputs, use_cache=True)
            fp_kv = outputs.past_key_values

            # Create prefix KV (all but last position)
            prefix_kv_fp = DynamicCache()
            prefix_kv_q = DynamicCache()

            for layer_idx in range(len(fp_kv)):
                k = fp_kv.key_cache[layer_idx][:, :, :-1, :]
                v = fp_kv.value_cache[layer_idx][:, :, :-1, :]

                prefix_kv_fp.update(k.clone(), v.clone(), layer_idx)

                # Apply quantization function
                k_q, v_q = quant_fn(k.clone(), v.clone())
                prefix_kv_q.update(k_q, v_q, layer_idx)

            # Run with FP prefix
            last_token = inputs.input_ids[:, -1:]
            torch.manual_seed(42)
            out_fp = model(input_ids=last_token, past_key_values=prefix_kv_fp, use_cache=False)
            logits_fp = out_fp.logits[:, -1, :]

            # Run with quantized prefix
            torch.manual_seed(42)
            out_q = model(input_ids=last_token, past_key_values=prefix_kv_q, use_cache=False)
            logits_q = out_q.logits[:, -1, :]

            # Metrics
            mse = F.mse_loss(logits_fp, logits_q).item()
            cos = F.cosine_similarity(logits_fp, logits_q, dim=-1).item()

            mse_list.append(mse)
            cos_list.append(cos)

    return np.mean(mse_list), np.mean(cos_list)


def uniform_quant(num_bits):
    """Uniform quantization function."""
    def quant_fn(k, v):
        return quantize_tensor(k, num_bits), quantize_tensor(v, num_bits)
    return quant_fn


def zone_quant(zone_fraction, zone_bits, rest_bits):
    """Zone-based quantization (Sink 6-4-4 style)."""
    def quant_fn(k, v):
        seq_len = k.shape[2]
        zone_end = max(1, int(seq_len * zone_fraction))

        k_out = k.clone()
        v_out = v.clone()

        # Quantize zone
        k_out[:, :, :zone_end, :] = quantize_tensor(k[:, :, :zone_end, :], zone_bits)
        v_out[:, :, :zone_end, :] = quantize_tensor(v[:, :, :zone_end, :], zone_bits)

        # Quantize rest
        if zone_end < seq_len:
            k_out[:, :, zone_end:, :] = quantize_tensor(k[:, :, zone_end:, :], rest_bits)
            v_out[:, :, zone_end:, :] = quantize_tensor(v[:, :, zone_end:, :], rest_bits)

        return k_out, v_out
    return quant_fn


def allocation_quant(allocation):
    """Per-bin allocation quantization."""
    num_bins = len(allocation)

    def quant_fn(k, v):
        seq_len = k.shape[2]
        k_out = k.clone()
        v_out = v.clone()

        for pos in range(seq_len):
            rel_pos = pos / seq_len
            bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
            num_bits = allocation[bin_idx]

            k_out[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], num_bits)
            v_out[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], num_bits)

        return k_out, v_out
    return quant_fn


def measure_attention_importance(model, tokenizer, prompts, num_bins=20):
    """Measure attention importance w(p) = E[Σ_t α_{t,p}²]."""
    device = next(model.parameters()).device

    importance_by_bin = [[] for _ in range(num_bins)]

    for prompt in tqdm(prompts, desc="Measuring attention importance"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len < 10:
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions

            attn_stack = torch.stack(attentions)
            attn_squared = attn_stack ** 2
            w_p = attn_squared.sum(dim=3)  # sum over query positions
            w_p = w_p.mean(dim=(0, 2))  # avg over layers and heads
            w_p = w_p[0].cpu().numpy()

            for p in range(seq_len):
                rel_pos = p / seq_len
                bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
                importance_by_bin[bin_idx].append(w_p[p])

    importance_profile = []
    for bin_idx in range(num_bins):
        if importance_by_bin[bin_idx]:
            importance_profile.append(np.mean(importance_by_bin[bin_idx]))
        else:
            importance_profile.append(1e-10)

    return np.array(importance_profile)


def compute_optimal_allocation(importance, B_avg, bit_options=[3, 4, 5, 6, 8], min_bits=3, max_bits=8):
    """
    Compute provably optimal bit allocation.

    b*(p) = 0.5 * log2(w(p)) + C
    where C = B_avg - mean(0.5 * log2(w(p)))
    """
    L = len(importance)
    w = np.maximum(importance, 1e-10)

    log_w = 0.5 * np.log2(w)
    C = B_avg - np.mean(log_w)
    b_continuous = log_w + C
    b_continuous = np.clip(b_continuous, min_bits, max_bits)

    valid_options = [b for b in bit_options if min_bits <= b <= max_bits]
    b_discrete = np.array([min(valid_options, key=lambda x: abs(x - b)) for b in b_continuous])

    # Greedy refinement
    current_avg = np.mean(b_discrete)
    for _ in range(100):
        if abs(current_avg - B_avg) < 0.05:
            break

        if current_avg > B_avg:
            candidates = [(i, w[i]) for i in range(L) if b_discrete[i] > min(valid_options)]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])
            idx = candidates[0][0]
            lower_options = [b for b in valid_options if b < b_discrete[idx]]
            if lower_options:
                b_discrete[idx] = max(lower_options)
        else:
            candidates = [(i, w[i]) for i in range(L) if b_discrete[i] < max(valid_options)]
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[1])
            idx = candidates[0][0]
            higher_options = [b for b in valid_options if b > b_discrete[idx]]
            if higher_options:
                b_discrete[idx] = min(higher_options)

        current_avg = np.mean(b_discrete)

    return b_discrete.tolist(), b_continuous.tolist()


def main():
    print("=" * 70)
    print("Final Optimal Bit Allocation Test (Corrected Methodology)")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Need for output_attentions
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
    ]

    test_prompts = [
        "Deep learning models require large amounts of training data and compute.",
        "The ancient Romans built roads that still exist today in many parts of Europe.",
        "Photosynthesis is the process by which green plants make food using sunlight.",
        "The stock market experienced significant volatility during the pandemic period.",
        "Quantum computers promise to revolutionize cryptography and drug discovery.",
        "Natural language processing has advanced rapidly with transformer architectures.",
    ]

    # Step 1: Verify error is monotonic
    print("\n" + "=" * 70)
    print("STEP 1: Verify Monotonic Error (Uniform Quantization)")
    print("=" * 70)

    for bits in [3, 4, 5, 6, 8]:
        mse, cos = evaluate_kv_quantization(model, tokenizer, test_prompts, uniform_quant(bits))
        print(f"  Uniform {bits}-bit: MSE = {mse:.4f}, CosSim = {cos:.4f}")

    # Step 2: Measure attention importance
    print("\n" + "=" * 70)
    print("STEP 2: Measure Attention Importance")
    print("=" * 70)

    importance = measure_attention_importance(model, tokenizer, calibration_prompts, num_bins=20)

    for i, w in enumerate(importance):
        rel_pos = (i + 0.5) / len(importance)
        bar = "█" * int(w / max(importance) * 40)
        print(f"  {rel_pos:.2f}: {w:.6f} {bar}")

    early_imp = np.mean(importance[:4])
    rest_imp = np.mean(importance[4:])
    print(f"\n  Early (0-20%): {early_imp:.6f}")
    print(f"  Rest (20-100%): {rest_imp:.6f}")
    print(f"  Ratio: {early_imp/rest_imp:.1f}x")

    # Step 3: Compute optimal allocations
    print("\n" + "=" * 70)
    print("STEP 3: Compute Optimal Allocations")
    print("=" * 70)

    for B_avg in [4.0, 4.2, 5.0]:
        b_discrete, _ = compute_optimal_allocation(importance, B_avg)
        actual_avg = np.mean(b_discrete)
        print(f"\n  Target B_avg = {B_avg}, Actual = {actual_avg:.2f}")
        print(f"  Allocation: {b_discrete}")

    # Step 4: Compare configurations
    print("\n" + "=" * 70)
    print("STEP 4: Compare Configurations")
    print("=" * 70)

    B_avg = 4.2
    optimal_alloc, _ = compute_optimal_allocation(importance, B_avg)

    configs = [
        ("Uniform 4-bit", uniform_quant(4), 4.0),
        ("Sink 6-4-4", zone_quant(0.1, 6, 4), 4.2),
        ("Sink 8-4-4", zone_quant(0.1, 8, 4), 4.4),
        ("Optimal", allocation_quant(optimal_alloc), np.mean(optimal_alloc)),
        ("Uniform 5-bit", uniform_quant(5), 5.0),
    ]

    print(f"\n  Optimal allocation for B_avg={B_avg}: {optimal_alloc}")

    results = []
    for name, quant_fn, avg_bits in configs:
        mse, cos = evaluate_kv_quantization(model, tokenizer, test_prompts, quant_fn)
        results.append({"name": name, "avg_bits": avg_bits, "mse": mse, "cos_sim": cos})
        print(f"\n  {name} (avg={avg_bits:.2f} bits):")
        print(f"    MSE = {mse:.4f}, CosSim = {cos:.4f}")

    # Results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<20} | {'Avg Bits':>10} | {'MSE':>12} | {'Cos Sim':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} | {r['avg_bits']:>10.2f} | {r['mse']:>12.4f} | {r['cos_sim']:>10.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    optimal = next(r for r in results if r["name"] == "Optimal")
    sink = next(r for r in results if r["name"] == "Sink 6-4-4")
    uniform4 = next(r for r in results if r["name"] == "Uniform 4-bit")
    uniform5 = next(r for r in results if r["name"] == "Uniform 5-bit")

    print(f"\n1. Optimal vs Sink 6-4-4:")
    if optimal["mse"] < sink["mse"]:
        improvement = (sink["mse"] - optimal["mse"]) / sink["mse"] * 100
        print(f"   Optimal is {improvement:.1f}% BETTER (lower MSE)")
    elif abs(optimal["mse"] - sink["mse"]) / sink["mse"] < 0.1:
        print(f"   Results are EQUIVALENT (within 10%)")
    else:
        degradation = (optimal["mse"] - sink["mse"]) / sink["mse"] * 100
        print(f"   Optimal is {degradation:.1f}% WORSE")

    print(f"\n2. Optimal vs Uniform 4-bit:")
    improvement = (uniform4["mse"] - optimal["mse"]) / uniform4["mse"] * 100
    print(f"   Optimal is {improvement:.1f}% better")

    print(f"\n3. Optimal vs Uniform 5-bit:")
    if optimal["mse"] <= uniform5["mse"] * 1.05:
        memory_savings = (5.0 - optimal["avg_bits"]) / 5.0 * 100
        print(f"   Optimal MATCHES Uniform 5-bit with {memory_savings:.0f}% fewer bits!")
    else:
        gap = (optimal["mse"] - uniform5["mse"]) / uniform5["mse"] * 100
        print(f"   Optimal is {gap:.1f}% worse than Uniform 5-bit")

    print("\n" + "=" * 70)
    print("MATHEMATICAL DERIVATION")
    print("=" * 70)
    print("""
THEOREM: Optimal Position-Aware KV Quantization

Objective: min_{b(p)} Σ_p w(p) · 2^{-2b(p)}
Constraint: (1/L) Σ_p b(p) = B_avg

where w(p) = Σ_t α_{t,p}² (attention importance)

Solution: b*(p) = 0.5 · log₂(w(p)) + C
where C = B_avg - mean(0.5 · log₂(w(p)))

PROOF: Standard Lagrangian optimization (see RESEARCH_METHODOLOGY.md)
""")

    print("=" * 70)
    print("Test Complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
