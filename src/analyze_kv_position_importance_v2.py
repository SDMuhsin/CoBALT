#!/usr/bin/env python3
"""
Phase 1.2 Analysis: KV Cache Position Importance (V2)

More robust analysis focusing on:
1. Attention weight distribution across positions
2. Output sensitivity to position-specific quantization
3. Long-context position importance patterns
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict


def analyze_attention_weight_distribution(model, tokenizer, prompts):
    """
    Analyze how attention weights are distributed across positions.
    Key question: Do early positions receive more attention?
    """
    device = next(model.parameters()).device

    all_attn_by_relative_pos = defaultdict(list)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (layer_attention) for each layer
        # Each layer_attention: (batch, num_heads, seq_len, seq_len)

        for layer_idx, layer_attn in enumerate(outputs.attentions):
            attn = layer_attn[0]  # (num_heads, seq_len, seq_len)

            # For each query position, look at attention to keys
            # Focus on the last query position (most relevant for generation)
            last_query_attn = attn[:, -1, :].mean(dim=0)  # Average over heads: (seq_len,)

            for pos in range(seq_len):
                rel_pos = pos / seq_len
                all_attn_by_relative_pos[round(rel_pos, 2)].append(last_query_attn[pos].item())

    return all_attn_by_relative_pos


def analyze_output_sensitivity_to_kv_quantization(model, tokenizer, prompt, num_bits=4):
    """
    Quantize different position ranges of KV cache and measure output difference.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    # Get FP reference
    with torch.no_grad():
        fp_outputs = model(**inputs, use_cache=True)
        fp_logits = fp_outputs.logits[:, -1, :]
        fp_kv = fp_outputs.past_key_values

    results = []

    # Test quantizing each quartile
    quartiles = [
        (0, seq_len // 4, "Q1 (0-25%)"),
        (seq_len // 4, seq_len // 2, "Q2 (25-50%)"),
        (seq_len // 2, 3 * seq_len // 4, "Q3 (50-75%)"),
        (3 * seq_len // 4, seq_len, "Q4 (75-100%)"),
    ]

    for start, end, name in quartiles:
        # Quantize this range only
        q_kv = quantize_kv_range(fp_kv, start, end, num_bits)

        # Get output with quantized KV
        with torch.no_grad():
            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
            q_logits = q_outputs.logits[:, -1, :]

        # Compute metrics
        mse = F.mse_loss(fp_logits, q_logits).item()

        # Cosine similarity
        cos_sim = F.cosine_similarity(fp_logits, q_logits, dim=-1).mean().item()

        # Top-k agreement
        fp_top10 = fp_logits.topk(10, dim=-1).indices
        q_top10 = q_logits.topk(10, dim=-1).indices
        top10_overlap = len(set(fp_top10[0].tolist()) & set(q_top10[0].tolist())) / 10

        results.append({
            "range": name,
            "start": start,
            "end": end,
            "mse": mse,
            "cosine_similarity": cos_sim,
            "top10_overlap": top10_overlap,
        })

    return results


def quantize_kv_range(kv_cache, start, end, num_bits):
    """Quantize only a specific position range of the KV cache."""
    from transformers.cache_utils import DynamicCache

    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]

        k_q = k.clone()
        v_q = v.clone()

        # Quantize only positions [start:end]
        k_q[:, :, start:end, :] = quantize_tensor(k[:, :, start:end, :], num_bits)
        v_q[:, :, start:end, :] = quantize_tensor(v[:, :, start:end, :], num_bits)

        quantized.update(k_q, v_q, layer_idx)

    return quantized


def quantize_tensor(x, num_bits):
    """Simple min-max quantization."""
    if x.numel() == 0:
        return x

    x_min = x.min()
    x_max = x.max()

    if x_max == x_min:
        return x

    scale = (x_max - x_min) / (2**num_bits - 1)
    x_q = torch.round((x - x_min) / scale) * scale + x_min

    return x_q


def analyze_long_context_attention_patterns(model, tokenizer, context_lengths=[64, 128, 256, 512]):
    """
    Analyze attention patterns at different context lengths.
    Focus on: Do sink tokens (first few positions) receive disproportionate attention?
    """
    device = next(model.parameters()).device

    results = []

    base_text = "This is a test sentence for analyzing attention patterns in language models. " * 50

    for target_len in context_lengths:
        # Tokenize and truncate to target length
        tokens = tokenizer.encode(base_text)[:target_len]
        inputs = {"input_ids": torch.tensor([tokens], device=device)}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Analyze attention from last position to all positions
        # Average across all layers and heads
        all_attn = torch.stack([layer_attn[0] for layer_attn in outputs.attentions])
        # all_attn: (num_layers, num_heads, seq_len, seq_len)

        last_query_attn = all_attn[:, :, -1, :].mean(dim=(0, 1))  # (seq_len,)

        # Compute statistics
        first_3_attn = last_query_attn[:3].sum().item()
        first_10pct_attn = last_query_attn[:max(1, len(last_query_attn)//10)].sum().item()
        last_10pct_attn = last_query_attn[-max(1, len(last_query_attn)//10):].sum().item()

        results.append({
            "context_length": target_len,
            "first_3_tokens_attn": first_3_attn,
            "first_10pct_attn": first_10pct_attn,
            "last_10pct_attn": last_10pct_attn,
            "attn_distribution": last_query_attn.cpu().numpy()
        })

    return results


def analyze_full_vs_partial_quantization(model, tokenizer, prompt, num_bits=4):
    """
    Compare: quantizing all positions vs only early/late positions.
    If early positions are more important, quantizing them should hurt more.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    # Get FP reference
    with torch.no_grad():
        fp_outputs = model(**inputs, use_cache=True)
        fp_logits = fp_outputs.logits[:, -1, :]
        fp_kv = fp_outputs.past_key_values

    configs = [
        ("Full quantization", 0, seq_len),
        ("First 25% only", 0, seq_len // 4),
        ("Last 25% only", 3 * seq_len // 4, seq_len),
        ("First 50% only", 0, seq_len // 2),
        ("Last 50% only", seq_len // 2, seq_len),
    ]

    results = []

    for name, start, end in configs:
        q_kv = quantize_kv_range(fp_kv, start, end, num_bits)

        with torch.no_grad():
            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            q_outputs = model(input_ids=dummy_input, past_key_values=q_kv, use_cache=False)
            q_logits = q_outputs.logits[:, -1, :]

        mse = F.mse_loss(fp_logits, q_logits).item()
        cos_sim = F.cosine_similarity(fp_logits, q_logits, dim=-1).mean().item()

        # Normalize by number of positions quantized
        positions_quantized = end - start
        mse_per_position = mse / positions_quantized if positions_quantized > 0 else 0

        results.append({
            "config": name,
            "positions_quantized": positions_quantized,
            "total_mse": mse,
            "mse_per_position": mse_per_position,
            "cosine_similarity": cos_sim,
        })

    return results


def main():
    print("=" * 70)
    print("Phase 1.2: KV Cache Position Importance Analysis (V2)")
    print("=" * 70)

    # Load model
    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()

    # Long test prompt
    test_prompt = """The history of artificial intelligence began in the mid-20th century when
    Alan Turing proposed the concept of machine intelligence. In 1950, Turing published his
    seminal paper introducing the Turing Test. The field was officially born at the Dartmouth
    Conference in 1956, where John McCarthy coined the term "artificial intelligence." Early
    AI research focused on symbolic reasoning and problem-solving. The 1980s saw the rise of
    expert systems, while the 1990s brought machine learning advances. The deep learning
    revolution began in 2012 with AlexNet, and large language models emerged in the 2020s."""

    print(f"\nTest prompt length: {len(tokenizer.encode(test_prompt))} tokens")

    # Analysis 1: Long-context attention patterns
    print("\n" + "=" * 70)
    print("Analysis 1: Attention Distribution Across Context Lengths")
    print("=" * 70)

    attn_results = analyze_long_context_attention_patterns(model, tokenizer)

    print(f"\n{'Context Len':>12} | {'First 3 Tokens':>15} | {'First 10%':>12} | {'Last 10%':>12}")
    print("-" * 60)
    for r in attn_results:
        print(f"{r['context_length']:>12} | {r['first_3_tokens_attn']:>15.4f} | "
              f"{r['first_10pct_attn']:>12.4f} | {r['last_10pct_attn']:>12.4f}")

    # Analysis 2: Output sensitivity to position-specific quantization
    print("\n" + "=" * 70)
    print("Analysis 2: Output Sensitivity to Position-Specific 4-bit Quantization")
    print("=" * 70)

    sensitivity_results = analyze_output_sensitivity_to_kv_quantization(
        model, tokenizer, test_prompt, num_bits=4
    )

    print(f"\n{'Range':>15} | {'MSE':>12} | {'Cos Sim':>10} | {'Top-10 Overlap':>15}")
    print("-" * 60)
    for r in sensitivity_results:
        print(f"{r['range']:>15} | {r['mse']:>12.6f} | {r['cosine_similarity']:>10.6f} | "
              f"{r['top10_overlap']:>15.2%}")

    # Analysis 3: Full vs partial quantization comparison
    print("\n" + "=" * 70)
    print("Analysis 3: Full vs Partial Quantization Impact")
    print("=" * 70)

    partial_results = analyze_full_vs_partial_quantization(
        model, tokenizer, test_prompt, num_bits=4
    )

    print(f"\n{'Config':>20} | {'Positions':>10} | {'Total MSE':>12} | {'MSE/Position':>14}")
    print("-" * 65)
    for r in partial_results:
        print(f"{r['config']:>20} | {r['positions_quantized']:>10} | "
              f"{r['total_mse']:>12.6f} | {r['mse_per_position']:>14.8f}")

    # KEY FINDINGS
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Check sink token effect
    if attn_results:
        r = attn_results[-1]  # Longest context
        sink_ratio = r['first_3_tokens_attn'] / (r['last_10pct_attn'] + 1e-10)
        print(f"\n1. Sink Token Effect (at {r['context_length']} tokens):")
        print(f"   First 3 tokens receive {sink_ratio:.2f}x more attention than last 10%")
        if sink_ratio > 3:
            print("   -> STRONG sink token effect detected (supports Gap 7)")
        elif sink_ratio > 1.5:
            print("   -> MODERATE sink token effect detected")
        else:
            print("   -> WEAK sink token effect (Gap 7 may have limited benefit)")

    # Check position sensitivity
    if sensitivity_results:
        early_mse = sensitivity_results[0]['mse']  # Q1
        late_mse = sensitivity_results[-1]['mse']  # Q4
        ratio = early_mse / late_mse if late_mse > 0 else float('inf')
        print(f"\n2. Position Sensitivity (4-bit quantization):")
        print(f"   Early (Q1) MSE / Late (Q4) MSE = {ratio:.2f}x")
        if ratio > 2:
            print("   -> Early positions ARE more sensitive (supports Gap 7)")
        elif ratio < 0.5:
            print("   -> Early positions are LESS sensitive (contradicts Gap 7)")
        else:
            print("   -> Position sensitivity is SIMILAR (Gap 7 benefit may be limited)")

    # Check MSE per position
    if partial_results:
        first_25_mse_per_pos = partial_results[1]['mse_per_position']  # First 25%
        last_25_mse_per_pos = partial_results[2]['mse_per_position']   # Last 25%
        per_pos_ratio = first_25_mse_per_pos / last_25_mse_per_pos if last_25_mse_per_pos > 0 else float('inf')
        print(f"\n3. MSE-per-Position Analysis:")
        print(f"   First 25% MSE/pos / Last 25% MSE/pos = {per_pos_ratio:.2f}x")
        if per_pos_ratio > 1.5:
            print("   -> Early positions cause MORE error per position (supports Gap 7)")
        elif per_pos_ratio < 0.67:
            print("   -> Early positions cause LESS error per position (contradicts Gap 7)")
        else:
            print("   -> Error contribution is UNIFORM across positions")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
