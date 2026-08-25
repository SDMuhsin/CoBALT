"""MATH adapter (Hendrycks competition math, greedy generation).

Answer extraction (\\boxed{} with nested braces, last-number fallback),
normalization, matching, generation, and the by-level / by-type breakdowns are
ported VERBATIM from /workspace/CRB/eval_math.py. Return keys match the spec's
CSV schema ({"accuracy","correct","total","by_level","by_type"}); accuracies are
0-1 fractions.
"""

import re

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache, subsample


def extract_boxed_answer(text):
    """Extract the last \\boxed{...} answer from text. Handles nested braces. (Verbatim CRB.)"""
    # Find the last occurrence of \boxed{
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None

    # Extract content handling nested braces
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i-1].strip()
    return None


def extract_last_number(text):
    """Extract the last number from text as a fallback. (Verbatim CRB.)"""
    # Find all numbers (including decimals, negatives, fractions)
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer):
    """Normalize a math answer for comparison. (Verbatim CRB.)"""
    if answer is None:
        return None

    answer = answer.strip()

    # Remove surrounding $ signs
    answer = answer.strip('$')

    # Remove trailing period
    if answer.endswith('.'):
        answer = answer[:-1]

    # Remove \text{} wrappers
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

    # Remove spaces
    answer = answer.replace(' ', '')

    # Try fraction conversion
    frac_match = re.match(r'^(-?\d+)/(\d+)$', answer)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            answer = str(num / den)

    # Try \\frac{a}{b} conversion
    frac_match = re.match(r'^\\frac\{(-?\d+)\}\{(\d+)\}$', answer)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            answer = str(num / den)

    # Try float conversion for numeric comparison
    try:
        val = float(answer)
        # Round to avoid floating point issues
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}"
    except (ValueError, OverflowError):
        return answer.lower()


def answers_match(predicted, ground_truth):
    """Check if predicted and ground truth answers match. (Verbatim CRB.)"""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm is None or gt_norm is None:
        return False

    # Exact string match after normalization
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison with tolerance
    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        return abs(pred_val - gt_val) < 1e-4
    except (ValueError, OverflowError):
        return False


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate MATH (greedy generation, boxed-answer matching).

    Returns {"accuracy", "correct", "total", "by_level", "by_type"} with
    accuracies as 0-1 fractions.
    """
    if verbose:
        print("Evaluating on MATH (greedy generation) ...")

    ensure_seqlen(model, seqlen)

    # Load MATH dataset (all subjects combined)
    math_configs = ['algebra', 'counting_and_probability', 'geometry',
                    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_examples = []
    for cfg in math_configs:
        split_data = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        for ex in split_data:
            ex['type'] = cfg
            all_examples.append(ex)
    dataset = subsample(all_examples, limit)
    if verbose:
        print(f"  MATH test set: {len(dataset)} problems across {len(math_configs)} categories")

    # Enable cache for efficient generation
    use_cache = set_use_cache(model, True)
    model.to(device)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    correct = 0
    total = 0
    by_level = {}
    by_type = {}

    try:
        for i, example in enumerate(dataset):
            problem = example['problem']
            solution = example['solution']
            level = example.get('level', 'Unknown')
            prob_type = example.get('type', 'Unknown')

            # Extract ground truth answer from solution
            gt_answer = extract_boxed_answer(solution)
            if gt_answer is None:
                gt_answer = extract_last_number(solution)
            if gt_answer is None:
                continue

            # Format prompt
            prompt = f"Problem: {problem}\nSolution:"
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            # Truncate prompt if too long (leave room for generation)
            max_prompt_len = model.seqlen - 512
            if max_prompt_len < 1:
                max_prompt_len = model.seqlen // 2
            if input_ids.shape[1] > max_prompt_len:
                input_ids = input_ids[:, -max_prompt_len:]

            # Generate
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode only the generated tokens
            generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                         skip_special_tokens=True)

            # Extract predicted answer
            pred_answer = extract_boxed_answer(generated)
            if pred_answer is None:
                pred_answer = extract_last_number(generated)

            is_correct = answers_match(pred_answer, gt_answer)

            if is_correct:
                correct += 1
            total += 1

            # Track by level and type
            for grouping, key in [(by_level, level), (by_type, prob_type)]:
                if key not in grouping:
                    grouping[key] = {"correct": 0, "total": 0}
                grouping[key]["total"] += 1
                if is_correct:
                    grouping[key]["correct"] += 1

            if verbose and (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")

        accuracy = correct / total if total > 0 else 0.0
    finally:
        model.config.use_cache = use_cache

    # Compute breakdowns
    level_accuracies = {}
    for key in sorted(by_level.keys()):
        d = by_level[key]
        level_accuracies[key] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    type_accuracies = {}
    for key in sorted(by_type.keys()):
        d = by_type[key]
        type_accuracies[key] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    if verbose:
        print(f"MATH Overall accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_level": {k: {"accuracy": level_accuracies[k], **v} for k, v in by_level.items()},
        "by_type": {k: {"accuracy": type_accuracies[k], **v} for k, v in by_type.items()},
    }
