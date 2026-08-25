"""ARC adapter (ARC-Easy + ARC-Challenge).

Scoring math (prompt format, length-normalized completion log-likelihood,
answerKey letter/number mapping, argmax) is ported VERBATIM from
/workspace/CRB/eval_arc.py. The CRB function loops BOTH configs and returns
{"ARC-Easy": {...}, "ARC-Challenge": {...}}; this single adapter preserves that
shape. The suite splits the result into two CSV rows (arc_easy, arc_challenge).
"""

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache, subsample, compute_completion_ll


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate ARC-Easy and ARC-Challenge (0-shot, length-normalized LL).

    Returns {"ARC-Easy": {"accuracy","correct","total"},
             "ARC-Challenge": {"accuracy","correct","total"}} with 0-1 fractions.
    """
    if verbose:
        print("Evaluating on ARC (0-shot) ...")

    ensure_seqlen(model, seqlen)

    use_cache = set_use_cache(model, False)
    model.to(device)

    all_results = {}

    try:
        for split_name in ["ARC-Easy", "ARC-Challenge"]:
            if verbose:
                print(f"\n  Evaluating {split_name} ...")
            dataset = load_dataset("allenai/ai2_arc", split_name, split="test")
            dataset = subsample(dataset, limit)

            correct = 0
            total = 0

            for i, example in enumerate(dataset):
                question = example['question']
                choices_text = example['choices']['text']
                choices_labels = example['choices']['label']
                answer_key = example['answerKey']

                # Find ground truth index
                try:
                    gt_idx = choices_labels.index(answer_key)
                except ValueError:
                    # answerKey might be numeric (1,2,3,4) instead of letter (A,B,C,D)
                    letter_map = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
                    mapped = letter_map.get(answer_key, answer_key)
                    try:
                        gt_idx = choices_labels.index(mapped)
                    except ValueError:
                        continue

                # Format prompt
                prompt = f"Question: {question}\nAnswer:"
                prompt_ids = tokenizer(prompt, return_tensors='pt')
                prompt_len = prompt_ids.input_ids.shape[1]

                best_score = float('-inf')
                best_idx = 0

                for j, choice_text in enumerate(choices_text):
                    full_text = prompt + " " + choice_text
                    full_ids = tokenizer(full_text, return_tensors='pt').input_ids.to(device)

                    # Truncate if too long
                    if full_ids.shape[1] > model.seqlen:
                        overshoot = full_ids.shape[1] - model.seqlen
                        full_ids = full_ids[:, overshoot:]
                        adj_prompt_len = max(1, prompt_len - overshoot)
                    else:
                        adj_prompt_len = prompt_len

                    total_ll, n_tokens = compute_completion_ll(model, full_ids, adj_prompt_len)

                    # Length-normalize
                    score = total_ll / n_tokens if n_tokens > 0 else float('-inf')

                    if score > best_score:
                        best_score = score
                        best_idx = j

                if best_idx == gt_idx:
                    correct += 1
                total += 1

                if verbose and (i + 1) % 500 == 0:
                    print(f"    [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")

            accuracy = correct / total if total > 0 else 0.0
            if verbose:
                print(f"  {split_name} accuracy: {accuracy:.4f} ({correct}/{total})")
            all_results[split_name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
    finally:
        model.config.use_cache = use_cache

    return all_results
