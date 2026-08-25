"""HellaSwag adapter.

Scoring math (preprocess, length-normalized completion log-likelihood, argmax)
is ported VERBATIM from /workspace/CRB/eval_hellaswag.py. Only the I/O boundary
is adapted to the uniform run(...) contract: tokenizer passed in, seqlen set,
optional limit subsample, no GLOBAL_*.json side effects, no model.cpu().
"""

import re

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache, subsample, compute_completion_ll


def preprocess_hellaswag(text):
    """Clean up HellaSwag text artifacts. (Verbatim from CRB.)"""
    text = text.strip()
    text = re.sub(r'\[header\]\s*', '', text)
    text = re.sub(r'\[.*?\]\s*', '', text)
    text = text.strip()
    return text


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate HellaSwag (0-shot, length-normalized completion log-likelihood).

    Returns {"accuracy", "correct", "total"} with accuracy as a 0-1 fraction.
    """
    if verbose:
        print("Evaluating on HellaSwag (0-shot) ...")

    ensure_seqlen(model, seqlen)

    # Load HellaSwag — use validation split (test labels are hidden)
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    dataset = subsample(dataset, limit)

    use_cache = set_use_cache(model, False)
    model.to(device)

    correct = 0
    total = 0

    try:
        for i, example in enumerate(dataset):
            ctx = preprocess_hellaswag(example['ctx'])
            endings = [preprocess_hellaswag(e) for e in example['endings']]
            label = int(example['label'])

            # Tokenize context to find prompt length
            ctx_ids = tokenizer(ctx, return_tensors='pt')
            prompt_len = ctx_ids.input_ids.shape[1]

            best_score = float('-inf')
            best_idx = 0

            for j, ending in enumerate(endings):
                full_text = ctx + " " + ending
                full_ids = tokenizer(full_text, return_tensors='pt').input_ids.to(device)

                # Truncate if too long
                if full_ids.shape[1] > model.seqlen:
                    # Truncate from the left but keep at least some completion tokens
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

            if best_idx == label:
                correct += 1
            total += 1

            if verbose and (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")

        accuracy = correct / total if total > 0 else 0.0
    finally:
        model.config.use_cache = use_cache

    if verbose:
        print(f"HellaSwag Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
