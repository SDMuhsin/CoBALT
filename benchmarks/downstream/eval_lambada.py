"""LAMBADA adapter (last-word prediction accuracy).

Scoring math (context/target split, greedy argmax over all target tokens) is
ported VERBATIM from /workspace/CRB/eval_lambada.opt_eval_lambada. CRB returns
accuracy as 0-100; this adapter normalizes to a 0-1 fraction. Dataset id is the
parquet-friendly ``EleutherAI/lambada_openai`` (the bare ``lambada`` script
loader breaks on datasets 5).
"""

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache, subsample


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate LAMBADA last-word prediction accuracy.

    Returns {"accuracy", "correct", "total"} with accuracy as a 0-1 fraction
    (CRB returns 0-100; divided by 100 here).
    """
    if verbose:
        print("Evaluating on LAMBADA ...")

    ensure_seqlen(model, seqlen)

    dataset = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    dataset = subsample(dataset, limit)

    use_cache = set_use_cache(model, False)
    model.to(device)

    correct = 0
    total = 0

    try:
        for i, example in enumerate(dataset):
            text = example["text"]

            # Split into context and target (last word)
            last_space = text.rfind(" ")
            if last_space == -1:
                continue
            context = text[:last_space]

            # Tokenize full text and context to find boundary
            full_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)

            ctx_len = ctx_ids.shape[1]
            full_len = full_ids.shape[1]
            n_target = full_len - ctx_len

            if n_target <= 0:
                continue

            # Forward pass
            logits = model(full_ids).logits

            # logits[0, t] predicts token at position t+1
            target_tokens = full_ids[0, ctx_len:full_len]
            pred_tokens = logits[0, ctx_len - 1 : full_len - 1].argmax(dim=-1)

            if torch.all(pred_tokens == target_tokens):
                correct += 1
            total += 1

            if verbose and (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct}/{total} = {correct/total*100:.2f}%")

        accuracy_pct = correct / total * 100 if total > 0 else 0.0
    finally:
        model.config.use_cache = use_cache

    accuracy = accuracy_pct / 100.0

    if verbose:
        print(f"LAMBADA Accuracy: {accuracy_pct:.2f}% ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
