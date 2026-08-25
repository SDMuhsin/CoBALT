"""OpenBookQA adapter (0-shot, length-normalized completion log-likelihood).

Elementary-science 4-choice QA. Standard in the lm-eval commonsense suite. Same
contract as eval_arc: score each choice's length-normalized LL given the question
stem, argmax. (4-choice → 25% floor; may floor at extreme compression.) 0-1."""
import torch
from datasets import load_dataset
from .common import ensure_seqlen, set_use_cache, subsample, compute_completion_ll


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    if verbose:
        print("Evaluating on OpenBookQA (0-shot) ...")
    ensure_seqlen(model, seqlen)
    dataset = load_dataset("allenai/openbookqa", "main", split="validation")
    dataset = subsample(dataset, limit)
    set_use_cache(model, False)
    model.to(device)
    correct = total = 0
    for i, ex in enumerate(dataset):
        stem = ex["question_stem"].strip()
        texts = ex["choices"]["text"]; labels = ex["choices"]["label"]
        answer = ex["answerKey"]
        prompt_len = tokenizer(stem, return_tensors="pt").input_ids.shape[1]
        best, best_idx = float("-inf"), 0
        for j, ch in enumerate(texts):
            full_ids = tokenizer(stem + " " + ch.strip(), return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] > model.seqlen:
                over = full_ids.shape[1] - model.seqlen
                full_ids = full_ids[:, over:]; pl = max(1, prompt_len - over)
            else:
                pl = prompt_len
            ll, n = compute_completion_ll(model, full_ids, pl)
            score = ll / n if n > 0 else float("-inf")
            if score > best:
                best, best_idx = score, j
        correct += int(labels[best_idx] == answer); total += 1
        if verbose and (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")
    acc = correct / total if total else 0.0
    if verbose:
        print(f"OpenBookQA accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}
