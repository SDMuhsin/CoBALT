"""PIQA adapter (0-shot, length-normalized completion log-likelihood).

Physical-commonsense 2-choice QA (goal + sol1/sol2). Standard in the lm-eval
commonsense suite reported by GPTQ/AWQ/SparseGPT/OmniQuant. Same scoring contract
as eval_hellaswag/eval_arc: for each candidate solution, score the length-normalized
LL of the solution given the goal, argmax. Dataset via the parquet mirror (datasets 5.x
dropped script loaders). Returns {"accuracy","correct","total"} (0-1)."""
import torch
from datasets import load_dataset
from .common import ensure_seqlen, set_use_cache, subsample, compute_completion_ll


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    if verbose:
        print("Evaluating on PIQA (0-shot) ...")
    ensure_seqlen(model, seqlen)
    dataset = load_dataset("piqa", revision="refs/convert/parquet", split="validation")
    dataset = subsample(dataset, limit)
    set_use_cache(model, False)
    model.to(device)
    correct = total = 0
    for i, ex in enumerate(dataset):
        goal = ex["goal"].strip()
        choices = [ex["sol1"].strip(), ex["sol2"].strip()]
        label = int(ex["label"])
        prompt_len = tokenizer(goal, return_tensors="pt").input_ids.shape[1]
        best, best_idx = float("-inf"), 0
        for j, ch in enumerate(choices):
            full_ids = tokenizer(goal + " " + ch, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] > model.seqlen:
                over = full_ids.shape[1] - model.seqlen
                full_ids = full_ids[:, over:]; pl = max(1, prompt_len - over)
            else:
                pl = prompt_len
            ll, n = compute_completion_ll(model, full_ids, pl)
            score = ll / n if n > 0 else float("-inf")
            if score > best:
                best, best_idx = score, j
        correct += int(best_idx == label); total += 1
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")
    acc = correct / total if total else 0.0
    if verbose:
        print(f"PIQA accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}
