"""BoolQ adapter (0-shot yes/no via answer-token log-prob).

Reading-comprehension yes/no QA. Standard in the lm-eval commonsense suite. Scoring:
prompt = passage + question + "Answer:", compare LL of " yes" vs " no" (first target
token); predict the higher. NOTE: BoolQ is class-imbalanced (~62% yes), so a degraded
model can score ~62% by always answering yes — interpret with that floor in mind. 0-1."""
import torch
from datasets import load_dataset
from .common import ensure_seqlen, set_use_cache, subsample, compute_completion_ll


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    if verbose:
        print("Evaluating on BoolQ (0-shot) ...")
    ensure_seqlen(model, seqlen)
    dataset = load_dataset("google/boolq", split="validation")
    dataset = subsample(dataset, limit)
    set_use_cache(model, False)
    model.to(device)
    correct = total = 0
    for i, ex in enumerate(dataset):
        passage = ex["passage"]; question = ex["question"]; ans = bool(ex["answer"])
        prompt = f"{passage}\nQuestion: {question}?\nAnswer:"
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        if prompt_len >= model.seqlen:  # keep last window incl. the answer slot
            pass
        scores = {}
        for word in (" yes", " no"):
            full_ids = tokenizer(prompt + word, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] > model.seqlen:
                over = full_ids.shape[1] - model.seqlen
                full_ids = full_ids[:, over:]; pl = max(1, prompt_len - over)
            else:
                pl = prompt_len
            ll, n = compute_completion_ll(model, full_ids, pl)
            scores[word] = ll / n if n > 0 else float("-inf")
        pred_yes = scores[" yes"] > scores[" no"]
        correct += int(pred_yes == ans); total += 1
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")
    acc = correct / total if total else 0.0
    if verbose:
        print(f"BoolQ accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}
