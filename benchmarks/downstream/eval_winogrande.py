"""WinoGrande adapter (0-shot, partial-context length-normalized LL).

Standard commonsense coreference (fill-the-blank, 2-choice). lm-eval scoring:
the context is everything BEFORE the "_"; for each option the completion is
option + everything AFTER the blank; pick the option whose completion has higher
length-normalized LL. winogrande_xl config, validation split. 0-1 accuracy."""
import torch
from datasets import load_dataset
from .common import ensure_seqlen, set_use_cache, subsample, compute_completion_ll


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    if verbose:
        print("Evaluating on WinoGrande (0-shot) ...")
    ensure_seqlen(model, seqlen)
    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    dataset = subsample(dataset, limit)
    set_use_cache(model, False)
    model.to(device)
    correct = total = 0
    for i, ex in enumerate(dataset):
        sent = ex["sentence"]; opts = [ex["option1"], ex["option2"]]
        ans = ex["answer"]
        if ans not in ("1", "2"):
            continue
        label = int(ans) - 1
        idx = sent.find("_")
        if idx == -1:
            continue
        context = sent[:idx]; tail = sent[idx + 1:]
        prompt_len = tokenizer(context, return_tensors="pt").input_ids.shape[1]
        best, best_idx = float("-inf"), 0
        for j, opt in enumerate(opts):
            full_ids = tokenizer(context + opt + tail, return_tensors="pt").input_ids.to(device)
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
        print(f"WinoGrande accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}
