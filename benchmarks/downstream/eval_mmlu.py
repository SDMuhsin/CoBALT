"""MMLU adapter (5-shot, answer-letter log-prob).

Prompt formatting and scoring math (5-shot prompt with shot-reduction on
overflow, answer-letter token scoring at the last position) are ported VERBATIM
from /workspace/CRB/eval_mmlu.py. Return keys are renamed to the spec's CSV
schema ({"accuracy","correct","total","subject_accuracies"}); accuracies are
0-1 fractions.

When ``limit`` is given, the *test* set is subsampled (deterministic first-N);
the few-shot pool (validation/dev) is always taken in full so every kept test
question still gets up to 5 in-subject shots.
"""

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache, subsample


def format_mmlu_question(question, choices):
    """Format a single MMLU question with lettered choices. (Verbatim CRB.)"""
    letters = ['A', 'B', 'C', 'D']
    formatted = question + "\n"
    for letter, choice in zip(letters, choices):
        formatted += f"{letter}. {choice}\n"
    formatted += "Answer:"
    return formatted


def format_mmlu_prompt(subject, few_shot_examples, question, choices):
    """Format full MMLU prompt with few-shot examples. (Verbatim CRB.)"""
    subject_name = subject.replace('_', ' ')
    prompt = f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"

    letters = ['A', 'B', 'C', 'D']
    for ex in few_shot_examples:
        prompt += format_mmlu_question(ex['question'], ex['choices'])
        prompt += f" {letters[ex['answer']]}\n\n"

    prompt += format_mmlu_question(question, choices)
    return prompt


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate MMLU (5-shot).

    Returns {"accuracy", "correct", "total", "subject_accuracies"} with
    accuracies as 0-1 fractions.
    """
    if verbose:
        print("Evaluating on MMLU (5-shot) ...")

    ensure_seqlen(model, seqlen)

    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all")
    test_data = dataset['test']
    # Few-shot examples come from validation split
    try:
        dev_data = dataset['validation']
    except KeyError:
        dev_data = dataset['dev']

    test_data = subsample(test_data, limit)

    use_cache = set_use_cache(model, False)
    model.to(device)

    # Get token IDs for answer letters (with space prefix)
    answer_tokens = []
    for letter in ['A', 'B', 'C', 'D']:
        token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        answer_tokens.append(token_ids[-1])

    # Group dev examples by subject for few-shot
    dev_by_subject = {}
    for ex in dev_data:
        subj = ex['subject']
        if subj not in dev_by_subject:
            dev_by_subject[subj] = []
        dev_by_subject[subj].append(ex)

    # Evaluate
    subject_correct = {}
    subject_total = {}
    total_correct = 0
    total = 0

    try:
        for i, example in enumerate(test_data):
            subject = example['subject']
            question = example['question']
            choices = example['choices']
            answer = example['answer']  # int 0-3

            # Try with 5-shot, reduce if prompt is too long
            few_shot_pool = dev_by_subject.get(subject, [])
            input_ids = None
            for n_shots in range(min(5, len(few_shot_pool)), -1, -1):
                few_shot = few_shot_pool[:n_shots]
                prompt = format_mmlu_prompt(subject, few_shot, question, choices)
                ids = tokenizer.encode(prompt, return_tensors='pt')
                if ids.shape[1] <= model.seqlen:
                    input_ids = ids.to(device)
                    break

            if input_ids is None:
                # Even 0-shot is too long; truncate from the left
                prompt = format_mmlu_prompt(subject, [], question, choices)
                input_ids = tokenizer.encode(prompt, return_tensors='pt')
                input_ids = input_ids[:, -model.seqlen:].to(device)

            # Forward pass
            logits = model(input_ids).logits[0, -1, :]  # [vocab_size]

            # Score answer letters
            scores = logits[answer_tokens]
            prediction = scores.argmax().item()

            correct = (prediction == answer)
            total_correct += correct
            total += 1

            if subject not in subject_correct:
                subject_correct[subject] = 0
                subject_total[subject] = 0
            subject_correct[subject] += correct
            subject_total[subject] += 1

            if verbose and (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{len(test_data)}] Running accuracy: {total_correct/total:.4f}")

        overall_accuracy = total_correct / total if total > 0 else 0.0
    finally:
        model.config.use_cache = use_cache

    # Per-subject accuracy
    subject_accuracies = {}
    for subj in sorted(subject_correct.keys()):
        subject_accuracies[subj] = subject_correct[subj] / subject_total[subj]

    if verbose:
        print(f"MMLU Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total})")
        print(f"  Subjects evaluated: {len(subject_accuracies)}")

    return {
        "accuracy": overall_accuracy,
        "correct": int(total_correct),
        "total": total,
        "subject_accuracies": subject_accuracies,
    }
