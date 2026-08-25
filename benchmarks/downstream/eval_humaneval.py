"""HumanEval adapter (pass@1, greedy generation + code execution).

Generation, truncation, and the self-contained code-execution checker are ported
VERBATIM from /workspace/CRB/eval_humaneval.py (tempfile + subprocess.run with a
10s per-problem timeout). The adapter ALWAYS runs generation+execution when
called; the SUITE decides whether to call it at all (gated on allow_codeexec).
"""

import os
import tempfile
import subprocess

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache, subsample


def truncate_at_stop_patterns(text):
    """Truncate generated code at common function-end patterns. (Verbatim CRB.)"""
    stop_patterns = ['\nclass ', '\ndef ', '\n# ', '\nif __name__', '\nprint(']
    min_idx = len(text)
    for pattern in stop_patterns:
        idx = text.find(pattern)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def check_correctness(prompt, completion, test, entry_point, timeout=10):
    """Execute generated code with test cases and check correctness.

    Returns True if all test cases pass, False otherwise. (Verbatim CRB.)
    """
    full_code = prompt + completion + "\n" + test
    # The test string should call check(entry_point), but add it if missing
    if f"check({entry_point})" not in full_code:
        full_code += f"\ncheck({entry_point})\n"

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            f.flush()
            tmp_file = f.name

        result = subprocess.run(
            ['python3', tmp_file],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate HumanEval (pass@1, greedy generation + execution).

    Returns {"pass_at_1", "passed", "total"} with pass_at_1 as a 0-1 fraction.
    """
    if verbose:
        print("Evaluating on HumanEval (pass@1, greedy) ...")

    ensure_seqlen(model, seqlen)

    # Load HumanEval dataset
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = subsample(dataset, limit)

    # Enable cache for efficient autoregressive generation
    use_cache = set_use_cache(model, True)
    model.to(device)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    passed = 0
    total = 0

    try:
        for i, example in enumerate(dataset):
            prompt = example['prompt']
            test = example['test']
            entry_point = example['entry_point']

            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            # Truncate prompt if too long (leave room for generation)
            max_prompt_len = model.seqlen - 256
            if input_ids.shape[1] > max_prompt_len:
                input_ids = input_ids[:, -max_prompt_len:]

            # Generate
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode only the generated tokens
            generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                         skip_special_tokens=True)

            # Truncate at stop patterns (end of function)
            generated = truncate_at_stop_patterns(generated)

            # Check correctness
            is_correct = check_correctness(prompt, generated, test, entry_point,
                                           timeout=10)

            if is_correct:
                passed += 1
            total += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dataset)}] Running pass@1: {passed/total:.4f}")

        pass_at_1 = passed / total if total > 0 else 0.0
    finally:
        model.config.use_cache = use_cache

    if verbose:
        print(f"HumanEval pass@1: {pass_at_1:.4f} ({passed}/{total})")

    return {
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
    }
