"""MRR adapter (Mean Reciprocal Rank of the correct next token, PTB test).

Scoring math (concatenate test sentences -> token stream -> chunk by seqlen,
rank = #tokens with strictly higher logit + 1, mean of 1/rank over all
positions) is ported VERBATIM from /workspace/CRB/eval_mrr.opt_eval_mrr.

Dataset: the PTB script loaders (``ptb_text_only`` / ``penn_treebank``) often
break on datasets 5, so we use a fallback chain and fall back to wikitext-2.
``limit`` caps the number of seqlen-sized chunks evaluated.
"""

import torch
from datasets import load_dataset

from .common import ensure_seqlen, set_use_cache


def _load_mrr_text(verbose=False):
    """Load PTB test text with a datasets-5-safe fallback chain to wikitext-2.

    Returns (list_of_strings, source_name). The strings are joined with " " by
    the caller (matching CRB's ``" ".join(testdata['sentence'])``).
    """
    # 1) PTB via ptb_text_only / penn_treebank (CRB's original loader).
    try:
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        if verbose:
            print("  MRR dataset: ptb_text_only/penn_treebank (test)")
        return list(testdata['sentence']), 'ptb_text_only'
    except Exception as e:
        if verbose:
            print(f"  ptb_text_only load failed ({type(e).__name__}: {e}); falling back to wikitext-2")

    # 2) Fallback: wikitext-2-raw-v1 test split.
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if verbose:
        print("  MRR dataset: wikitext/wikitext-2-raw-v1 (test) [fallback]")
    return list(testdata['text']), 'wikitext-2-raw-v1'


@torch.no_grad()
def run(model, tokenizer, device, limit=None, seqlen=2048, verbose=False, **kwargs):
    """Evaluate Mean Reciprocal Rank of the correct next token.

    Returns {"mrr", "total"} where ``total`` is the number of next-token
    positions scored. ``mrr`` is a plain float in (0, 1].
    """
    if verbose:
        print("Evaluating MRR on PTB test set ...")

    ensure_seqlen(model, seqlen)

    sentences, _source = _load_mrr_text(verbose=verbose)
    testenc = tokenizer(" ".join(sentences), return_tensors='pt')
    test_ids = testenc.input_ids  # [1, total_tokens]

    seqlen = model.seqlen
    nsamples = test_ids.numel() // seqlen
    if limit is not None:
        nsamples = min(nsamples, limit)

    if verbose:
        print(f"  Test set: {test_ids.numel()} tokens, {nsamples} chunks of {seqlen}")

    use_cache = set_use_cache(model, False)
    model.to(device)

    reciprocal_ranks_sum = 0.0
    total_positions = 0

    try:
        for i in range(nsamples):
            input_ids = test_ids[:, i * seqlen : (i + 1) * seqlen].to(device)

            logits = model(input_ids).logits  # [1, seqlen, vocab_size]

            # shift: logits[0, :-1] predicts tokens at positions [1:]
            shift_logits = logits[0, :-1, :]  # [seqlen-1, vocab_size]
            shift_labels = input_ids[0, 1:]   # [seqlen-1]

            # Get the logit value for each correct token
            correct_logits = shift_logits[
                torch.arange(shift_logits.size(0), device=device), shift_labels
            ]  # [seqlen-1]

            # Rank = number of tokens with strictly higher logit + 1
            ranks = (shift_logits > correct_logits.unsqueeze(1)).sum(dim=1).float() + 1.0

            reciprocal_ranks = 1.0 / ranks
            reciprocal_ranks_sum += reciprocal_ranks.sum().item()
            total_positions += ranks.numel()

            if verbose and ((i + 1) % 5 == 0 or i == 0):
                running_mrr = reciprocal_ranks_sum / total_positions
                print(f"  [chunk {i+1}/{nsamples}] Running MRR: {running_mrr:.6f}")

        mrr = reciprocal_ranks_sum / total_positions if total_positions > 0 else 0.0
    finally:
        model.config.use_cache = use_cache

    if verbose:
        print(f"MRR: {mrr:.6f} ({total_positions} positions)")

    return {
        "mrr": mrr,
        "total": total_positions,
    }
