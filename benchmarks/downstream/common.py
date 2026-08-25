"""Shared helpers for the downstream-task adapters.

These are thin utilities every adapter uses to honour the uniform ``run(...)``
contract: pin ``model.seqlen``, toggle ``use_cache`` (restoring it afterwards),
deterministically subsample a dataset, and a couple of log-likelihood helpers
ported verbatim from the CRB reference (kept here so the per-task modules can
share one copy of the identical scoring math).
"""

import torch


def ensure_seqlen(model, seqlen):
    """Set ``model.seqlen`` (CRB evals read this) and return the value used."""
    model.seqlen = seqlen
    return seqlen


def set_use_cache(model, val):
    """Set ``model.config.use_cache`` and return the previous value (to restore)."""
    old = model.config.use_cache
    model.config.use_cache = val
    return old


def subsample(ds, limit):
    """Deterministic first-``limit`` slice of an iterable/HF dataset (no shuffle).

    ``limit=None`` returns the dataset unchanged. HF ``Dataset`` objects support
    ``.select`` (fast, lazy); anything else falls back to a plain list slice.
    """
    if limit is None:
        return ds
    try:
        n = len(ds)
    except TypeError:
        # Not sized (e.g. a generator) -> materialize then slice.
        return list(ds)[:limit]
    n = min(limit, n)
    select = getattr(ds, "select", None)
    if select is not None:
        try:
            return select(range(n))
        except Exception:
            pass
    return ds[:n]


def compute_completion_ll(model, input_ids, prompt_len):
    """Total log-likelihood and token count for completion tokens.

    Ported verbatim from CRB ``eval_hellaswag.compute_completion_ll`` /
    ``eval_arc.compute_completion_ll`` (identical in both).

    Args:
        model: language model
        input_ids: [1, total_len] tensor on device
        prompt_len: number of prompt tokens (completion starts here)

    Returns:
        (total_log_likelihood, num_completion_tokens)
    """
    n_completion = input_ids.shape[1] - prompt_len
    if n_completion <= 0:
        return 0.0, 0

    outputs = model(input_ids)
    logits = outputs.logits[0]  # [total_len, vocab_size]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # For completion token at position j, the predicting logit is at position j-1
    completion_ids = input_ids[0, prompt_len:]  # [n_completion]
    prediction_log_probs = log_probs[prompt_len - 1:-1]  # [n_completion, vocab_size]
    token_log_probs = prediction_log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)

    total_ll = token_log_probs.sum().item()
    return total_ll, n_completion
