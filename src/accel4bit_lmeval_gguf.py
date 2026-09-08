#!/usr/bin/env python
"""In-process lm_eval model for GGUF files via llama-cpp-python (accel4bit gguf arm).

Why: llama-server's /v1/completions has no prompt logprobs (echo is ignored), so lm_eval `local-completions`
cannot score loglikelihoods against it; the llama-cpp-python *server* supports echo but spends ~12 s/request
in Python building top_logprobs over the 262k Gemma vocab.  This class scores loglikelihoods directly from
`Llama.scores` (logits_all=True) with KV-prefix reuse across requests that share a context (the 4 choices of
a multiple-choice question re-evaluate only their continuation tokens).  Tokenization = llama.cpp's own
tokenizer of the GGUF (BOS added, like lm_eval does for gemma HF/vLLM models).  Scoring semantics mirror
lm_eval's HFLM (left-truncate to max_length, drop last token, rolling windows for wikitext).

usage: python src/accel4bit_lmeval_gguf.py --model_path X.gguf --tasks arc_easy medqa_4options ... \
         --output_dir results/... --tag own [--limit_medmcqa 1000] [--include_path scripts/accel4bit_lmeval_tasks]
"""
import argparse, json, os, sys, time
import numpy as np
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval import utils as lm_utils


def _score_rows(logits, tgt):
    """logits: (n, V) float32 numpy view; tgt: (n,) ints -> (sum logprob of tgt, is_greedy).
    torch (CPU, multi-threaded) logsumexp; a numpy exp over n x 262144 took ~20 s per 4096-token window."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(logits))
    t = torch.from_numpy(np.asarray(tgt, dtype=np.int64))
    lse = torch.logsumexp(x, dim=1)
    tok_lp = x[torch.arange(x.shape[0]), t] - lse
    greedy = bool((x.argmax(dim=1) == t).all())
    return float(tok_lp.sum()), greedy


@register_model("llamacpp_gguf")
class LlamaCppGGUF(TemplateLM):
    def __init__(self, model_path, n_ctx=4096, n_gpu_layers=99, n_batch=512, n_threads=16, max_length=4096, add_bos=True, **kw):
        super().__init__()
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu_layers), n_batch=int(n_batch),
                         n_threads=int(n_threads), logits_all=True, verbose=False)
        self.max_length = int(max_length)
        self.add_bos = add_bos
        self._prev = []          # tokens currently in the KV cache (for prefix reuse)
        self.batch_size = 1
        self.n_evals = 0; self.n_tok_evald = 0; self.n_tok_reused = 0

    @property
    def eot_token_id(self):
        return self.llm.token_eos()

    @property
    def prefix_token_id(self):
        return self.llm.token_bos()

    def tok_encode(self, string, add_special_tokens=None, **kw):
        add_bos = self.add_bos if add_special_tokens is None else bool(add_special_tokens)
        return self.llm.tokenize(string.encode("utf-8"), add_bos=add_bos, special=True)

    def _eval_prefix_reuse(self, inp):
        lp = 0
        for a, b in zip(self._prev, inp):
            if a != b:
                break
            lp += 1
        if lp == len(inp) and lp > 0:      # identical to what is cached: scores rows still valid
            lp = len(inp) - 1              # re-evaluate the last token (cheap) to be safe
        self.llm.n_tokens = lp
        self.llm.eval(inp[lp:])
        self._prev = list(inp)
        self.n_evals += 1; self.n_tok_evald += len(inp) - lp; self.n_tok_reused += lp

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kw):
        # requests: list of ((ctx_str, cont_str), ctx_enc, cont_enc)
        order = sorted(range(len(requests)), key=lambda i: tuple(requests[i][1] + requests[i][2]))
        res = [None] * len(requests)
        for i in tqdm(order, disable=disable_tqdm, desc="loglikelihood"):
            _, ctx_enc, cont_enc = requests[i]
            full = list(ctx_enc) + list(cont_enc)
            inp = full[-(self.max_length + 1):][:-1]          # HFLM semantics: drop the last token, left-truncate
            n_cont = len(cont_enc)
            assert n_cont >= 1 and len(inp) >= n_cont
            self._eval_prefix_reuse(inp)
            logits = self.llm.scores[len(inp) - n_cont: len(inp), :]
            res[i] = _score_rows(logits, cont_enc)
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        out = []
        for (string,) in tqdm([r.args for r in requests], disable=disable_tqdm, desc="rolling"):
            windows = list(map(lm_utils.make_disjoint_window,
                               lm_utils.get_rolling_token_windows(token_list=self.tok_encode(string),
                                                                  prefix_token=self.prefix_token_id,
                                                                  max_seq_len=self.max_length, context_len=1)))
            reqs = [((None, None), c, k) for c, k in windows]
            nlls = self._loglikelihood_tokens(reqs, disable_tqdm=True)
            out.append(float(sum(x[0] for x in nlls)))
        return out

    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for (ctx, gen_kwargs) in tqdm([r.args for r in requests], disable=disable_tqdm, desc="generate"):
            until = gen_kwargs.get("until", None)
            max_toks = int(gen_kwargs.get("max_gen_toks", 256))
            self._prev = []; self.llm.reset()
            r = self.llm.create_completion(ctx, max_tokens=max_toks, temperature=0.0, stop=until)
            out.append(r["choices"][0]["text"])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default="own")
    ap.add_argument("--limit_medmcqa", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="global limit (validation only)")
    ap.add_argument("--include_path", default="/workspace/PTQResearch/scripts/accel4bit_lmeval_tasks")
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--n_gpu_layers", type=int, default=99)
    ap.add_argument("--n_threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import lm_eval
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=a.include_path)
    lm = LlamaCppGGUF(a.model_path, n_ctx=a.n_ctx, n_gpu_layers=a.n_gpu_layers, n_threads=a.n_threads, max_length=a.n_ctx)
    os.makedirs(a.output_dir, exist_ok=True)
    summary = {}
    for task in a.tasks:
        limit = a.limit if a.limit is not None else (a.limit_medmcqa if task == "medmcqa" else None)
        t0 = time.time(); lm.n_evals = lm.n_tok_evald = lm.n_tok_reused = 0
        r = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, task_manager=tm, log_samples=True,
                                    random_seed=a.seed, numpy_random_seed=a.seed, torch_random_seed=a.seed, fewshot_random_seed=a.seed)
        dt = time.time() - t0
        r["accel4bit"] = dict(model_path=a.model_path, tag=a.tag, task=task, limit=limit, seconds=dt, n_evals=lm.n_evals,
                              tokens_evaluated=lm.n_tok_evald, tokens_prefix_reused=lm.n_tok_reused, n_ctx=a.n_ctx,
                              backend="llama-cpp-python in-process (src/accel4bit_lmeval_gguf.py)")
        samples = r.pop("samples", {})
        sfx = "" if a.tag == "own" else f"_{a.tag}"          # own arm: eval_<task>.json / eval_summary.json (same names as the other arms)
        out = os.path.join(a.output_dir, f"eval_{task}{sfx}.json")
        with open(out, "w") as f:
            json.dump(r, f, indent=1, default=str)
        with open(os.path.join(a.output_dir, f"eval_{task}{sfx}.samples.jsonl"), "w") as f:
            for s in samples.get(task, []):
                f.write(json.dumps(s, default=str) + "\n")
        summary[task] = {k: v for k, v in r["results"][task].items() if k != "alias"}
        summary[task]["sample_len"] = r["n-samples"][task]["effective"]
        # merge into the cumulative summary file (same JSON shape as results/accel4bit/<model>/gptq/eval_summary.json)
        sp = os.path.join(a.output_dir, f"eval_summary{sfx}.json")
        cur = json.load(open(sp)) if os.path.exists(sp) else {}
        cur[task] = summary[task]
        json.dump(cur, open(sp, "w"), indent=2)
        print(f"RESULT {a.tag} {task} n={r['n-samples'][task]} {dt:.0f}s evals={lm.n_evals} tok_eval={lm.n_tok_evald} tok_reused={lm.n_tok_reused}",
              json.dumps({k: v for k, v in r["results"][task].items() if not k.startswith("alias")}), flush=True)
    print("SUMMARY", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
