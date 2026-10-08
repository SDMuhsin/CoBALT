#!/usr/bin/env python
"""Medical-domain calibration corpus for BioMistral (BACKLOG #7), TRAIN splits only, no eval-split text:
  MedQA-USMLE train (question + options + answer), MedMCQA train (question + options + explanation; the lm_eval task
  evaluates on VALIDATION, which is excluded), PubMedQA labeled fold0 train+validation (context + question + long answer;
  the task evaluates on the 500-doc test split, excluded). Items are shuffled (seed 42) and written '\n\n'-separated in the
  same format as calib_ultrachat_512x2048.txt, enough for >= 528 x 2048 tokens.
"""
import json, os, random
import pandas as pd
from transformers import AutoTokenizer

HUB = "/scratch/ckp908/prism_hf/hub"
OUT = "/workspace/PTQResearch/results/accel4bit/calib_med_512x2048.txt"
TOK = "/scratch/root/PTQResearch/accel4bit_models/biomistral-7b/hf_bf16"
NEED = 530 * 2048

items = []
mq = [json.loads(l) for l in open(f"{HUB}/datasets--GBaker--MedQA-USMLE-4-options-hf/snapshots/17af9355ef89fba60de966eabaeba797c695f86e/train.json") if l.strip()]
for d in mq:
    q = d.get("sent1") or d.get("question") or ""
    opts = [d.get(f"ending{i}") for i in range(4)]
    if not all(opts):
        opts = list(d.get("options", {}).values()) if isinstance(d.get("options"), dict) else opts
    ans = d.get("label", d.get("answer_idx", ""))
    txt = q + "\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts) if o)
    if isinstance(ans, int) and 0 <= ans < 4 and opts[ans]:
        txt += f"\nAnswer: {opts[ans]}"
    items.append(("medqa", txt))
n_mq = len(items)
mm = pd.read_parquet(f"{HUB}/datasets--openlifescienceai--medmcqa/snapshots/91c6572c454088bf71b679ad90aa8dffcd0d5868/data/train-00000-of-00001.parquet")
for r in mm.itertuples():
    txt = f"{r.question}\nA. {r.opa}\nB. {r.opb}\nC. {r.opc}\nD. {r.opd}"
    if isinstance(r.exp, str) and len(r.exp) > 20:
        txt += f"\nExplanation: {r.exp}"
    items.append(("medmcqa", txt))
n_mm = len(items) - n_mq
pq = f"{HUB}/datasets--bigbio--pubmed_qa/snapshots/13a7d15476092370cbabb6475390e7e69b74d2f2/pubmed_qa_labeled_fold0_source"
for split in ("train", "validation"):
    df = pd.read_parquet(f"{pq}/{split}/0000.parquet")
    for r in df.itertuples():
        ctx = r.CONTEXTS
        ctx = " ".join(ctx) if not isinstance(ctx, str) else ctx
        txt = f"{ctx}\n{r.QUESTION}\n{r.LONG_ANSWER}"
        items.append(("pubmedqa", txt))
n_pq = len(items) - n_mq - n_mm
random.Random(42).shuffle(items)
tok = AutoTokenizer.from_pretrained(TOK)
out, ntok = [], 0
for src, t in items:
    t = t.replace("\n\n", "\n").strip()
    if not t:
        continue
    out.append(t)
    ntok += len(tok(t, add_special_tokens=False)["input_ids"]) + 1
    if ntok >= NEED:
        break
open(OUT, "w", encoding="utf-8").write("\n\n".join(out) + "\n")
used = {k: sum(1 for s, _ in items[:len(out)] if s == k) for k in ("medqa", "medmcqa", "pubmedqa")}
print(f"sources medqa={n_mq} medmcqa={n_mm} pubmedqa={n_pq}; wrote {len(out)} items, {ntok} tokens -> {OUT}; used {used}")
