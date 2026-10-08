#!/usr/bin/env python
"""Medical calibration corpus v2 (BACKLOG #47 'larger/more diverse medical corpus'): the round-1 QA items
(MedQA train, MedMCQA train+explanations, PubMedQA fold0 train/val — eval splits excluded) INTERLEAVED 1:1 with
PubMed abstracts from PubMedQA pqa_artificial (211k) + pqa_unlabeled (61k), excluding every PMID of pqa_labeled
(which contains the 500 evaluated test docs). Items shuffled (seed 43), '\n\n'-separated, >= 1040 x 2048 tokens."""
import glob, random
import pandas as pd
from transformers import AutoTokenizer

HUB = "/scratch/ckp908/prism_hf/hub"
OUT = "/workspace/PTQResearch/results/accel4bit/calib_med2_1024x2048.txt"
TOK = "/scratch/root/PTQResearch/accel4bit_models/biomistral-7b/hf_bf16"
NEED = 1045 * 2048
S = f"{HUB}/datasets--qiaojin--PubMedQA/snapshots/9001f2853fb87cab8d220904e0de81ac6973b318"

qa = [s for s in open("/workspace/PTQResearch/results/accel4bit/calib_med_512x2048.txt", encoding="utf-8").read().split("\n\n") if s.strip()]
lab = set(pd.read_parquet(glob.glob(f"{S}/pqa_labeled/*.parquet")[0])["pubid"].tolist())
pm = []
for sp in ("pqa_artificial", "pqa_unlabeled"):
    df = pd.read_parquet(glob.glob(f"{S}/{sp}/*.parquet")[0])
    for r in df.itertuples():
        if r.pubid in lab:
            continue
        ctx = r.context["contexts"] if isinstance(r.context, dict) else r.context
        ctx = " ".join(ctx) if not isinstance(ctx, str) else ctx
        pm.append(f"{ctx}\n{r.question}\n{r.long_answer}".replace("\n\n", "\n").strip())
rng = random.Random(43)
rng.shuffle(pm); rng.shuffle(qa)
tok = AutoTokenizer.from_pretrained(TOK)
out, ntok, i_pm, i_qa = [], 0, 0, 0
while ntok < NEED:
    for src in ("pm", "qa"):
        t = pm[i_pm] if src == "pm" else qa[i_qa % len(qa)]
        if src == "pm": i_pm += 1
        else: i_qa += 1
        out.append(t); ntok += len(tok(t, add_special_tokens=False)["input_ids"]) + 1
open(OUT, "w", encoding="utf-8").write("\n\n".join(out) + "\n")
print(f"pubmed pool {len(pm)} (labeled excluded {len(lab)}), qa pool {len(qa)}; wrote {len(out)} items "
      f"({i_pm} pubmed, {i_qa} qa; qa reuse={'yes' if i_qa > len(qa) else 'no'}), {ntok} tokens -> {OUT}")
