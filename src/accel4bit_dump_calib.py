#!/usr/bin/env python
"""Materialise the accel4bit PROTOCOL calibration set as plain text (shared by all arms; used by
llama-imatrix for the gguf arm).

Recipe (= llm-compressor default GPTQ/AWQ example recipe):
  HuggingFaceH4/ultrachat_200k, split train_sft, ds.shuffle(seed=42).select(range(512)),
  text = tokenizer.apply_chat_template(messages, tokenize=False)   (Gemma-3 tokenizer)
  ids  = tokenizer(text, add_special_tokens=False, truncation=True, max_length=2048)
Each sample is decoded back to text (special tokens kept as literal strings so that
`llama-imatrix --parse-special` re-parses <start_of_turn>/<end_of_turn>); the leading "<bos>" emitted by
the chat template is stripped because llama-imatrix adds BOS itself (add_bos=true for Gemma).
Samples are separated by one blank line.  A JSON sidecar records per-sample token counts.
"""
import argparse, json, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF snapshot dir of the Gemma-3 tokenizer (medgemma-27b-text-it)")
    ap.add_argument("--out", default="results/accel4bit/calib_ultrachat_512x2048.txt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        print(f"[dump_calib] {a.out} exists -> reuse (use --force to regenerate)")
        return 0

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=a.seed).select(range(a.n))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    counts, total, trunc = [], 0, 0
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            text = tok.apply_chat_template(ex["messages"], tokenize=False)
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=a.seq)["input_ids"]
            if len(ids) == a.seq:
                trunc += 1
            dec = tok.decode(ids, skip_special_tokens=False)
            if dec.startswith("<bos>"):
                dec = dec[len("<bos>"):]
            dec = dec.strip("\n")
            # samples are blank-line separated: collapse internal blank lines so the separator is unambiguous
            dec = "\n".join(l if l.strip() else " " for l in dec.split("\n"))
            f.write(dec + "\n\n")
            counts.append(len(ids)); total += len(ids)
    os.replace(a.out + ".tmp", a.out)
    meta = dict(dataset="HuggingFaceH4/ultrachat_200k", split="train_sft", seed=a.seed, n=a.n, seq=a.seq,
                tokenizer=a.tokenizer, tokens_total=total, samples_truncated_at_seq=trunc,
                tokens_min=min(counts), tokens_max=max(counts), tokens_mean=total / len(counts),
                note="leading <bos> stripped; special tokens kept literal (use llama-imatrix --parse-special)")
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
