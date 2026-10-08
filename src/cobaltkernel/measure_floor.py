#!/usr/bin/env python
"""Measure a model's OWN bf16 argmax-agreement floor: HF eager attention vs HF sdpa attention
on the verify_kernel prompt (same tokens, same length).  verify_kernel's FLOOR_ARGMAX was
measured on gemma-3-4b; a different model needs its own number before gate (i) means anything.

  python measure_floor.py --model <hf dir> [--prompt-len 1152]
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobaltkernel.verify_kernel import get_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=1152)
    ap.add_argument("--prompt-seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    ids = torch.tensor(get_ids(a.model, a.prompt_len, seed=a.prompt_seed), device="cuda")[None]
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                              attn_implementation="eager").cuda().eval()
    with torch.no_grad():
        lg_e = hf(ids).logits[0].float()
        hf.config._attn_implementation = "sdpa"
        for m in hf.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "sdpa"
        lg_s = hf(ids).logits[0].float()
    agree = (lg_e.argmax(-1) == lg_s.argmax(-1)).float().mean().item()
    d = (lg_e - lg_s).abs()
    print(f"FLOOR {os.path.basename(a.model.rstrip('/'))} eager-vs-sdpa bf16: argmax agreement "
          f"{100*agree:.4f}% ({int(agree*a.prompt_len)}/{a.prompt_len}) max|diff|={d.max().item():.4f} "
          f"mean={d.mean().item():.5f}", flush=True)


if __name__ == "__main__":
    main()
