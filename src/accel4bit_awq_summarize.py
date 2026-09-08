#!/usr/bin/env python
"""Collect results/accel4bit/<model>/awq/{eval_*.json,speed_summary.json,bpw.json,kernel_dispatch.txt}
into RESULTS_TABLE.md (markdown) + results_summary.json.  Usage: summarize.py <results_dir> <MIG-UUID>"""
import glob, json, os, sys
r, mig = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "?"
out = {"slice": mig}
rows = []
for fn in sorted(glob.glob(os.path.join(r, "eval_*.json"))):
    try:
        j = json.load(open(fn))
    except Exception as e:
        rows.append((os.path.basename(fn), f"UNREADABLE {e}")); continue
    for task, m in j.get("results", {}).items():
        keep = {k: v for k, v in m.items() if any(k.startswith(p) for p in ("acc", "word_perplexity", "byte_perplexity", "bits_per_byte")) and not k.endswith("_stderr")}
        n = j.get("n-samples", {}).get(task, {}).get("effective")
        out[task] = {"metrics": keep, "n": n}
        rows.append((task, ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in keep.items()) + f" (n={n})"))
sp = os.path.join(r, "speed_summary.json")
if os.path.exists(sp):
    out["speed"] = json.load(open(sp))
bp = os.path.join(r, "bpw.json")
if os.path.exists(bp):
    try:
        out["bpw"] = json.load(open(bp))
    except Exception:
        pass
kd = os.path.join(r, "kernel_dispatch.txt")
if os.path.exists(kd):
    out["kernel_dispatch_head"] = open(kd).read()[:2000]
json.dump(out, open(os.path.join(r, "results_summary.json"), "w"), indent=1); json.dump(out, open(os.path.join(r, "eval_summary.json"), "w"), indent=1)
L = [f"# AWQ results ({os.path.basename(os.path.dirname(r))}) — slice {mig}", "", "| task | metrics |", "|---|---|"]
L += [f"| {a} | {b} |" for a, b in rows]
s = out.get("speed")
if s:
    L += ["", "| speed (vLLM, bs=1 512->128 unless noted) | value |", "|---|---|"]
    for k in ("latency_512_128_s", "ttft_512_s", "prefill_tok_s", "decode_tok_s", "e2e_tok_s_bs1", "peak_gpu_mem_MiB_nvidia_smi"):
        if k in s: L.append(f"| {k} | {s[k]} |")
    t = s.get("throughput")
    if t:
        L += [f"| throughput (256 prompts 512->128, max-num-seqs 32) | {t.get('tokens_per_second','?'):.1f} tok/s total, {t.get('requests_per_second','?'):.2f} req/s, elapsed {t.get('elapsed_time','?'):.1f}s |"]
b = out.get("bpw")
if b:
    L += ["", f"effective bpw (quantized linears) = {b['bpw_quantized_linears']:.4f}; whole artifact = {b['bpw_all_tensors']:.3f} bits/param ({b['n_quantized_linears']} linears, {b['quantized_params']/1e9:.2f}B quantized params)"]
open(os.path.join(r, "RESULTS_TABLE.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
