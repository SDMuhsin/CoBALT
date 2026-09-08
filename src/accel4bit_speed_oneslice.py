"""accel4bit PHASE 3 helper: fold one arm's raw one-slice speed logs/JSONs into a uniform <arm>.json, and build
results/accel4bit/<model>/speed_oneslice/SPEED_TABLE.md from all <arm>.json in that directory.

Sub-commands (called by scripts/run_accel4bit_speed_oneslice.sh):
  vllm  --arm A --model M --slice UUID --outdir D --artifact PATH --latency-json F --ttft-json F --throughput-json F
        --latency-log F [--ttft-log F --throughput-log F] --peak-samples F --python PY [--extra-note S] [--status S]
  gguf  --arm gguf[_tag] --model M --slice UUID --outdir D --artifact PATH --bench-json F [--bench-cublas-json F]
        --batched-log F --peak-samples F --bench-stderr F [--status S]
  pending --arm A --model M --slice UUID --outdir D --reason S
  table --outdir D --model M

Uniform per-arm JSON schema (all numbers floats; missing -> null):
  single_stream: ttft_ms, prefill_tok_s, decode_tok_s, e2e_latency_s, e2e_gen_tok_s (+ vLLM bench-latency avg/p50/p99)
  batched:       req_s, total_tok_s (in+out), output_tok_s, elapsed_s, config
  like_for_like: decode_tok_s / prefill_tok_s chosen so vLLM and llama.cpp measure the SAME thing (see notes in the table).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

OUT_LEN = 128
IN_LEN = 512
N_PROMPTS = 256


def load(p):
    if p and os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception as e:  # noqa
            return {"_load_error": f"{p}: {e}"}
    return None


def peak_from_samples(p):
    if not p or not os.path.exists(p):
        return None
    vals = []
    for line in open(p):
        line = line.strip()
        if line.isdigit():
            vals.append(int(line))
    return max(vals) if vals else None


def dir_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for root, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith((".safetensors", ".gguf", ".bin", ".pt")):
                tot += os.path.getsize(os.path.join(root, f))
    return tot


def kernel_lines(log):
    """Kernel-selection lines from a vLLM INFO/DEBUG log (deduplicated, in order)."""
    if not log or not os.path.exists(log):
        return []
    pat = re.compile(r"(Using \S*Kernel for [^\n]*|Selected \S*Kernel for [^\n]*|Using FlashAttention version \d+|"
                     r"Using \S+ backend|Attention backend[^\n]*|Using Marlin[^\n]*|marlin[^\n]{0,60}|NVFP4 GEMM[^\n]*)")
    seen, out = set(), []
    for line in open(log, errors="replace"):
        if "compilation/backends" in line or "site-packages" in line:
            continue
        m = pat.search(line)
        if m:
            s = m.group(0).strip()
            s = re.sub(r"\s+", " ", s)
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def kernel_summary(lines, arm):
    j = " | ".join(lines)
    if "NvFp4" in j or "NVFP4" in j:
        return "vLLM FlashInferCutlassNvFp4LinearKernel (CUTLASS SM120 block-scaled FP4 GEMM, W4A4)"
    if "MarlinLinearKernel" in j:
        return "vLLM MarlinLinearKernel (gptq_marlin / awq_marlin W4A16, sm_120)"
    if "FP8ScaledMM" in j or "ScaledMM" in j:
        return "vLLM CutlassFP8ScaledMMLinearKernel (W8A8 FP8 CUTLASS scaled_mm)"
    if arm in ("bf16", "ref_bf16"):
        return "bf16 cuBLAS GEMM (no quantization)"
    return "UNKNOWN (no kernel-selection line found)"


def versions(py):
    code = ("import json,sys\nd={'python':sys.version.split()[0]}\n"
            "for m in ['vllm','torch','transformers','compressed_tensors','flashinfer','lm_eval']:\n"
            "    try:\n        mod=__import__(m); d[m]=getattr(mod,'__version__','?')\n    except Exception as e: d[m]=None\n"
            "print(json.dumps(d))")
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300,
                             env={**os.environ, "PYTHONPATH": ""})
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa
        return {"error": str(e)}


def base(a, runtime):
    return {
        "arm": a.arm, "model": a.model, "slice": a.slice, "slice_type": a.slice_type,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "runtime": runtime,
        "artifact": {"path": a.artifact, "bytes": dir_bytes(a.artifact) if a.artifact and os.path.exists(a.artifact) else None},
        "protocol": {"single_stream": f"{IN_LEN}->{OUT_LEN}, bs=1, greedy", "batched": f"{N_PROMPTS} prompts {IN_LEN}->{OUT_LEN}, max concurrency 32"},
    }


def cmd_vllm(a):
    lat = load(a.latency_json)
    ttft = load(a.ttft_json)
    thr = load(a.throughput_json)
    out = base(a, {"name": "vLLM", "python": a.python, "venv": os.path.dirname(os.path.dirname(a.python)),
                   "versions": versions(a.python)})
    lines = kernel_lines(a.latency_log) + [l for l in kernel_lines(a.ttft_log) if l not in kernel_lines(a.latency_log)]
    out["kernel"] = {"summary": kernel_summary(lines, a.arm), "lines": lines}
    ss = {"ttft_ms": None, "prefill_tok_s": None, "decode_tok_s": None, "e2e_latency_s": None, "e2e_gen_tok_s": None,
          "bench_latency_avg_s": None, "bench_latency_p50_s": None, "bench_latency_p99_s": None,
          "source": "ttft/prefill/decode from src/accel4bit_ttft.py (median of 10 after 3 warmup; TTFT = max_tokens=1 request on the "
                    "same 512-token prompt; decode = 127/(e2e-TTFT)); e2e = vllm bench latency avg over 10 iters"}
    if lat and "avg_latency" in lat:
        ss["bench_latency_avg_s"] = lat["avg_latency"]
        ss["bench_latency_p50_s"] = lat.get("percentiles", {}).get("50")
        ss["bench_latency_p99_s"] = lat.get("percentiles", {}).get("99")
        ss["e2e_latency_s"] = lat["avg_latency"]
        ss["e2e_gen_tok_s"] = OUT_LEN / lat["avg_latency"]
    if ttft and "ttft_s_median" in ttft:
        ss["ttft_ms"] = ttft["ttft_s_median"] * 1e3
        ss["prefill_tok_s"] = ttft["prefill_tok_s_from_ttft"]
        ss["decode_tok_s"] = ttft["decode_tok_s"]
        ss["ttft_script_e2e_s"] = ttft["e2e_s_median_128"]
        if ss["e2e_latency_s"] is None:
            ss["e2e_latency_s"] = ttft["e2e_s_median_128"]
            ss["e2e_gen_tok_s"] = ttft["gen_tok_s_e2e"]
    out["single_stream"] = ss
    bt = {"req_s": None, "total_tok_s": None, "output_tok_s": None, "elapsed_s": None,
          "config": f"vllm bench throughput --input-len {IN_LEN} --output-len {OUT_LEN} --num-prompts {N_PROMPTS} --max-num-seqs 32 "
                    "(continuous batching; vLLM tokens_per_second counts INPUT+OUTPUT tokens)"}
    if thr and "requests_per_second" in thr:
        bt["req_s"] = thr["requests_per_second"]
        bt["total_tok_s"] = thr["tokens_per_second"]
        bt["output_tok_s"] = thr["requests_per_second"] * OUT_LEN
        bt["elapsed_s"] = thr["elapsed_time"]
        bt["num_requests"] = thr.get("num_requests")
        bt["total_num_tokens"] = thr.get("total_num_tokens")
    out["batched"] = bt
    out["peak_mem_MiB"] = {"max_over_all_stages": peak_from_samples(a.peak_samples),
                           "note": "sum of nvidia-smi compute-apps used_memory over this run's processes, 1 s samples; vLLM pre-allocates "
                                   "gpu_memory_utilization=0.85 of the slice for KV cache, so this is the budget, not the model's need",
                           "torch_max_memory_allocated_GB_ttft_run": (ttft or {}).get("torch_max_memory_allocated_GB")}
    out["like_for_like"] = {"decode_tok_s": ss["decode_tok_s"], "prefill_tok_s": ss["prefill_tok_s"],
                            "e2e_latency_s": ss["e2e_latency_s"],
                            "note": "decode = tokens 2..128 after a 512-token prompt (KV ~512-640); prefill = 512/TTFT"}
    out["raw"] = {"latency": lat, "ttft": ttft, "throughput": thr}
    out["extra_note"] = a.extra_note
    ok = ss["e2e_latency_s"] is not None and ss["decode_tok_s"] is not None and bt["output_tok_s"] is not None
    out["status"] = a.status or ("OK" if ok else "PARTIAL")
    json.dump(out, open(os.path.join(a.outdir, f"{a.arm}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("arm", "status", "kernel", "single_stream", "batched", "peak_mem_MiB")}, indent=1))


def parse_batched(log):
    rows = []
    if not log or not os.path.exists(log):
        return rows
    for line in open(log, errors="replace"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10 or not cells[0].isdigit():
            continue
        try:
            pp, tg, b, nkv, tpp, spp, ttg, stg, t, s = [float(x) for x in cells[:10]]
        except ValueError:
            continue
        rows.append({"PP": int(pp), "TG": int(tg), "B": int(b), "N_KV": int(nkv), "T_PP_s": tpp, "S_PP_tok_s": spp,
                     "T_TG_s": ttg, "S_TG_tok_s": stg, "T_s": t, "S_tok_s": s})
    return rows


def bench_entry(j, n_prompt, n_gen):
    for e in j or []:
        if e.get("n_prompt") == n_prompt and e.get("n_gen") == n_gen:
            return e
    return None


def cmd_gguf(a):
    bj = load(a.bench_json)
    cj = load(a.bench_cublas_json)
    rows = parse_batched(a.batched_log)
    build = {}
    if isinstance(bj, list) and bj:
        build = {"build_commit": bj[0].get("build_commit"), "build_number": bj[0].get("build_number"), "gpu_info": bj[0].get("gpu_info"),
                 "n_threads": bj[0].get("n_threads"), "flash_attn": bj[0].get("flash_attn"), "n_batch": bj[0].get("n_batch"), "n_ubatch": bj[0].get("n_ubatch")}
    out = base(a, {"name": "llama.cpp", "binaries": os.path.dirname(a.bench_json) if False else "/workspace/PTQResearch/temp/llama.cpp/build/bin",
                   "build": build, "cuda_toolkit": "/scratch/root/PTQResearch/cuda-13 (nvcc 13.0.88), CMAKE_CUDA_ARCHITECTURES=120 (sm_120a)"})
    pp = bench_entry(bj, IN_LEN, 0)
    tg = bench_entry(bj, 0, OUT_LEN)
    cpp = bench_entry(cj, IN_LEN, 0)
    ctg = bench_entry(cj, 0, OUT_LEN)
    b1 = next((r for r in rows if r["B"] == 1), None)
    b8 = next((r for r in rows if r["B"] == 8), None)
    b32 = next((r for r in rows if r["B"] == 32), None)
    mmq_line = None
    if a.bench_stderr and os.path.exists(a.bench_stderr):
        for line in open(a.bench_stderr, errors="replace"):
            if "ggml_cuda_init" in line or "compute capability" in line:
                mmq_line = line.strip()
    k = {"summary": "llama.cpp CUDA MMQ (int8 tensor-core mul_mat_q, Q4_K/Q6_K) for prefill; MMVQ mat-vec for bs=1 decode; "
                    "FA on, CUDA graphs on. Proof: forced-cuBLAS counterfactual build below",
         "cuda_init": mmq_line,
         "mmq_vs_forced_cublas": {"pp512_mmq_tok_s": pp and pp.get("avg_ts"), "pp512_cublas_tok_s": cpp and cpp.get("avg_ts"),
                                  "tg128_mmq_tok_s": tg and tg.get("avg_ts"), "tg128_cublas_tok_s": ctg and ctg.get("avg_ts"),
                                  "pp_speedup_mmq_over_cublas": (pp["avg_ts"] / cpp["avg_ts"]) if pp and cpp and cpp.get("avg_ts") else None,
                                  "note": "GGML_CUDA_FORCE_CUBLAS is compile-time at this commit -> separate build temp/llama.cpp/build-cublas"}}
    out["kernel"] = k
    ss = {"ttft_ms": (b1["T_PP_s"] * 1e3) if b1 else None,
          "prefill_tok_s": b1["S_PP_tok_s"] if b1 else (pp and pp.get("avg_ts")),
          "decode_tok_s": b1["S_TG_tok_s"] if b1 else None,
          "e2e_latency_s": b1["T_s"] if b1 else None,
          "e2e_gen_tok_s": (OUT_LEN / b1["T_s"]) if b1 else None,
          "llama_bench_pp512_tok_s": pp and pp.get("avg_ts"), "llama_bench_pp512_std": pp and pp.get("stddev_ts"),
          "llama_bench_tg128_tok_s": tg and tg.get("avg_ts"), "llama_bench_tg128_std": tg and tg.get("stddev_ts"),
          "source": "ttft/prefill/decode/e2e from llama-batched-bench B=1 row (pp512 then tg128 in ONE request = same shape as the vLLM "
                    "runs); llama_bench_* from llama-bench -p 512 -n 128 -r 5 (tg128 there is decode from an EMPTY context, not after a prompt)"}
    out["single_stream"] = ss
    bt = {"req_s": (32 / b32["T_s"]) if b32 else None, "total_tok_s": b32["S_tok_s"] if b32 else None,
          "output_tok_s": (32 * OUT_LEN / b32["T_s"]) if b32 else None, "elapsed_s": b32 and b32["T_s"],
          "decode_tok_s_at_B32": b32 and b32["S_TG_tok_s"], "prefill_tok_s_at_B32": b32 and b32["S_PP_tok_s"],
          "config": "llama-batched-bench -npp 512 -ntg 128 -npl 32 (ONE static batch of 32 sequences; NOT 256 continuously-batched requests) "
                    "-> req_s = 32/T, output_tok_s = 32*128/T, total_tok_s = S (pp+tg tokens / T)",
          "rows": rows}
    out["batched"] = bt
    out["peak_mem_MiB"] = {"max_over_all_stages": peak_from_samples(a.peak_samples),
                           "note": "sum of nvidia-smi compute-apps used_memory over this run's processes, 1 s samples (llama.cpp allocates "
                                   "only what it needs: weights + KV for the requested context)"}
    out["like_for_like"] = {"decode_tok_s": ss["decode_tok_s"], "prefill_tok_s": ss["prefill_tok_s"], "e2e_latency_s": ss["e2e_latency_s"],
                            "note": "from llama-batched-bench B=1 (decode of 128 tokens after a 512-token prompt) = same shape as vLLM decode"}
    out["raw"] = {"llama_bench_mmq": bj, "llama_bench_cublas": cj, "batched_rows": rows}
    out["extra_note"] = a.extra_note
    ok = ss["decode_tok_s"] is not None and ss["llama_bench_tg128_tok_s"] is not None and bt["output_tok_s"] is not None
    out["status"] = a.status or ("OK" if ok else "PARTIAL")
    json.dump(out, open(os.path.join(a.outdir, f"{a.arm}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("arm", "status", "single_stream", "batched", "peak_mem_MiB")}, indent=1, default=str))


def cmd_pending(a):
    out = base(a, None)
    out.update({"status": "pending", "reason": a.reason, "kernel": None, "single_stream": None, "batched": None, "peak_mem_MiB": None,
                "like_for_like": None})
    p = os.path.join(a.outdir, f"{a.arm}.json")
    if os.path.exists(p):
        try:
            old = json.load(open(p))
            if old.get("status") in ("OK", "PARTIAL"):
                print(f"[pending] keeping existing {p} (status {old['status']})")
                return
        except Exception:
            pass
    json.dump(out, open(p, "w"), indent=1)
    print(f"[pending] {a.arm}: {a.reason}")


ARM_ORDER = ["bf16", "ref_fp8", "gptq", "awq", "nvfp4", "gguf", "gguf_unsloth_q4km", "gguf_unsloth_udq4kxl"]
ARM_LABEL = {"bf16": "bf16 reference (vLLM)", "ref_fp8": "FP8-dynamic W8A8 reference (vLLM)", "gptq": "A gptq W4A16 g128 (vLLM gptq_marlin)",
             "awq": "D awq W4A16 asym g128 (vLLM awq_marlin)", "nvfp4": "C nvfp4 W4A4 (vLLM CUTLASS FP4)",
             "gguf": "B gguf own imatrix Q4_K_M (llama.cpp MMQ)", "gguf_unsloth_q4km": "B' unsloth prebuilt Q4_K_M (llama.cpp MMQ)",
             "gguf_unsloth_udq4kxl": "B'' unsloth prebuilt UD-Q4_K_XL (llama.cpp MMQ)"}


def f(x, nd=1):
    if x is None:
        return "pending"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def cmd_table(a):
    arms = {}
    for p in glob.glob(os.path.join(a.outdir, "*.json")):
        try:
            j = json.load(open(p))
            if "arm" in j:
                arms[j["arm"]] = j
        except Exception:
            pass
    order = [x for x in ARM_ORDER if x in arms] + sorted(x for x in arms if x not in ARM_ORDER)
    L = [f"# accel4bit — ONE-SLICE cross-arm speed table — {a.model}", ""]
    sl = next((arms[x].get("slice") for x in order if arms[x].get("slice")), "?")
    st = next((arms[x].get("slice_type") for x in order if arms[x].get("slice_type")), "?")
    L += [f"Slice: **{sl}** ({st}). All arms run SEQUENTIALLY on this one slice, one process at a time "
          f"(PROTOCOL phase 3). Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `src/accel4bit_speed_oneslice.py table` from `{a.outdir}/<arm>.json`.",
          "Protocol: single-stream = 512-token prompt -> 128 generated tokens, bs=1, greedy; batched = 256 requests 512->128, max concurrency 32 "
          "(llama.cpp: one static batch of 32, see caveats).", ""]
    L += ["## Single-stream (bs=1, 512 -> 128)", "",
          "| arm | runtime / kernel | TTFT ms | prefill tok/s | **decode tok/s (like-for-like)** | e2e latency s (512->128) | e2e gen tok/s | peak GPU mem MiB | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in order:
        j = arms[x]
        ss = j.get("single_stream") or {}
        pm = (j.get("peak_mem_MiB") or {}).get("max_over_all_stages")
        L.append(f"| {ARM_LABEL.get(x, x)} | {(j.get('kernel') or {}).get('summary', 'pending')} | {f(ss.get('ttft_ms'))} | {f(ss.get('prefill_tok_s'), 0)} | "
                 f"**{f(ss.get('decode_tok_s'))}** | {f(ss.get('e2e_latency_s'), 3)} | {f(ss.get('e2e_gen_tok_s'))} | {f(pm, 0)} | {j.get('status')} |")
    L += ["", "## Batched throughput (256 x 512->128, max concurrency 32)", "",
          "| arm | req/s | total tok/s (in+out) | **output tok/s** | elapsed s | note |", "|---|---|---|---|---|---|"]
    for x in order:
        j = arms[x]
        bt = j.get("batched") or {}
        note = "vLLM continuous batching, 256 requests" if not x.startswith("gguf") else \
            f"llama-batched-bench single batch B=32 (decode {f(bt.get('decode_tok_s_at_B32'))} tok/s at B=32, prefill {f(bt.get('prefill_tok_s_at_B32'), 0)} tok/s); B=8 / B=1 rows in JSON"
        L.append(f"| {ARM_LABEL.get(x, x)} | {f(bt.get('req_s'), 2)} | {f(bt.get('total_tok_s'), 0)} | **{f(bt.get('output_tok_s'), 0)}** | {f(bt.get('elapsed_s'))} | {note} |")
    L += ["", "## llama.cpp native numbers (for reference; NOT like-for-like with vLLM)", "",
          "| arm | llama-bench pp512 tok/s | llama-bench tg128 tok/s (decode from EMPTY context) | forced-cuBLAS pp512 | forced-cuBLAS tg128 | MMQ/cuBLAS prefill speedup |",
          "|---|---|---|---|---|---|"]
    for x in order:
        if not x.startswith("gguf"):
            continue
        j = arms[x]
        ss = j.get("single_stream") or {}
        mk = ((j.get("kernel") or {}).get("mmq_vs_forced_cublas") or {})
        L.append(f"| {ARM_LABEL.get(x, x)} | {f(ss.get('llama_bench_pp512_tok_s'), 0)} ± {f(ss.get('llama_bench_pp512_std'), 0)} | "
                 f"{f(ss.get('llama_bench_tg128_tok_s'))} ± {f(ss.get('llama_bench_tg128_std'))} | {f(mk.get('pp512_cublas_tok_s'), 0)} | "
                 f"{f(mk.get('tg128_cublas_tok_s'))} | {f(mk.get('pp_speedup_mmq_over_cublas'), 2)}x |")
    L += ["", "## Kernel dispatch evidence", ""]
    for x in order:
        j = arms[x]
        k = j.get("kernel") or {}
        if not k:
            L.append(f"- **{x}**: pending")
            continue
        L.append(f"- **{x}**: {k.get('summary')}")
        for ln in (k.get("lines") or [])[:6]:
            L.append(f"    - `{ln}`")
        if k.get("cuda_init"):
            L.append(f"    - `{k['cuda_init']}`")
    L += ["", "## Runtime / artifact provenance", "", "| arm | runtime | artifact | artifact GB | date |", "|---|---|---|---|---|"]
    for x in order:
        j = arms[x]
        rt = j.get("runtime") or {}
        if rt.get("name") == "vLLM":
            v = rt.get("versions") or {}
            r = f"vLLM {v.get('vllm')} / torch {v.get('torch')} / compressed-tensors {v.get('compressed_tensors')} ({rt.get('venv')})"
        elif rt.get("name") == "llama.cpp":
            b = rt.get("build") or {}
            r = f"llama.cpp {b.get('build_commit')} (b{b.get('build_number')}), CUDA 13.0 build sm_120a"
        else:
            r = "pending"
        art = j.get("artifact") or {}
        gb = art.get("bytes")
        L.append(f"| {x} | {r} | `{art.get('path')}` | {f(gb / 1e9, 2) if gb else 'pending'} | {j.get('date')} |")
    L += ["", "## Measurement caveats (read before comparing columns)", "",
          "1. **vLLM vs llama.cpp measure differently.** `vllm bench latency` is the END-TO-END wall time of one 512->128 request (prefill + 128 decode "
          "steps + scheduler/sampler overhead); `llama-bench tg128` is pure decode of 128 tokens starting from an EMPTY KV cache (no prompt), and "
          "`pp512` is prompt processing alone. The like-for-like columns above therefore use, for vLLM, `src/accel4bit_ttft.py` "
          "(decode tok/s = 127 / (e2e - TTFT), TTFT = max_tokens=1 request on the same prompt) and, for llama.cpp, the `llama-batched-bench` "
          "B=1 row (pp512 then tg128 inside one request: S_TG = decode after the 512-token prompt, T_PP = TTFT). Both are single-process, "
          "greedy, bs=1, same 512/128 shape, same slice.",
          "2. **vLLM throughput `tokens_per_second` counts input+output tokens** (256 x 640 = 163,840 tokens); the output tok/s column is "
          "requests_per_second x 128. llama-batched-bench runs ONE static batch of 32 sequences (not 256 continuously-batched requests); its "
          "`S t/s` also counts pp+tg tokens; output tok/s = 32x128/T. The two batched columns are the closest available shapes, not identical workloads.",
          "3. **Peak GPU memory** for vLLM is dominated by the KV-cache pre-allocation (`gpu_memory_utilization=0.85` of the slice) and is NOT the "
          "model's footprint; use the artifact GB / engine-log weight size for footprint. llama.cpp allocates only weights + requested KV.",
          "4. Kernel dispatch is taken from vLLM's kernel-selection log lines (INFO) in the latency/TTFT runs of THIS pass; for llama.cpp, "
          "MMQ use is proven by the forced-cuBLAS counterfactual build (`GGML_CUDA_FORCE_CUBLAS` is compile-time at commit 74a7c897).",
          "5. Per-arm DEVELOPMENT speed numbers in `results/accel4bit/<model>/<arm>/` were taken on 1g.24gb slices (46 SMs) and are NOT comparable "
          "to this table (94 SMs); only this table is cross-arm comparable.",
          "6. Missing arms show `pending` (artifact not yet produced, or the run failed — see `<arm>.json`.reason / `.status`).", ""]
    open(os.path.join(a.outdir, "SPEED_TABLE.md"), "w").write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vllm", "gguf", "pending", "table"])
    ap.add_argument("--arm"); ap.add_argument("--model"); ap.add_argument("--slice"); ap.add_argument("--slice-type", default="2g.48gb")
    ap.add_argument("--outdir", required=True); ap.add_argument("--artifact")
    ap.add_argument("--latency-json"); ap.add_argument("--ttft-json"); ap.add_argument("--throughput-json")
    ap.add_argument("--latency-log"); ap.add_argument("--ttft-log"); ap.add_argument("--throughput-log")
    ap.add_argument("--peak-samples"); ap.add_argument("--python"); ap.add_argument("--extra-note", default="")
    ap.add_argument("--status"); ap.add_argument("--reason", default="")
    ap.add_argument("--bench-json"); ap.add_argument("--bench-cublas-json"); ap.add_argument("--batched-log"); ap.add_argument("--bench-stderr")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    {"vllm": cmd_vllm, "gguf": cmd_gguf, "pending": cmd_pending, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
