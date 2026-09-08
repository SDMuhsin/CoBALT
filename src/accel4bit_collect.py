"""accel4bit PHASE 3 collector: aggregate every arm's quality JSONs, bpw JSONs and the ONE-SLICE speed JSONs into
results/accel4bit/BASELINE.md (+ BASELINE.json with the same numbers).

Usage: python src/accel4bit_collect.py [--models gemma-3-4b medgemma-27b] [--out results/accel4bit/BASELINE.md]
Only reads files; never launches anything. Missing cells are printed as "pending" (never dropped).

Per-arm file layouts handled (all under results/accel4bit/<model>/<arm>/):
  ref_bf16 / ref_fp8 : eval_summary.json {"tasks": {task: {"metrics": {...}}}}   or eval_<task>.json (raw lm_eval: results[task])
  gptq               : eval_summary.json {task: {...}}                            or eval_<task>.json {"results": {...}}
  nvfp4              : eval_<task>.json {"results": {...}} / summary.json {"eval": {task: {...}}}
  awq                : eval_<task>.json (raw lm_eval results file)
  gguf               : eval_summary.json / eval_<task>.json (own) and eval_summary_<tag>.json / eval_<task>_<tag>.json (unsloth prebuilts),
                       ppl_native_<tag>.log ("Final estimate: PPL = x +/- y")
  bpw                : bpw.json (bpw_quantized_linears) / quant_summary[_<tag>].json (bpw_linear) / safetensors header (fp8) / 16 (bf16)
  speed              : <model>/speed_oneslice/<arm>.json written by src/accel4bit_speed_oneslice.py
"""
import argparse
import glob
import json
import os
import re
import struct
import subprocess
import time

ROOT = "/workspace/PTQResearch"
RESROOT = f"{ROOT}/results/accel4bit"
ARTROOT = "/scratch/root/PTQResearch/accel4bit_models"
TASKS = ["wikitext", "arc_easy", "medqa_4options", "pubmedqa", "medmcqa"]

# arm id -> (results subdir, quality tag suffix or None, artifact subpath, label)
ARMS = [
    ("bf16", "ref_bf16", None, "text_bf16", "bf16 reference"),
    ("ref_fp8", "ref_fp8", None, "ref_fp8", "FP8-dynamic W8A8 reference"),
    ("gptq", "gptq", None, "gptq", "A gptq: GPTQ W4A16 sym g128 -> vLLM gptq_marlin"),
    ("awq", "awq", None, "awq", "D awq: AWQ W4A16 asym g128 -> vLLM awq_marlin"),
    ("nvfp4", "nvfp4", None, "nvfp4", "C nvfp4: NVFP4 W4A4 -> vLLM CUTLASS FP4"),
    ("gguf", "gguf", "own", "gguf/<name>-Q4_K_M.gguf", "B gguf: own imatrix Q4_K_M -> llama.cpp MMQ"),
    ("gguf_unsloth_q4km", "gguf", "unsloth_q4km", "gguf/unsloth_prebuilt/medgemma-27b-text-it-Q4_K_M.gguf", "B' unsloth prebuilt Q4_K_M -> llama.cpp MMQ"),
    ("gguf_unsloth_udq4kxl", "gguf", "unsloth_udq4kxl", "gguf/unsloth_prebuilt/medgemma-27b-text-it-UD-Q4_K_XL.gguf", "B'' unsloth prebuilt UD-Q4_K_XL -> llama.cpp MMQ"),
]
GGUF_NAME = {"gemma-3-4b": "gemma-3-4b-it", "medgemma-27b": "medgemma-27b-text-it"}
MODEL_LABEL = {"gemma-3-4b": "gemma-3-4b (SMOKE: unsloth/gemma-3-4b-it, text model only, 34 layers, 4.55 B params)",
               "medgemma-27b": "medgemma-27b (TARGET: unsloth/medgemma-27b-text-it b780610, Gemma3ForCausalLM, 62 layers, 27.0 B params)"}


# ----------------------------------------------------------------------------------------------------- helpers
def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def metric_block(j, task):
    """Find the lm_eval metric dict for `task` inside any of the layouts listed in the module docstring."""
    if not isinstance(j, dict):
        return None
    if "results" in j and isinstance(j["results"], dict):
        r = j["results"]
        if task in r and isinstance(r[task], dict):
            return r[task]
        if any(k.endswith(",none") for k in r):
            return r
    if "tasks" in j and isinstance(j["tasks"], dict) and task in j["tasks"]:
        t = j["tasks"][task]
        return t.get("metrics") if isinstance(t, dict) else None
    if "eval" in j and isinstance(j["eval"], dict) and task in j["eval"]:
        return j["eval"][task]
    if task in j and isinstance(j[task], dict) and any(k.endswith(",none") for k in j[task]):
        return j[task]
    return None


def task_metrics(resdir, task, tag=None):
    """Return (metrics dict or None, source file)."""
    cands = []
    if tag == "own":      # gguf arm writes untagged files for its own GGUF (eval_summary.json / eval_<task>.json), older runs used _own
        cands += [f"{resdir}/eval_summary.json", f"{resdir}/eval_{task}.json", f"{resdir}/eval_{task}_own.json"]
    elif tag:             # unsloth prebuilts: eval_summary_<tag>.json / eval_<task>_<tag>.json
        cands += [f"{resdir}/eval_summary_{tag}.json", f"{resdir}/eval_{task}_{tag}.json"]
    else:
        cands += [f"{resdir}/eval_summary.json", f"{resdir}/eval_{task}.json", f"{resdir}/summary.json"]
    for c in cands:
        j = jload(c)
        m = metric_block(j, task)
        if m:
            return m, c
    return None, None


def n_samples(resdir, task, tag=None):
    if tag == "own":
        cl = [f"{resdir}/eval_{task}.json", f"{resdir}/eval_summary.json", f"{resdir}/eval_{task}_own.json"]
    elif tag:
        cl = [f"{resdir}/eval_{task}_{tag}.json", f"{resdir}/eval_summary_{tag}.json"]
    else:
        cl = [f"{resdir}/eval_{task}.json", f"{resdir}/eval_summary.json"]
    for c in cl:
        j = jload(c)
        if not j:
            continue
        ns = j.get("n-samples") or j.get("n_samples")
        if isinstance(ns, dict):
            if task in ns and isinstance(ns[task], dict):
                return ns[task].get("effective")
            if "effective" in ns:
                return ns["effective"]
        m = metric_block(j, task)
        if m and "sample_len" in m:
            return m["sample_len"]
        if "tasks" in j and task in j["tasks"]:
            return (j["tasks"][task].get("metrics") or {}).get("sample_len")
    return None


def native_ppl(resdir, tag):
    p = f"{resdir}/ppl_native_{tag}.log"
    if not os.path.exists(p):
        return None
    for line in open(p, errors="replace"):
        m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def safetensors_headers(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update(h)
    return out


def fp8_bpw(d):
    """bits per weight of the FP8 linears from safetensors headers: (weight bytes + scale bytes) / numel(weight)."""
    try:
        h = safetensors_headers(d)
    except Exception:
        return None
    nbytes = 0
    numel = 0
    for k, v in h.items():
        if v.get("dtype") == "F8_E4M3" and k.endswith(".weight"):
            s, e = v["data_offsets"]
            nbytes += e - s
            n = 1
            for x in v["shape"]:
                n *= x
            numel += n
            sc = h.get(k + "_scale") or h.get(k.replace(".weight", ".weight_scale"))
            if sc:
                s, e = sc["data_offsets"]
                nbytes += e - s
    return (8.0 * nbytes / numel) if numel else None


def arm_bpw(model, arm, resdir, tag):
    if arm == "bf16":
        return 16.0, "bf16 (no quantization)"
    if arm == "ref_fp8":
        b = fp8_bpw(f"{ARTROOT}/{model}/ref_fp8")
        return b, "safetensors header: F8_E4M3 weights + per-channel scales"
    if arm.startswith("gguf"):
        p = f"{resdir}/quant_summary.json" if tag == "own" else f"{resdir}/quant_summary_{tag}.json"
        j = jload(p)
        return (j or {}).get("bpw_linear"), f"{os.path.basename(p)}: bits/elements of attn_*/ffn_* tensors (Q4_K_M mix: Q4_K + Q6_K for attn_v/ffn_down subset)"
    j = jload(f"{resdir}/bpw.json")
    return (j or {}).get("bpw_quantized_linears"), "bpw.json: 8*bytes(packed weight + scales [+ zero points / global scales]) / numel over quantized Linears"


def artifact_path(model, arm, sub):
    if arm == "bf16" and model == "medgemma-27b":   # bf16 27B = the HF snapshot itself (no extraction needed)
        return "/scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65"
    if arm == "gguf":
        return f"{ARTROOT}/{model}/gguf/{GGUF_NAME[model]}-Q4_K_M.gguf"
    return f"{ARTROOT}/{model}/{sub}"


def artifact_bytes(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    if os.path.isdir(p):
        tot = 0
        for root, _, files in os.walk(p, followlinks=True):
            for f in files:
                if f.endswith((".safetensors", ".gguf")):
                    tot += os.path.getsize(os.path.join(root, f))
        return tot or None
    return None


def fmt(x, nd=3, pending="pending"):
    if x is None:
        return pending
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def pts(q, ref, nd=1):
    if q is None or ref is None:
        return ""
    return f" ({(q - ref) * 100:+.{nd}f} pt)"


def dppl(q, ref):
    if q is None or ref is None:
        return ""
    return f" ({q - ref:+.2f}, {100 * (q / ref - 1):+.1f}%)"


# ----------------------------------------------------------------------------------------------------- quality
def quality_row(model, arm, sub, tag):
    resdir = f"{RESROOT}/{model}/{sub}"
    row = {"arm": arm, "resdir": resdir, "tasks": {}, "n": {}, "source": {}}
    for t in TASKS:
        m, src = task_metrics(resdir, t, tag)
        row["tasks"][t] = m
        row["source"][t] = src
        row["n"][t] = n_samples(resdir, t, tag) if m else None
    row["native_ppl"] = native_ppl(resdir, tag) if arm.startswith("gguf") else None
    row["bpw"], row["bpw_src"] = arm_bpw(model, arm, resdir, tag)
    return row


def g(m, k):
    return (m or {}).get(k)


def quality_table(model, rows):
    ref = next((r for r in rows if r["arm"] == "bf16"), None)
    rt = ref["tasks"] if ref else {}
    L = ["| arm | wikitext word-PPL (lm_eval) | llama-perplexity wiki.test.raw c2048 (GGUF only, native) | arc_easy acc / acc_norm | medqa_4options acc | pubmedqa acc | medmcqa@1000 acc | eff. bpw (quantized linears) | artifact GB |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["tasks"]
        w = g(t["wikitext"], "word_perplexity,none")
        ae, aen = g(t["arc_easy"], "acc,none"), g(t["arc_easy"], "acc_norm,none")
        mq, pq, mm = g(t["medqa_4options"], "acc,none"), g(t["pubmedqa"], "acc,none"), g(t["medmcqa"], "acc,none")
        rw, rae, raen = g(rt.get("wikitext"), "word_perplexity,none"), g(rt.get("arc_easy"), "acc,none"), g(rt.get("arc_easy"), "acc_norm,none")
        rmq, rpq, rmm = g(rt.get("medqa_4options"), "acc,none"), g(rt.get("pubmedqa"), "acc,none"), g(rt.get("medmcqa"), "acc,none")
        is_ref = r["arm"] == "bf16"
        cell_w = fmt(w) + ("" if is_ref else dppl(w, rw))
        cell_ae = (f"{fmt(ae, 4)} / {fmt(aen, 4)}" if ae is not None else "pending") + ("" if is_ref else (pts(ae, rae) + pts(aen, raen)).replace(") (", " / "))
        cell_mq = fmt(mq, 4) + ("" if is_ref else pts(mq, rmq))
        cell_pq = fmt(pq, 4) + ("" if is_ref else pts(pq, rpq))
        cell_mm = fmt(mm, 4) + ("" if is_ref else pts(mm, rmm))
        npl = r["native_ppl"]
        cell_npl = (f"{npl[0]:.4f} ± {npl[1]:.4f}" if npl else ("pending" if r["arm"].startswith("gguf") else "n/a"))
        ab = r.get("artifact_bytes")
        L.append(f"| {r['arm']} | {cell_w} | {cell_npl} | {cell_ae} | {cell_mq} | {cell_pq} | {cell_mm} | {fmt(r['bpw'], 3)} | {fmt(ab / 1e9, 2) if ab else 'pending'} |")
    L.append("")
    L.append("Deltas in parentheses are vs the bf16 row of the SAME model (accuracy in percentage points; PPL absolute and relative). "
             "lm_eval 0.4.13, seed 1234, 0-shot task defaults, `--include_path scripts/accel4bit_lmeval_tasks` (shared pubmedqa parquet yaml, 500 docs); "
             "medmcqa `--limit 1000`. n per task: " + ", ".join(f"{t}={next((r['n'][t] for r in rows if r['n'][t]), 'pending')}" for t in TASKS) + ".")
    return L


# ----------------------------------------------------------------------------------------------------- speed
def speed_rows(model):
    d = f"{RESROOT}/{model}/speed_oneslice"
    out = {}
    for p in glob.glob(f"{d}/*.json"):
        j = jload(p)
        if j and "arm" in j and "status" in j:
            out[j["arm"]] = j
    return out


def sf(x, nd=1):
    return "pending" if x is None else (x if isinstance(x, str) else f"{x:,.{nd}f}")


def speed_table(model, sp, arms):
    L = ["| arm | status | TTFT ms | prefill tok/s | **decode tok/s (like-for-like)** | e2e latency s (512->128) | batched output tok/s (256x512->128, conc. 32) | batched total tok/s (in+out) | peak GPU mem MiB | kernel dispatched |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        j = sp.get(arm)
        if not j:
            L.append(f"| {arm} | pending | pending | pending | pending | pending | pending | pending | pending | pending |")
            continue
        ss = j.get("single_stream") or {}
        bt = j.get("batched") or {}
        pm = (j.get("peak_mem_MiB") or {}).get("max_over_all_stages")
        k = (j.get("kernel") or {}).get("summary", "pending")
        if arm.startswith("gguf") and bt.get("output_tok_s") is not None:
            bt_note = f"{sf(bt.get('output_tok_s'), 0)} (single batch B=32)"
        else:
            bt_note = sf(bt.get("output_tok_s"), 0)
        L.append(f"| {arm} | {j.get('status')} | {sf(ss.get('ttft_ms'))} | {sf(ss.get('prefill_tok_s'), 0)} | **{sf(ss.get('decode_tok_s'))}** | {sf(ss.get('e2e_latency_s'), 3)} | "
                 f"{bt_note} | {sf(bt.get('total_tok_s'), 0)} | {sf(pm, 0)} | {k} |")
    sl = next((j.get("slice") for j in sp.values() if j.get("slice")), None)
    L.append("")
    L.append(f"Slice: {sl or 'pending'} (2g.48gb, 94 SMs) — every row measured sequentially on this ONE slice by `scripts/run_accel4bit_speed_oneslice.sh {model} <MIG>`; "
             f"details, raw JSON and the llama.cpp native (`llama-bench`) numbers in `results/accel4bit/{model}/speed_oneslice/SPEED_TABLE.md`. "
             "Like-for-like decode = tokens 2..128 after the 512-token prompt (vLLM: `src/accel4bit_ttft.py`; llama.cpp: `llama-batched-bench` B=1 row). "
             "vLLM's batched `tokens_per_second` counts input+output tokens; output tok/s = req/s x 128. Peak memory for vLLM = the 0.85 KV-cache pre-allocation, not the footprint.")
    ref_arm = "bf16" if model == "gemma-3-4b" else "ref_fp8"
    rj = sp.get(ref_arm)
    if rj and (rj.get("single_stream") or {}).get("decode_tok_s"):
        rd = rj["single_stream"]["decode_tok_s"]
        rp = rj["single_stream"].get("prefill_tok_s")
        ro = (rj.get("batched") or {}).get("output_tok_s")
        L.append("")
        L.append(f"Speedup vs the {ref_arm} reference on the same slice (decode / prefill / batched output): " + "; ".join(
            f"{a}: {sf((sp[a]['single_stream'] or {}).get('decode_tok_s', 0) / rd, 2)}x / "
            f"{sf(((sp[a]['single_stream'] or {}).get('prefill_tok_s') or 0) / rp, 2) if rp else 'n/a'} / "
            f"{sf(((sp[a].get('batched') or {}).get('output_tok_s') or 0) / ro, 2) if ro else 'n/a'}"
            for a in arms if a in sp and a != ref_arm and (sp[a].get("single_stream") or {}).get("decode_tok_s")) + ".")
    return L


# ----------------------------------------------------------------------------------------------------- provenance
def git_head(d):
    try:
        return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()[:12]
    except Exception:
        return "?"


def provenance(model, sp):
    art = f"{ARTROOT}/{model}"
    res = f"{RESROOT}/{model}"
    gmeta = jload(f"{art}/gptq/accel4bit_quant_meta.json") or {}
    nmeta = jload(f"{art}/nvfp4/accel4bit_quant_meta.json") or {}
    ameta = jload(f"{art}/awq/accel4bit_awq_recipe.json") or {}
    llcpp = git_head(f"{ROOT}/temp/llama.cpp")
    lc_build = ((sp.get("gguf") or {}).get("runtime") or {}).get("build") or {}
    fp8cfg = jload(f"{art}/ref_fp8/config.json") or {}
    qc = fp8cfg.get("quantization_config") or {}
    imx = None
    p = f"{res}/gguf/imatrix.log"
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            m = re.search(r"computing over (\d+) chunks, n_ctx=(\d+)", line)
            if m:
                imx = f"{m.group(1)} chunks of {m.group(2)} tokens"
    v = lambda d, k: d.get("versions", {}).get(k) if "versions" in d else d.get(k)  # noqa
    rows = [
        ("bf16", "unsloth/gemma-3-4b-it snapshot bf46152c (multimodal) -> TEXT model extracted by `src/accel4bit_extract_text.py` (444/444 tensors torch.equal; fp32 logits max|diff| 0)" if model == "gemma-3-4b"
                 else "unsloth/medgemma-27b-text-it snapshot b780610b (bf16, 54.02 GB); quality via vLLM PREFETCH LAYER OFFLOAD (31/62 layers in pinned host RAM, enforce_eager) because it does not fit a slice",
         "none (16-bit)", "n/a", "vLLM 0.28.0 (bf16 cuBLAS GEMM, FLASH_ATTN)", "env_accel_ref (vllm 0.28.0, lm_eval 0.4.13, transformers 5.16.1, compressed-tensors 0.17.0)",
         f"`scripts/run_accel4bit_ref.sh {model} MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0 " + ("extract_text bf16_quality bf16_speed`" if model == "gemma-3-4b" else "bf16_quality`  (REF27_OFFLOAD=prefetch)")),
        ("ref_fp8", ("turnio/medgemma-27b-text-it-FP8-Dynamic snapshot 84294e5b (third-party, llmcompressor 0.13.0; verified = per-channel FP8 RTN of the same base weights as the unsloth bf16 mirror)" if model == "medgemma-27b" else "n/a for the 4b smoke (no FP8 reference produced)"),
         f"QuantizationModifier FP8_DYNAMIC: weights FP8 E4M3 per-channel static, activations per-token dynamic, ignore [lm_head]" if qc else "pending",
         "none (RTN, data-free)", "vLLM 0.28.0 CutlassFP8ScaledMMLinearKernel (W8A8 CUTLASS scaled_mm SM120)", "env_accel_ref",
         f"`scripts/run_accel4bit_ref.sh medgemma-27b MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0 fp8_download fp8_quality fp8_speed`"),
        ("gptq", f"llm-compressor {v(gmeta, 'llmcompressor') or 'pending'} (pip; clone in temp/llm-compressor) `GPTQModifier`; src/accel4bit_gptq_quant.py",
         f"targets Linear, scheme W4A16 (int4 sym, group 128), dampening_frac {gmeta.get('recipe', {}).get('dampening_frac', 'pending')}, actorder {gmeta.get('recipe', {}).get('actorder', 'pending')}, block_size 128, ignore lm_head/embed/vision; quant {fmt(gmeta.get('quant_minutes'), 1)} min",
         f"{(gmeta.get('calibration') or {}).get('dataset', 'pending')} {(gmeta.get('calibration') or {}).get('split', '')}, {(gmeta.get('calibration') or {}).get('num_samples', '?')} x {(gmeta.get('calibration') or {}).get('max_seq_length', '?')} tok, chat-templated, shuffle={(gmeta.get('calibration') or {}).get('shuffle')}, seed {(gmeta.get('calibration') or {}).get('seed')}",
         "vLLM 0.28.0 MarlinLinearKernel for CompressedTensorsWNA16 (gptq_marlin, sm_120)",
         f"env_accel_gptq_arm (= env_accel_vllm site-packages + lm_eval 0.4.13: vllm 0.28.0, torch {v(gmeta, 'torch')}, transformers {v(gmeta, 'transformers')}, compressed-tensors {v(gmeta, 'compressed_tensors')})",
         f"`STAGES=\"quant sanity quality speed bpw\" bash scripts/run_accel4bit_gptq.sh {model} <MIG>`"),
        ("awq", f"llm-compressor {(ameta.get('versions') or {}).get('llmcompressor', 'pending')} `AWQModifier` + `QuantizationModifier`; src/accel4bit_awq_quant.py",
         f"scheme {ameta.get('args', {}).get('scheme', 'pending')} (int4 ASYM, group 128, zero-points), duo_scaling {ameta.get('args', {}).get('duo_scaling')}, n_grid {ameta.get('args', {}).get('n_grid')}, "
         f"mappings input_layernorm->qkv, v_proj->o_proj, pre_ffn_norm->gate/up, up->down (v->o SKIPPED on every layer: GQA shape mismatch, see caveats); ignore lm_head/vision; oneshot {fmt(ameta.get('oneshot_minutes'), 1)} min",
         f"{(ameta.get('calib') or {}).get('dataset', 'pending')} {(ameta.get('calib') or {}).get('split', '')}, 512 x 2048, shuffle_seed {(ameta.get('calib') or {}).get('shuffle_seed')}, {(ameta.get('calib') or {}).get('tokens', '?')} tokens",
         "vLLM 0.28.0 MarlinLinearKernel for CompressedTensorsWNA16 (awq_marlin path: uint4 + zp, sm_120)",
         f"env_accel_awq (overlay on env_accel_vllm: vllm 0.28.0, torch {(ameta.get('versions') or {}).get('torch')}, transformers {(ameta.get('versions') or {}).get('transformers')}, compressed-tensors {(ameta.get('versions') or {}).get('compressed_tensors')})",
         f"`bash scripts/run_accel4bit_awq.sh {model} <MIG> quant,gen,eval,speed,bpw`"),
        ("nvfp4", f"llm-compressor {nmeta.get('llmcompressor', 'pending')} `QuantizationModifier(scheme=NVFP4)`; src/accel4bit_nvfp4_quant.py",
         f"scheme NVFP4 = W4A4 FP4 E2M1, 16-element blocks with FP8 E4M3 block scales + FP32 global scales (weights static, activations dynamic per-token/block); ignore lm_head/embed/vision; wall {fmt((nmeta.get('wall_s') or 0) / 60, 1) if nmeta.get('wall_s') else 'pending'} min",
         f"{nmeta.get('dataset', 'pending')}, {nmeta.get('num_samples', '?')} x {nmeta.get('max_seq_len', '?')} tok, seed {nmeta.get('seed')} (llm-compressor default shuffle)",
         "vLLM 0.28.0 FlashInferCutlassNvFp4LinearKernel (CUTLASS SM120 block-scaled FP4 GEMM; torch.profiler: cutlass MainloopSm120TmaWarpSpecializedBlockScaled kernels)",
         f"env_accel_nvfp4 (own venv: vllm 0.28.0, torch {nmeta.get('torch')}, transformers {nmeta.get('transformers')}, compressed-tensors {nmeta.get('compressed_tensors')}, flashinfer 0.6.16.post3)",
         f"`bash scripts/run_accel4bit_nvfp4.sh {model} <MIG> all`"),
        ("gguf", f"llama.cpp {llcpp} (b{lc_build.get('build_number', '10820')}) `convert_hf_to_gguf.py --outtype bf16` -> `llama-imatrix` -> `llama-quantize --imatrix ... Q4_K_M`",
         f"Q4_K_M with imatrix (Q4_K for most attn/ffn, Q6_K for the attn_v/ffn_down subset + token_embd per the Q4_K_M mix, F32 norms); imatrix `-c 512 --parse-special`, {imx or 'pending'}",
         "same ultrachat 512x2048 text dumped by src/accel4bit_dump_calib.py (`results/accel4bit/calib_ultrachat_512x2048.txt`, shuffle(seed=42).select(512), 584,705 HF tokens)",
         "llama.cpp CUDA MMQ (int8 tensor-core mul_mat_q) prefill + MMVQ decode, sm_120a, CUDA 13.0 toolkit; proven vs forced-cuBLAS build",
         "env_accel_llamacpp (python glue only: gguf 0.19.0, llama-cpp-python 0.3.35 for lm_eval scoring, lm_eval 0.4.13); binaries temp/llama.cpp/build/bin",
         f"`bash scripts/run_accel4bit_gguf.sh {model} all`  (MIG via env MIG=<uuid>)"),
    ]
    if model == "medgemma-27b":
        rows += [("gguf_unsloth_q4km", "unsloth/medgemma-27b-text-it-GGUF prebuilt Q4_K_M (third-party, downloaded)", "unsloth Q4_K_M (their imatrix/recipe, not ours)", "unsloth's (unknown)",
                  "llama.cpp CUDA MMQ (same binaries)", "as gguf", "`TAG=unsloth_q4km bash scripts/run_accel4bit_gguf.sh medgemma-27b ppl bench batched eval`"),
                 ("gguf_unsloth_udq4kxl", "unsloth/medgemma-27b-text-it-GGUF prebuilt UD-Q4_K_XL (unsloth dynamic 2.0, third-party)", "unsloth UD-Q4_K_XL (mixed per-layer types)", "unsloth's (unknown)",
                  "llama.cpp CUDA MMQ (same binaries)", "as gguf", "`TAG=unsloth_udq4kxl bash scripts/run_accel4bit_gguf.sh medgemma-27b ppl bench batched eval`")]
    L = ["| arm | repo / version / quantizer | exact quant config | calibration | runtime + kernel | venv | repro command |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return L


# ----------------------------------------------------------------------------------------------------- caveats
def caveats(model, rows, sp):
    C = []
    C.append("**MIG slices.** Every number above was produced on MIG slices of an RTX PRO 6000 Blackwell (SM 12.0), never a full GPU: "
             "quality/quantization per arm on its own 1g.24gb slice (46 SMs) — gptq MIG-bef7a31e, gguf MIG-475dbed1, nvfp4 quant MIG-6ec7b494 / eval MIG-71f8f46a, "
             "awq MIG-12daede5, references MIG-1d47bdbe (2g.48gb, 94 SMs). The cross-arm SPEED table is the only cross-arm-comparable speed data: "
             "one slice (2g.48gb), sequential, one process at a time. Per-arm development speed numbers in each arm's NOTES.md are 1g.24gb numbers and must not be mixed with it. "
             "MIG was never reconfigured.")
    C.append("**No bf16 speed for medgemma-27b.** bf16 weights are 54.02 GB > 47.38 GiB visible on the largest slice; the bf16 27B QUALITY row comes from vLLM prefetch "
             "layer offload (same weights/kernels, only placement differs, `enforce_eager`), and the 27B SPEED reference is the FP8-dynamic W8A8 checkpoint on the same slice. "
             "4-bit 27B speedups are therefore vs FP8, not vs bf16; the gemma-3-4b block gives the vs-bf16 picture.")
    C.append("**Cross-slice numerics.** Quality tasks are loglikelihood-based and were bit-exact-seeded, but this project has measured that the same vLLM cell can differ across "
             "MIG slices (memory `mig-slice-numerics-confound`: ~9% PPL / 0.4 pt across slices in another suite). Quality rows here were NOT all produced on the same slice "
             "(see the slice list above); treat sub-1-pt accuracy differences and sub-2% PPL differences between arms as within noise.")
    C.append("**Calibration subsets differ across arms although the pool is the same.** All calibrated arms use HuggingFaceH4/ultrachat_200k train_sft, 512 x 2048, chat-templated, seed 42, "
             "but gptq took the FIRST 512 rows unshuffled (`shuffle_calibration_samples=False`), nvfp4 used llm-compressor's default (shuffled) selection, awq recorded `shuffle_seed 42` "
             "(575,810 tokens) and the GGUF imatrix text is `shuffle(seed=42).select(range(512))` (584,705 HF tokens, 1167 llama.cpp chunks of 512). Same distribution, not the identical 512 rows. "
             "The FP8 reference is data-free RTN.")
    C.append("**AWQ GQA mapping skip.** llm-compressor's AWQ `v_proj -> o_proj` smoothing mapping is skipped on EVERY layer for Gemma-3 (`_set_resolved_mappings | WARNING - 34 mappings were "
             "skipped due to incompatible shapes` on the 4b, 62 on the 27b): v_proj output (kv_heads x head_dim) != o_proj input (heads x head_dim) under GQA. o_proj is therefore "
             "quantized without AWQ scaling (plain RTN asym g128 after the other three mappings). The AWQ artifact also keeps the multimodal `Gemma3ForConditionalGeneration` config "
             "(vision tower excluded from quantization, `--limit-mm-per-prompt {\"image\":0}` at serve time), so its artifact GB includes the bf16 vision tower and its eval loads it.")
    C.append("**GGUF quality backend deviates from PROTOCOL (labelled).** llama-server has no prompt logprobs (`echo` unsupported), and llama-cpp-python's server takes ~12 s/request, so the "
             "gguf lm_eval numbers come from `src/accel4bit_lmeval_gguf.py`, an in-process lm_eval TemplateLM over llama-cpp-python 0.3.35 (its vendored llama.cpp, NOT the b10820 build used "
             "for speed) scoring the SAME Q4_K_M file. Validation vs HF fp32 on gemma-3-1b: argmax agreement 35/40 on arc_easy. `llama-perplexity` (native, c=2048 on wiki.test.raw) is a "
             "different PPL definition from lm_eval word-PPL and is reported in its own column only.")
    C.append("**Eval-config drift between arms (all lm_eval 0.4.13, seed 1234, task defaults, medmcqa --limit 1000, same pubmedqa yaml):** gptq/ref/nvfp4 pass `enable_prefix_caching=False` "
             "to vLLM, the awq runner does not (and runs one lm_eval process per task); nvfp4's vLLM model_args omit `seed=1234` (lm_eval `--seed 1234` is set). None of these affect "
             "loglikelihood scoring in principle; they are listed for completeness. The 27B bf16 reference ran eager (no torch.compile) under offload.")
    C.append("**Speed measurement caveats** (details in each model's SPEED_TABLE.md): vLLM `bench latency` is end-to-end 512->128 (prefill + 128 decode steps + scheduler); llama-bench `tg128` is "
             "decode from an EMPTY context and `pp512` prefill only — the like-for-like columns use `src/accel4bit_ttft.py` (vLLM) and `llama-batched-bench` B=1 (llama.cpp). vLLM throughput "
             "`tokens_per_second` counts input+output tokens; llama-batched-bench B=32 is one static batch, not 256 continuously-batched requests. vLLM peak memory is the 0.85 KV pre-allocation. "
             "vLLM runs use `VLLM_USE_FLASHINFER_SAMPLER=0` (torch sampler; greedy) and torch.compile + CUDA graphs (FULL_AND_PIECEWISE); llama.cpp uses FA + CUDA graphs.")
    C.append("**Third-party artifacts.** ref_fp8 (turnio) and the unsloth prebuilt GGUFs were not produced here; provenance of the FP8 one was verified against our bf16 weights "
             "(bit-identical embeddings/norms; per-channel RTN rel-err matches), the unsloth GGUF recipes are unknown (reported as extra data points only).")
    bp = {r["arm"]: r.get("bpw") for r in rows}
    C.append("**Quality is NOT bit-matched across the 4-bit arms.** Effective bits per weight on the quantized linears differ: "
             + ", ".join(f"{a} {bp[a]:.3f}" for a in ("gptq", "awq", "nvfp4", "gguf") if bp.get(a)) +
             " (gptq = int4 + bf16 group-128 scale; awq adds int4 zero-points; nvfp4 = FP4 + FP8 16-element block scales; gguf Q4_K_M mixes Q4_K with Q6_K "
             "for the attn_v/ffn_down subset). A higher-bpw arm buys quality with bits; compare quality per bpw, and compare speed per artifact GB, "
             "rather than reading the quality table as a like-for-like 4.0-bit comparison.")
    if model == "medgemma-27b":
        C.append("**gguf 27B batched throughput: only the one-slice number is valid.** The gguf arm's own `llama-batched-bench -npl 1,8,32` on its 24 GB "
                 "development slice OOM'd at B=32 (16.5 GB weights + 32 x 640-token KV + compute buffers exceed the 1g.24gb slice), so the arm's NOTES carry "
                 "no valid B=32 point; the batched gguf column above comes from the 48 GB one-slice pass (`speed_oneslice/gguf*.json`, B=1/8/32 all completed).")
    # dynamic: pending status
    pend = [r["arm"] for r in rows if any(r["tasks"][t] is None for t in TASKS)]
    if pend:
        C.append(f"**Pending quality cells ({model}):** " + ", ".join(f"{a} [" + ", ".join(t for t in TASKS if next(r for r in rows if r['arm'] == a)['tasks'][t] is None) + "]" for a in pend) + ".")
    spend = [a for a, j in sp.items() if j.get("status") != "OK"]
    if spend:
        C.append(f"**Speed rows not OK ({model}):** " + ", ".join(f"{a} = {sp[a].get('status')} ({sp[a].get('reason', '')})" for a in spend) + ".")
    # per-arm NOTES flags
    for arm_dir in ["gptq", "gguf", "awq", "nvfp4", "ref_bf16", "ref_fp8"]:
        p = f"{RESROOT}/{model}/{arm_dir}/NOTES.md"
        if not os.path.exists(p):
            C.append(f"NOTES.md for `{model}/{arm_dir}` does not exist yet (arm still running or notes pending).")
    return C


# ----------------------------------------------------------------------------------------------------- main
def build(models, out):
    doc = ["# accel4bit — BASELINE: 4-bit PTQ of MedGemma-27B that is actually GPU-accelerated", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `src/accel4bit_collect.py` from `results/accel4bit/<model>/<arm>/*` and "
           "`results/accel4bit/<model>/speed_oneslice/*.json`. Per-arm recipe, protocol and environment are in the "
           "\"(c) Recipe / provenance\" section of each model below. Cells marked `pending` have no data yet "
           "(arm not finished) and are never silently dropped.",
           "",
           "Arms: **A gptq** (GPTQ W4A16 g128 -> vLLM gptq_marlin), **B gguf** (llama.cpp imatrix Q4_K_M -> CUDA MMQ), **C nvfp4** (NVFP4 W4A4 -> vLLM CUTLASS SM120 FP4), "
           "optional **D awq** (AWQ W4A16 asym g128 -> vLLM awq_marlin); references **bf16** (quality; speed only where it fits) and **FP8-dynamic** (27B speed+quality on the 48 GB slice). "
           "Hardware: RTX PRO 6000 Blackwell Server Edition (SM 12.0) in MIG mode — six 1g.24gb + one 2g.48gb slices; MIG never reconfigured.", ""]
    allj = {}
    for model in models:
        arms = [a for a in ARMS if not (model == "gemma-3-4b" and a[0] in ("ref_fp8", "gguf_unsloth_q4km", "gguf_unsloth_udq4kxl"))]
        rows = []
        for arm, sub, tag, asub, label in arms:
            r = quality_row(model, arm, sub, tag)
            ap = artifact_path(model, arm, asub)
            r["artifact"] = ap
            r["artifact_bytes"] = artifact_bytes(ap)
            r["label"] = label
            rows.append(r)
        sp = speed_rows(model)
        doc += [f"## {MODEL_LABEL.get(model, model)}", ""]
        doc += ["### (a) Quality", ""] + quality_table(model, rows) + [""]
        doc += ["### (b) Speed — ONE slice (MIG-1d47bdbe, 2g.48gb), all arms sequential", ""] + speed_table(model, sp, [a[0] for a in arms]) + [""]
        doc += ["### (c) Recipe / provenance", ""] + provenance(model, sp) + [""]
        doc += ["### (d) Caveats (honest)", ""] + [f"{i + 1}. {c}" for i, c in enumerate(caveats(model, rows, sp))] + [""]
        allj[model] = {"quality": [{k: v for k, v in r.items() if k != "source"} for r in rows], "speed": sp}
    doc += ["## Files", "",
            "- Per-arm raw results, logs and NOTES: `results/accel4bit/<model>/{gptq,gguf,nvfp4,awq,ref_bf16,ref_fp8}/`",
            "- One-slice speed pass: `results/accel4bit/<model>/speed_oneslice/` (`<arm>.json`, `SPEED_TABLE.md`, raw `vllm bench` / `llama-bench` JSON + logs)",
            "- Runners: `scripts/run_accel4bit_{gptq,gguf,nvfp4,awq,ref}.sh`, `scripts/run_accel4bit_speed_oneslice.sh`; glue `src/accel4bit_*.py`",
            "- Artifacts: `/scratch/root/PTQResearch/accel4bit_models/<model>/<arm>/`; venvs `/scratch/root/PTQResearch/env_accel_*`; llama.cpp `temp/llama.cpp` (build/bin, build-cublas/bin)",
            "- Regenerate this file: `python src/accel4bit_collect.py`", ""]
    open(out, "w").write("\n".join(doc))
    json.dump(allj, open(out.replace(".md", ".json"), "w"), indent=1, default=str)
    print(f"wrote {out} and {out.replace('.md', '.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["medgemma-27b", "gemma-3-4b"])
    ap.add_argument("--out", default=f"{RESROOT}/BASELINE.md")
    a = ap.parse_args()
    build(a.models, a.out)


if __name__ == "__main__":
    main()
