#!/usr/bin/env python
"""One table: task quality x computational metrics for every KERNEL arm we have on BioMistral-7B, vs llama.cpp Q4_K_M.
Generated from the speed JSONs (results/*/speed_oneslice/*.json) and the lm_eval JSONs; never hand-edited.
  python src/cobaltkernel/holistic_table.py  -> results/biomistral/HOLISTIC.md (and stdout)
"""
import json, os, time

ROOT = "/workspace/PTQResearch"
CK = f"{ROOT}/results/cobaltkernel/biomistral-7b"
AC = f"{ROOT}/results/accel4bit/biomistral-7b"
TASKS = [("wikitext", "word_perplexity,none"), ("arc_easy", "acc,none"),
         ("medqa_4options", "acc,none"), ("pubmedqa", "acc,none"), ("medmcqa", "acc,none")]

# arm -> (speed json [same-window run 1], quiet-window in-kernel json or None, bpw, quality dir, note)
ARMS = [
    ("llama.cpp Q4_K_M (MMQ/MMVQ, imatrix)", f"{AC}/speed_oneslice/gguf.json", None, 4.797, f"{AC}/gguf", "the bar"),
    ("CoBALT DENSE4, sp0.5 global mask", f"{CK}/speed_oneslice/cobalt_dense4.json", f"{CK}/speed_oneslice/cobalt_dense4_idle1.json", 4.1875, f"{CK}/cobalt_dense4_e4", "control"),
    ("CoBALT 16:32 b4 (shipped kernel: tuned + MMA qkv/o)", f"{CK}/speed_oneslice/cobalt_b1632_4.json", f"{CK}/speed_oneslice/cobalt_b1632_4_mma5.json", 3.1875, f"{CK}/cobalt_blk32_b4_e4", "speed headline"),
    ("CoBALT 16:32 b4, gptq+clip quantizer", None, "same as above", 3.1875, f"{CK}/cobalt_blk32_b4_gptq_aw_e4", "same bytes/layout -> speed inherited"),
    ("CoBALT quant-only sp0 DENSE4 (RTN)", f"{CK}/speed_oneslice/cobalt_sp0_dense4.json", f"{CK}/speed_oneslice/cobalt_dense4_idle1.json", 4.1875, f"{CK}/cobalt_sp0_b4_e4", ""),
    ("CoBALT sp0 DENSE4, gptq+clip + medical calib", None, "same as above", 4.1875, f"{CK}/cobalt_sp0_b4_gptq_aw_medcal_e4", "same bytes/layout -> speed inherited"),
    ("CoBALT mixed A (16:32 qkv+gate|up, DENSE4 o+down)", f"{CK}/speed_oneslice/cobalt_mixA.json", None, 3.5337, f"{CK}/cobalt_mixA_b4_e4", "v2 prefill protocol: TTFT n/c"),
    ("CoBALT mixed A, gptq+clip + medical (3-draw mean)", None, "same as above", 3.5337, ["cobalt_mixA_b4_gptq_aw_medcal_e4", "cobalt_mixA_b4_gptq_aw_medcal_off144_e4", "cobalt_mixA_b4_gptq_aw_medcal_off288_e4"], "speed inherited"),
    ("CoBALT mixed B (16:32 gate|up, DENSE4 attn+down)", f"{CK}/speed_oneslice/cobalt_mixB.json", None, 3.6490, f"{CK}/cobalt_mixB_b4_e4", "v2 prefill protocol: TTFT n/c"),
    ("CoBALT 16:32 b4 shipped kernel, re-measured as CONTROL in the mixD window (host load 19-24)", f"{CK}/speed_oneslice/cobalt_b1632_4_mixD_ctl.json", f"{CK}/speed_oneslice/cobalt_b1632_4_mixD_ctl.json", 3.1875, f"{CK}/cobalt_blk32_b4_e4", "same-window control for the mixD row"),
    ("CoBALT 16:32 b4, gptq+clip + medical + BLOCK RECONSTRUCTION (reconB) — PACKED (bytes identical to shipped), verified 32/32, own window", f"{CK}/speed_oneslice/cobalt_b1632_4_reconB_reconB1.json", f"{CK}/speed_oneslice/cobalt_b1632_4_reconB_reconB1.json", 3.1875, f"{CK}/cobalt_blk32_b4_gptq_aw_medcal_reconB_e4", "round-2 literal-layout row (draw 0); host load 22 in its window -> host-loop numbers jittery, in-kernel step 6.72 = control 6.71"),
    ("CoBALT 16:32 b4 shipped kernel, re-measured as CONTROL in the reconB window", f"{CK}/speed_oneslice/cobalt_b1632_4_reconB_ctl.json", f"{CK}/speed_oneslice/cobalt_b1632_4_reconB_ctl.json", 3.1875, f"{CK}/cobalt_blk32_b4_e4", "same-window control for the reconB row"),
    ("CoBALT mixed D (16:32 q/k/v/o/gate\\|up, DENSE4 down_proj), gptq+clip + medical (3-draw mean) — PACKED, verified 32/32, MMA knobs", f"{CK}/speed_oneslice/cobalt_mixD_medcal_mixD1.json", f"{CK}/speed_oneslice/cobalt_mixD_medcal_mixD1.json", 3.4567, ["cobalt_mixD_b4_gptq_aw_medcal_e4", "cobalt_mixD_b4_gptq_aw_medcal_off144_e4", "cobalt_mixD_b4_gptq_aw_medcal_off288_e4"], "round-2 headline; own window (host load 19-24) with llama.cpp + shipped-16:32 control; pass 2 in cobalt_mixD_medcal_mixD2.json"),
    ("CoBALT mixed B, gptq+clip + medical (3-draw mean) — PACKED, verified, v3 prefill", f"{CK}/speed_oneslice/cobalt_mixB_medcal_p1.json", f"{CK}/speed_oneslice/cobalt_mixB_medcal_p1.json", 3.6490, ["cobalt_mixB_b4_gptq_aw_medcal_e4", "cobalt_mixB_b4_gptq_aw_medcal_off144_e4", "cobalt_mixB_b4_gptq_aw_medcal_off288_e4"], "quantizer-gap headline; own quiet-window speed (pass 1), llama.cpp same window in speed_oneslice_mixBmed1"),
]
# llama.cpp measured in the SAME window as an arm's own passes (host-loop decode tok/s), keyed by a substring of the arm's speed json
Q4_MIXB_WINDOW = [f"{AC}/speed_oneslice_mixBmed1/gguf.json", f"{AC}/speed_oneslice_mixBmed2/gguf.json"]
Q4_WINDOWS = {"mixB_medcal": Q4_MIXB_WINDOW,
              "mixD_medcal": [f"{AC}/speed_oneslice_mixD1/gguf.json", f"{AC}/speed_oneslice_mixD2/gguf.json"],
              "b1632_4_mixD_ctl": [f"{AC}/speed_oneslice_mixD1/gguf.json", f"{AC}/speed_oneslice_mixD2/gguf.json"],
              "b1632_4_reconB_reconB": [f"{AC}/speed_oneslice_reconB1/gguf.json", f"{AC}/speed_oneslice_reconB2/gguf.json"],
              "b1632_4_reconB_ctl": [f"{AC}/speed_oneslice_reconB1/gguf.json", f"{AC}/speed_oneslice_reconB2/gguf.json"]}


def own_window(path):
    for k, w in Q4_WINDOWS.items():
        if k in (path or "") and os.path.exists(w[0]):
            return w
    return None
# quiet-window llama.cpp host-loop numbers of record (SPEED.md final protocol / protocol 2), tok/s
Q4_QUIET = 143.6


def metric(d, task, key):
    p = os.path.join(d, f"eval_{task}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p)); r = j.get("results", j)
    return r.get(task, {}).get(key)


def quality(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    vals = {}
    for t, k in TASKS:
        xs = [metric(d if d.startswith("/") else f"{CK}/{d}", t, k) for d in dirs]
        xs = [x for x in xs if x is not None]
        vals[t] = sum(xs) / len(xs) if xs else None
    return vals


def main():
    bf16 = quality(f"{CK}/ref_bf16")
    bavg = 100 * (bf16["medqa_4options"] + bf16["pubmedqa"] + bf16["medmcqa"]) / 3
    g = json.load(open(ARMS[0][1]))
    q4_dec = g["single_stream"]["decode_tok_s"]; q4_pre = g["single_stream"]["prefill_tok_s"]; q4_gb = g["artifact"]["bytes"] / 1e9
    q4_peak = g.get("peak_gpu_mem_mib") or (g.get("peak_mem_MiB") or {}).get("max_over_all_stages")
    q4q = quality(f"{AC}/gguf"); q4avg = 100 * (q4q["medqa_4options"] + q4q["pubmedqa"] + q4q["medmcqa"]) / 3
    hdr = ("| arm | bpw | GB (x Q4_K_M) | peak MiB | decode tok/s same-window run-1 (x Q4_K_M) | quiet-window in-kernel tok/s (x Q4_K_M 143.6) | "
           "prefill tok/s (x) | wiki PPL | ARC-e | MedQA | PubMedQA | MedMCQA@1000 | Med avg | d bf16 | d Q4_K_M | note |")
    out = ["# BioMistral-7B — holistic table: task quality x computational metrics, every kernel arm vs llama.cpp Q4_K_M", "",
           f"Generated {time.strftime('%F %T')} by `src/cobaltkernel/holistic_table.py`. Speed: ONE 2g.48gb slice (MIG-1d47bdbe), 512->128, "
           "bs=1 greedy; 'same-window run-1' = every arm measured in one host-contended window (host load ~20; relative numbers only); "
           "'quiet-window in-kernel' = the final protocol (host load <4, clock64 kernel step) where measured. Quality = fakequant under vLLM "
           f"(kernel-independent), bf16 Med avg {bavg:.2f}, Q4_K_M {q4avg:.2f}. bpw = kernel-layout accounting on decoder linears. "
           "Arms marked 'speed inherited' change only the quantizer/calibration, not bytes or layout, so the kernel row above applies.", "",
           hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for arm, sp, quiet, bpw, qdir, note in ARMS:
        r = json.load(open(sp)) if (sp and os.path.exists(sp)) else None
        if sp and not os.path.exists(sp):
            r = None; gb_s = dec_s = pre_s = peak_s = "pending"
        if r:
            ss = r["single_stream"]; gb = r["artifact"]["bytes"] / 1e9
            peak = r.get("peak_gpu_mem_mib") or (r.get("peak_mem_MiB") or {}).get("max_over_all_stages")
            dec = ss["decode_tok_s"]; pre = ss.get("prefill_tok_s")
            gb_s = f"{gb:.2f} ({gb/q4_gb:.2f}x)"
            if own_window(sp):
                # never in the contended run-1 window: its own passes are same-window records vs the same-window llama.cpp
                gq = json.load(open(own_window(sp)[0]))["single_stream"]
                dec_s = f"n/a run-1 (own-window host loop {dec:.1f} = {dec/gq['decode_tok_s']:.3f}x llama.cpp {gq['decode_tok_s']:.1f})"
                pre_s = f"{pre:.0f} ({pre/gq['prefill_tok_s']:.2f}x same-window llama.cpp {gq['prefill_tok_s']:.0f}; TTFT {ss['ttft_ms']:.0f} vs {gq['ttft_ms']:.0f} ms)"
            else:
                dec_s = f"{dec:.1f} ({dec/q4_dec:.2f}x)"
                pre_s = f"{pre:.0f} ({pre/q4_pre:.2f}x)" if (pre and pre > 1000) else "n/c (v2 protocol)"
            peak_s = f"{peak} ({peak/q4_peak:.2f}x)" if (peak and q4_peak) else str(peak)
        elif sp is None:
            gb_s = dec_s = pre_s = peak_s = "= row above"
        if quiet and quiet != "same as above" and os.path.exists(quiet):
            qr = json.load(open(quiet)); qk = qr["single_stream"].get("decode_tok_s_inkernel") or qr["single_stream"].get("decode_tok_s")
            ref = Q4_QUIET
            if own_window(quiet):
                ref = json.load(open(own_window(quiet)[0]))["single_stream"]["decode_tok_s"]
            quiet_s = f"{qk:.1f} ({qk/ref:.3f}x vs same-window llama.cpp {ref:.1f})"
        elif quiet == "same as above":
            quiet_s = "= row above"
        elif arm.startswith("llama.cpp"):
            quiet_s = f"{Q4_QUIET} host loop (1.000x)"
        else:
            quiet_s = "not in quiet protocol"
        v = quality(qdir)
        avg = 100 * (v["medqa_4options"] + v["pubmedqa"] + v["medmcqa"]) / 3
        ppl = f"{v['wikitext']:.3f} ({100*(v['wikitext']/bf16['wikitext']-1):+.1f}%)"
        out.append(f"| {arm} | {bpw} | {gb_s} | {peak_s} | {dec_s} | {quiet_s} | {pre_s} | {ppl} | {v['arc_easy']:.4f} | "
                   f"{v['medqa_4options']:.4f} | {v['pubmedqa']:.4f} | {v['medmcqa']:.4f} | {avg:.2f} | {avg-bavg:+.2f} | {avg-q4avg:+.2f} | {note} |")
    txt = "\n".join(out) + "\n"
    open(f"{ROOT}/results/biomistral/HOLISTIC.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
