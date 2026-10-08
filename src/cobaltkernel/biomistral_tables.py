#!/usr/bin/env python
"""Build results/biomistral/QUALITY.md and SPEED.md from the measured JSON records (never
hand-transcribed).  Rows are only printed when their file exists; missing arms are listed
as `pending`, never silently dropped.

  python src/cobaltkernel/biomistral_tables.py
"""
import glob, json, os, time

ROOT = "/workspace/PTQResearch"
MODEL = "biomistral-7b"
Q_DIRS = {   # arm label -> directory holding eval_<task>.json
    "bf16 (reference)": f"{ROOT}/results/cobaltkernel/{MODEL}/ref_bf16",
    "llama.cpp Q4_K_M (imatrix)": f"{ROOT}/results/accel4bit/{MODEL}/gguf",
    "CoBALT DENSE4 (control)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_dense4_e4",
    "CoBALT 16:32 b4": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk32_b4_e4",
    "CoBALT 16:32 b4 +o_proj": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk32_b4_ohyb_e4",
    "CoBALT 16:32 b6 +o_proj@4 (matched bytes)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk32_b6_ohyb4_e4",
    # --- decomposition / recipe levers (diagnostic rows) ---
    "DIAG prune-only: 16:32 mask, 8-bit codes": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk32_b8_e4",
    "DIAG quant-only: sp0, 4-bit": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_sp0_b4_e4",
    "DIAG 16:32 b4, beta=0": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk32_b4_beta0_e4",
    "DIAG 16:32 b4, beta=1": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk32_b4_beta1_e4",
    "CoBALT mixed A: 16:32 qkv+gateup, DENSE o_proj+down (3.54 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixA_b4_e4",
    "CoBALT mixed B: 16:32 gateup only, DENSE attention+down (3.65 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixB_b4_e4",
    # --- quantizer-gap study 2026-10-07 (full grid: results/biomistral/qgap/TABLE.md) ---
    "llama.cpp Q4_K_M (MEDICAL imatrix; matched-lever control)": f"{ROOT}/results/accel4bit/{MODEL}/gguf_medcal",
    "QGAP sp0 4-bit, gptq+clip (4.19 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_sp0_b4_gptq_aw_e4",
    "QGAP sp0 4-bit, gptq+clip, MEDICAL calib (4.19 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_sp0_b4_gptq_aw_medcal_e4",
    "QGAP mixed B, gptq+clip (3.65 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixB_b4_gptq_aw_e4",
    "QGAP mixed B, gptq+clip, MEDICAL calib draw 1 (3.65 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixB_b4_gptq_aw_medcal_e4",
    "QGAP mixed B, gptq+clip, MEDICAL calib draw 2 (3.65 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixB_b4_gptq_aw_medcal_off144_e4",
    "QGAP mixed B, gptq+clip, MEDICAL calib draw 3 (3.65 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixB_b4_gptq_aw_medcal_off288_e4",
    "QGAP mixed B, gptq+clip+act-order, MEDICAL calib (3.65 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixB_b4_gptq_aw_ao_medcal_e4",
    "QGAP mixed A, gptq+clip, MEDICAL calib draw 1 (3.53 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixA_b4_gptq_aw_medcal_e4",
    "QGAP mixed A, gptq+clip, MEDICAL calib draw 2 (3.53 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixA_b4_gptq_aw_medcal_off144_e4",
    "QGAP mixed A, gptq+clip, MEDICAL calib draw 3 (3.53 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_mixA_b4_gptq_aw_medcal_off288_e4",
    "QGAP 20:32 all matrices, gptq+clip (3.69 bpw; NO kernel layout yet)": f"{ROOT}/results/cobaltkernel/{MODEL}/cobalt_blk2032_b4_gptq_aw_e4",
}
S_FILES = {  # arm label -> speed json
    "llama.cpp Q4_K_M (MMQ/MMVQ)": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice/gguf.json",
    "CoBALT DENSE4 (control, start)": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_dense4.json",
    "CoBALT 16:32 b4": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4.json",
    "CoBALT DENSE4 (control, end)": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_dense4_end.json",
    "CoBALT mixed A (3.53 bpw; v2 protocol, TTFT n/c)": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixA.json",
    "CoBALT quant-only sp0 DENSE4 (4.19 bpw)": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_sp0_dense4.json",
    "CoBALT mixed B (3.65 bpw; v2 protocol, TTFT n/c; pinned)": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixB.json",
    # --- quiet-window passes for the mixed-B MEDICAL artifact (2026-10-07; v3 prefill via PF_BLK1632_QKV/_D) ---
    "llama.cpp Q4_K_M, mixB-medical window pass 1": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_mixBmed1/gguf.json",
    "CoBALT mixed B MEDICAL (3.65 bpw; v3 prefill; shipped knobs) pass 1": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixB_medcal_p1.json",
    "CoBALT mixed B RTN artifact, v3 prefill control": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixB_v3_p1.json",
    "llama.cpp Q4_K_M, mixB-medical window pass 2": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_mixBmed2/gguf.json",
    "CoBALT mixed B MEDICAL (3.65 bpw; v3 prefill; shipped knobs) pass 2": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixB_medcal_p2.json",
    # --- same-window passes for the mixed-D MEDICAL artifact (2026-10-07 round 2; host load 19-24 from another tenant, in-kernel numbers host-immune) ---
    "llama.cpp Q4_K_M, mixD-medical window pass 1": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_mixD1/gguf.json",
    "CoBALT mixed D MEDICAL (3.46 bpw: DENSE4 down_proj only; shipped knobs + MMA) pass 1": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixD_medcal_mixD1.json",
    "CoBALT 16:32 b4 shipped kernel, same-window CONTROL": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_mixD_ctl.json",
    "llama.cpp Q4_K_M, mixD-medical window pass 2": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_mixD2/gguf.json",
    "CoBALT mixed D MEDICAL (3.46 bpw: DENSE4 down_proj only; shipped knobs + MMA) pass 2": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_mixD_medcal_mixD2.json",
    # --- same-window passes for the all-16:32 reconB artifact (medical + block reconstruction; byte-identical layout to the shipped kernel) ---
    "llama.cpp Q4_K_M, reconB window pass 1": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_reconB1/gguf.json",
    "CoBALT 16:32 b4 reconB (3.19 bpw; medical + block recon; shipped knobs + MMA) pass 1": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_reconB_reconB1.json",
    "CoBALT 16:32 b4 shipped kernel, reconB-window CONTROL": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_reconB_ctl.json",
    "llama.cpp Q4_K_M, reconB window pass 2": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_reconB2/gguf.json",
    "CoBALT 16:32 b4 reconB (3.19 bpw; medical + block recon; shipped knobs + MMA) pass 2": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_reconB_reconB2.json",
    # --- round 3 (2026-10-08): the FAST build on the shipped artifact (tensor-core decode attention, LUT4 selector+mask table,
    #     one-iteration lookahead in the row decoder); same bytes as cobalt_b1632_4; quiet-window passes with the shipped control ---
    "llama.cpp Q4_K_M, fast window pass 1": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_fast1/gguf.json",
    "CoBALT 16:32 b4 FAST (3.19 bpw; TC attention + LUT4 + lookahead; same artifact) pass 1": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_fast_fast1.json",
    "CoBALT 16:32 b4 shipped kernel, fast-window CONTROL": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_fast_ctl.json",
    "llama.cpp Q4_K_M, fast window pass 2": f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_fast2/gguf.json",
    "CoBALT 16:32 b4 FAST (3.19 bpw; TC attention + LUT4 + lookahead; same artifact) pass 2": f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_fast_fast2.json",
}
TASKS = [("wikitext", "word_perplexity,none"), ("arc_easy", "acc,none"),
         ("medqa_4options", "acc,none"), ("pubmedqa", "acc,none"), ("medmcqa", "acc,none")]


def metric(d, task, key):
    p = os.path.join(d, f"eval_{task}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    r = j.get("results", j)
    return r.get(task, {}).get(key)


def quality():
    rows, base = [], {}
    for t, k in TASKS:
        base[t] = metric(Q_DIRS["bf16 (reference)"], t, k)
    out = ["# BioMistral-7B — quality (lm_eval 0.4.13, seed 1234, 0-shot, medmcqa --limit 1000)", "",
           f"Generated {time.strftime('%F %T')} by `src/cobaltkernel/biomistral_tables.py` from "
           "`results/cobaltkernel/biomistral-7b/<tag>/eval_*.json` and `results/accel4bit/biomistral-7b/gguf/eval_*.json`.",
           "CoBALT rows = fakequant HF checkpoint under vLLM (kernel-independent, embed/lm_head DENSE4). "
           "`Med avg` = mean(MedQA, PubMedQA, MedMCQA) in points; deltas vs the bf16 row.", "",
           "| arm | wiki PPL | ARC-e | MedQA | PubMedQA | MedMCQA@1000 | Med avg | d vs bf16 |",
           "|---|---|---|---|---|---|---|---|"]
    bavg = None
    if all(base[t] is not None for t in ("medqa_4options", "pubmedqa", "medmcqa")):
        bavg = 100 * (base["medqa_4options"] + base["pubmedqa"] + base["medmcqa"]) / 3
    for arm, d in Q_DIRS.items():
        vals = {t: metric(d, t, k) for t, k in TASKS}
        if all(v is None for v in vals.values()):
            out.append(f"| {arm} | pending | | | | | | |")
            continue
        def f(t, pct=True):
            v = vals[t]
            if v is None:
                return "pending"
            if t == "wikitext":
                return f"{v:.3f}" + (f" ({100*(v/base[t]-1):+.1f}%)" if base[t] else "")
            return f"{v:.4f}"
        avg = None
        if all(vals[t] is not None for t in ("medqa_4options", "pubmedqa", "medmcqa")):
            avg = 100 * (vals["medqa_4options"] + vals["pubmedqa"] + vals["medmcqa"]) / 3
        out.append(f"| {arm} | {f('wikitext')} | {f('arc_easy')} | {f('medqa_4options')} | {f('pubmedqa')} | "
                   f"{f('medmcqa')} | {'%.2f' % avg if avg is not None else 'pending'} | "
                   f"{('%+.2f' % (avg - bavg)) if (avg is not None and bavg is not None) else ''} |")
    return "\n".join(out) + "\n"


def speed():
    out = ["# BioMistral-7B — speed, ONE slice, sequential (512->128, bs=1 greedy; batched as noted)", "",
           f"Generated {time.strftime('%F %T')} by `src/cobaltkernel/biomistral_tables.py` from the speed JSONs. "
           "DENSE4 control repeated at both ends of the run (REQUIREMENTS G7).", "",
           "| arm | slice | bpw | artifact GB | decode tok/s | x Q4_K_M | TTFT ms | prefill tok/s | batched out tok/s (config) | peak MiB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    q4 = None
    recs = {}
    for arm, p in S_FILES.items():
        if os.path.exists(p):
            recs[arm] = json.load(open(p))
    g = recs.get("llama.cpp Q4_K_M (MMQ/MMVQ)")
    if g:
        q4 = (g.get("single_stream") or {}).get("decode_tok_s")
    for arm, p in S_FILES.items():
        r = recs.get(arm)
        if not r:
            out.append(f"| {arm} | pending | | | | | | | | |")
            continue
        ss = r.get("single_stream") or {}
        b = r.get("batched") or {}
        art = r.get("artifact") or {}
        gb = (art.get("bytes") or 0) / 1e9
        dec = ss.get("decode_tok_s")
        ratio = f"{dec/q4:.3f}" if (dec and q4) else ""
        bpw = art.get("bpw") or ""
        if not bpw and art.get("layout"):
            bpw = {"DENSE4": "4.1875", "BLK1632_4": "3.1875", "BLK1632_6": "4.1875"}.get(art["layout"], "")
            if "mixed A" in arm:
                bpw = "3.5337"
            if "mixed B" in arm:
                bpw = "3.6490"
            if "mixB-medical window" in arm:
                bpw = "4.797"
        peak = r.get("peak_gpu_mem_mib") or (r.get("peak_mem_MiB") or {}).get("max_over_all_stages")
        out.append(f"| {arm} | {r.get('slice_type','?')} {str(r.get('slice','?'))[:12]} | {bpw} | {gb:.2f} | "
                   f"{dec if dec is not None else 'pending'} | {ratio} | {ss.get('ttft_ms','')} | {ss.get('prefill_tok_s','')} | "
                   f"{b.get('output_tok_s','')} ({str(b.get('config',''))[:40]}) | {peak} |")
    return "\n".join(out) + "\n"


def speed_repeats():
    """Interleaved repeats (pinned + niced) -> per-arm list of decode tok/s and the median."""
    import statistics
    R = f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice"
    arms = {
        "CoBALT DENSE4 (control)": [f"{R}/cobalt_dense4_r{i}.json" for i in (2, 3, 4)],
        "CoBALT 16:32 b4": [f"{R}/cobalt_b1632_4_r{i}.json" for i in (2, 3, 5)],
        "llama.cpp Q4_K_M": [f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_r2/gguf.json"],
    }
    out = ["", "## Interleaved repeats (nice -5, taskset 0-7 for EVERY arm; A-B-A-B-A order; host load ~20 from another tenant)", "",
           "| arm | decode tok/s per repeat | median | median x Q4_K_M |", "|---|---|---|---|"]
    med = {}
    for arm, files in arms.items():
        vals = []
        for f in files:
            if os.path.exists(f):
                v = (json.load(open(f)).get("single_stream") or {}).get("decode_tok_s")
                if v:
                    vals.append(round(float(v), 2))
        med[arm] = statistics.median(vals) if vals else None
    q = med.get("llama.cpp Q4_K_M")
    for arm, files in arms.items():
        vals = [round(float((json.load(open(f)).get("single_stream") or {}).get("decode_tok_s") or 0), 2)
                for f in files if os.path.exists(f)]
        m = med[arm]
        out.append(f"| {arm} | {', '.join(str(v) for v in vals) or 'pending'} | {m if m else ''} | "
                   f"{(m/q):.3f} |" if (m and q) else f"| {arm} | {', '.join(str(v) for v in vals) or 'pending'} | {m if m else ''} | |")
    out.append("")
    out.append("Run-1 (unpinned, host-contended) numbers are kept in the table above for the record; the repeats are the numbers of record.")
    return "\n".join(out) + "\n"


def knob_table():
    """Every knob-sweep record on the 2g slice (clock64 step; M=1-only builds where noted)."""
    import glob, re
    R = f"{ROOT}/results/cobaltkernel/{MODEL}/knob_sweep"
    rows = []
    for f in sorted(glob.glob(f"{R}/*.json")):
        b = os.path.basename(f)[:-5]
        if b.startswith(("attn_test", "cs_test", "fuse_test", "spill_test")) or "INVALID" in b or "SPILL_" in b:
            continue                       # 1g-slice records: not comparable, see LOG
        j = json.load(open(f)); pb = j.get("phase_breakdown_ms", {}); ss = j.get("single_stream", {})
        if j.get("status") != "OK" or "total_measured" not in pb:
            continue
        rows.append((pb["total_measured"], b, pb, ss, j.get("sm_clock_mhz_for_stamps")))
    rows.sort()
    out = ["", "## Knob sweep records (2g slice MIG-1d47bdbe, BioMistral 16:32 b4; sorted by GPU-side kernel step)", "",
           "| config | step ms | qkv | attn | o_proj | gate\|up | down | misc | in-kernel tok/s | SM MHz |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for t, b, pb, ss, mhz in rows:
        out.append(f"| `{b}` | **{t:.3f}** | {pb.get('qkv',0):.3f} | {pb.get('attn',0):.3f} | {pb.get('o_proj',0):.3f} | "
                   f"{pb.get('gateup_geglu',0):.3f} | {pb.get('down_proj',0):.3f} | {pb.get('norms_and_misc',0):.3f} | "
                   f"{ss.get('decode_tok_s_inkernel','')} | {mhz or '2430 (assumed)'} |")
    out.append("")
    out.append("All records here use the MEASURED SM clock for the stamps. The first-sweep records (converted at an assumed "
               "2.43 GHz, ~3% optimistic) live in `knob_sweep/sweep1_assumedclock/` and are excluded from selection.")
    return "\n".join(out) + "\n"


def final_protocol():
    """Quiet-window final protocol records (speed_oneslice_idle*, *_idle{1,2}.json)."""
    import glob
    out = ["", "## FINAL quiet-window protocol (host load < 4, every arm nice -5 / cores 0-7, interleaved)", "",
           "| arm | run | decode tok/s (host loop) | in-kernel tok/s | kernel step ms | TTFT ms | prefill tok/s |",
           "|---|---|---|---|---|---|---|"]
    n = 0
    for i in (1, 2):
        g = f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_idle{i}/gguf.json"
        if os.path.exists(g):
            ss = json.load(open(g))["single_stream"]; n += 1
            out.append(f"| llama.cpp Q4_K_M | {i} | {ss.get('decode_tok_s')} | — | — | {ss.get('ttft_ms')} | {ss.get('prefill_tok_s')} |")
        for arm, lab in (("cobalt_b1632_4", "CoBALT 16:32 b4 (tuned)"), ("cobalt_dense4", "CoBALT DENSE4 control")):
            f = f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/{arm}_idle{i}.json"
            if os.path.exists(f):
                j = json.load(open(f)); ss = j["single_stream"]; pb = j.get("phase_breakdown_ms", {}); n += 1
                out.append(f"| {lab} | {i} | {ss.get('decode_tok_s')} | {ss.get('decode_tok_s_inkernel')} | {pb.get('total_measured','')} | {ss.get('ttft_ms')} | {ss.get('prefill_tok_s')} |")
    if n == 0:
        out.append("| pending | | | | | | |")
    out.append("")
    out.append("Pass 1 ran at host load ≈2 (number of record); pass 2 ran at host load ≈18 (another tenant's job started) — "
               "llama.cpp's host-side loop degrades under load, the in-kernel path does not.")
    # ---- second protocol: the tensor-core decoder on the L2-resident phases (qkv, o_proj), same window rules
    out += ["", "## Protocol 2: + tensor-core decode on qkv / o_proj (`COBALT_BLK1632_MMA=1`, phases qkv|o, PF 1)", "",
            "Passes 1–2: `COBALT_MMA_UPW=4` (unit sizing default at the time); passes 3–4: `COBALT_MMA_UPW=2` (the shipped value); "
            "passes 5–6 (if present): `COBALT_MMA_UPW=2 COBALT_MMA_UPW_O=1`. Same window rules as above (host load annotated in the log).", "",
            "| arm | run | decode tok/s (host loop) | in-kernel tok/s | kernel step ms | TTFT ms | prefill tok/s |",
            "|---|---|---|---|---|---|---|"]
    n2 = 0
    for i in (1, 2, 3, 4, 5, 6):
        g = f"{ROOT}/results/accel4bit/{MODEL}/speed_oneslice_mma{i}/gguf.json"
        if os.path.exists(g):
            j = json.load(open(g)); ss = j["single_stream"]; n2 += 1
            out.append(f"| llama.cpp Q4_K_M | {i} | {ss.get('decode_tok_s')} | — | — | {ss.get('ttft_ms')} | {ss.get('prefill_tok_s')} |")
        f = f"{ROOT}/results/cobaltkernel/{MODEL}/speed_oneslice/cobalt_b1632_4_mma{i}.json"
        if os.path.exists(f):
            j = json.load(open(f)); ss = j["single_stream"]; pb = j.get("phase_breakdown_ms", {}); n2 += 1
            out.append(f"| CoBALT 16:32 b4 (tuned + MMA qkv/o) | {i} | {ss.get('decode_tok_s')} | {ss.get('decode_tok_s_inkernel')} | {pb.get('total_measured','')} | {ss.get('ttft_ms')} | {ss.get('prefill_tok_s')} |")
    if n2 == 0:
        out.append("| pending | | | | | | |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    os.makedirs(f"{ROOT}/results/biomistral", exist_ok=True)
    open(f"{ROOT}/results/biomistral/QUALITY.md", "w").write(quality())
    open(f"{ROOT}/results/biomistral/SPEED.md", "w").write(speed() + speed_repeats() + final_protocol() + knob_table())
    print(open(f"{ROOT}/results/biomistral/QUALITY.md").read())
    print(open(f"{ROOT}/results/biomistral/SPEED.md").read())
