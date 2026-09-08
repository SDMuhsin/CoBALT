"""cobaltkernel BASELINE collector: aggregate the accel4bit reference arms (bf16, ref_fp8, gptq,
awq, nvfp4, gguf) together with the CoBALT-kernel arms (cobalt_dense4, cobalt_sparse4x, ...) into
results/cobaltkernel/BASELINE.md (+ BASELINE.json), in the SAME table columns/formatting as
results/accel4bit/BASELINE.md, so the two sets of arms sit row-for-row. Reuses
src/accel4bit_collect.py's table-rendering functions verbatim (imported, never modified) for both
the reference rows AND the cobalt rows -- only the ROW-BUILDING for cobalt arms is new here.

Usage:
  python src/cobaltkernel/collect_baseline.py [--models gemma-3-4b medgemma-27b]
      [--arms-json results/cobaltkernel/arms.json] [--out results/cobaltkernel/BASELINE.md]

Only reads files; never launches anything. Missing cells render "pending" (never dropped) --
this must work today, before the 27B quality eval finishes and before any arm is kernel-packed.

Inputs:
  - Reference arms: results/accel4bit/BASELINE.json (quality + speed), produced by
    `python src/accel4bit_collect.py`. If that file (or a model's key in it) is missing, this
    script re-derives the reference rows the same way accel4bit_collect.py does (imported
    functions `quality_row`/`speed_rows`, applied to the same ARMS list) so the table is never
    silently short a reference arm.
  - CoBALT arms: `results/cobaltkernel/arms.json` maps arm id -> {model, layout, quality_tag,
    manifest_path, raw_manifest_path, speed_json, label, notes}:
      * quality_tag         -> results/cobaltkernel/<model>/<quality_tag>/eval_*.json (lm_eval
                                0.4.13 raw output, same accel4bit PROTOCOL; parsed with
                                accel4bit_collect.task_metrics/n_samples, tag=None layout).
                                Partial evals (only some eval_<task>.json written so far) are
                                handled: missing tasks render "pending", present ones render.
      * manifest_path       -> the PACKED CBK1 artifact manifest (src/cobaltkernel/pack_cobalt.py
                                output; see docs/FORMAT.md). Effective bpw and
                                artifact GB come from its top-level `bpw.bpw_packed` /
                                `bpw.packed_bytes`. Missing manifest -> bpw/artifact GB "pending".
      * raw_manifest_path   -> the pre-pack quantizer manifest (src/cobaltkernel/quantize_cobalt.py
                                output) -- used only for the provenance/recipe table (exact
                                quantizer config: sparsity, bits, beta, group_size, hull, calib).
      * speed_json          -> a one-slice speed JSON in the schema documented in
                                results/cobaltkernel/SPEED_JSON_SCHEMA.md (a superset of the
                                accel4bit speed_oneslice schema: same single_stream/batched/
                                peak_mem_MiB/kernel/slice/status fields, so it is fed straight
                                into accel4bit_collect.speed_table's `sp` dict). Missing/absent
                                file -> that arm's speed row is "pending" (accel4bit_collect
                                already renders missing sp entries as pending).

Adds:
  - (e) "Speedup vs Q4_K_M gguf and vs ref_fp8" section, one line per model, computed from the
    same speed dict; independent of, and in addition to, accel4bit_collect.speed_table's own
    "vs bf16/ref_fp8" line inside section (b).
  - (d) Caveats: the standard accel4bit-style caveats for the reference arms (reused verbatim),
    plus cobaltkernel-specific caveats, plus verbatim text from results/cobaltkernel/CAVEATS.md
    if that file exists (else a placeholder line saying it does not exist yet).
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/workspace/PTQResearch/src")
import accel4bit_collect as A4  # noqa: E402  (reuse table logic/formatting verbatim; never modified)

ROOT = "/workspace/PTQResearch"
RES_CK = f"{ROOT}/results/cobaltkernel"
RES_A4 = f"{ROOT}/results/accel4bit"
TASKS = A4.TASKS
REF_ARM_IDS = ["bf16", "ref_fp8", "gptq", "awq", "nvfp4", "gguf"]  # subset of A4.ARMS we pull in, in this order


# ----------------------------------------------------------------------------------------------------- reference rows
def reference_rows_and_speed(model):
    """Reference (accel4bit) quality rows + speed dict for `model`, preferring BASELINE.json
    (fast, no re-parsing) and falling back to re-deriving via accel4bit_collect functions for
    any arm/model missing from it, so a stale or absent BASELINE.json never silently drops a
    reference arm."""
    baseline = A4.jload(f"{RES_A4}/BASELINE.json") or {}
    have = {r["arm"]: r for r in baseline.get(model, {}).get("quality", [])}
    sp = dict(baseline.get(model, {}).get("speed", {}))
    rows = []
    a4_arms = {a[0]: a for a in A4.ARMS}
    for arm_id in REF_ARM_IDS:
        if model == "gemma-3-4b" and arm_id == "ref_fp8":
            continue  # accel4bit_collect.build() excludes ref_fp8 for the 4b smoke too
        if arm_id in have:
            rows.append(have[arm_id])
            continue
        if arm_id not in a4_arms:
            continue
        arm, sub, tag, asub, label = a4_arms[arm_id]
        r = A4.quality_row(model, arm, sub, tag)
        ap = A4.artifact_path(model, arm, asub)
        r["artifact"] = ap
        r["artifact_bytes"] = A4.artifact_bytes(ap)
        r["label"] = label
        rows.append(r)
    if not sp:
        sp = A4.speed_rows(model)
    return rows, sp


# ----------------------------------------------------------------------------------------------------- cobalt rows
def cobalt_quality_row(model, arm_id, cfg):
    tag = cfg.get("quality_tag")
    resdir = f"{RES_CK}/{model}/{tag}" if tag else None
    tasks, n, source = {}, {}, {}
    for t in TASKS:
        m, src = (None, None) if not resdir else A4.task_metrics(resdir, t, tag=None)
        tasks[t] = m
        source[t] = src
        n[t] = A4.n_samples(resdir, t, tag=None) if (resdir and m) else None

    manifest = A4.jload(cfg.get("manifest_path") or "")
    bpw, bpw_src, artifact_bytes = None, "pending (packed CBK1 artifact not built yet)", None
    if manifest:
        mb = manifest.get("bpw") or {}
        bpw = mb.get("bpw_packed") or mb.get(f"bpw_{(cfg.get('layout') or '').lower()}")
        artifact_bytes = mb.get("packed_bytes")
        if artifact_bytes is None:
            embed_bytes = (manifest.get("embed") or {}).get("bytes") or 0
            misc_bytes = 0
            mp = os.path.dirname(cfg["manifest_path"])
            for f in glob.glob(os.path.join(mp, "layer_*.bin")):
                misc_bytes += os.path.getsize(f)
            artifact_bytes = embed_bytes + misc_bytes or None
        bpw_src = f"{os.path.basename(cfg['manifest_path'])}: bpw.bpw_packed (layout {manifest.get('layout')}, weighted over all packed decoder matrices; embedding {manifest.get('embed_layout')} counted separately in artifact bytes)"
    elif cfg.get("bpw_override") is not None:
        # ADDITIVE (2026-09-07): an arm may be quality-measured before any packer/kernel path for
        # its layout exists (the mixed-layout o_proj hybrids). Only consulted when the PACKED manifest is
        # absent, so no packed arm's bpw/GB can change; `bpw_override_src` carries the provenance verbatim.
        bpw = cfg["bpw_override"]
        artifact_bytes = cfg.get("artifact_bytes_override")
        bpw_src = "NOT from a packed manifest -- " + (cfg.get("bpw_override_src") or "no source recorded")

    return {
        "arm": arm_id, "resdir": resdir, "tasks": tasks, "n": n, "source": source,
        "native_ppl": None, "bpw": bpw, "bpw_src": bpw_src,
        "artifact": cfg.get("manifest_path"), "artifact_bytes": artifact_bytes,
        "label": cfg.get("label", arm_id),
    }


def cobalt_speed_row(cfg):
    p = cfg.get("speed_json")
    if not p:
        return None
    full = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return A4.jload(full)


# ----------------------------------------------------------------------------------------------------- provenance (cobalt rows appended to A4's table)
def cobalt_provenance_rows(model, arms_cfg):
    rows = []
    for arm_id, cfg in arms_cfg.items():
        raw = A4.jload(cfg.get("raw_manifest_path") or "") or {}
        rc = raw.get("config", {})
        manifest = A4.jload(cfg.get("manifest_path") or "")
        pack_state = f"packed ({cfg.get('layout')})" if manifest else f"NOT PACKED YET (target layout {cfg.get('layout')})"
        if cfg.get("mixed_layout"):
            det = "; ".join(f"{k} = {v}" for k, v in (cfg.get("layout_detail") or {}).items())
            pack_state += " -- MIXED-LAYOUT artifact (not a single layout): " + (det or cfg.get("layout") or "")
        rows.append((
            arm_id,
            "src/cobaltkernel/quantize_cobalt.py (streamed layer-wise CoBALT reproduction of src/nosink.py) "
            "+ src/cobaltkernel/pack_cobalt.py -> custom CUDA megakernel; " + pack_state,
            f"sparsity {rc.get('sparsity', 'pending')}, bits {rc.get('bits', 'pending')}, beta {rc.get('beta', 'pending')}, group_size {rc.get('group_size', 'pending')}, "
            f"hull {rc.get('hull', 'pending')} (D2: survivor hull by default), damping_frac {rc.get('damping_frac', 'pending')}, method: {rc.get('method', 'pending')}",
            f"{rc.get('calib', 'pending')}, {rc.get('n_calib', '?')} x {rc.get('seq_len', '?')} tok, seq_inputs={rc.get('seq_inputs')} (D3: dense-model calibration); source: {rc.get('calib_source', 'pending')}",
            "custom persistent CUDA megakernel (src/cobaltkernel/csrc/megakernel.cu[h]), grid.sync phases per layer (D5); quality of record measured via the fakequant HF checkpoint under vLLM, kernel-independent (D6)",
            "env_accel_ref (fakequant-quality path) / cobaltkernel CUDA build (CUDA 13.0 toolkit, temp or in-tree csrc; speed path)",
            f"quality: `bash scripts/run_cobaltkernel_quality.sh {model} {cfg.get('quality_tag')}`; pack: `python src/cobaltkernel/pack_cobalt.py --raw {cfg.get('raw_manifest_path', 'pending').replace('/manifest.json', '')} --layout {cfg.get('layout')} --out <packed_dir>`",
        ))
    return rows


def append_cobalt_provenance(model, base_lines, arms_cfg):
    rows = cobalt_provenance_rows(model, arms_cfg)
    return base_lines + ["| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |" for r in rows]


# ----------------------------------------------------------------------------------------------------- speed-table footnotes
def speed_footnotes(sp, arms_cfg):
    """Additive: a footnote under section (b) for arms whose registry entry sets
    `prefill_is_decode_rate: true` (their TTFT/prefill/batched cells are NOT prefill/continuous-batching
    measurements). Renders nothing when no such arm has a speed row, so no existing arm's output changes."""
    # ADDITIVE (2026-09-07): also render the footnote for any arm carrying a `speed_note`,
    # not only for arms flagged `prefill_is_decode_rate`. Renders nothing for arms with neither key.
    flagged = [a for a, c in arms_cfg.items()
               if (c.get("prefill_is_decode_rate") or c.get("speed_note")) and a in sp]
    if not flagged:
        return []
    lines = [""]
    for a in flagged:
        note = arms_cfg[a].get("speed_note") or ""
        lines.append(f"> **HOW TO READ THE `{a}` ROW.** {note}")
    return lines


# ----------------------------------------------------------------------------------------------------- (e) speedup vs gguf / ref_fp8
def speedup_section(model, sp, arms_cfg=None):
    ref_ids = [a for a in ("gguf", "ref_fp8") if a in sp and (sp[a].get("single_stream") or {}).get("decode_tok_s")]
    lines = ["### (e) Speedup vs Q4_K_M gguf and vs ref_fp8 (same slice)", ""]
    if not ref_ids:
        lines.append("pending (no gguf/ref_fp8 speed row on this slice yet for this model).")
        return lines
    cobalt_ids = [a for a in sp if a.startswith("cobalt_")]
    if not cobalt_ids:
        lines.append("pending (no cobaltkernel speed_oneslice/*.json for this model yet -- see results/cobaltkernel/SPEED_JSON_SCHEMA.md).")
        return lines
    for arm in cobalt_ids:
        ss = sp[arm].get("single_stream") or {}
        if not ss.get("decode_tok_s"):
            lines.append(f"- **{arm}**: pending (speed row present but status={sp[arm].get('status')}).")
            continue
        parts = []
        for ref in ref_ids:
            rss = sp[ref]["single_stream"] or {}
            rd, rp = rss.get("decode_tok_s"), rss.get("prefill_tok_s")
            d = ss.get("decode_tok_s")
            pf = ss.get("prefill_tok_s")
            dstr = A4.sf(d / rd, 2) + "x" if (d and rd) else "n/a"
            if (arms_cfg or {}).get(arm, {}).get("prefill_is_decode_rate"):
                pstr = "n/a (this arm has no prefill measurement -- its prefill/TTFT cells are decode-rate numbers)"
            else:
                pstr = A4.sf(pf / rp, 2) + "x" if (pf and rp) else "n/a"
            parts.append(f"vs {ref}: decode {dstr} / prefill {pstr}")
        lines.append(f"- **{arm}**: " + "; ".join(parts) + ".")
    return lines


# ----------------------------------------------------------------------------------------------------- (d) caveats
def cobaltkernel_caveats(model, ref_rows, cobalt_rows, sp, arms_cfg):
    C = list(A4.caveats(model, ref_rows, {a: j for a, j in sp.items() if not a.startswith("cobalt_")}))
    C.append("**CoBALT-kernel arms are a work in progress (2026-09-06).** `cobalt_dense4` / `cobalt_sparse4x` share the SAME "
             "quantized weights (packing layout only changes the on-disk/VRAM layout, not the values) "
             "so they share one `quality_tag`/lm_eval run per model; quality-of-record is measured on the fakequant HF checkpoint "
             "under vLLM (D6), independent of whether the packed kernel artifact exists yet.")
    pend_pack = [a for a, c in arms_cfg.items() if not A4.jload(c.get("manifest_path") or "")]
    if pend_pack:
        C.append(f"**Not yet packed ({model}):** " + ", ".join(pend_pack) + " -- bpw/artifact GB rows are `pending` until "
                 "`src/cobaltkernel/pack_cobalt.py` writes their manifest.json (see results/cobaltkernel/arms.json for the target path/layout).")
    mixed = [a for a, c in arms_cfg.items() if c.get("mixed_layout")]
    if mixed:
        C.append(f"**Mixed-layout arms ({model}):** " + ", ".join(mixed) + " serve `self_attn.o_proj` on a DIFFERENT "
                 "mask (and, where the arm's name ends `_ohyb4`, a different bit-width) from the other six quantized "
                 "matrices, so they are NOT single-layout artifacts; their `eff. bpw` is a per-matrix weighted figure "
                 "(see each arm's `bpw_override_src` in results/cobaltkernel/arms.json), NOT a packed-manifest reading, "
                 "and no packer or kernel path for a mixed-layout model exists yet -- every speed cell is `pending` and "
                 "their speed class is UNMEASURED. Quality is unaffected by this: it is measured on the fakequant HF "
                 "checkpoint under vLLM (D6), which is kernel- and packing-independent.")
    pend_speed = [a for a in arms_cfg if a not in sp]
    if pend_speed:
        C.append(f"**No speed measurement yet ({model}):** " + ", ".join(pend_speed) + " -- `scripts/run_cobaltkernel_speed_oneslice.sh` "
                 "(schema: results/cobaltkernel/SPEED_JSON_SCHEMA.md) has not been run for these arms; speed/kernel-dispatched/peak-memory "
                 "cells render `pending`, never dropped.")
    for r in cobalt_rows:
        if any(r["tasks"][t] is None for t in TASKS):
            missing = [t for t in TASKS if r["tasks"][t] is None]
            C.append(f"**Partial/pending quality ({model}/{r['arm']}):** missing task(s) " + ", ".join(missing) +
                     f" in `{r['resdir']}` (eval may still be running -- see run.log/eval.log there).")
    cav_md = f"{RES_CK}/CAVEATS.md"
    if os.path.exists(cav_md):
        C.append("**From results/cobaltkernel/CAVEATS.md:** " + " ".join(l.strip() for l in open(cav_md) if l.strip() and not l.startswith("#")))
    else:
        C.append("results/cobaltkernel/CAVEATS.md does not exist yet (no additional kernel-engineering caveats recorded).")
    return C


# ----------------------------------------------------------------------------------------------------- main
def build(models, arms_json, out):
    arms_all = json.load(open(arms_json))
    doc = ["# cobaltkernel — BASELINE: CoBALT-quantized MedGemma-27B on a custom CUDA megakernel", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `src/cobaltkernel/collect_baseline.py` from "
           "`results/accel4bit/BASELINE.json` (reference arms), `results/cobaltkernel/<model>/<tag>/eval_*.json` (CoBALT quality), "
           "`results/cobaltkernel/arms.json` (arm registry) and `results/cobaltkernel/<model>/speed_oneslice/*.json` (CoBALT speed, "
           "schema: `results/cobaltkernel/SPEED_JSON_SCHEMA.md`). Weight format: `docs/FORMAT.md`. "
           "Table columns/formatting reused VERBATIM from `src/accel4bit_collect.py` "
           "(imported, not modified) so CoBALT-kernel arms sit row-for-row against the accel4bit arms. "
           "Cells marked `pending` have no data yet and are never silently dropped.",
           "",
           "Reference arms (from accel4bit, same snapshot/protocol): **bf16**, **FP8-dynamic** (27B speed+quality reference), "
           "**A gptq**, **D awq**, **C nvfp4**, **B gguf** (llama.cpp imatrix Q4_K_M, the primary llama.cpp-MMQ comparison target). "
           "CoBALT-kernel arms (this project): **E cobalt-dense4** (primary, DENSE4 packing, D7) and **F cobalt-sparse4x** "
           "(ablation, SPARSE4X packing -- does exploiting the 50% sparsity pay?). Hardware: RTX PRO 6000 Blackwell (SM 12.0) "
           "in MIG mode, same 2g.48gb slice `MIG-1d47bdbe` as accel4bit; MIG never reconfigured.", ""]
    allj = {}
    for model in models:
        arms_cfg = arms_all.get(model, {})
        ref_rows, sp = reference_rows_and_speed(model)
        cobalt_rows = [cobalt_quality_row(model, arm_id, cfg) for arm_id, cfg in arms_cfg.items()]
        for arm_id, cfg in arms_cfg.items():
            j = cobalt_speed_row(cfg)
            if j:
                sp[arm_id] = j
        all_rows = ref_rows + cobalt_rows
        arm_order = [r["arm"] for r in ref_rows] + list(arms_cfg.keys())

        doc += [f"## {A4.MODEL_LABEL.get(model, model)}", ""]
        doc += ["### (a) Quality", ""] + A4.quality_table(model, all_rows) + [""]
        doc += ["### (b) Speed — ONE slice (MIG-1d47bdbe, 2g.48gb), all arms sequential", ""] + A4.speed_table(model, sp, arm_order) + speed_footnotes(sp, arms_cfg) + [""]
        doc += ["### (c) Recipe / provenance", ""]
        prov = A4.provenance(model, {a: j for a, j in sp.items() if not a.startswith("cobalt_")})
        doc += append_cobalt_provenance(model, prov, arms_cfg) + [""]
        doc += speedup_section(model, sp, arms_cfg) + [""]
        doc += ["### (d) Caveats (honest)", ""] + [f"{i + 1}. {c}" for i, c in enumerate(cobaltkernel_caveats(model, ref_rows, cobalt_rows, sp, arms_cfg))] + [""]

        allj[model] = {
            "quality": [{k: v for k, v in r.items() if k != "source"} for r in all_rows],
            "speed": sp,
            "arms_cfg": arms_cfg,
        }
    doc += ["## Files", "",
            "- Reference arm data: `results/accel4bit/<model>/{gptq,gguf,nvfp4,awq,ref_bf16,ref_fp8}/`, `results/accel4bit/BASELINE.{md,json}`",
            "- CoBALT quality runs: `results/cobaltkernel/<model>/<quality_tag>/` (`eval_*.json`, `run.log`, `eval.log`)",
            "- CoBALT packed artifacts + manifests: paths in `results/cobaltkernel/arms.json` (`manifest_path`); format spec `docs/FORMAT.md`",
            "- CoBALT one-slice speed: `results/cobaltkernel/<model>/speed_oneslice/<arm>.json`, schema `results/cobaltkernel/SPEED_JSON_SCHEMA.md`",
            "- Arm registry: `results/cobaltkernel/arms.json`; kernel source: `src/cobaltkernel/csrc/`; quantizer/packer: `src/cobaltkernel/{quantize_cobalt,pack_cobalt}.py`",
            "- Regenerate this file: `python src/cobaltkernel/collect_baseline.py`", ""]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(doc))
    json.dump(allj, open(out.replace(".md", ".json"), "w"), indent=1, default=str)
    print(f"wrote {out} and {out.replace('.md', '.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["medgemma-27b", "gemma-3-4b"])
    ap.add_argument("--arms-json", default=f"{RES_CK}/arms.json")
    ap.add_argument("--out", default=f"{RES_CK}/BASELINE.md")
    a = ap.parse_args()
    build(a.models, a.arms_json, a.out)


if __name__ == "__main__":
    main()
