#!/usr/bin/env python
"""Quantizer-gap study table (BioMistral-7B): every arm with its effective bpw (kernel layout accounting on the
6.979B decoder-linear params), held-out / calibration output-error means from the quantizer manifest, and the
lm_eval numbers. Generated, never hand-edited:  python src/cobaltkernel/qgap_table.py  -> results/biomistral/qgap/TABLE.md
"""
import glob, json, os, time

ROOT = "/workspace/PTQResearch"
R = f"{ROOT}/results/cobaltkernel/biomistral-7b"
QG = f"{ROOT}/results/biomistral/qgap"
ART = "/scratch/root/PTQResearch/accel4bit_models/biomistral-7b"
TASKS = [("wikitext", "word_perplexity,none"), ("arc_easy", "acc,none"),
         ("medqa_4options", "acc,none"), ("pubmedqa", "acc,none"), ("medmcqa", "acc,none")]
# per-layer param shares of the 7 decoder linears (Mistral-7B: h4096, kv1024, inter14336)
SHARE = {"self_attn.q_proj": 4096 * 4096, "self_attn.k_proj": 4096 * 1024, "self_attn.v_proj": 4096 * 1024,
         "self_attn.o_proj": 4096 * 4096, "mlp.gate_proj": 4096 * 14336, "mlp.up_proj": 4096 * 14336,
         "mlp.down_proj": 4096 * 14336}
TOT = sum(SHARE.values())
META = 0.1875   # fp16 scale + u8 zero per 128-group (kernel layouts DENSE_b / BLK1632_b)


def layout_bpw(bits, blk1632):
    return (0.5 * bits + 1.0 if blk1632 else bits) + META


def arm_bpw(man):
    """effective bpw from the manifest's per-matrix (bits, mask_block, sparsity_target)."""
    L = man["layers"]
    if not L:
        return float("nan")
    acc, n = 0.0, 0
    for li, mats in L.items():
        for name, st in mats.items():
            sp = st.get("sparsity_target", man["config"]["sparsity"])
            mb = st.get("mask_block", man["config"].get("mask_block", 0))
            bits = st.get("bits", man["config"]["bits"])
            if sp == 0.0:
                b = layout_bpw(bits, False)
            elif mb == 32:
                # k-of-32 block layout: survivor codes + a 32-bit selector per block (1 bit/weight) + metadata.
                # Only 16:32 is implemented in the kernel; other k are fakequant-only (flagged in the arm name).
                b = bits * (1.0 - int(32 * sp) / 32.0) + 1.0 + META
            else:   # unstructured (global top-k) mask: the kernel has no bitmap layout -> served DENSE_b
                b = layout_bpw(bits, False)
            acc += b * SHARE[name]; n += SHARE[name]
    return acc / n


def metric(d, task, key):
    p = os.path.join(d, f"eval_{task}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p)); r = j.get("results", j)
    return r.get(task, {}).get(key)


def row_vals(d):
    return [metric(d, t, k) for t, k in TASKS]


def main():
    base = row_vals(f"{R}/ref_bf16")
    bavg = 100 * sum(base[2:]) / 3
    rows = []
    # fixed reference rows
    refs = [("bf16", f"{R}/ref_bf16", "16", None), ("llama.cpp Q4_K_M (imatrix)", f"{ROOT}/results/accel4bit/biomistral-7b/gguf", "4.797", None),
            ("llama.cpp Q4_K_M (MEDICAL imatrix, matched-lever control)", f"{ROOT}/results/accel4bit/biomistral-7b/gguf_medcal", "4.797", None),
            ("llama.cpp Q4_K_M (imatrix) — MedMCQA FULL 4183 [fullmm]", f"{ROOT}/results/accel4bit/biomistral-7b/gguf_fullmm", "4.797", None)]
    for lab, d, bpw, man in refs:
        rows.append((lab, bpw, man, row_vals(d)))
    # every quality dir that has a manifest in qgap/ or a raw artifact we can map
    for d in sorted(glob.glob(f"{R}/cobalt_*")):
        tag = os.path.basename(d)
        man = None
        mp = f"{QG}/{tag.replace('_fullmm', '')}.manifest.json"   # *_fullmm = same checkpoint re-scored on FULL MedMCQA (4,183)
        if os.path.exists(mp):
            man = json.load(open(mp))
        else:
            rm = json.load(open(f"{d}/run_meta.json")) if os.path.exists(f"{d}/run_meta.json") else {}
            ck = rm.get("ckpt", "")
            raw = ck.split("/")[-1].split("_fakequant")[0] if ck else ""
            if raw and os.path.exists(f"{ART}/{raw}/manifest.json"):
                man = json.load(open(f"{ART}/{raw}/manifest.json"))
        bpw = f"{arm_bpw(man):.4f}" if man and man.get("layers") else "?"
        rows.append((tag, bpw, man, row_vals(d)))
    out = ["# BioMistral-7B quantizer-gap study (lm_eval 0.4.13, seed 1234, 0-shot, medmcqa@1000; fakequant under vLLM)", "",
           f"Generated {time.strftime('%F %T')} by `src/cobaltkernel/qgap_table.py`. bpw = kernel-layout accounting on the "
           "decoder linears (DENSE_b = b+0.1875, BLK1632_b = b/2+1+0.1875; embed/lm_head excluded, as in Q4_K_M's 4.797). "
           "`e-ho` / `e-cal` = mean per-matrix held-out / calibration output-error ratio tr(DHD)/tr(WHW) from the quantizer "
           "(held-out = 16 disjoint blocks of the same calibration stream). Med avg = mean(MedQA, PubMedQA, MedMCQA) in points; d vs bf16 "
           f"({bavg:.2f}); d vs Q4_K_M target is within 1.0.", "",
           "| arm | bpw | emb/head | quantizer | e-ho | e-cal | wiki PPL | ARC-e | MedQA | PubMedQA | MedMCQA | Med avg | d bf16 | d Q4_K_M |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    q4 = None
    for lab, bpw, man, v in rows:
        qz = ""
        eho = ecal = ""
        if man:
            c = man["config"]
            qz = (c.get("quant_mode", "rtn") + ("" if c.get("clip", "none") == "none" else "+clip:" + c["clip"])
                  + ("+actorder" if c.get("actorder") else "")
                  + (f" damp{c['damping_frac']}" if c.get("damping_frac", 0.01) != 0.01 else "")
                  + (f" ncal{c['n_calib']}" if c.get("n_calib", 128) != 128 else "")
                  + (f" g{c['group_size']}" if c.get("group_size", 128) != 128 else "")
                  + (f" bits{c['bits_override']}" if c.get("bits_override") else ""))
            if "eout_ho_mean" in man:
                eho = f"{man['eout_ho_mean']:.4f}"; ecal = f"{man['eout_calib_mean']:.4f}"
        ppl, arc, mq, pq, mm = v
        if None in (mq, pq, mm):
            avg = None
        else:
            avg = 100 * (mq + pq + mm) / 3
        if lab == "llama.cpp Q4_K_M (imatrix)" and avg is not None:
            q4 = avg          # the bar is the SHIPPED (ultrachat-imatrix) Q4_K_M row only
        f = lambda x, p=4: "pending" if x is None else f"{x:.{p}f}"
        pplt = "pending" if ppl is None else f"{ppl:.3f} ({100*(ppl/base[0]-1):+.1f}%)"
        suf = lab.replace("_fullmm", "").rsplit("_", 1)[-1] if lab.startswith("cobalt_") else ("bf16" if lab == "bf16" else "Q4_K/Q6_K")
        eh = {"e4": "4/4"}.get(suf, suf.replace("e", "").replace("h", "/") if suf.startswith("e") else suf)
        if "fullmm" in lab:
            qz = (qz + " **MedMCQA FULL** (Med avg not comparable to @1000 rows)").strip()
        rows_s = (f"| {lab} | {bpw} | {eh} | {qz} | {eho} | {ecal} | {pplt} | {f(arc)} | {f(mq)} | {f(pq)} | {f(mm)} | "
                  f"{f(avg, 2)} | {'' if avg is None else f'{avg-bavg:+.2f}'} | "
                  f"{'' if (avg is None or q4 is None) else f'{avg-q4:+.2f}'} |")
        out.append(rows_s)
    os.makedirs(QG, exist_ok=True)
    open(f"{QG}/TABLE.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
