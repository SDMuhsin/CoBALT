"""cobaltkernel quality collector: turn results/cobaltkernel/<model>/<tag>/eval_*.json (written by
scripts/run_cobaltkernel_quality.sh via the shared src/accel4bit_lmeval.py driver) into a markdown table row in
the SAME format/columns as results/accel4bit/BASELINE.md section (a), with deltas vs the bf16 row taken from
results/accel4bit/BASELINE.json for the same model.

Usage:
  python src/cobaltkernel/collect_quality.py --model gemma-3-4b --tag <tag>
      -> appends/updates the row for <tag> in results/cobaltkernel/<model>/QUALITY.md (also prints it)
  python src/cobaltkernel/collect_quality.py --model gemma-3-4b --tag <tag> --compare-only
      -> just prints one line per task comparing this run to the bf16 and gguf BASELINE rows (no file write)

Only reads files; never launches anything.
"""
import argparse
import glob
import json
import os

ROOT = "/workspace/PTQResearch"
RES_CK = f"{ROOT}/results/cobaltkernel"
BASELINE_JSON = f"{ROOT}/results/accel4bit/BASELINE.json"
TASKS = ["wikitext", "arc_easy", "medqa_4options", "pubmedqa", "medmcqa"]


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def task_metrics(resdir, task):
    """Raw lm_eval eval_<task>.json (as written by src/accel4bit_lmeval.py) -> res["results"][task] dict."""
    j = jload(os.path.join(resdir, f"eval_{task}.json"))
    if not j:
        return None
    r = j.get("results", {})
    return r.get(task)


def baseline_arm(model, arm):
    d = jload(BASELINE_JSON)
    if not d or model not in d:
        return None
    for row in d[model].get("quality", []):
        if row.get("arm") == arm:
            return row
    return None


def fmt_ppl_delta(x, base):
    if x is None:
        return "n/a"
    if base is None:
        return f"{x:.3f}"
    d = x - base
    pct = 100.0 * d / base if base else float("nan")
    sign = "+" if d >= 0 else ""
    return f"{x:.3f} ({sign}{d:.2f}, {sign}{pct:.1f}%)"


def fmt_acc_delta(x, base):
    if x is None:
        return "n/a"
    if base is None:
        return f"{x:.4f}"
    d = (x - base) * 100.0
    sign = "+" if d >= 0 else ""
    return f"{x:.4f} ({sign}{d:.1f} pt)"


def gather(model, tag):
    resdir = f"{RES_CK}/{model}/{tag}"
    out = {"resdir": resdir, "tasks": {}}
    for t in TASKS:
        m = task_metrics(resdir, t)
        if m:
            out["tasks"][t] = m
    meta = jload(f"{resdir}/run_meta.json") or {}
    out["meta"] = meta
    return out


def row_for(model, tag, run):
    """Build one markdown row, columns matching BASELINE.md (a): arm | wikitext word-PPL | llama-perplexity (n/a
    here) | arc_easy acc/acc_norm | medqa_4options acc | pubmedqa acc | medmcqa@1000 acc | eff bpw | artifact GB."""
    bf16 = baseline_arm(model, "bf16") or {}
    bf16_t = bf16.get("tasks", {})
    t = run["tasks"]

    def g(task, key):
        return (t.get(task) or {}).get(key)

    def gb(task, key):
        return (bf16_t.get(task) or {}).get(key)

    wiki = fmt_ppl_delta(g("wikitext", "word_perplexity,none"), gb("wikitext", "word_perplexity,none"))
    arc_acc = g("arc_easy", "acc,none")
    arc_norm = g("arc_easy", "acc_norm,none")
    arc_acc_b = gb("arc_easy", "acc,none")
    arc_norm_b = gb("arc_easy", "acc_norm,none")
    if arc_acc is None:
        arc_cell = "n/a"
    else:
        parts = [f"{arc_acc:.4f}"]
        if arc_acc_b is not None:
            d = (arc_acc - arc_acc_b) * 100
            parts[0] += f" ({'+' if d>=0 else ''}{d:.1f} pt)"
        if arc_norm is not None:
            p2 = f"{arc_norm:.4f}"
            if arc_norm_b is not None:
                d2 = (arc_norm - arc_norm_b) * 100
                p2 += f" ({'+' if d2>=0 else ''}{d2:.1f} pt)"
            parts.append(p2)
        arc_cell = " / ".join(parts)
    medqa = fmt_acc_delta(g("medqa_4options", "acc,none"), gb("medqa_4options", "acc,none"))
    pubmed = fmt_acc_delta(g("pubmedqa", "acc,none"), gb("pubmedqa", "acc,none"))
    medmcqa = fmt_acc_delta(g("medmcqa", "acc,none"), gb("medmcqa", "acc,none"))

    meta = run.get("meta", {})
    artifact_gb = meta.get("artifact_gb", "n/a")
    bpw = "n/a (fakequant, bf16-format checkpoint; see FORMAT.md for the packed on-disk bpw)"

    return f"| {tag} | {wiki} | n/a | {arc_cell} | {medqa} | {pubmed} | {medmcqa} | {bpw} | {artifact_gb} |"


def compare_only(model, tag):
    run = gather(model, tag)
    bf16 = baseline_arm(model, "bf16") or {}
    gguf = baseline_arm(model, "gguf") or {}
    bf16_t, gguf_t = bf16.get("tasks", {}), gguf.get("tasks", {})
    t = run["tasks"]
    if not t:
        print(f"[collect_quality] no eval_*.json found yet in {run['resdir']}")
        return
    print(f"[collect_quality] {model} tag={tag}  (baseline arms: bf16 resdir={bf16.get('resdir')}, gguf resdir={gguf.get('resdir')})")
    for task in TASKS:
        cur = t.get(task)
        if not cur:
            print(f"  {task}: pending")
            continue
        b = bf16_t.get(task, {})
        g = gguf_t.get(task, {})
        if task == "wikitext":
            cv, bv, gv = cur.get("word_perplexity,none"), b.get("word_perplexity,none"), g.get("word_perplexity,none")
            db = f"{cv-bv:+.3f}" if (cv is not None and bv is not None) else "n/a"
            dg = f"{cv-gv:+.3f}" if (cv is not None and gv is not None) else "n/a"
            print(f"  wikitext word-PPL: cobalt={cv:.3f}  bf16={bv}  gguf={gv}  (Δbf16={db}, Δgguf={dg})" if cv is not None else f"  wikitext: n/a")
        elif task == "arc_easy":
            cv, bv, gv = cur.get("acc,none"), b.get("acc,none"), g.get("acc,none")
            print(f"  arc_easy acc: cobalt={cv}  bf16={bv}  gguf={gv}")
        else:
            cv, bv, gv = cur.get("acc,none"), b.get("acc,none"), g.get("acc,none")
            print(f"  {task} acc: cobalt={cv}  bf16={bv}  gguf={gv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma-3-4b", "medgemma-27b", "biomistral-7b"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--out", default=None, help="markdown file to write/update (default results/cobaltkernel/<model>/QUALITY.md)")
    a = ap.parse_args()

    if a.compare_only:
        compare_only(a.model, a.tag)
        return

    run = gather(a.model, a.tag)
    row = row_for(a.model, a.tag, run)
    print(row)

    out_path = a.out or f"{RES_CK}/{a.model}/QUALITY.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ("| arm | wikitext word-PPL (lm_eval) | llama-perplexity wiki.test.raw c2048 (GGUF only, native) | "
              "arc_easy acc / acc_norm | medqa_4options acc | pubmedqa acc | medmcqa@1000 acc | eff. bpw | artifact GB |\n"
              "|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    if os.path.exists(out_path):
        for ln in open(out_path):
            if ln.startswith(f"| {a.tag} |"):
                continue  # replace stale row for this tag
            lines.append(ln.rstrip("\n"))
    if not lines:
        lines = [f"# cobaltkernel quality — {a.model}",
                 "",
                 "Same lm_eval protocol/columns as `results/accel4bit/BASELINE.md` §(a); deltas vs that file's bf16 row.",
                 "",
                 header.rstrip("\n")]
    lines.append(row)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[collect_quality] wrote {out_path}")


if __name__ == "__main__":
    main()
