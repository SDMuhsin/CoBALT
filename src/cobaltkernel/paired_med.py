#!/usr/bin/env python
"""Paired per-question comparison of two arms on the Med tasks (MedQA / PubMedQA / MedMCQA@1000).
  python src/cobaltkernel/paired_med.py <dirA> <dirB> [--reps 10000]
A dir is either results/cobaltkernel/biomistral-7b/<tag> (samples_<task>.json from --log_samples) or
results/accel4bit/biomistral-7b/gguf (eval_<task>.samples.jsonl). Reports per-task paired deltas, the Med-avg delta,
a question-resampling bootstrap 95% CI and the paired SE (the right yardstick: unpaired Med-avg SE is ~1.0 pt).
"""
import argparse, json, os, random, statistics

TASKS = ["medqa_4options", "pubmedqa", "medmcqa"]


def load(d, task):
    p1 = os.path.join(d, f"samples_{task}.json")
    p2 = os.path.join(d, f"eval_{task}.samples.jsonl")
    if os.path.exists(p1):
        j = json.load(open(p1))
        recs = j[task] if isinstance(j, dict) else j
    elif os.path.exists(p2):
        recs = [json.loads(l) for l in open(p2) if l.strip()]
    else:
        return None
    return {r["doc_id"]: float(r["acc"]) for r in recs}


def main():
    global TASKS
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma list; e.g. medmcqa alone for the full-set dirs")
    args = ap.parse_args()
    TASKS = [t for t in args.tasks.split(",") if t]
    rng = random.Random(1234)
    per = {}
    for t in TASKS:
        A, B = load(args.a, t), load(args.b, t)
        if A is None or B is None:
            print(f"{t}: samples missing ({'A' if A is None else 'B'})"); return
        ids = sorted(set(A) & set(B))
        per[t] = [(A[i], B[i]) for i in ids]
        da = statistics.mean(a for a, _ in per[t]); db = statistics.mean(b for _, b in per[t])
        disc = sum(1 for a, b in per[t] if a != b)
        print(f"{t:16s} n={len(ids):5d}  A={100*da:.2f}  B={100*db:.2f}  B-A={100*(db-da):+.2f}  discordant={disc}")
    def medavg(sample):
        return 100 * statistics.mean(statistics.mean(b - a for a, b in s) for s in sample)
    d0 = medavg([per[t] for t in TASKS])
    boots = []
    for _ in range(args.reps):
        boots.append(medavg([[per[t][rng.randrange(len(per[t]))] for _ in per[t]] for t in TASKS]))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    se = statistics.pstdev(boots)
    print(f"Med avg  B-A = {d0:+.2f}  paired SE {se:.2f}  bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"{'SIGNIFICANT' if lo > 0 or hi < 0 else 'not significant'}")


if __name__ == "__main__":
    main()
