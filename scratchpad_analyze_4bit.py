import csv, os
from collections import defaultdict

BASE = "results/benchmark_4bit"
CHANCE = {"mnli":33.3, "imagenet":0.1, "arc_easy":25.0, "hellaswag":25.0, "piqa":50.0, "winogrande":50.0}
BASELINES = {"sparsegpt","wanda-awq","wanda-sinq","jsq-wo","slim"}

def load(fn, scale):
    rows=[]
    p=os.path.join(BASE,fn)
    if not os.path.exists(p): return rows
    with open(p) as f:
        for r in csv.DictReader(f):
            try: v=float(r["value"])
            except: continue
            rows.append(dict(model=r["model"], method=r["method"], bits=int(float(r["bits"])),
                             sp=float(r["sparsity"]), task=r["task"], val=v*scale))
    return rows

rows = load("glue.csv",1.0) + load("vit.csv",1.0) + load("llm.csv",100.0)
rows = [r for r in rows if r["bits"]==4 and r["sp"]>0]

# group by (model, task, sp)
cells=defaultdict(dict)
for r in rows:
    cells[(r["model"],r["task"],r["sp"])][r["method"]]=r["val"]

def valid_baseline_max(d, ch):
    # exclude divergence artifacts (val > 100 for pct-scaled) and missing
    cands={}
    for m,v in d.items():
        if m not in BASELINES: continue
        if v>100.5: continue  # divergence guard
        cands[m]=v
    if not cands: return None, None
    m=max(cands, key=cands.get)
    return m, cands[m]

print(f"{'model':<42}{'task':<11}{'sp':<5}{'cobalt':>8}{'best_base':>10}{'Δ':>8}  verdict")
print("-"*100)
by_model=defaultdict(list)
for (model,task,sp),d in sorted(cells.items()):
    ch=CHANCE.get(task, 0)
    cob=d.get("cobalt")
    bm,bv=valid_baseline_max(d,ch)
    if cob is None or bv is None:
        continue
    if cob>100.5:  # cobalt divergence artifact
        print(f"{model:<42}{task:<11}{sp:<5.2f}{cob:>8.2f}{bv:>10.2f}{'DIVERG':>8}  cobalt-DIVERGED (excluded)")
        continue
    delta=cob-bv
    both_chance = cob<ch+2 and bv<ch+2
    if both_chance: verdict="CHANCE"
    elif abs(delta)<0.7: verdict="TIE"
    elif delta>0: verdict="WIN"
    else: verdict="LOSS"
    tag=f"({bm})"
    print(f"{model:<42}{task:<11}{sp:<5.2f}{cob:>8.2f}{bv:>10.2f}{delta:>+8.2f}  {verdict} {tag}")
    by_model[model].append((task,sp,delta,verdict,cob,bv,ch))

print("\n=== per-model win/loss summary ===")
for model,lst in by_model.items():
    w=sum(1 for x in lst if x[3]=="WIN"); L=sum(1 for x in lst if x[3]=="LOSS")
    t=sum(1 for x in lst if x[3]=="TIE"); c=sum(1 for x in lst if x[3]=="CHANCE")
    print(f"{model:<42} WIN {w}  LOSS {L}  TIE {t}  CHANCE {c}  (n={len(lst)})")
