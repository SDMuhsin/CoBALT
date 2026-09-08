"""Split lm_eval results_*.json files under <res_dir>/lm_eval_* into per-task eval_<task>.json (PROTOCOL)."""
import glob
import json
import os
import sys

res = sys.argv[1]
summary = {}
for f in sorted(glob.glob(os.path.join(res, "lm_eval_*", "**", "results_*.json"), recursive=True)):
    d = json.load(open(f))
    for task, r in d["results"].items():
        out = dict(task=task, source=f, results=r, versions=d.get("versions", {}).get(task),
                   n_samples=d.get("n-samples", {}).get(task), config=d.get("config"),
                   task_config=d.get("configs", {}).get(task))
        json.dump(out, open(os.path.join(res, f"eval_{task}.json"), "w"), indent=2)
        summary[task] = {k: v for k, v in r.items() if isinstance(v, (int, float))}
json.dump(summary, open(os.path.join(res, "eval_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
