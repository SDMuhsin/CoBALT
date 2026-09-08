"""accel4bit shared quality-eval driver: load the model ONCE, run tasks in the given priority
order, write one raw lm-eval JSON per task to <out_dir>/eval_<task>.json (so a killed job still leaves the
finished tasks), plus <out_dir>/eval_summary.json.

Usage:
  python accel4bit_lmeval.py --backend vllm --pretrained <dir> --out_dir <dir> \
      --tasks wikitext medqa_4options arc_easy pubmedqa medmcqa:1000 \
      --model_args max_model_len=4096,gpu_memory_utilization=0.85,dtype=bfloat16[,quantization=...]
  Task syntax: name[:limit]. Backends: vllm (lm_eval VLLM) | hf (lm_eval HFLM, e.g. device_map=auto offload).
Seed 1234 everywhere; fewshot = task defaults (never overridden).
"""
import argparse, json, os, sys, time, platform, subprocess

import torch


def parse_model_args(s):
    out = {}
    for kv in filter(None, s.split(",")):
        k, v = kv.split("=", 1)
        if v.startswith("{"):  # dict literal without commas, e.g. limit_mm_per_prompt={"image":0} (awq arm, multimodal gemma-3-4b)
            v = json.loads(v)
        elif v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[k] = v
    return out


def versions():
    d = {"python": platform.python_version(), "torch": torch.__version__}
    for m in ("lm_eval", "vllm", "transformers", "compressed_tensors", "accelerate"):
        try:
            d[m] = __import__(m).__version__
        except Exception as e:  # noqa
            d[m] = f"n/a ({e.__class__.__name__})"
    try:
        d["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader"],
                                         capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    d["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--model_args", default="")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_samples", action="store_true")
    ap.add_argument("--include_path", default="/workspace/PTQResearch/scripts/accel4bit_lmeval_tasks",
                    help="lm_eval task override dir (shared pubmedqa parquet fix); '' to disable")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    import lm_eval
    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=a.include_path or None)
    print("[accel4bit_lmeval] include_path:", a.include_path, flush=True)

    margs = parse_model_args(a.model_args)
    margs["pretrained"] = a.pretrained
    print("[accel4bit_lmeval] versions:", json.dumps(versions()), flush=True)
    print("[accel4bit_lmeval] backend:", a.backend, "model_args:", margs, "batch_size:", a.batch_size, flush=True)
    t0 = time.time()
    if a.backend == "vllm":
        from lm_eval.models.vllm_causallms import VLLM
        bs = a.batch_size
        lm = VLLM(batch_size=bs, **margs)
    else:
        from lm_eval.models.huggingface import HFLM
        bs = int(a.batch_size) if str(a.batch_size).isdigit() else a.batch_size
        lm = HFLM(batch_size=bs, **margs)
    load_s = time.time() - t0
    print(f"[accel4bit_lmeval] model loaded in {load_s:.0f}s", flush=True)

    summary = {"pretrained": a.pretrained, "backend": a.backend, "model_args": margs, "batch_size": a.batch_size,
               "seed": a.seed, "load_s": load_s, "include_path": a.include_path, "versions": versions(), "tasks": {}}
    for spec in a.tasks:
        name, _, lim = spec.partition(":")
        limit = int(lim) if lim else None
        out_json = os.path.join(a.out_dir, f"eval_{name}.json")
        print(f"[accel4bit_lmeval] === task {name} limit={limit} -> {out_json}", flush=True)
        t1 = time.time()
        try:
            res = simple_evaluate(model=lm, tasks=[name], limit=limit, random_seed=a.seed, numpy_random_seed=a.seed,
                                  torch_random_seed=a.seed, fewshot_random_seed=a.seed, log_samples=a.log_samples,
                                  confirm_run_unsafe_code=True, task_manager=tm)
        except Exception as e:  # keep going with remaining tasks; report verbatim
            import traceback
            tb = traceback.format_exc()
            print(f"[accel4bit_lmeval] TASK FAILED {name}: {e}\n{tb}", flush=True)
            summary["tasks"][name] = {"status": "FAILED", "error": repr(e), "traceback": tb, "limit": limit}
            json.dump(summary, open(os.path.join(a.out_dir, "eval_summary.json"), "w"), indent=2, default=handle_non_serializable)
            continue
        dt = time.time() - t1
        if a.log_samples and "samples" in res:
            json.dump(res.pop("samples"), open(os.path.join(a.out_dir, f"samples_{name}.json"), "w"),
                      default=handle_non_serializable)
        res["accel4bit"] = {"task": name, "limit": limit, "eval_s": dt, "pretrained": a.pretrained,
                            "backend": a.backend, "model_args": margs, "versions": summary["versions"], "include_path": a.include_path}
        if torch.cuda.is_available():
            try:
                res["accel4bit"]["torch_max_memory_allocated_GB"] = torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                pass
        json.dump(res, open(out_json, "w"), indent=2, default=handle_non_serializable)
        metrics = res["results"].get(name, {})
        summary["tasks"][name] = {"status": "OK", "eval_s": dt, "limit": limit, "metrics": metrics}
        print(f"[accel4bit_lmeval] {name} done in {dt:.0f}s: {json.dumps(metrics, default=handle_non_serializable)}", flush=True)
        json.dump(summary, open(os.path.join(a.out_dir, "eval_summary.json"), "w"), indent=2, default=handle_non_serializable)
    print("[accel4bit_lmeval] ALL_DONE total %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
