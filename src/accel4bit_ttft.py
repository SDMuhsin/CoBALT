"""Single-stream TTFT / prefill / decode breakdown with vLLM offline LLM (PROTOCOL speed eval, 512 -> 128, bs=1, greedy).
TTFT is measured as wall time of a prefill-only request (max_tokens=1) on the same 512-token prompt; decode tok/s =
(128-1) / (t_128 - t_1). Reports median over --iters after --warmup. Extra CLI args are forwarded to vllm.LLM as
--key value (e.g. --quantization fp8, --cpu-offload-gb 20).
"""
import argparse, json, os, time, statistics, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    a, rest = ap.parse_known_args()
    extra = {}
    i = 0
    while i < len(rest):
        k = rest[i].lstrip("-").replace("-", "_")
        v = rest[i + 1] if i + 1 < len(rest) and not rest[i + 1].startswith("--") else True
        if isinstance(v, str) and v[:1] in "{[":   # JSON values, e.g. --limit-mm-per-prompt '{"image":0}' (phase-3 one-slice pass)
            v = json.loads(v)
            extra[k] = v
            i += 2
            continue
        try:
            v = int(v)
        except (TypeError, ValueError):
            try:
                v = float(v)
            except (TypeError, ValueError):
                pass
        extra[k] = v
        i += 2 if v is not True else 1

    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.max_model_len, seed=1234,
              gpu_memory_utilization=a.gpu_memory_utilization, enable_prefix_caching=False, **extra)
    # deterministic 512-token prompt via token ids
    prompt = {"prompt_token_ids": list(range(10, 10 + a.input_len))}
    sp_full = SamplingParams(temperature=0.0, max_tokens=a.output_len, ignore_eos=True)
    sp_one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    def run(sp):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t, len(out[0].outputs[0].token_ids)

    for _ in range(a.warmup):
        run(sp_full); run(sp_one)
    t1, tN = [], []
    for _ in range(a.iters):
        d1, n1 = run(sp_one); assert n1 == 1
        dN, nN = run(sp_full); assert nN == a.output_len, nN
        t1.append(d1); tN.append(dN)
    ttft = statistics.median(t1); tfull = statistics.median(tN)
    res = {
        "model": a.model, "input_len": a.input_len, "output_len": a.output_len, "iters": a.iters, "warmup": a.warmup,
        "extra_llm_args": extra, "dtype": a.dtype, "max_model_len": a.max_model_len,
        "ttft_s_median": ttft, "ttft_s_all": t1,
        "e2e_s_median_128": tfull, "e2e_s_all_128": tN,
        "prefill_tok_s_from_ttft": a.input_len / ttft,
        "decode_tok_s": (a.output_len - 1) / (tfull - ttft),
        "gen_tok_s_e2e": a.output_len / tfull,
        "torch_max_memory_allocated_GB": torch.cuda.max_memory_allocated() / 1e9,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        res["nvidia_smi_used_MiB"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                                                    capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_all") and not k.endswith("_all_128")}, indent=1))


if __name__ == "__main__":
    main()
