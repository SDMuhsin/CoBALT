"""accel4bit: vLLM generation sanity check + single-stream (bs=1, greedy) speed measurement.

Single-stream protocol: prompt 512 tokens -> 128 new tokens, bs=1, greedy.
TTFT is measured as the wall time of a max_tokens=1 request on the same prompt (prefill-only), decode tokens/s as
(128-1)/(t_128 - t_1); prefill tokens/s = 512/TTFT. Complements `vllm bench latency` (which reports end-to-end
latency only). Run with VLLM_LOGGING_LEVEL=DEBUG to record the dispatched kernel from the log.
"""
import argparse
import json
import os
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="compressed-tensors")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true", help="no torch.compile/CUDA graphs (27B on a 1g.24gb slice: graph capture OOMs)")
    args = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, quantization=args.quantization, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=False, seed=0, enforce_eager=args.enforce_eager)
    tok = llm.get_tokenizer()

    # --- sanity generations ---
    prompts = ["The capital of France is", "Aspirin is commonly used to treat",
               "Question: A 45-year-old man presents with crushing chest pain radiating to the left arm. "
               "The most likely diagnosis is"]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=48))
    sanity = []
    for p, o in zip(prompts, outs):
        print(f"[sanity] {p!r} -> {o.outputs[0].text!r}", flush=True)
        sanity.append(dict(prompt=p, output=o.outputs[0].text))
    res = dict(model=args.model, quantization=args.quantization, vllm=vllm.__version__, torch=torch.__version__,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), sanity=sanity)
    if args.sanity_only:
        json.dump(res, open(args.out_json, "w"), indent=2)
        print("SANITY_OK", flush=True)
        return

    # --- single-stream timing: exactly input_len prompt tokens (token ids) ---
    text = ("The history of medicine is long and varied. " * 400)
    ids = tok(text, add_special_tokens=False).input_ids[: args.input_len - 1]
    ids = [tok.bos_token_id] + ids
    assert len(ids) == args.input_len, len(ids)
    prompt = [dict(prompt_token_ids=ids)]
    sp_full = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    sp_one = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

    def timed(sp):
        t = time.perf_counter()
        o = llm.generate(prompt, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        n = len(o[0].outputs[0].token_ids)
        return dt, n

    for _ in range(args.warmup):
        timed(sp_full); timed(sp_one)
    t_full, t_one = [], []
    for _ in range(args.iters):
        dt, n = timed(sp_full); assert n == args.output_len, n; t_full.append(dt)
        dt, n = timed(sp_one); assert n == 1, n; t_one.append(dt)
    ttft = statistics.median(t_one)
    e2e = statistics.median(t_full)
    decode_tps = (args.output_len - 1) / (e2e - ttft)
    res.update(dict(
        input_len=args.input_len, output_len=args.output_len, iters=args.iters, warmup=args.warmup,
        ttft_s_median=ttft, ttft_s_all=t_one, e2e_s_median=e2e, e2e_s_all=t_full,
        prefill_tok_s=args.input_len / ttft, decode_tok_s=decode_tps,
        e2e_output_tok_s=args.output_len / e2e,
    ))
    print(f"[single-stream] TTFT {ttft*1000:.1f} ms (prefill {args.input_len/ttft:.0f} tok/s), "
          f"e2e {e2e:.3f} s, decode {decode_tps:.1f} tok/s", flush=True)
    json.dump(res, open(args.out_json, "w"), indent=2)
    print("SINGLE_STREAM_OK", flush=True)


if __name__ == "__main__":
    main()
