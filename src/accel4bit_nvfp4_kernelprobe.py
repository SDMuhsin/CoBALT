#!/usr/bin/env python
"""Kernel attribution for the NVFP4 arm: which GEMM kernel does vLLM 0.28.0 actually run on
SM 12.0 for a compressed-tensors NVFP4 checkpoint?

Two independent proofs:
 1. vLLM's own selector log line `Using <Kernel> for NVFP4 GEMM` (vllm/model_executor/kernels/
    linear/__init__.py::init_nvfp4_linear_kernel) and the Marlin fallback warning
    "Weight-only FP4 compression will be used leveraging the Marlin kernel" — captured at
    VLLM_LOGGING_LEVEL=DEBUG.
 2. torch.profiler CUDA kernel names during one generate() call, with the engine run IN-PROCESS
    (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the profiler sees the worker's kernels. We list every
    kernel whose name matches fp4|nvfp4|cutlass|marlin|scaled_mm|blockscaled with its total time.
"""
import argparse
import json
import os
import re
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--gen", type=int, default=32)
    a = p.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              **({"max_num_batched_tokens": a.max_num_batched_tokens} if a.max_num_batched_tokens else {}),
              enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    # long prompt so prefill GEMMs have M>>1 (where W4A4 vs W4A16 differs) plus decode steps
    text = "The patient presented with chest pain and shortness of breath. " * 200
    ids = tok(text).input_ids[: a.prompt_len]
    prompt = tok.decode(ids)
    sp = SamplingParams(max_tokens=a.gen, temperature=0.0)
    llm.generate([prompt], sp)  # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        out = llm.generate([prompt], sp)
        torch.cuda.synchronize()
    print("GEN:", repr(out[0].outputs[0].text[:200]))
    rows = []
    pat = re.compile(r"fp4|nvfp4|cutlass|marlin|scaled_mm|blockscaled|gemm", re.I)
    for ev in prof.key_averages():
        if ev.device_type is not None and "CUDA" in str(ev.device_type) or getattr(ev, "self_device_time_total", 0) > 0:
            name = ev.key
            if pat.search(name):
                rows.append({"kernel": name[:220], "count": ev.count,
                             "device_time_us": float(getattr(ev, "self_device_time_total",
                                                             getattr(ev, "self_cuda_time_total", 0)))})
    rows.sort(key=lambda r: -r["device_time_us"])
    for r in rows[:40]:
        print(f"{r['device_time_us']:12.1f} us  x{r['count']:6d}  {r['kernel']}")
    verdict = {"marlin_kernels": [r for r in rows if "marlin" in r["kernel"].lower()],
               "fp4_cutlass_kernels": [r for r in rows if re.search(r"fp4|nvfp4|blockscaled", r["kernel"], re.I)
                                       and "marlin" not in r["kernel"].lower()]}
    json.dump({"model": a.model, "matched_kernels": rows, "verdict": verdict,
               "generated": out[0].outputs[0].text}, open(a.out_json, "w"), indent=1)
    print("N_MARLIN", len(verdict["marlin_kernels"]), "N_FP4_NATIVE", len(verdict["fp4_cutlass_kernels"]))


if __name__ == "__main__":
    main()
