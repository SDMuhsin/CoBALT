#!/usr/bin/env python
"""vLLM generation sanity for a compressed-tensors W4A16 artifact + report which linear kernel
was picked.  Run with VLLM_LOGGING_LEVEL=DEBUG and grep the log for marlin/machete/cutlass."""
import json
import os
import sys
import time


def main():
    path = sys.argv[1]
    quant = sys.argv[2] if len(sys.argv) > 2 else "compressed-tensors"
    from vllm import LLM, SamplingParams
    import vllm
    t0 = time.time()
    kw = {}
    with open(os.path.join(path, "config.json")) as f:
        if "Gemma3ForConditionalGeneration" in json.load(f).get("architectures", []):
            kw["limit_mm_per_prompt"] = {"image": 0}   # multimodal ckpt: skip dummy-image profiling
    util = float(os.environ.get("GPU_UTIL", "0.85"))
    try:
        llm = LLM(model=path, quantization=quant, max_model_len=4096, gpu_memory_utilization=util,
                  dtype="bfloat16", seed=0, **kw)
    except Exception as e:  # memory-budget failure at startup -> one retry at 0.92 (DEVIATION, labelled)
        msg = repr(e)
        if util < 0.92 and any(k in msg.lower() for k in ("memory", "engine core initialization failed")):
            print(f"GEN_RETRY gpu_memory_utilization=0.92 after: {msg[:300]}", flush=True)
            llm = LLM(model=path, quantization=quant, max_model_len=4096, gpu_memory_utilization=0.92,
                      dtype="bfloat16", seed=0, **kw)
        else:
            raise
    print(f"LOAD_S {time.time()-t0:.1f} vllm={vllm.__version__}", flush=True)
    # introspect the kernel actually used by the quantized linears
    try:
        core = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except Exception:
        core = None
    if core is not None:
        seen = {}
        for name, mod in core.named_modules():
            qm = getattr(mod, "quant_method", None)
            if qm is not None:
                k = type(qm).__name__
                kern = getattr(qm, "kernel", None) or getattr(getattr(qm, "scheme", None), "kernel", None)
                key = (k, type(kern).__name__ if kern is not None else None)
                seen[key] = seen.get(key, 0) + 1
        print("QUANT_METHODS " + json.dumps({f"{a}/{b}": n for (a, b), n in seen.items()}), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    prompts = ["The capital of France is", "Aspirin is commonly used to treat",
               "A 45-year-old man presents with crushing chest pain radiating to the left arm. The most likely diagnosis is"]
    outs = llm.generate(prompts, sp)
    for o in outs:
        print("GEN", repr(o.prompt), "->", repr(o.outputs[0].text), flush=True)
    print("ACCEL4BIT_AWQ_GEN_OK", flush=True)


if __name__ == "__main__":
    main()
