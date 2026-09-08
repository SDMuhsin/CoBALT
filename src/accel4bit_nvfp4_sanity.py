"""accel4bit NVFP4 arm: sanity generation on a compressed-tensors checkpoint (vLLM).
Real file (not a `python -` heredoc) because VLLM_WORKER_MULTIPROC_METHOD=spawn
re-imports __main__ in the EngineCore process and cannot re-open <stdin>.
Usage: python accel4bit_nvfp4_sanity.py <model_dir>
"""
import sys, time

if __name__ == "__main__":
    import argparse
    from vllm import LLM, SamplingParams
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None); a = ap.parse_args()
    extra = {"max_num_batched_tokens": a.max_num_batched_tokens} if a.max_num_batched_tokens else {}
    t = time.time()
    llm = LLM(model=a.model, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=False, **extra)
    print("LOAD_S", round(time.time() - t, 1))
    sp = SamplingParams(max_tokens=48, temperature=0.0)
    for p in ["The capital of France is", "Aspirin is commonly used to treat",
              "Question: What is the first-line treatment for type 2 diabetes?\nAnswer:"]:
        o = llm.generate([p], sp)[0].outputs[0].text
        print("GEN", repr(p), "->", repr(o))
    print("SANITY_OK")
