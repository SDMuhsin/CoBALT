#!/usr/bin/env python
"""Smoke test: torch.utils.cpp_extension.load() JIT for sm_120a on this box.
Run after `source scripts/cobaltkernel_env.sh 2g`."""
import os, time, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
ext = load(
    name="cobaltkernel_ext_smoke",
    sources=[os.path.join(HERE, "ext_smoke.cu")],
    extra_cuda_cflags=[
        "-O3",   # NOTE: torch appends its own -std=c++20; do not pass --std here
        "-gencode", "arch=compute_120a,code=sm_120a",
        "--use_fast_math",
    ],
    extra_cflags=["-O3"],
    verbose=True,
)
print(f"[build] {time.time()-t0:.1f}s")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

x = torch.ones(1 << 20, dtype=torch.bfloat16, device="cuda")
y = torch.zeros_like(x)
ext.axpy(x, y, 2.0)
assert y.float().mean().item() == 2.0, y.float().mean().item()
print("axpy                 : OK")
v = ext.mma_probe().item()
print(f"bf16 m16n8k16 mma    : {v} (expect 16.0)  {'OK' if v == 16.0 else 'MISMATCH'}")
