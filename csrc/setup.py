"""
Setup script for building CPR-SINQ CUDA extension.

Build with:
    cd /workspace/SINQ/csrc && pip install -e .

Or build in place:
    cd /workspace/SINQ/csrc && python setup.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))
cpr_dir = os.path.join(this_dir, "cpr")

# Source files
sources = [
    os.path.join(cpr_dir, "bindings.cpp"),
    os.path.join(cpr_dir, "pack_kernels.cu"),
    os.path.join(cpr_dir, "dequant_kernels.cu"),
    os.path.join(cpr_dir, "matmul_kernels.cu"),
    os.path.join(cpr_dir, "tiled_matmul.cu"),
    os.path.join(cpr_dir, "mma_kernel.cu"),
    os.path.join(cpr_dir, "wmma_tiled_kernel.cu"),
    os.path.join(cpr_dir, "wmma_pipelined_kernel.cu"),
    os.path.join(cpr_dir, "wmma_optimized_kernel.cu"),
    os.path.join(cpr_dir, "wmma_heavy_kernel.cu"),
    os.path.join(cpr_dir, "wmma_v2_kernel.cu"),
    os.path.join(cpr_dir, "wmma_warp_special_kernel.cu"),
    os.path.join(cpr_dir, "wmma_small_tile_kernel.cu"),
]

# Compiler flags
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_86,code=sm_86",  # A40, RTX 3090
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
        "-lineinfo",  # For profiling
    ],
}

setup(
    name="cpr_kernels",
    version="0.1.0",
    description="CPR-SINQ CUDA kernels for mixed-precision quantized matmul",
    author="SINQ Team",
    ext_modules=[
        CUDAExtension(
            name="cpr_kernels",
            sources=sources,
            include_dirs=[cpr_dir],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
