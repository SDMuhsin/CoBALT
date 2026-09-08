"""Toolchain / device configuration for the CoBALT CUDA kernels.

The research tree assumed one specific box (a fixed CUDA install, a fixed MIG slice,
`sm_120a` hard-coded).  Deployment cannot assume any of that, so this module derives
everything it can from the running machine and only falls back to an override when the
environment sets one explicitly.

    from prod import env
    env.configure()          # idempotent; safe to call more than once
    print(env.preflight())   # dict; raises PreflightError on a hard blocker

What `configure()` sets
-----------------------
  TORCH_CUDA_ARCH_LIST      compute capability of the visible device (e.g. "12.0a")
  COBALTKERNEL_NVCC_ARCH    the matching -gencode flag for the raw-nvcc test harnesses
  TORCH_EXTENSIONS_DIR      JIT build cache (default: $COBALT_CACHE/torch_ext)

Nothing else is touched.  If you already export any of these, yours wins.
"""
from __future__ import annotations

import os
import shutil
import subprocess

# Minimum compute capability. The decode GEMV needs only sm_70-era features, but the
# prefill path uses `mma.sync.m16n8k16.bf16`, which is sm_80+.
MIN_CC = (8, 0)

# Capabilities with an arch-SPECIFIC target ("a" suffix). On these, the plain target
# silently drops the arch-specific tensor-core/fp8 mma variants.
ARCH_SPECIFIC = {(9, 0), (10, 0), (12, 0)}


class PreflightError(RuntimeError):
    """A blocker that will make the build or the run fail later, reported early."""


def _cache_root() -> str:
    return os.environ.get("COBALT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "cobalt")


def detect_arch(index: int = 0) -> str:
    """TORCH_CUDA_ARCH_LIST string for the visible device, e.g. '12.0a' / '8.6'."""
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("no CUDA device visible (torch.cuda.is_available() is False)")
    cc = torch.cuda.get_device_capability(index)
    return f"{cc[0]}.{cc[1]}" + ("a" if cc in ARCH_SPECIFIC else "")


def configure(index: int = 0, cache_dir: str | None = None) -> dict:
    """Set the build environment for the CUDA extensions. Existing exports win."""
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or detect_arch(index)
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    if "COBALTKERNEL_NVCC_ARCH" not in os.environ:
        tag = arch.replace(".", "")
        os.environ["COBALTKERNEL_NVCC_ARCH"] = (
            f"-gencode arch=compute_{tag},code=sm_{tag}")
    ext = cache_dir or os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(
        _cache_root(), "torch_ext")
    os.environ["TORCH_EXTENSIONS_DIR"] = ext
    os.makedirs(ext, exist_ok=True)
    return {"TORCH_CUDA_ARCH_LIST": arch,
            "COBALTKERNEL_NVCC_ARCH": os.environ["COBALTKERNEL_NVCC_ARCH"],
            "TORCH_EXTENSIONS_DIR": ext}


def _nvcc_version() -> str | None:
    nvcc = shutil.which("nvcc") or (
        os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")
        if os.environ.get("CUDA_HOME") else None)
    if not nvcc or not os.path.exists(nvcc):
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[1].split(",")[0].strip()
    except Exception:
        pass
    return "unknown"


def preflight(index: int = 0, strict: bool = True) -> dict:
    """Check every prerequisite the kernels need. Returns a report; raises on blockers.

    Blockers (strict=True): no CUDA device, compute capability below MIN_CC, no nvcc,
    no ninja, no cooperative-launch support. Each message says what to do about it.
    """
    import torch
    rep: dict = {"blockers": [], "warnings": []}
    rep["torch"] = torch.__version__
    rep["torch_cuda"] = torch.version.cuda

    if not torch.cuda.is_available():
        rep["blockers"].append("No CUDA device visible. Check CUDA_VISIBLE_DEVICES.")
        if strict:
            raise PreflightError(rep["blockers"][0])
        return rep

    props = torch.cuda.get_device_properties(index)
    cc = (props.major, props.minor)
    rep.update(device=props.name, compute_capability=f"{cc[0]}.{cc[1]}",
               sm_count=props.multi_processor_count,
               total_mem_gib=round(props.total_memory / 2**30, 2),
               arch=detect_arch(index))
    if cc < MIN_CC:
        rep["blockers"].append(
            f"compute capability {cc[0]}.{cc[1]} < {MIN_CC[0]}.{MIN_CC[1]}; the prefill "
            "kernel's mma.sync.m16n8k16.bf16 will not compile. Decode-only use would "
            "need the prefill path disabled (CobaltModel(..., prefill_kernel=False)).")

    ver = _nvcc_version()
    rep["nvcc"] = ver
    if ver is None:
        rep["blockers"].append(
            "nvcc not found. Install a CUDA toolkit whose version matches torch "
            f"({torch.version.cuda}) and export CUDA_HOME, or put nvcc on PATH.")

    rep["ninja"] = shutil.which("ninja")
    if not rep["ninja"]:
        rep["blockers"].append(
            "ninja not found on PATH. `pip install ninja` -- torch's cpp_extension "
            "shells out to the binary, so the venv's bin/ must be on PATH.")

    # Cooperative launch: the whole design is one cudaLaunchCooperativeKernel per step.
    try:
        coop = torch.cuda.get_device_properties(index).cooperative
    except AttributeError:
        coop = None
    rep["cooperative_launch"] = coop
    if coop is False:
        rep["blockers"].append(
            "cooperativeLaunch is not supported on this device; the megakernel cannot run.")

    rep["env"] = configure(index)
    if rep["blockers"] and strict:
        raise PreflightError("; ".join(rep["blockers"]))
    return rep
