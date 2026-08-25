#!/usr/bin/env python3
"""Print the UUID of a FREE MIG slice.

The GPUs on this box are MIG-sliced and SHARED between jobs. The simple
``nvidia-smi -L | grep MIG- | head -1`` that the venv activation uses always
grabs the first slice, which may already be busy -> OOM. This script instead
picks a MIG slice belonging to the physical GPU with the MOST free memory.

Logic:
  * Parse ``nvidia-smi -L`` to map each MIG-UUID to its parent physical GPU
    index. The output looks like::

        GPU 0: NVIDIA ... (UUID: GPU-....)
          MIG 1g.24gb     Device  0: (UUID: MIG-....)
          MIG 1g.24gb     Device  1: (UUID: MIG-....)
        GPU 1: NVIDIA ... (UUID: GPU-....)
          MIG 2g.48gb     Device  0: (UUID: MIG-....)

  * Get per-physical-GPU free memory from
    ``nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits``.
  * Print the MIG-UUID of the physical GPU with the most free memory.
    Ties resolve to the first such GPU; within a GPU, the first listed slice.

If no MIG slices exist, print nothing and exit 0. Parsing is best-effort: any
unexpected line is ignored rather than crashing.

stdlib + subprocess only.
"""

import re
import subprocess
import sys


# "GPU 0: NVIDIA ... (UUID: GPU-9cbecbe4-...)"
_GPU_RE = re.compile(r"^GPU\s+(\d+):.*\(UUID:\s*(GPU-[0-9a-fA-F-]+)\)")
# "  MIG 1g.24gb     Device  0: (UUID: MIG-bef7a31e-...)"
_MIG_RE = re.compile(r"\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)")


def _run(cmd):
    """Run a command, returning stdout (str) or '' on any failure."""
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def _parse_mig_to_gpu(listing):
    """Map each MIG-UUID -> parent physical GPU index, preserving listing order.

    Returns a list of (mig_uuid, gpu_index) tuples in the order encountered.
    """
    pairs = []
    current_gpu = None
    for line in listing.splitlines():
        gpu_m = _GPU_RE.match(line.strip())
        if gpu_m:
            try:
                current_gpu = int(gpu_m.group(1))
            except (TypeError, ValueError):
                current_gpu = None
            continue
        mig_m = _MIG_RE.search(line)
        if mig_m and current_gpu is not None:
            pairs.append((mig_m.group(1), current_gpu))
    return pairs


def _parse_free_mem(csv_text):
    """Map physical GPU index -> free memory (MiB, int) from query-gpu CSV."""
    free = {}
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            mem = int(float(parts[1]))
        except (TypeError, ValueError):
            continue
        free[idx] = mem
    return free


def pick_free_mig():
    """Return the MIG-UUID on the GPU with the most free memory, or None."""
    pairs = _parse_mig_to_gpu(_run(["nvidia-smi", "-L"]))
    if not pairs:
        return None

    free = _parse_free_mem(
        _run([
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ])
    )

    # Choose the best slice. Iterate in listing order so that ties (equal free
    # memory, or missing memory data) resolve to the first-listed slice/GPU.
    best_uuid = None
    best_mem = None
    for mig_uuid, gpu_idx in pairs:
        mem = free.get(gpu_idx, -1)  # unknown free mem sorts lowest
        if best_mem is None or mem > best_mem:
            best_mem = mem
            best_uuid = mig_uuid
    return best_uuid


def main():
    uuid = pick_free_mig()
    if uuid:
        sys.stdout.write(uuid + "\n")
    # No MIG slices -> print nothing, exit 0.
    return 0


if __name__ == "__main__":
    sys.exit(main())
