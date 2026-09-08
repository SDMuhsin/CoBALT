"""Import bridge to the research implementation in `src/cobaltkernel/`.

`prod` deliberately contains no copy of the quantizer, the packer or the CUDA sources:
one implementation, one place to fix a bug.  What lives here is the plumbing that makes
those modules importable, since several of them are written script-style (`import
cobalt_math`, not `from . import cobalt_math`) and expect `src/cobaltkernel` itself on
`sys.path`.
"""
from __future__ import annotations

import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # <repo>/src
REPO = os.path.dirname(SRC)
KERNEL_SRC = os.path.join(SRC, "cobaltkernel")


def install() -> None:
    """Make both `cobaltkernel.x` and the script-style bare `x` imports resolve."""
    for p in (SRC, KERNEL_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


def repo_path(p: str) -> str:
    """Resolve a repo-relative path (recipes store calibration paths that way)."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def run_module_main(module_name: str, argv: list[str]) -> None:
    """Call a research script's `main()` with a synthetic argv.

    Used instead of a subprocess so a failure raises a Python traceback the caller can
    see, and so the caller's CUDA context / environment (the pinned recipe knobs) apply.
    """
    install()
    import importlib
    mod = importlib.import_module(module_name)
    saved = sys.argv
    sys.argv = [module_name] + [str(a) for a in argv]
    try:
        mod.main()
    finally:
        sys.argv = saved
