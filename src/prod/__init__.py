"""CoBALT deployment package -- quantize, pack, verify, serve, benchmark.

A production-facing wrapper around the research implementation in `src/cobaltkernel/`.
No algorithm is reimplemented here; what this package adds is a single entry point per
stage, a pinned configuration per shipped arm, and a way to check a build before trusting
it.  See docs/REPRODUCTION.md.

    python -m prod doctor
    python -m prod recipes

    from prod import CobaltModel, recipes, bench
    m = CobaltModel.load("/models/medgemma-27b-cobalt-b1632-4", config_dir=HF)
    m.generate_text("A 54-year-old presents with", max_new_tokens=64)
"""
from .model import CobaltModel, recipe_for_artifact
from .recipes import Recipe, RECIPES, activate, describe, get, names

__all__ = ["CobaltModel", "recipe_for_artifact", "Recipe", "RECIPES",
           "activate", "describe", "get", "names",
           "env", "recipes", "quantize", "pack", "verify", "bench"]

__version__ = "1.0.0"


def __getattr__(name):
    # submodules are imported lazily: `prod.bench` pulls in torch, `prod.quantize`
    # pulls in the streaming quantizer -- neither should be a cost of `import prod`.
    if name in ("env", "recipes", "quantize", "pack", "verify", "bench", "cli"):
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(name)
