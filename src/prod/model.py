"""Stage 3 -- serving.  One cooperative kernel launch per prefill, one per decode step.

    from prod import CobaltModel
    m = CobaltModel.load_pretrained("artifacts/biomistral-7b-blk1632_b4")
    print(m.generate_text("A 54-year-old presents with", max_new_tokens=64))

The recipe (the layout the kernel decodes) comes from the artifact's `manifest.json`; the
model target (the per-model kernel tuning and the reference numbers) comes from the
`config.json` beside the weights -- or from `config_dir` when the artifact has none.

Shape of the runtime
--------------------
`prefill()` runs the whole prompt in ONE `cudaLaunchCooperativeKernel`; `step()` runs one
decode token in one more.  There is no per-layer kernel dispatch, no cuBLAS call and no
host round-trip inside a step -- token ids travel in the kernel parameter block.  The KV
cache is shared between the two kernels, so a prefill hands off to decode with no copy.

Batching: the decode kernel is instantiated for M in {1, 2, 4, 8}.  `batch` picks the
instantiation; it is a compile-time template parameter, so changing it rebuilds.
"""
from __future__ import annotations

import json
import os

from . import _bridge, recipes as _recipes

# packed manifest `layout` + o_proj layout -> recipe name
_BY_LAYOUT = {
    ("DENSE4", "DENSE4"): "dense4",
    ("BLK1632_4", "BLK1632_4"): "blk1632_b4",
    ("BLK1632_4", "DENSE4"): "blk1632_b4_oproj",
    ("BLK1632_6", "DENSE4"): "blk1632_b6_oproj4",
}


def recipe_for_artifact(artifact_dir: str):
    """Infer which recipe an already-packed artifact was built with.

    The manifest records the uniform layout and every matrix's own layout id, which is
    exactly what distinguishes the four arms -- so a partner cannot pair an artifact
    with the wrong kernel configuration by mistake.
    """
    man = json.load(open(os.path.join(artifact_dir, "manifest.json")))
    uniform = man["layout"]
    oproj = man["layers"][0]["matrices"]["o_proj"]["layout_name"]
    key = (uniform, oproj)
    if key not in _BY_LAYOUT:
        raise ValueError(
            f"artifact layout {key} matches no shipped recipe ({sorted(_BY_LAYOUT)}). "
            "Pass recipe= explicitly if this is a custom build.")
    return _recipes.get(_BY_LAYOUT[key])



def _model_cfg(d: str | None) -> dict | None:
    """The HF config.json (text_config unwrapped) next to `d`, for recipes.target_for()."""
    import json as _json
    if not d:
        return None
    p = os.path.join(d, "config.json")
    if not os.path.exists(p):
        return None
    c = _json.load(open(p))
    return c.get("text_config", c)

class CobaltModel:
    """A packed CoBALT artifact bound to the megakernel, in one shipped configuration."""

    def __init__(self, runner, recipe, artifact_dir, tokenizer=None):
        self._mcfg = None          # HF config dict -> recipes.target_for (set by load())
        self._r = runner
        self.recipe = recipe
        self.artifact_dir = artifact_dir
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, artifact_dir: str, config_dir: str | None = None, recipe=None,
             batch: int = 1, max_ctx: int = 1024, tokenizer=None, verbose: bool = False):
        """Build the kernel for `artifact_dir` and load its weights.

        artifact_dir : a CBK1 packed directory (stage 2 output)
        config_dir   : HF snapshot holding config.json, if the artifact has none
        recipe       : name or Recipe; inferred from the manifest when omitted
        batch        : in-kernel batch M, one of {1, 2, 4, 8} (compile-time)
        max_ctx      : KV cache capacity in tokens
        tokenizer    : optional HF tokenizer for the text-level helpers
        """
        from . import env as _env
        _env.configure()
        r = (_recipes.get(recipe) if isinstance(recipe, str)
             else recipe or recipe_for_artifact(artifact_dir))
        mcfg = _model_cfg(config_dir or artifact_dir)

        with _recipes.activate(r, mcfg):
            _bridge.install()
            # The recipe's knobs are COMPILE-TIME macros and, for the attention split, a
            # class attribute bound when runner.py is first imported -- so the import has
            # to happen inside activate(), and the split is re-applied explicitly in case
            # the module was already imported by something else in this process.
            from cobaltkernel.runner import KernelRunner
            KernelRunner.KEYS_PER_BLOCK = int(os.environ.get("COBALT_KPB", 128))
            runner = KernelRunner(artifact_dir, M=batch, max_ctx=max_ctx,
                                  config_dir=config_dir, verbose=verbose)
        obj = cls(runner, r, artifact_dir, tokenizer)
        obj._mcfg = mcfg
        return obj

    @classmethod
    def load_pretrained(cls, artifact_dir: str, config_dir: str | None = None, **kw):
        """`load()` plus the HF tokenizer from `config_dir` (or the artifact dir)."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config_dir or artifact_dir)
        return cls.load(artifact_dir, config_dir=config_dir, tokenizer=tok, **kw)

    # --------------------------------------------------------------- generate
    def reset(self):
        """Clear the KV cache. Call between unrelated prompts."""
        self._r.reset()
        return self

    def generate(self, prompt_ids, max_new_tokens: int, use_prefill_kernel: bool = True):
        """Greedy generation from token ids. Returns the generated ids (prompt excluded)."""
        self.reset()
        with _recipes.activate(self.recipe, self._mcfg):
            return self._r.generate(list(prompt_ids), max_new_tokens,
                                    use_prefill_kernel=use_prefill_kernel)

    def generate_text(self, prompt: str, max_new_tokens: int = 64, **kw) -> str:
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer; use CobaltModel.load_pretrained(...) "
                               "or pass tokenizer=")
        ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        out = self.generate(ids, max_new_tokens, **kw)
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def prefill(self, prompt_ids):
        """Fill the KV cache for `prompt_ids` in one launch; returns logits [1, vocab]."""
        with _recipes.activate(self.recipe, self._mcfg):
            return self._r.prefill_kernel(list(prompt_ids))

    def step(self, tokens, positions):
        """One decode launch. Returns (logits, argmax_ids)."""
        with _recipes.activate(self.recipe, self._mcfg):
            return self._r.step(list(tokens), list(positions))

    # ------------------------------------------------------------------ info
    @property
    def target(self):
        """The shipped model target this artifact matched, or None."""
        return _recipes.target_for(self._mcfg)

    @property
    def stats(self) -> dict:
        r = self._r
        t = self.target
        return {"recipe": self.recipe.name, "layout": self.recipe.layout,
                "bpw": self.recipe.bpw, "artifact": self.artifact_dir,
                "target": t.name if t else None,
                "model_type": (self._mcfg or {}).get("model_type"),
                "weight_bytes_per_token": r.weight_bytes(),
                "batch_M": r.M, "max_ctx": r.max_ctx, "grid_blocks": r.blocks,
                "layers": r.n_layers, "hidden": r.hidden, "vocab": r.vocab}

    def phase_times_ms(self) -> dict:
        """Per-phase device time of the LAST step run with timings enabled."""
        return {k: v / 1e3 for k, v in self._r.phase_times_us().items()}

    def __repr__(self):
        return (f"<CobaltModel {self.recipe.name} ({self.recipe.bpw:.4f} bpw) "
                f"M={self._r.M} ctx={self._r.max_ctx}>")
