"""Stage 5 -- speed measurement.

    from prod import bench
    print(bench.decode(artifact, config_dir=..., prompt=512, gen=128))

Protocol: a `prompt`-token prompt through the prefill kernel, then `gen` tokens through
the decode kernel, greedy, batch 1.

  ttft_ms       wall time to the FIRST generated token (the whole prompt)
  decode_tok_s  tokens 2..gen only -- TTFT excluded, so it is a like-for-like decode rate
  e2e_latency_s the whole thing

Two measurement rules this project learned the hard way, both worth keeping:

  1. **Benchmark the real kernel, never a proxy.**  A standalone GEMV benchmark is
     throughput-bound; a decode step runs 0.9-3 waves and is latency-bound.  Extrapolating
     the first to the second overpredicted decode by 20-28% here, twice.
  2. **Never screen on a smaller GPU than you report on.**  Halving the SM count changes
     warps per SM and inverts the ranking: a variant that placed first on a 1g slice
     placed last on the 2g slice it would ship on.

`compare_to_reference()` puts a measurement next to what we recorded for the same recipe.
On different hardware the absolute tok/s will differ; the ratio between arms is the part
that should reproduce.
"""
from __future__ import annotations

import gc
import time

from . import _bridge, env as _env, model as _model, recipes as _recipes


def _synthetic_prompt(n: int) -> list[int]:
    """Deterministic in-vocabulary token ids -- content does not affect decode cost."""
    return [(i * 7 + 3) % 1000 + 5 for i in range(n)]


def decode(artifact_dir: str, config_dir: str | None = None, recipe=None,
           prompt: int = 512, gen: int = 128, warmup: int = 16,
           use_prefill_kernel: bool = True, batch_M: tuple[int, ...] = ()) -> dict:
    """Measure the single-stream decode protocol. Returns a result dict."""
    import torch
    _env.configure()
    r = (_recipes.get(recipe) if isinstance(recipe, str)
         else recipe or _model.recipe_for_artifact(artifact_dir))

    m = _model.CobaltModel.load(artifact_dir, config_dir=config_dir, recipe=r,
                                batch=1, max_ctx=prompt + gen + 8)
    ids = _synthetic_prompt(prompt)
    out: dict = {"recipe": r.name, "layout": r.layout, "bpw": r.bpw,
                 "protocol": f"{prompt}->{gen}, batch 1, greedy",
                 "device": torch.cuda.get_device_name(0),
                 "weight_bytes_per_token": m.stats["weight_bytes_per_token"],
                 "layers": m.stats["layers"], "hidden": m.stats["hidden"]}

    with _recipes.activate(r):
        run = m._r
        # warmup also pays the one-off JIT build and allocation costs
        run.reset()
        for t in range(warmup):
            run.step([ids[t]], [t])
        if use_prefill_kernel:
            run.reset()
            run.prefill_kernel(ids)
        torch.cuda.synchronize()

        run.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_prefill_kernel:
            cur = int(run.prefill_kernel(ids)[0].argmax())
        else:
            _, nxt = run.prefill([ids])
            cur = int(nxt[0])
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        t1 = time.perf_counter()
        for s in range(gen - 1):
            _, nxt = run.step([cur], [prompt + s])
            cur = int(nxt.cpu()[0])
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t1
        e2e = time.perf_counter() - t0

        out.update(ttft_ms=round(ttft * 1e3, 3),
                   prefill_tok_s=round(prompt / ttft, 1),
                   decode_tok_s=round((gen - 1) / t_gen, 3),
                   e2e_latency_s=round(e2e, 4))
        bps = out["weight_bytes_per_token"] * out["decode_tok_s"]
        out["achieved_gb_s"] = round(bps / 1e9, 1)

        run.step([cur], [prompt + gen], timings=True)
        torch.cuda.synchronize()
        ph = run.phase_times_us()
        out["phase_ms"] = {k: round(v / 1e3, 4) for k, v in ph.items()}
        out["peak_mem_mib"] = int(torch.cuda.max_memory_allocated() / 2**20)

    del m
    gc.collect()
    torch.cuda.empty_cache()

    for M in batch_M:
        out.setdefault("batched", {})[str(M)] = _batched(
            artifact_dir, config_dir, r, M, prompt, gen)
    return out


def _batched(artifact_dir, config_dir, r, M, prompt, gen) -> dict:
    """In-kernel batch of M sequences: total tokens/s including the prompt."""
    import torch
    m = _model.CobaltModel.load(artifact_dir, config_dir=config_dir, recipe=r,
                                batch=M, max_ctx=prompt + gen + 8)
    ids = _synthetic_prompt(prompt)
    with _recipes.activate(r):
        run = m._r
        run.reset()
        for t in range(8):
            run.step([ids[t]] * M, [t] * M)
        torch.cuda.synchronize()
        run.reset()
        t0 = time.perf_counter()
        for t in range(prompt):
            _, nxt = run.step([ids[t]] * M, [t] * M)
        cur = nxt.cpu().tolist()
        for s in range(gen - 1):
            _, nxt = run.step(cur, [prompt + s] * M)
            cur = nxt.cpu().tolist()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    res = {"M": M, "output_tok_s": round(M * gen / dt, 1),
           "total_tok_s": round(M * (prompt + gen) / dt, 1)}
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return res


def compare_to_reference(result: dict) -> str:
    """Render a measurement next to the numbers recorded for the same recipe."""
    r = _recipes.get(result["recipe"])
    ref = r.reference
    rows = [("decode tok/s", result.get("decode_tok_s"), ref["decode_tok_s"]),
            ("TTFT ms", result.get("ttft_ms"), ref["ttft_ms"]),
            ("prefill tok/s", result.get("prefill_tok_s"), ref["prefill_tok_s"])]
    rm = _recipes.REFERENCE_MODEL
    same = (result.get("layers") == rm["layers"] and result.get("hidden") == rm["hidden"])
    lines = [f"recipe {r.name} ({r.bpw:.4f} bpw) on {result.get('device')}",
             f"{'metric':<16}{'measured':>12}{'reference':>12}{'ratio':>9}"]
    for nm, got, exp in rows:
        ratio = (f"{got / exp:.3f}" if same and got and exp else "n/a")
        lines.append(f"{nm:<16}{got!s:>12}{exp!s:>12}{ratio:>9}")
    if not same:
        lines += ["", f"NOTE: this run is a {result.get('layers')}-layer / "
                      f"{result.get('hidden')}-hidden model; the reference is "
                      f"{rm['name']} ({rm['layers']} layers / {rm['hidden']} hidden). "
                      "Ratios are withheld because they would not compare like with like."]
    lines += ["", "reference numbers were measured on:", "  " + _recipes.REFERENCE_NOTE]
    return "\n".join(lines)
