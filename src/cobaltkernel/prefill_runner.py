"""Python runner for the Gemma3 PREFILL megakernel (CBK1 DENSE4 weights).

One cooperative kernel launch processes a whole prompt:

    r = PrefillRunner(packed_dir, config_dir=..., max_tokens=1152, max_ctx=1200)
    logits = r.prefill(ids)                  # [1, vocab] fp32 (last position)
    logits = r.prefill(ids, all_logits=True) # [T, vocab] fp32

The KV cache it fills is byte-compatible with the decode megakernel
(`runner.KernelRunner`): [n_layers][B][kv_heads][max_ctx][head_dim] bf16, post-RoPE K
and raw V.  The contract is `pf::kv_off()` in csrc/prefill_kernel.cuh.
"""

import json
import math
import os

import torch

try:
    from cobaltkernel import arch
except ImportError:          # run from inside src/cobaltkernel
    import arch

_EXT = None


def _build_ext(verbose=False):
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "csrc")
    arch = os.environ.get("COBALTKERNEL_NVCC_ARCH",
                          "-gencode arch=compute_120a,code=sm_120a").split()
    flags = []
    # CoBALT-16:32 arm.  The SAME env var the decode megakernel uses
    # (runner.py), so one setting selects the arm for both kernels; 0 = DENSE4 control.
    blk = int(os.environ.get("COBALT_BLK1632", 0))
    assert blk in (0, 4, 6), "COBALT_BLK1632 must be 0, 4 or 6"
    if blk:
        flags.append(f"-DPF_BLK1632={blk}")
    # MIXED-LAYOUT o_proj-hybrid arms: COBALT_BLK1632_O is the BLK code
    # width of the o_proj GEMM alone.  Unset = same as COBALT_BLK1632 = the shipped build.
    blko = os.environ.get("COBALT_BLK1632_O")
    if blko is not None:
        assert int(blko) in (0, 4, 6), "COBALT_BLK1632_O must be 0, 4 or 6"
        flags.append(f"-DPF_BLK1632_O={int(blko)}")
    # per-matrix overrides for the fused qkv GEMM / the down_proj GEMM (mixed arms B/C); unset = COBALT_BLK1632
    for env, macro in (("COBALT_BLK1632_QKV", "PF_BLK1632_QKV"), ("COBALT_BLK1632_D", "PF_BLK1632_D")):
        v = os.environ.get(env)
        if v is not None:
            assert int(v) in (0, 4, 6), f"{env} must be 0, 4 or 6"
            flags.append(f"-D{macro}={int(v)}")
    if os.environ.get("COBALT_PF_BLK_LUT"):
        flags.append(f"-DCBK_GEMM_BLK_LUT={int(os.environ['COBALT_PF_BLK_LUT'])}")
    if os.environ.get("COBALT_PF_TAILSPLIT"):
        flags.append("-DPF_TAILSPLIT=1")
    minb = os.environ.get("COBALT_PF_BATCH_MINB")
    if minb:
        flags.append(f"-DPF_BATCH_MINB={int(minb)}")
    _EXT = load(
        name="cbk_prefill" + ("_" + "_".join(
            f.split("-D")[-1].replace("=", "").replace("PF_BLK1632_QKV", "q").replace("PF_BLK1632_D", "d")
             .replace("PF_BLK1632_O", "o").replace("PF_BLK1632", "").replace("CBK_GEMM_BLK_LUT", "lut")
             .replace("PF_TAILSPLIT", "ts").replace("PF_BATCH_MINB", "mb")
            for f in flags) if flags else ""),
        sources=[os.path.join(src, "prefill_bindings.cpp"),
                 os.path.join(src, "prefill_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", *flags, *arch],
        extra_include_paths=[src],
        verbose=verbose,
    )
    return _EXT


class PrefillRunner:
    PHASES = ["embed", "qkv", "rope_kv", "attn", "o_proj", "addnorm1", "gateup",
              "down", "addnorm2"]

    def __init__(self, model_dir, config_dir=None, max_tokens=2048, max_ctx=None,
                 max_logit_rows=None, device="cuda", verbose=False, kv=None,
                 layers_limit=None, kv_batch=1):
        self.ext = _build_ext(verbose)
        self.device = device
        cfg = json.load(open(os.path.join(config_dir or model_dir, "config.json")))
        self.cfg = cfg
        H = cfg["hidden_size"]
        self.hidden = H
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", H // self.n_heads)
        self.inter = cfg["intermediate_size"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.sliding_window = arch.sliding_window(cfg)
        self.nq_dim = self.n_heads * self.head_dim
        self.nkv_dim = self.n_kv * self.head_dim
        self.nqkv = self.nq_dim + 2 * self.nkv_dim
        self.kv_group = self.n_heads // self.n_kv
        self.attn_scale = arch.attn_scale(cfg)
        self.arch = arch.flags(cfg)
        self.embed_scale = self.arch["embed_scale"]
        self.norm_names = arch.norm_names(cfg)
        self.max_tokens = max_tokens
        self.max_ctx = max_ctx or max_tokens
        self.max_logit_rows = max_logit_rows or max_tokens
        self.layer_types = arch.layer_types(cfg)

        self.kv_batch = kv_batch
        self._load_packed(model_dir)
        self._alloc(kv)
        # debug lever: run only the first `layers_limit` decoder layers
        self.n_run_layers = layers_limit or self.n_layers
        self._configure()

    # ------------------------------------------------------------------ weights
    @staticmethod
    def _desc(blob, entry):
        base = blob.data_ptr()
        arr = entry["arrays"]
        g = lambda k: (base + arr[k]["off"]) if (k in arr and arr[k]["bytes"]) else 0
        return [g("data"), g("row_off"), g("scale"), g("zero"), g("col_scale"),
                entry["K"], entry["N"], entry["N"] // 128, entry["layout"]]

    # layout id the decoder GEMMs must carry, given the build's COBALT_BLK1632 setting
    @staticmethod
    def _want_layout(name=None):
        blk = int(os.environ.get("COBALT_BLK1632", 0))
        over = {"o_proj": "COBALT_BLK1632_O", "qkv": "COBALT_BLK1632_QKV", "down_proj": "COBALT_BLK1632_D"}.get(name)
        if over and os.environ.get(over) is not None:
            blk = int(os.environ[over])
        return {0: 1, 4: 6, 6: 7}[blk]

    def _load_packed(self, model_dir):
        man = json.load(open(os.path.join(model_dir, "manifest.json")))
        assert man.get("fuse_qkv"), "the prefill kernel needs a --fuse-qkv artifact (*_dense4f)"
        dev = self.device

        def blob(name):
            with open(os.path.join(model_dir, name), "rb") as f:
                b = f.read()
            return torch.frombuffer(bytearray(b), dtype=torch.uint8).to(dev).contiguous()

        self.blobs = {"embed": blob(man["embed"]["file"])}
        self.embed_desc = self._desc(self.blobs["embed"], man["embed"])
        assert self.embed_desc[6] == self.hidden and self.embed_desc[5] >= self.vocab
        assert man["embed"]["layout"] in (1, 2), "embedding must be DENSE4/DENSE8"
        if man.get("lm_head"):
            self.blobs["lm_head"] = blob(man["lm_head"]["file"])
            self.lm_head_desc = self._desc(self.blobs["lm_head"], man["lm_head"])
            assert man["lm_head"]["layout"] in (1, 2), "lm_head must be DENSE4/DENSE8"
        else:
            assert self.arch["tied"], "untied model but no packed lm_head in the artifact"
            self.lm_head_desc = list(self.embed_desc)

        misc = blob(man["misc"]["file"])
        self.norms = {}
        for name, e in man["misc"]["arrays"].items():
            n = e["bytes"] // 4
            v = misc[e["off"]: e["off"] + e["bytes"]].view(torch.float32)[:n]
            self.norms[name] = v.to(torch.bfloat16).contiguous()
        del misc
        self.final_norm = self.norms["model.norm"]
        self._rope_tables()

        assert len(man["layers"]) == self.n_layers
        rows = []
        self.keep = []
        for i, lay in enumerate(man["layers"]):
            b = blob(lay["file"])
            self.blobs[i] = b
            row = []
            for nm in ("qkv", "o_proj", "gateup", "down_proj"):
                e = lay["matrices"][nm]
                assert e["layout"] == self._want_layout(nm), (
                    f"{nm}: layout {e['layout']} but the prefill kernel was built for "
                    f"layout {self._want_layout(nm)} (COBALT_BLK1632="
                    f"{os.environ.get('COBALT_BLK1632', 0)}, _O={os.environ.get('COBALT_BLK1632_O')}, "
                    f"_QKV={os.environ.get('COBALT_BLK1632_QKV')}, _D={os.environ.get('COBALT_BLK1632_D')})")
                assert e["N"] % 128 == 0, f"{nm}: N must be a multiple of KT=128"
                row += self._desc(b, e)
            ns = [self.norms[f"{i}.{n}"] if n else None for n in self.norm_names]
            self.keep.append([t for t in ns if t is not None])
            row += [t.data_ptr() if t is not None else 0 for t in ns]
            row += [1 if self.layer_types[i] == "sliding_attention" else 0, 0]
            assert len(row) == 44
            rows.append(row)
        # fused-qkv column-scale rows are picked per BR=256 row tile -> the q|k|v row
        # boundaries must be multiples of 256.
        assert self.nq_dim % 256 == 0 and self.nkv_dim % 256 == 0, \
            "fused qkv row ranges must align to the BR=256 tile"
        self.layers_tbl = torch.tensor(rows, dtype=torch.int64, device=dev).contiguous()

    def _rope_inv(self, params):
        base = float(params["rope_theta"])
        D = self.head_dim
        inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        if params.get("rope_type", "default") == "linear":
            inv = inv / float(params["factor"])
        elif params.get("rope_type", "default") != "default":
            raise NotImplementedError(params)
        return inv.to(self.device)

    def _rope_tables(self):
        rp = arch.rope_params(self.cfg)
        self.inv_local = self._rope_inv(rp["sliding_attention"]).contiguous()
        self.inv_global = self._rope_inv(rp["full_attention"]).contiguous()

    # ------------------------------------------------------------------ buffers
    def _alloc(self, kv):
        dev, dt = self.device, torch.bfloat16
        M, H, D = self.max_tokens, self.hidden, self.head_dim
        z = lambda *s, d=dt: torch.zeros(*s, dtype=d, device=dev)
        if kv is not None:
            self.kcache, self.vcache = kv
            self.kv_b = self.kcache.shape[1]
            assert self.kcache.shape[0] == self.n_layers
            assert self.kcache.shape[2] == self.n_kv and self.kcache.shape[4] == D
            self.max_ctx = self.kcache.shape[3]
        else:
            self.kv_b = self.kv_batch
            self.kcache = z(self.n_layers, self.kv_b, self.n_kv, self.max_ctx, D)
            self.vcache = z(self.n_layers, self.kv_b, self.n_kv, self.max_ctx, D)
        self.tokens = torch.zeros(M, dtype=torch.int32, device=dev)
        self.positions = torch.zeros(M, dtype=torch.int32, device=dev)
        self.rope_cs = z(2, M, D // 2, d=torch.float32)
        self.rope_sn = z(2, M, D // 2, d=torch.float32)
        self.h = z(M, H); self.xn = z(M, H)
        self.qkvb = z(M, self.nqkv); self.qb = z(M, self.nq_dim)
        self.ao = z(M, self.nq_dim); self.ob = z(M, H)
        self.act = z(M, self.inter); self.db = z(M, H)
        R = self.max_logit_rows
        self.hg = z(R, H)
        self.logits = z(R, self.vocab, d=torch.float32)
        self.logit_rows = torch.zeros(R, dtype=torch.int32, device=dev)
        self.timings = torch.zeros(self.n_layers * 8 + 8, dtype=torch.int64, device=dev)

    def _configure(self):
        self.r = self.ext.PrefillRunner()
        iv = [self.hidden, self.n_run_layers, self.n_heads, self.n_kv, self.head_dim,
              self.inter, self.vocab, self.max_tokens, 1, 0, self.kv_b, 0,
              self.max_ctx, self.sliding_window, self.nq_dim, self.nkv_dim, self.nqkv,
              self.kv_group, self.arch["norm_plus_one"], self.arch["act_gelu"]]
        fv = [self.eps, self.attn_scale, self.embed_scale]
        self.r.configure(self.layers_tbl, self.embed_desc, self.lm_head_desc, self.final_norm,
                         self.inv_local, self.inv_global, self.rope_cs, self.rope_sn,
                         self.kcache, self.vcache, self.tokens, self.positions,
                         self.h, self.xn,
                         self.qkvb, self.qb, self.ao, self.ob, self.act, self.db,
                         self.hg, self.logits, self.logit_rows, iv, fv)
        self.blocks = self.r.num_blocks()
        self.blocks_per_sm = self.r.num_blocks_per_sm()
        self.smem = self.r.smem_bytes()
        self.blocks_b = self.r.num_blocks_batch()
        self.blocks_per_sm_b = self.r.num_blocks_per_sm_batch()
        self.smem_b = self.r.smem_bytes_batch()

    # ------------------------------------------------------------------ run
    def reset(self):
        self.kcache.zero_(); self.vcache.zero_()

    @torch.no_grad()
    def prefill(self, ids, pos0=0, all_logits=False, rows=None, timings=False, seq=0):
        """ids: list/1-D tensor of token ids.  Returns fp32 logits [R, vocab]."""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        T = len(ids)
        assert T <= self.max_tokens, f"{T} tokens > max_tokens {self.max_tokens}"
        assert pos0 + T <= self.max_ctx
        if rows is None:
            rows = list(range(T)) if all_logits else [T - 1]
        R = len(rows)
        assert R <= self.max_logit_rows
        self.tokens[:T].copy_(torch.tensor(ids, dtype=torch.int32))
        self.logit_rows[:R].copy_(torch.tensor(rows, dtype=torch.int32))
        self.r.set_run(T, R, pos0, seq)
        self.r.run(self.timings if timings else None)
        return self.logits[:R]

    @torch.no_grad()
    def batch_step(self, tokens, positions, timings=False):
        """ONE decode step for len(tokens) INDEPENDENT sequences: row b is sequence b
        (KV-cache slot b) at absolute position positions[b].  Returns fp32 logits
        [B, vocab], one row per sequence."""
        B = len(tokens)
        assert B == len(positions)
        assert B <= self.kv_b, f"{B} sequences > KV batch {self.kv_b}"
        assert B <= self.max_tokens and B <= self.max_logit_rows
        assert max(positions) < self.max_ctx
        self.tokens[:B].copy_(torch.tensor(list(tokens), dtype=torch.int32))
        self.positions[:B].copy_(torch.tensor(list(positions), dtype=torch.int32))
        self.logit_rows[:B].copy_(torch.arange(B, dtype=torch.int32))
        self.r.set_run(B, B, 0, 0)
        self.r.run_batch(self.timings if timings else None)
        return self.logits[:B]

    def phase_times_us(self, sm_clock_hz=2.43e9):
        t = self.timings.cpu().tolist()
        n = self.n_layers * 8 + 3
        d = [(t[i + 1] - t[i]) / sm_clock_hz * 1e6 for i in range(n)]
        agg = {"embed": d[0]}
        names = self.PHASES[1:]
        for k in names:
            agg[k] = 0.0
        for L in range(self.n_layers):
            for j, k in enumerate(names):
                agg[k] += d[1 + L * 8 + j]
        base = 1 + self.n_layers * 8
        agg["gather"] = d[base]
        agg["lm_head"] = d[base + 1]
        agg["total"] = (t[n] - t[0]) / sm_clock_hz * 1e6
        return agg
