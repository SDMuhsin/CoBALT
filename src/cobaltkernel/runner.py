"""Python runner for the Gemma3 decode megakernel (BF16 dense weights).

One cooperative kernel launch per decode step; `prefill()` currently loops the
decode kernel token by token (the mma.sync prefill path is a later phase --
see `prefill_hook` below).

    r = KernelRunner("/path/to/gemma-3-4b/text_bf16", M=1, max_ctx=1200)
    r.reset()
    logits, nxt = r.step([tok], [pos])       # logits [M, vocab] fp32 on device
"""

import json
import math
import os

import torch


def cbk_chunk_len(N, cmax=2048):
    """Mirror of cbk::chunk_len() in gemv_api.cuh."""
    return N if N < cmax else cmax


_EXT = None


def _build_ext(verbose=False):
    """JIT-build the extension.

    MINB (min blocks/SM => the per-thread register budget) is a COMPILE-TIME macro and
    each value gets its own build directory, so `COBALT_MINB=3` recompiles once and is
    then cached.  `COBALT_M1ONLY=1` builds only the M=1 kernel (~2x faster compile) for
    development iterations.
    """
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "csrc")
    arch = os.environ.get("COBALTKERNEL_NVCC_ARCH",
                          "-gencode arch=compute_120a,code=sm_120a").split()
    minb = int(os.environ.get("COBALT_MINB", 2))
    m1 = bool(os.environ.get("COBALT_M1ONLY"))
    prof = bool(os.environ.get("COBALT_PROF"))
    pvb = int(os.environ.get("COBALT_PVB", 4))
    rtol = int(os.environ.get("COBALT_RTOL", 11))
    # CoBALT-16:32 arm: 0 = OFF (the DENSE4 control build, bit-for-bit as shipped),
    # 4 or 6 = also instantiate the BLK1632 multi-row GEMV at that code width.
    blk = int(os.environ.get("COBALT_BLK1632", 0))
    assert blk in (0, 4, 6), "COBALT_BLK1632 must be 0, 4 or 6"
    bft = int(os.environ.get("COBALT_BLK1632_FT", 1))
    grp = int(os.environ.get("COBALT_GATEUP_RP", 0))
    bpx = int(os.environ.get("COBALT_BLK1632_PFX", 1))
    gsp = int(os.environ.get("COBALT_GATEUP_SPLIT", 0))
    # BLK16_32 per-granule dependency-chain levers.  0 = shipped path.
    #   COBALT_BLK1632_LUT : 0 global const table / 1 __constant__ / 2 __shared__ / 3 arithmetic
    #   COBALT_BLK1632_NOXB: DIAGNOSTIC, drops the 2nd gate/up activation stream (WRONG numerics)
    # DEFAULT 2 for the BLK arms: MEASURED (2g slice, 27B, 512->128) the
    # __shared__ selector table is worth +11.9 % decode at b=4 (36.94 -> 41.36 tok/s) and
    # +10.9 % at b=6 (30.90 -> 34.28).  Left at 0 when COBALT_BLK1632=0 so the DENSE4
    # control build stays bit-for-bit the originally shipped one (no staging, no shared
    # array).  Set COBALT_BLK1632_LUT explicitly to reproduce H's numbers (=0).
    blut = int(os.environ.get("COBALT_BLK1632_LUT", 2 if blk else 0))
    assert blut in (0, 1, 2, 3), "COBALT_BLK1632_LUT must be 0..3"
    bnox = int(os.environ.get("COBALT_BLK1632_NOXB", 0))
    # COBALT_BLK1632_ZF: DENSE4-style zero-fill of pruned slots in the multi-row BLK
    # decoder (drops the per-row masked x-sum).  Changes rounding -> re-gate numerics.
    # DEFAULT 1 for the BLK arms: MEASURED +12.7 % decode at b=4 (40.89 -> 46.09 tok/s)
    # and +5.8 % at b=6.  It is a NUMERICS change of the SAME CLASS the shipped DENSE4 arm
    # already has (pruned weights carried as the zero code, cancelled in `qs - zf*sum(x)`),
    # and it costs 0.17 pt of gemma-3-4b prefill argmax at b=4 / 0.09 pt at b=6 -- see
    # verify_kernel.py.  COBALT_BLK1632_ZF=0 restores the exact-zero decoder.
    bzf = int(os.environ.get("COBALT_BLK1632_ZF", 1 if blk else 0))
    # COBALT_BLK1632_HALFJ: DIAGNOSTIC, halves the (extract, cvt, FMA) triples in the
    # BLK decoder's inner loop.  WRONG numerics; it is the measured CEILING for
    # compressing x onto the 16 survivor lanes instead of expanding codes to 32 slots.
    bhj = int(os.environ.get("COBALT_BLK1632_HALFJ", 0))
    # COBALT_BLK1632_NOOVH: DIAGNOSTIC, deletes the per-granule selector/mask overhead.
    bnv = int(os.environ.get("COBALT_BLK1632_NOOVH", 0))
    # COBALT_XSYNC: DIAGNOSTIC, n extra empty grid.sync per layer.  The slope prices a
    # cooperative barrier in this kernel (what folding the RMS into the GEMV would save).
    xsy = int(os.environ.get("COBALT_XSYNC", 0))
    # COBALT_ATTN_DIAG: DIAGNOSTIC, 1 = no q.k, 2 = no p.V, 3 = neither.  WRONG numerics.
    adg = int(os.environ.get("COBALT_ATTN_DIAG", 0))
    # COBALT_KVONCE: with FUSEPREP, only the key-split block whose range
    # contains `pos` appends k/v to the cache instead of all `split` of them writing the
    # same bytes.  BIT-IDENTICAL by construction.  1 = on (default), 0 = the old behaviour.
    kv1 = int(os.environ.get("COBALT_KVONCE", 1))
    # COBALT_BLK1632_BIAS: magic-bias (prmt) byte->float in the BLK decoder; needs ZF.
    # DEFAULT 1 for the BLK arms (measured, 2026-09-06).  One __byte_perm places the
    # code byte into the mantissa of a float with exponent byte 0x43, giving exactly
    # 128 + c, so the cvt disappears: 2 ops per decoded position instead of 3.  The +128
    # is removed once by folding it into the zero-point term ZF already computes,
    # acc += sf*(qs - (zf+128)*sum(x)) -- hence the hard dependency on ZF.
    # MEASURED (2g slice, 27B, 512->128): +1.9 % decode at b=4 (46.05 -> 46.93) and
    # +1.7 % at b=6.  Unit suite 150/150, worst rel err 7.188e-07.  gemma-3-4b
    # verify_kernel.py --ref-packed: gate (i) 98.7847 % vs CONTROL B 98.8715 % and the
    # FLOOR_ARGMAX bar 98.6100 % => harness PASS, where the ZF-only build FAILS at
    # 98.4375 %.  The honest statement is that BIAS costs nothing at the gate and brings
    # the ZF build's one failing gate back inside the bar -- NOT that it is more accurate
    # (argmax agreement is coarse and tie-sensitive and no mechanism is established).
    # COBALT_BLK1632_BIAS=0 is the revert.
    bbi = int(os.environ.get("COBALT_BLK1632_BIAS", 1 if (blk and bzf) else 0))
    # COBALT_BLK1632_NOLD: DIAGNOSTIC, 1 = no mask load, 2 = no nibble load, 3 = neither.
    bnl = int(os.environ.get("COBALT_BLK1632_NOLD", 0))
    # COBALT_BLK1632_MSCHED: 0 shipped / 1 masks-first / 2 mask lookahead one iteration.
    bms = int(os.environ.get("COBALT_BLK1632_MSCHED", 0))
    fp = int(os.environ.get("COBALT_FUSEPREP", 1))
    flags = ["-O3", "--extended-lambda", "-DCBK_X_INTERLEAVED", "-lineinfo",
             f"-DCBK_MINB={minb}", f"-DCBK_PVB={pvb}", f"-DCBK_RTOL={rtol}",
             f"-DCBK_FUSEPREP={fp}", f"-DCBK_BLK1632_ARM={blk}",
             f"-DCBK_BLK1632_FT={bft}",
             f"-DCBK_GATEUP_RP={grp}",
             f"-DCBK_BLK1632_PFX={bpx}",
             f"-DCBK_GATEUP_SPLIT={gsp}",
             f"-DCBK_BLK1632_LUT={blut}",
             f"-DCBK_BLK1632_NOXB={bnox}",
             f"-DCBK_BLK1632_ZF={bzf}",
             f"-DCBK_BLK1632_HALFJ={bhj}",
             f"-DCBK_BLK1632_NOOVH={bnv}",
             f"-DCBK_XSYNC={xsy}",
             f"-DCBK_ATTN_DIAG={adg}",
             f"-DCBK_KVONCE={kv1}",
             f"-DCBK_BLK1632_BIAS={bbi}",
             f"-DCBK_BLK1632_NOLD={bnl}",
             f"-DCBK_BLK1632_MSCHED={bms}", *arch]
    if m1:
        flags.append("-DCBK_M1ONLY")
    if prof:
        # sub-phase clock64 stamps inside the attention / GEMV phases (block 0 only).
        # Separate build so the shipping kernel carries none of it.
        flags.append("-DCBK_PROF")
    _EXT = load(
        name=(f"cobalt_megakernel_b{minb}" + (f"_pv{pvb}" if pvb != 4 else "")
              + (f"_rt{rtol}" if rtol != 11 else "") + ("_fp" if fp else "_nofp")
              + (f"_blk{blk}" + ("" if bft else "_noft") if blk else "")
              + (f"_rp{grp}" if grp else "") + (f"_pfx{bpx}" if bpx != 1 else "") + ("_gsp" if gsp else "")
              + (f"_lut{blut}" if blut else "") + ("_noxb" if bnox else "") + ("_zf" if bzf else "")
              + ("_hj" if bhj else "") + ("_nov" if bnv else "") + (f"_xs{xsy}" if xsy else "") + (f"_ad{adg}" if adg else "") + ("" if kv1 else "_nokv1") + ("_bias" if bbi else "") + (f"_nold{bnl}" if bnl else "") + (f"_ms{bms}" if bms else "")
              + ("_m1" if m1 else "") + ("_prof" if prof else "")),
        sources=[os.path.join(src, "bindings.cpp"), os.path.join(src, "megakernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=flags,
        extra_include_paths=[src],
        verbose=verbose,
    )
    return _EXT


# --------------------------------------------------------------------------
class KernelRunner:
    # Dynamic shared memory holds ONE staged activation chunk: chunk_len(N) * M bf16,
    # and chunk_len(N) <= CBK_CHUNK_MAX = 2048 for every matrix.  It does NOT scale
    # with hidden_size, so the 27B fits at every M.
    #   M=1 -> 4 KiB  -> 6 blocks/SM (thread-limited)
    #   M=8 -> 32 KiB -> 2-3 blocks/SM
    CHUNK_MAX = 2048
    SPLIT_MAX = 64
    LAYERW_FIELDS = 44          # 4 MatDesc x 9 + 6 norm ptrs + is_sliding + pad
    # v2 attention: a block owns (sequence, kv-head, key-split) and its 8 warps take
    # 32-key chunks, so ~256 keys per block keeps every lane busy.  MUST NOT depend on
    # M or on the grid size (bit-exactness of M=4 vs M=1).
    KEYS_PER_BLOCK = int(os.environ.get("COBALT_KPB", 128))
    # Grid size is a real knob (a phase costs ceil(rows/warps) row-times).  MEASURED on
    # the 27B / 2g at MINB=2: 188 blocks 40.8 tok/s, 141 39.5, 94 39.8.
    DEFAULT_BLOCKS = os.environ.get("COBALT_BLOCKS")

    def __init__(self, model_dir, M=1, max_ctx=1024, device="cuda",
                 smem_bytes=None, verbose=False, config_dir=None,
                 force_bf16=()):
        assert M in (1, 2, 4, 8), "megakernel is instantiated for M in {1,2,4,8}"
        self.ext = _build_ext(verbose)
        self.device = device
        self.M = M
        self.max_ctx = max_ctx
        # names of packed matrices to dequantize to plain bf16 (debug bisect)
        self.force_bf16 = set(force_bf16 or ())
        self._smem_req = smem_bytes
        self.smem = 0
        self.xcap = 0

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
        self.sliding_window = cfg.get("sliding_window", 4096)
        self.nq_dim = self.n_heads * self.head_dim
        self.nkv_dim = self.n_kv * self.head_dim
        self.nqkv = self.nq_dim + 2 * self.nkv_dim
        self.kv_group = self.n_heads // self.n_kv
        self.attn_scale = float(cfg.get("query_pre_attn_scalar", self.head_dim)) ** -0.5
        # HF rounds the embedding scale to bf16 BEFORE the multiply
        self.embed_scale = float(
            torch.tensor(math.sqrt(H), dtype=torch.bfloat16).float())

        assert self.head_dim % 32 == 0 and self.head_dim <= 256, "head_dim must be 32*k, k<=8"
        # dynamic smem = max(one staged activation chunk, the in-block attention
        # reduce buffer [WARPS][head_dim] floats + 2*WARPS)
        # one staged activation chunk, or the in-block attention combine buffer
        # v2: dynamic smem is only the attention block-combine buffer plus the
        # block's KG query vectors.  The GEMV phases use NO shared memory.
        need = (8 * self.head_dim + 2 * 8) * 4 + self.kv_group * self.head_dim * 2
        self.smem = int(self._smem_req) if self._smem_req else need
        self.xcap = self.smem // 2
        assert self.smem >= need, f"smem {self.smem} < required {need}"

        # ---- layer types -------------------------------------------------
        pat = cfg.get("sliding_window_pattern", cfg.get("_sliding_window_pattern", 6))
        if cfg.get("layer_types"):
            lt = list(cfg["layer_types"])
        else:
            lt = ["sliding_attention" if bool((i + 1) % pat) else "full_attention"
                  for i in range(self.n_layers)]
        self.layer_types = lt

        self.model_dir = model_dir
        self.config_dir = config_dir
        self._pf = None
        self._load_weights(model_dir)
        self._alloc()
        self._configure()

    # ------------------------------------------------------------------ load
    def _rope_inv(self, params):
        base = float(params["rope_theta"])
        D = self.head_dim
        inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        if params.get("rope_type", "default") == "linear":
            inv = inv / float(params["factor"])
        elif params.get("rope_type", "default") != "default":
            raise NotImplementedError(params)
        return inv.to(self.device)

    # A MatDesc is 9 int64: data, row_off, scale, zero, col_scale, K, N, G, layout.
    PLAIN_BF16 = -1

    @staticmethod
    def _desc_bf16(t):
        """MatDesc for a row-major bf16 [K, N] weight."""
        assert t.dtype == torch.bfloat16 and t.is_contiguous()
        K, N = t.shape
        return [t.data_ptr(), 0, 0, 0, 0, K, N, 0, KernelRunner.PLAIN_BF16]

    @staticmethod
    def _desc_packed(blob, entry):
        """MatDesc for a CBK1 packed matrix described by manifest.json."""
        base = blob.data_ptr()
        arr = entry["arrays"]
        g = lambda k: (base + arr[k]["off"]) if (k in arr and arr[k]["bytes"]) else 0
        return [g("data"), g("row_off"), g("scale"), g("zero"), g("col_scale"),
                entry["K"], entry["N"], entry["N"] // 128, entry["layout"]]

    def _load_weights(self, model_dir):
        if os.path.exists(os.path.join(model_dir, "manifest.json")):
            self.quantized = True
            self._load_packed(model_dir)
        else:
            self.quantized = False
            self._load_bf16(model_dir)

    def _load_bf16(self, model_dir):
        from safetensors.torch import load_file

        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            shards = sorted(set(json.load(open(idx))["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        W = {}
        for s in shards:
            for k, v in load_file(os.path.join(model_dir, s)).items():
                W[k] = v
        dev, dt = self.device, torch.bfloat16

        self.embed = W["model.embed_tokens.weight"][: self.vocab].to(dev, dt).contiguous()
        self.final_norm = W["model.norm.weight"].to(dev, dt).contiguous()
        self.embed_desc = self._desc_bf16(self.embed)
        self._rope_tables()

        self.lw = []
        rows = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            mats = []
            keep = []
            # q/k/v are ONE fused row space (the packed artifact's `qkv`), so the
            # bf16 arm concatenates them the same way.
            qkvw = torch.cat([W[p + n + ".weight"] for n in
                              ("self_attn.q_proj", "self_attn.k_proj",
                               "self_attn.v_proj")], 0).to(dev, dt).contiguous()
            keep.append(qkvw)
            mats.append(self._desc_bf16(qkvw))
            for nm in ("self_attn.o_proj",):
                t = W[p + nm + ".weight"].to(dev, dt).contiguous()
                keep.append(t)
                mats.append(self._desc_bf16(t))
            g = W[p + "mlp.gate_proj.weight"]
            u = W[p + "mlp.up_proj.weight"]
            # row 2i = gate_i, row 2i+1 = up_i (same interleave the packer uses)
            gu = torch.stack([g, u], 1).reshape(2 * self.inter, self.hidden)
            gu = gu.to(dev, dt).contiguous()
            keep.append(gu)
            mats.append(self._desc_bf16(gu))
            dn = W[p + "mlp.down_proj.weight"].to(dev, dt).contiguous()
            keep.append(dn)
            mats.append(self._desc_bf16(dn))
            ns = [W[p + n].to(dev, dt).contiguous() for n in
                  ("input_layernorm.weight", "post_attention_layernorm.weight",
                   "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                   "self_attn.q_norm.weight", "self_attn.k_norm.weight")]
            keep += ns
            self.lw.append(keep)
            row = []
            for d in mats:
                row += d
            row += [t.data_ptr() for t in ns]
            row += [1 if self.layer_types[i] == "sliding_attention" else 0, 0]
            assert len(row) == self.LAYERW_FIELDS, len(row)
            rows.append(row)
            for kk in list(W.keys()):
                if kk.startswith(p):
                    del W[kk]
        del W
        self.layers_tbl = torch.tensor(rows, dtype=torch.int64, device=self.device).contiguous()

    def _load_packed(self, model_dir):
        """Load a CBK1 artifact (manifest.json + layer_XX.bin + embed.bin + misc.bin)."""
        man = json.load(open(os.path.join(model_dir, "manifest.json")))
        dev = self.device

        def blob(name):
            with open(os.path.join(model_dir, name), "rb") as f:
                b = f.read()
            t = torch.frombuffer(bytearray(b), dtype=torch.uint8).to(dev)
            return t.contiguous()

        self.blobs = {}
        self.blobs["embed"] = blob(man["embed"]["file"])
        self.embed_desc = self._desc_packed(self.blobs["embed"], man["embed"])
        assert self.embed_desc[6] == self.hidden and self.embed_desc[5] >= self.vocab

        # misc.bin: all RMSNorm weights as fp32 -> bf16 on device
        misc = blob(man["misc"]["file"])
        self.norms = {}
        for name, e in man["misc"]["arrays"].items():
            n = e["bytes"] // 4
            v = misc[e["off"]: e["off"] + e["bytes"]].view(torch.float32)[:n]
            self.norms[name] = v.to(torch.bfloat16).contiguous()
        del misc
        self.final_norm = self.norms["model.norm"]
        self._rope_tables()

        nlim = len(man["layers"])
        assert nlim == self.n_layers, (
            f"artifact has {nlim} layers, config says {self.n_layers}; use --layers-limit")
        self.lw = []
        rows = []
        NAMES = ("qkv", "o_proj", "gateup", "down_proj")
        assert man.get("fuse_qkv"), (
            "megakernel v2 needs a --fuse-qkv artifact (…_dense4f); "
            f"{model_dir} has no fused `qkv` matrix")
        for i, lay in enumerate(man["layers"]):
            b = blob(lay["file"])
            self.blobs[i] = b
            row = []
            sub = []
            for nm in NAMES:
                e = lay["matrices"][nm]
                if nm in self.force_bf16:
                    from cobaltkernel.dequant_ref import dequant_matrix
                    t = dequant_matrix(b, e, dev, torch.bfloat16).contiguous()
                    sub.append(t)
                    row += self._desc_bf16(t)
                    continue
                # layouts 1/2/4 (DENSE4/DENSE8/BF16) are column-sliceable; BLK16_32
                # (6/7) is not, but the decode megakernel never slices (it always calls
                # gemv_view with c0=0, len=N), so a full row is all it needs.
                assert e["layout"] in (1, 2, 4, 6, 7) or e["N"] == cbk_chunk_len(e["N"]), (
                    f"layer {i} {nm}: SPARSE layout cannot be column-sliced and "
                    f"N={e['N']} needs chunking")
                row += self._desc_packed(b, e)
            ns = [self.norms[f"{i}.{n}"] for n in
                  ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm",
                   "self_attn.q_norm", "self_attn.k_norm")]
            self.lw.append(ns + sub)
            row += [t.data_ptr() for t in ns]
            row += [1 if self.layer_types[i] == "sliding_attention" else 0, 0]
            assert len(row) == self.LAYERW_FIELDS, len(row)
            rows.append(row)
        self.layers_tbl = torch.tensor(rows, dtype=torch.int64, device=dev).contiguous()

    def _rope_tables(self):
        rp = self.cfg.get("rope_parameters")
        if rp is None:
            rp = {"full_attention": {"rope_type": "default",
                                     "rope_theta": self.cfg.get("rope_theta", 1e6)},
                  "sliding_attention": {"rope_type": "default",
                                        "rope_theta": self.cfg.get("rope_local_base_freq", 1e4)}}
            if self.cfg.get("rope_scaling"):
                rp["full_attention"].update(self.cfg["rope_scaling"])
        self.inv_local = self._rope_inv(rp["sliding_attention"]).contiguous()
        self.inv_global = self._rope_inv(rp["full_attention"]).contiguous()

    # ----------------------------------------------------------------- alloc
    def _alloc(self):
        dev, dt = self.device, torch.bfloat16
        M, H, D = self.M, self.hidden, self.head_dim
        z = lambda *s, d=dt: torch.zeros(*s, dtype=d, device=dev)
        self.kcache = z(self.n_layers, M, self.n_kv, self.max_ctx, D)
        self.vcache = z(self.n_layers, M, self.n_kv, self.max_ctx, D)
        self.tokens = torch.zeros(M, dtype=torch.int32, device=dev)
        self.positions = torch.zeros(M, dtype=torch.int32, device=dev)
        self.rope_cs = z(2, M, D // 2, d=torch.float32)
        self.rope_sn = z(2, M, D // 2, d=torch.float32)
        self.h = z(M, H); self.h2 = z(M, H)
        self.qkv = z(M, self.nqkv)
        self.attn_out = z(M, self.nq_dim)
        self.obuf = z(M, H); self.act = z(M, self.inter); self.dbuf = z(M, H)
        self.partials = z(M, self.n_heads, self.SPLIT_MAX, D + 2, d=torch.float32)
        self.logits = z(M, self.vocab, d=torch.float32)
        # amax needs (blocks+1)*M -- blocks known after configure; allocate max
        self.amax_val = z(4096 * M, d=torch.float32)
        self.amax_idx = z(4096 * M, d=torch.int32)
        # x' staging: up to 3 column-scaled copies of the hidden state (q/k/v)
        self.xbuf = z(3 * H * M)
        self.rsums = z(4096 * M, d=torch.float32)
        # + 8 tail stamps + 16 CBK_PROF sub-phase accumulators (see megakernel.cu)
        self.timings = torch.zeros(self.n_layers * self.STAMPS + 24, dtype=torch.int64,
                                   device=dev)
        self.dbg_h = None

    def _configure(self):
        self.r = self.ext.MegaRunner()
        iv = [self.hidden, self.n_layers, self.n_heads, self.n_kv, self.head_dim,
              self.inter, self.vocab, self.M, self.max_ctx, self.sliding_window,
              self.nq_dim, self.nkv_dim, self.nqkv, self.kv_group, self.xcap]
        fv = [self.eps, self.attn_scale, self.embed_scale]
        self.r.configure(self.layers_tbl, self.embed_desc, self.final_norm,
                         self.inv_local, self.inv_global, self.rope_cs, self.rope_sn,
                         self.kcache, self.vcache,
                         self.tokens, self.positions, self.h, self.h2, self.qkv,
                         self.attn_out, self.obuf, self.act, self.dbuf, self.partials,
                         self.logits, self.amax_val, self.amax_idx,
                         self.xbuf, self.rsums, iv, fv, self.smem,
                         int(os.environ.get("COBALT_MINB", 2)))
        # L2 prefetch of o_proj from the attention phase.
        # MEASURED on the 27B/1g: o_proj 3.37 -> 2.68 ms (-20%) but attention
        # 5.80 -> 6.16 ms, net -0.6% = within run-to-run noise.  Off by default;
        # COBALT_PREFETCH=1 re-enables (may pay off on 2g, whose L2 is 64 MiB).
        self.r.prefetch_o = 1 if os.environ.get('COBALT_PREFETCH') else 0
        # Grid size is a real tuning knob: a phase costs ceil(rows/warps) row-times,
        # so MORE blocks can mean a WORSE last-wave tail on the per-layer matrices.
        _b = os.environ.get('COBALT_BLOCKS')
        if _b:
            # never exceed the co-resident limit -- a cooperative launch above it fails
            self.r.blocks = min(int(_b), self.r.num_blocks())
        self.blocks = self.r.num_blocks()
        self.blocks_per_sm = self.r.num_blocks_per_sm()
        assert self.blocks + 1 <= 4096, "grid too large for the amax buffer"

    # ------------------------------------------------------------------ run
    def reset(self):
        self.kcache.zero_(); self.vcache.zero_()

    def enable_debug(self):
        if self.dbg_h is None:
            self.dbg_h = torch.zeros(self.n_layers + 1, self.M, self.hidden,
                                     dtype=torch.bfloat16, device=self.device)
        return self.dbg_h

    def _split_for(self, maxpos):
        """Number of BLOCK-level key splits per (sequence, kv-head).  Each block's 8
        warps subdivide its range in 32-key chunks and reduce in shared memory; this is
        the number of cross-block partials, and 1 means no reduce phase at all.

        MUST depend only on the context length -- NOT on M and NOT on the grid size --
        or the online-softmax combine order changes with the batch size and a sequence
        decoded at M=4 stops being bit-identical to M=1.
        """
        S = maxpos + 1
        return int(min(self.SPLIT_MAX,
                       max(1, (S + self.KEYS_PER_BLOCK - 1) // self.KEYS_PER_BLOCK)))

    def step(self, tokens, positions, debug=False, timings=False):
        """tokens/positions: list of length <= M (padded with the last entry)."""
        M = self.M
        tk = list(tokens) + [0] * (M - len(tokens))
        ps = list(positions) + [0] * (M - len(positions))
        assert max(ps) < self.max_ctx, f"position {max(ps)} >= max_ctx {self.max_ctx}"
        # tokens/positions travel in the kernel PARAMETER BLOCK -> a decode step is
        # exactly one cudaLaunchCooperativeKernel and zero memcpys.
        self.r.step(tk, ps, self._split_for(max(ps)),
                    self.enable_debug() if debug else None,
                    self.timings if timings else None)
        nxt = self.amax_idx[self.blocks * M: self.blocks * M + M]
        return self.logits, nxt

    def prefill(self, ids_batch):
        """ids_batch: list (len <= M) of equal-length token-id lists.
        Runs the decode kernel token by token.  Returns logits at the last
        position, [M, vocab]."""
        T = len(ids_batch[0])
        assert all(len(x) == T for x in ids_batch), "batch prefill needs equal lengths"
        for t in range(T):
            lg, nxt = self.step([x[t] for x in ids_batch], [t] * len(ids_batch))
        return lg, nxt

    # ------------------------------------------------------- prefill megakernel
    def prefill_runner(self, max_tokens=None):
        """The PREFILL megakernel (`prefill_runner.PrefillRunner`) bound to
        THIS runner's KV cache, sharing THIS runner's weight blobs so the model is not
        loaded twice (a second copy would be another 14 GB at the 27B).

        The KV layout contract is `pf::kv_off()` in csrc/prefill_kernel.cuh, verified
        32/32 by verify_prefill.py test (ii).
        """
        need = max_tokens or self.max_ctx
        pf = getattr(self, "_pf", None)
        if pf is not None and pf.max_tokens >= need:
            return pf
        assert self.quantized, "the prefill megakernel needs a CBK1 packed artifact"
        assert self.M == 1, "prefill-kernel generate() is single-sequence (M=1)"
        from cobaltkernel.prefill_runner import PrefillRunner

        # NB: capture the shared TENSORS, never `self` -- a closure over the decode
        # runner makes a reference cycle (runner -> _pf -> closure -> runner) that
        # refcounting cannot break, so `del runner` would leave the 14 GB of weights
        # resident and the measured peak memory would nearly double.
        shared = {"blobs": self.blobs, "embed_desc": list(self.embed_desc),
                  "norms": self.norms, "final_norm": self.final_norm}

        class _SharedWeights(PrefillRunner):
            """PrefillRunner that reuses the decode runner's already-resident blobs."""

            _shared = shared

            def _load_packed(self, model_dir):
                man = json.load(open(os.path.join(model_dir, "manifest.json")))
                assert man.get("fuse_qkv"), "prefill needs a --fuse-qkv artifact"
                sh = self._shared
                self.blobs = sh["blobs"]
                self.embed_desc = list(sh["embed_desc"])
                self.norms = sh["norms"]
                self.final_norm = sh["final_norm"]
                self._rope_tables()
                rows, self.keep = [], []
                for i, lay in enumerate(man["layers"]):
                    b = self.blobs[i]
                    row = []
                    for nm in ("qkv", "o_proj", "gateup", "down_proj"):
                        e = lay["matrices"][nm]
                        assert e["layout"] == self._want_layout(nm), (
                            f"{nm}: layout {e['layout']} vs prefill build layout "
                            f"{self._want_layout(nm)} (COBALT_BLK1632/_O)")
                        assert e["N"] % 128 == 0, f"{nm}: N must be a multiple of 128"
                        row += self._desc(b, e)
                    ns = [self.norms[f"{i}.{n}"] for n in
                          ("input_layernorm", "post_attention_layernorm",
                           "pre_feedforward_layernorm", "post_feedforward_layernorm",
                           "self_attn.q_norm", "self_attn.k_norm")]
                    self.keep.append(ns)
                    row += [t.data_ptr() for t in ns]
                    row += [1 if self.layer_types[i] == "sliding_attention" else 0, 0]
                    assert len(row) == 44
                    rows.append(row)
                assert self.nq_dim % 256 == 0 and self.nkv_dim % 256 == 0
                self.layers_tbl = torch.tensor(rows, dtype=torch.int64,
                                               device=self.device).contiguous()

        self._pf = _SharedWeights(self.model_dir, config_dir=self.config_dir,
                                  max_tokens=need, max_logit_rows=1,
                                  kv=(self.kcache, self.vcache))
        return self._pf

    def prefill_kernel(self, ids, pos0=0):
        """Fill the KV cache for `ids` with ONE prefill-megakernel launch.
        Returns the fp32 logits [1, vocab] at the LAST prompt position."""
        return self.prefill_runner(len(ids)).prefill(list(ids), pos0=pos0)

    def generate(self, prompt_ids, n_new, use_prefill_kernel=True):
        """End-to-end greedy generation: ONE prefill-megakernel launch for the prompt,
        then ONE decode-megakernel launch per generated token.
        Returns the list of `n_new` generated token ids."""
        prompt_ids = list(prompt_ids)
        T = len(prompt_ids)
        assert T + n_new <= self.max_ctx
        if use_prefill_kernel:
            lg = self.prefill_kernel(prompt_ids)
            cur = int(lg[0].argmax())
        else:
            _, nxt = self.prefill([prompt_ids])
            cur = int(nxt[0])
        out = [cur]
        for i in range(n_new - 1):
            _, nxt = self.step([cur], [T + i])
            cur = int(nxt[0])
            out.append(cur)
        return out[:n_new]

    # Hook for the future mma.sync chunked-prefill kernel: replace this method
    # with a call into a second entry point that processes T tokens per launch.
    prefill_hook = None

    def weight_bytes(self):
        """Bytes of weights streamed per decode step (what the BW ceiling divides)."""
        if self.quantized:
            n = sum(int(b.numel()) for b in self.blobs.values())
        else:
            n = self.embed.numel() * 2
            for keep in self.lw:
                for t in keep:
                    n += t.numel() * t.element_size()
        n += self.final_norm.numel() * 2
        return n

    PHASES = ["h", "qkv", "attn", "attn_reduce", "o_proj", "h2", "gateup", "down"]
    STAMPS = 11                 # grid.sync phases per layer in megakernel.cu v2
    # stamp index within a layer -> reported phase
    _MAP = {"h": (0,), "qkv": (1, 2), "attn": (3, 4), "attn_reduce": (5,),
            "o_proj": (6,), "h2": (7,), "gateup": (8, 9), "down": (10,)}

    def phase_times_us(self, sm_clock_hz=2.43e9):
        t = self.timings.cpu().tolist()
        n = self.n_layers * self.STAMPS + 4
        d = [(t[i + 1] - t[i]) / sm_clock_hz * 1e6 for i in range(n)]
        agg = {k: 0.0 for k in self.PHASES}
        for L in range(self.n_layers):
            for k, idx in self._MAP.items():
                for j in idx:
                    agg[k] += d[L * self.STAMPS + j]
        base = self.n_layers * self.STAMPS
        agg["tail_h"] = d[base]
        agg["lm_head"] = d[base + 1] + d[base + 2]
        agg["argmax"] = d[base + 3]
        agg["total"] = (t[n] - t[0]) / sm_clock_hz * 1e6
        return agg

    # CBK_PROF accumulators: block 0's OWN work inside a phase, summed over layers.
    # The grid.sync-delimited phase time minus this is the barrier wait (i.e. how much
    # of the phase block 0 spends waiting for the slowest block).
    PROF_SLOTS = ["attn_prep", "attn_sq", "attn_walk", "attn_combine", "attn_reduce",
                  "qkv_gemv", "o_gemv", "gateup_gemv", "down_gemv", "lm_head_gemv"]

    def prof_times_us(self, sm_clock_hz=2.43e9):
        t = self.timings.cpu().tolist()
        base = self.n_layers * self.STAMPS + 8
        return {k: t[base + i] / sm_clock_hz * 1e6
                for i, k in enumerate(self.PROF_SLOTS)}

    def reset_timings(self):
        self.timings.zero_()
