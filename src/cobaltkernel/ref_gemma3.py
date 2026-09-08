"""Dependency-light pure-torch reference implementation of the Gemma3 *text*
forward pass (Gemma3ForCausalLM / gemma3_text).

No `transformers` import at runtime -- only torch + safetensors + json.
This is the executable specification that the CUDA kernel must reproduce
bit-for-bit-ish; this file IS the specification (see docs/KERNELS.md).

Public API
----------
    state = load_state(model_dir, device="cuda", dtype=torch.bfloat16)
    logits, kv = prefill(state, ids)                 # ids: LongTensor [T]
    logits      = forward_step(state, token_id, pos, kv)

`kv` is a list (one entry per layer) of dicts {"k": [1,H_kv,T,D], "v": ...}.
Everything is batch-size 1 (the kernel target is bs=1 decode); the math is
written so that a batch dim could be added trivially.
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
class Gemma3RefConfig:
    """Normalises BOTH config dialects:

    * old (medgemma-27b, transformers 4.51): `rope_theta`, `rope_local_base_freq`,
      `rope_scaling`, `sliding_window_pattern`, no `layer_types`.
    * new (gemma-3-4b, transformers >=5.14): `rope_parameters` dict keyed by
      layer type, explicit `layer_types` list.
    """

    def __init__(self, cfg: dict):
        self.hidden_size = cfg["hidden_size"]
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.num_attention_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = cfg["intermediate_size"]
        self.vocab_size = cfg["vocab_size"]
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.sliding_window = cfg.get("sliding_window", 4096)
        self.query_pre_attn_scalar = cfg.get("query_pre_attn_scalar", 256)
        self.attn_logit_softcapping = cfg.get("attn_logit_softcapping", None)
        self.final_logit_softcapping = cfg.get("final_logit_softcapping", None)
        self.tie_word_embeddings = cfg.get("tie_word_embeddings", True)
        self.hidden_activation = cfg.get("hidden_activation", "gelu_pytorch_tanh")
        assert self.hidden_activation == "gelu_pytorch_tanh", self.hidden_activation

        # ---- layer types -------------------------------------------------
        pattern = cfg.get("sliding_window_pattern", cfg.get("_sliding_window_pattern", 6))
        if cfg.get("layer_types"):
            self.layer_types = list(cfg["layer_types"])
        else:
            # transformers Gemma3TextConfig.__post_init__:
            #   "sliding_attention" if bool((i + 1) % pattern) else "full_attention"
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layer_types) == self.num_hidden_layers

        # ---- rope --------------------------------------------------------
        rp = cfg.get("rope_parameters")
        if rp is None:
            rp = {
                "full_attention": {"rope_type": "default", "rope_theta": cfg.get("rope_theta", 1e6)},
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": cfg.get("rope_local_base_freq", 1e4),
                },
            }
            if cfg.get("rope_scaling"):
                rp["full_attention"].update(cfg["rope_scaling"])
        self.rope_parameters = rp

        # ---- derived scalars used by the kernel --------------------------
        self.attn_scale = float(self.query_pre_attn_scalar) ** -0.5
        self.embed_scale = float(self.hidden_size) ** 0.5
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads


def _inv_freq(params: dict, head_dim: int, device) -> torch.Tensor:
    """inv_freq per transformers `compute_default_rope_parameters` /
    `_compute_linear_scaling_rope_parameters`.  attention_scaling is 1.0 for
    both 'default' and 'linear', so cos/sin are NOT rescaled."""
    base = float(params["rope_theta"])
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    rtype = params.get("rope_type", "default")
    if rtype == "linear":
        inv = inv / float(params["factor"])
    elif rtype != "default":
        raise NotImplementedError(f"rope_type={rtype}")
    return inv


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load_state(model_dir, device="cuda", dtype=torch.bfloat16):
    cfg = Gemma3RefConfig(json.load(open(os.path.join(model_dir, "config.json"))))

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    W = {}
    if os.path.exists(idx_path):
        shards = sorted(set(json.load(open(idx_path))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    for s in shards:
        for k, v in load_file(os.path.join(model_dir, s)).items():
            W[k] = v.to(device=device, dtype=dtype)

    if cfg.tie_word_embeddings:
        # HF ties lm_head to embed_tokens; a materialised lm_head.weight in the
        # checkpoint (the gemma-3-4b text extraction has one) is ignored, exactly
        # as `from_pretrained` does.  Drop it so we do not pay 1.3 GB twice.
        W.pop("lm_head.weight", None)
        W["lm_head.weight"] = W["model.embed_tokens.weight"]
    assert "lm_head.weight" in W

    rope = {}
    for lt in sorted(set(cfg.layer_types)):
        rope[lt] = _inv_freq(cfg.rope_parameters[lt], cfg.head_dim, device)

    return {"cfg": cfg, "W": W, "rope": rope, "device": device, "dtype": dtype}


# --------------------------------------------------------------------------
# primitives (dtype policy copied from HF)
# --------------------------------------------------------------------------
def rms_norm(x, w, eps):
    """Gemma3RMSNorm: fp32 normalise, multiply by (1 + w) in fp32, cast back."""
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + w.float())
    return out.type_as(x)


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_cos_sin(inv_freq, positions, dtype):
    """positions: 1-D LongTensor.  Returns cos,sin of shape [T, head_dim] in
    `dtype` (HF casts to hidden_states.dtype == bf16 *before* applying)."""
    freqs = positions.float()[:, None] * inv_freq[None, :]      # [T, D/2] fp32
    emb = torch.cat((freqs, freqs), dim=-1)                     # [T, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: [1, H, T, D]; cos/sin: [T, D]
    cos = cos[None, None]
    sin = sin[None, None]
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


# --------------------------------------------------------------------------
# one decoder layer
# --------------------------------------------------------------------------
def _attention(cfg, W, p, h, cos, sin, kv_layer, layer_type, q_positions, append=True):
    """h: [1, T, hidden]  (already input_layernorm'ed).  Returns o_proj output."""
    T = h.shape[1]
    D = cfg.head_dim
    q = F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, T, cfg.num_attention_heads, D).transpose(1, 2)
    k = F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, T, cfg.num_key_value_heads, D).transpose(1, 2)
    v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, T, cfg.num_key_value_heads, D).transpose(1, 2)

    q = rms_norm(q, W[p + "self_attn.q_norm.weight"], cfg.rms_norm_eps)
    k = rms_norm(k, W[p + "self_attn.k_norm.weight"], cfg.rms_norm_eps)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if append:
        kv_layer["k"] = k if kv_layer["k"] is None else torch.cat([kv_layer["k"], k], dim=2)
        kv_layer["v"] = v if kv_layer["v"] is None else torch.cat([kv_layer["v"], v], dim=2)
    K, V = kv_layer["k"], kv_layer["v"]
    S = K.shape[2]

    Kr = repeat_kv(K, cfg.num_kv_groups)
    Vr = repeat_kv(V, cfg.num_kv_groups)

    attn = torch.matmul(q, Kr.transpose(2, 3)) * cfg.attn_scale     # bf16 matmul, fp32 accum
    if cfg.attn_logit_softcapping is not None:
        c = cfg.attn_logit_softcapping
        attn = torch.tanh(attn / c) * c

    kv_pos = torch.arange(S, device=q.device)
    qp = q_positions[:, None]
    allowed = kv_pos[None, :] <= qp                                 # causal
    if layer_type == "sliding_attention":
        allowed &= kv_pos[None, :] > qp - cfg.sliding_window        # kv > q - W
    attn = attn.masked_fill(~allowed[None, None], float("-inf"))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)  # fp32 softmax
    out = torch.matmul(attn, Vr)                                     # [1,H,T,D]
    out = out.transpose(1, 2).reshape(1, T, -1)
    return F.linear(out, W[p + "self_attn.o_proj.weight"])


def _mlp(cfg, W, p, h):
    g = F.linear(h, W[p + "mlp.gate_proj.weight"])
    u = F.linear(h, W[p + "mlp.up_proj.weight"])
    return F.linear(gelu_tanh(g) * u, W[p + "mlp.down_proj.weight"])


def _layer(cfg, W, i, h, cos_map, sin_map, kv, q_positions, append=True):
    p = f"model.layers.{i}."
    lt = cfg.layer_types[i]
    eps = cfg.rms_norm_eps

    residual = h
    x = rms_norm(h, W[p + "input_layernorm.weight"], eps)
    x = _attention(cfg, W, p, x, cos_map[lt], sin_map[lt], kv[i], lt, q_positions, append)
    x = rms_norm(x, W[p + "post_attention_layernorm.weight"], eps)   # post-norm on the branch
    h = residual + x

    residual = h
    x = rms_norm(h, W[p + "pre_feedforward_layernorm.weight"], eps)
    x = _mlp(cfg, W, p, x)
    x = rms_norm(x, W[p + "post_feedforward_layernorm.weight"], eps)
    h = residual + x
    return h


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
def new_kv(state):
    return [{"k": None, "v": None} for _ in range(state["cfg"].num_hidden_layers)]


def _embed(state, ids):
    cfg, W = state["cfg"], state["W"]
    e = W["model.embed_tokens.weight"][ids]
    # HF: embed_scale is an fp32 buffer cast to the *weight dtype* (bf16!)
    # before the multiply -- sqrt(2560)=50.59644 becomes bf16 50.5.
    scale = torch.tensor(cfg.embed_scale, dtype=state["dtype"], device=e.device)
    return (e * scale).unsqueeze(0)


def _run(state, ids, positions, kv, append=True):
    cfg, W = state["cfg"], state["W"]
    h = _embed(state, ids)
    cos_map, sin_map = {}, {}
    for lt, inv in state["rope"].items():
        cos_map[lt], sin_map[lt] = rope_cos_sin(inv, positions, state["dtype"])
    for i in range(cfg.num_hidden_layers):
        h = _layer(cfg, W, i, h, cos_map, sin_map, kv, positions, append)
    h = rms_norm(h, W["model.norm.weight"], cfg.rms_norm_eps)
    logits = F.linear(h, W["lm_head.weight"])
    if cfg.final_logit_softcapping is not None:
        c = cfg.final_logit_softcapping
        logits = torch.tanh(logits / c) * c
    return logits


@torch.no_grad()
def prefill(state, ids, kv=None):
    """ids: LongTensor [T].  Returns (logits [1,T,V], kv)."""
    if kv is None:
        kv = new_kv(state)
    ids = ids.to(state["device"])
    positions = torch.arange(ids.shape[0], device=state["device"])
    return _run(state, ids, positions, kv), kv


@torch.no_grad()
def forward_step(state, token_id, pos, kv):
    """One decode step.  token_id: int, pos: int (absolute position).
    Returns logits [V]."""
    ids = torch.tensor([token_id], device=state["device"], dtype=torch.long)
    positions = torch.tensor([pos], device=state["device"])
    return _run(state, ids, positions, kv)[0, 0]
