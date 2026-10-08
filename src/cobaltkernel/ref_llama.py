"""Dependency-light pure-torch reference forward pass for the Llama family
(MistralForCausalLM / LlamaForCausalLM), the executable specification the CUDA
kernels must reproduce for these models.  Same public API as ref_gemma3.py:

    state = load_state(model_dir, device="cuda", dtype=torch.bfloat16)
    logits, kv = prefill(state, ids)                 # ids: LongTensor [T]
    logits      = forward_step(state, token_id, pos, kv)

dtype policy copied from transformers 4.55 modeling_mistral / modeling_llama:
  * RMSNorm: normalise in fp32, cast to bf16, THEN multiply by w (bf16 x bf16).
  * RoPE: cos/sin fp32 -> bf16; q*cos + rotate_half(q)*sin in bf16.
  * attention: bf16 matmuls, fp32 softmax cast back to bf16 before @V.
  * MLP: down(silu(gate(x)) * up(x)) in bf16.
  * residual adds in bf16; no embedding scale; untied lm_head.
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class LlamaRefConfig:
    def __init__(self, cfg: dict):
        self.model_type = cfg.get("model_type", "llama")
        assert self.model_type in ("mistral", "llama"), self.model_type
        self.hidden_size = cfg["hidden_size"]
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.num_attention_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = cfg["intermediate_size"]
        self.vocab_size = cfg["vocab_size"]
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        self.sliding_window = cfg.get("sliding_window")          # None = full causal
        self.tie_word_embeddings = cfg.get("tie_word_embeddings", False)
        self.hidden_act = cfg.get("hidden_act", "silu")
        assert self.hidden_act == "silu", self.hidden_act
        self.attn_logit_softcapping = None
        self.final_logit_softcapping = None
        assert not cfg.get("attention_bias", False), "attention bias is not wired"
        rs = cfg.get("rope_scaling")
        if rs and rs.get("rope_type", rs.get("type", "default")) != "default":
            raise NotImplementedError(f"rope_scaling {rs}")
        self.rope_theta = float(cfg.get("rope_theta", 1e4))
        # one layer type for the whole model (uniform sliding window or none)
        lt = "sliding_attention" if self.sliding_window else "full_attention"
        self.layer_types = [lt] * self.num_hidden_layers
        if self.sliding_window is None:
            self.sliding_window = 1 << 30
        self.attn_scale = float(self.head_dim) ** -0.5
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads


def _inv_freq(theta, head_dim, device):
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))


def load_state(model_dir, device="cuda", dtype=torch.bfloat16):
    cfg = LlamaRefConfig(json.load(open(os.path.join(model_dir, "config.json"))))
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
        W.pop("lm_head.weight", None)
        W["lm_head.weight"] = W["model.embed_tokens.weight"]
    assert "lm_head.weight" in W, "untied model without lm_head.weight"
    inv = _inv_freq(cfg.rope_theta, cfg.head_dim, device)
    rope = {lt: inv for lt in set(cfg.layer_types)}
    return {"cfg": cfg, "W": W, "rope": rope, "device": device, "dtype": dtype}


# ---------------------------------------------------------------- primitives
def rms_norm(x, w, eps):
    """LlamaRMSNorm / MistralRMSNorm: fp32 normalise, cast back, then w * x."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_cos_sin(inv_freq, positions, dtype):
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    return (x * cos[None, None]) + (rotate_half(x) * sin[None, None])


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


# ---------------------------------------------------------------- one layer
def _attention(cfg, W, p, h, cos, sin, kv_layer, layer_type, q_positions, append=True):
    T = h.shape[1]
    D = cfg.head_dim
    q = F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, T, cfg.num_attention_heads, D).transpose(1, 2)
    k = F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, T, cfg.num_key_value_heads, D).transpose(1, 2)
    v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, T, cfg.num_key_value_heads, D).transpose(1, 2)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if append:
        kv_layer["k"] = k if kv_layer["k"] is None else torch.cat([kv_layer["k"], k], dim=2)
        kv_layer["v"] = v if kv_layer["v"] is None else torch.cat([kv_layer["v"], v], dim=2)
    K, V = kv_layer["k"], kv_layer["v"]
    S = K.shape[2]
    Kr = repeat_kv(K, cfg.num_kv_groups)
    Vr = repeat_kv(V, cfg.num_kv_groups)
    attn = torch.matmul(q, Kr.transpose(2, 3)) * cfg.attn_scale
    kv_pos = torch.arange(S, device=q.device)
    qp = q_positions[:, None]
    allowed = kv_pos[None, :] <= qp
    if layer_type == "sliding_attention":
        allowed &= kv_pos[None, :] > qp - cfg.sliding_window      # HF sliding_window_overlay
    attn = attn.masked_fill(~allowed[None, None], float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, Vr)
    out = out.transpose(1, 2).reshape(1, T, -1)
    return F.linear(out, W[p + "self_attn.o_proj.weight"])


def _mlp(cfg, W, p, h):
    g = F.linear(h, W[p + "mlp.gate_proj.weight"])
    u = F.linear(h, W[p + "mlp.up_proj.weight"])
    return F.linear(F.silu(g) * u, W[p + "mlp.down_proj.weight"])


def _layer(cfg, W, i, h, cos_map, sin_map, kv, q_positions, append=True):
    p = f"model.layers.{i}."
    lt = cfg.layer_types[i]
    eps = cfg.rms_norm_eps
    residual = h
    x = rms_norm(h, W[p + "input_layernorm.weight"], eps)
    x = _attention(cfg, W, p, x, cos_map[lt], sin_map[lt], kv[i], lt, q_positions, append)
    h = residual + x
    residual = h
    x = rms_norm(h, W[p + "post_attention_layernorm.weight"], eps)   # the PRE-MLP norm
    x = _mlp(cfg, W, p, x)
    h = residual + x
    return h


# ---------------------------------------------------------------- entry points
def new_kv(state):
    return [{"k": None, "v": None} for _ in range(state["cfg"].num_hidden_layers)]


def _embed(state, ids):
    return state["W"]["model.embed_tokens.weight"][ids].unsqueeze(0)


def _run(state, ids, positions, kv, append=True):
    cfg, W = state["cfg"], state["W"]
    h = _embed(state, ids)
    cos_map, sin_map = {}, {}
    for lt, inv in state["rope"].items():
        cos_map[lt], sin_map[lt] = rope_cos_sin(inv, positions, state["dtype"])
    for i in range(cfg.num_hidden_layers):
        h = _layer(cfg, W, i, h, cos_map, sin_map, kv, positions, append)
    h = rms_norm(h, W["model.norm.weight"], cfg.rms_norm_eps)
    return F.linear(h, W["lm_head.weight"])


@torch.no_grad()
def prefill(state, ids, kv=None):
    if kv is None:
        kv = new_kv(state)
    ids = ids.to(state["device"])
    positions = torch.arange(ids.shape[0], device=state["device"])
    return _run(state, ids, positions, kv), kv


@torch.no_grad()
def forward_step(state, token_id, pos, kv):
    ids = torch.tensor([token_id], device=state["device"], dtype=torch.long)
    positions = torch.tensor([pos], device=state["device"])
    return _run(state, ids, positions, kv)[0, 0]
