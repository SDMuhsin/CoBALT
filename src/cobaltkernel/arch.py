"""Model-family dispatch shared by the quantizer, packer, runners, oracle and verifier.

Two families are wired:

  gemma3  Gemma3 text (medgemma-27b, gemma-3-4b): dual RoPE (local/global), 1-in-N sliding
          layers, FOUR RMSNorms per layer with the (1 + w) Gemma convention, QK-norm, tied
          embedding * sqrt(hidden), GeGLU.
  llama   Mistral / Llama: one RoPE, uniform sliding window (Mistral) or none, TWO RMSNorms
          per layer with the plain w convention, no QK-norm, untied lm_head, no embedding
          scale, SwiGLU.

Everything here is read from config.json.  The kernel takes the differences as RUNTIME
values -- null norm pointers (bypass), `norm_plus_one`, `act_gelu`, a separate `lm_head`
MatDesc -- so a Gemma3 artifact runs through exactly the arithmetic it did before.

The six kernel norm ROLES, in LayerW order:
  in_ln, post_attn_ln, pre_ff_ln, post_ff_ln, q_norm, k_norm
"""
import math

import torch

ROLES = ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm")

# role -> HF parameter name (None = the kernel bypasses that norm: plain residual add /
# no QK-norm).  NOTE Mistral/Llama's `post_attention_layernorm` is the norm BEFORE the
# MLP, i.e. the kernel's `pre_ff_ln` role -- the name is the same as Gemma3's but the
# position in the layer is not.
_NORMS = {
    "gemma3": ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
               "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"),
    "llama": ("input_layernorm", None, "post_attention_layernorm", None, None, None),
}


def family_of(cfg):
    mt = cfg.get("model_type", "") if isinstance(cfg, dict) else getattr(cfg, "model_type", "")
    if mt.startswith("gemma3"):
        return "gemma3"
    if mt in ("mistral", "llama"):
        return "llama"
    raise NotImplementedError(f"model_type={mt!r}: only gemma3 / mistral / llama are wired")


def norm_names(cfg):
    """HF parameter names for the six kernel roles (None = bypassed)."""
    return _NORMS[family_of(cfg)]


def all_norm_names():
    """Every norm name any family uses (the packer copies whichever exist)."""
    out = []
    for names in _NORMS.values():
        for n in names:
            if n and n not in out:
                out.append(n)
    return out


def flags(cfg):
    """Runtime arch flags the kernels take."""
    fam = family_of(cfg)
    H = cfg["hidden_size"]
    if fam == "gemma3":
        act = cfg.get("hidden_activation", "gelu_pytorch_tanh")
        assert act == "gelu_pytorch_tanh", act
        # HF rounds the embedding scale to bf16 BEFORE the multiply
        es = float(torch.tensor(math.sqrt(H), dtype=torch.bfloat16).float())
        return dict(family=fam, norm_plus_one=1, act_gelu=1, embed_scale=es,
                    tied=bool(cfg.get("tie_word_embeddings", True)))
    act = cfg.get("hidden_act", "silu")
    assert act == "silu", act
    return dict(family=fam, norm_plus_one=0, act_gelu=0, embed_scale=1.0,
                tied=bool(cfg.get("tie_word_embeddings", False)))


def layer_types(cfg):
    fam = family_of(cfg)
    L = cfg["num_hidden_layers"]
    if cfg.get("layer_types"):
        lt = list(cfg["layer_types"])
        assert len(lt) == L
        return lt
    if fam == "gemma3":
        pat = cfg.get("sliding_window_pattern", cfg.get("_sliding_window_pattern", 6))
        return ["sliding_attention" if bool((i + 1) % pat) else "full_attention"
                for i in range(L)]
    # mistral: uniform sliding window iff config.sliding_window is set (HF uses the same rule)
    return ["sliding_attention" if cfg.get("sliding_window") else "full_attention"] * L


def sliding_window(cfg):
    return int(cfg.get("sliding_window") or 4096)


def rope_params(cfg):
    """{'sliding_attention': {...}, 'full_attention': {...}} in the ref_gemma3 dialect."""
    rp = cfg.get("rope_parameters")
    if rp and "full_attention" in rp:
        return rp
    if family_of(cfg) == "gemma3":
        rp = {"full_attention": {"rope_type": "default", "rope_theta": cfg.get("rope_theta", 1e6)},
              "sliding_attention": {"rope_type": "default",
                                    "rope_theta": cfg.get("rope_local_base_freq", 1e4)}}
        if cfg.get("rope_scaling"):
            rp["full_attention"].update(cfg["rope_scaling"])
        return rp
    # llama family: ONE rope for every layer type
    p = {"rope_type": "default", "rope_theta": float(cfg.get("rope_theta", 1e4))}
    rs = cfg.get("rope_scaling") or (rp if isinstance(rp, dict) else None)
    if rs:
        rt = rs.get("rope_type", rs.get("type", "default"))
        if rt != "default":
            raise NotImplementedError(f"rope_scaling {rs} not wired for the llama family")
    return {"full_attention": dict(p), "sliding_attention": dict(p)}


def attn_scale(cfg):
    hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return float(cfg.get("query_pre_attn_scalar", hd)) ** -0.5


def ref_module(cfg):
    """The executable torch specification for this family."""
    if family_of(cfg) == "gemma3":
        from cobaltkernel import ref_gemma3 as R
    else:
        from cobaltkernel import ref_llama as R
    return R
