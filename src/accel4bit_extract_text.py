"""Extract the text-only Gemma3ForCausalLM from a multimodal Gemma3ForConditionalGeneration checkpoint
(PROTOCOL: smoke model unsloth/gemma-3-4b-it is multimodal; quantize/eval the TEXT model only).
Usage: python accel4bit_extract_text.py <src_repo_or_dir> <out_dir>
"""
import sys, os, json, shutil, time, torch
from transformers import AutoConfig, AutoTokenizer, Gemma3ForConditionalGeneration, Gemma3ForCausalLM, Gemma3TextConfig

src, out = sys.argv[1], sys.argv[2]
t0 = time.time()
mm = Gemma3ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16, device_map="cpu")
tcfg = mm.config.text_config
if not isinstance(tcfg, Gemma3TextConfig):
    tcfg = Gemma3TextConfig(**tcfg.to_dict())
tcfg.architectures = ["Gemma3ForCausalLM"]
causal = Gemma3ForCausalLM(tcfg).to(torch.bfloat16)
lm_sd = mm.model.language_model.state_dict()
missing, unexpected = causal.model.load_state_dict(lm_sd, strict=True), None
causal.lm_head.weight = mm.lm_head.weight  # tied to embed_tokens in Gemma3
assert torch.equal(causal.lm_head.weight, mm.model.language_model.embed_tokens.weight), "lm_head not tied?"
# sanity: same next-token logits on a text prompt
tok = AutoTokenizer.from_pretrained(src)
ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    a = mm(input_ids=ids).logits[0, -1].float()
    b = causal(input_ids=ids).logits[0, -1].float()
print("max|logit diff| =", (a - b).abs().max().item(), "argmax mm/causal:", a.argmax().item(), b.argmax().item(), repr(tok.decode(b.argmax())))
os.makedirs(out, exist_ok=True)
causal.config.torch_dtype = "bfloat16"
causal.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(out)
gen = getattr(mm, "generation_config", None)
if gen is not None:
    gen.save_pretrained(out)
cfg = json.load(open(os.path.join(out, "config.json")))
print("saved config keys:", cfg.get("architectures"), cfg.get("model_type"), cfg.get("num_hidden_layers"), cfg.get("vocab_size"))
n_params = sum(p.numel() for p in causal.parameters())
print("n_params(text model, incl tied embed) =", n_params, "elapsed %.0fs" % (time.time() - t0))
print("EXTRACT_OK")
