#!/usr/bin/env python
"""accel4bit ARM C: NVFP4 (W4A4) / NVFP4A16 PTQ with llm-compressor QuantizationModifier.

Base recipe: temp/llm-compressor/examples/quantization_w4a4_fp4/llama3_example.py (+ the
multimodal ignore list from examples/quantization_w4a4_fp4/gemma4_example.py and
examples/multimodal_vision/gemma3_example.py). No hand-written quantizer: llm-compressor does
   * weights : FP4 E2M1, per-16 block FP8-E4M3 scales + per-tensor FP32 global scale (RTN)
   * activations (NVFP4 only): dynamic per-16 block scales at inference, static per-tensor
     global scale calibrated here from the calibration set (this is what needs the data).

Calibration per PROTOCOL: 512 x 2048 tokens of HuggingFaceH4/ultrachat_200k train_sft,
shuffled with seed 42, chat-templated with the model's own template.

Multimodal checkpoints (Gemma3ForConditionalGeneration, e.g. gemma-3-4b-it): the full
multimodal model is loaded, calibrated with TEXT ONLY (no pixel_values), and every Linear in
vision_tower / multi_modal_projector plus lm_head / embeddings is put on the `ignore` list, so
only the text decoder linears are quantized (vLLM then loads the vision tower in bf16).
"""
import argparse
import json
import os
import sys
import time

import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="local HF snapshot dir")
    p.add_argument("--out", required=True, help="output dir for compressed checkpoint")
    p.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "NVFP4A16"])
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pipeline", default=None, help="llm-compressor calibration pipeline "
                   "(None=inferred -> sequential; 'basic' loads the whole model on GPU)")
    p.add_argument("--sequential-targets", default=None,
                   help="comma-separated module class names for the sequential pipeline")
    p.add_argument("--offload-device", default="cpu",
                   help="sequential_offload_device for cached intermediates")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--split", default="train_sft")
    return p.parse_args()


def main():
    a = parse()
    t0 = time.time()
    torch.manual_seed(a.seed)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    import llmcompressor, compressed_tensors, transformers

    cfg = json.load(open(os.path.join(a.model_path, "config.json")))
    archs = cfg.get("architectures", [])
    multimodal = any("ConditionalGeneration" in x for x in archs)
    print(f"[quant] archs={archs} multimodal={multimodal} scheme={a.scheme}", flush=True)
    print(f"[quant] versions llmcompressor={llmcompressor.__version__} "
          f"compressed_tensors={compressed_tensors.__version__} "
          f"transformers={transformers.__version__} torch={torch.__version__}", flush=True)

    if multimodal:
        # TEXT-ONLY path (same as src/accel4bit_gptq_quant.py, ARM A): the multimodal class's forward
        # is untraceable by llm-compressor's sequential tracer (first attempt died at subgraph 35/35
        # with `NotImplementedError: register_meta for torch.nonzero()`; log
        # results/accel4bit/gemma-3-4b/nvfp4/quant_attempt1_FAILED_mm_class.log). Rebuild the text
        # model as Gemma3ForCausalLM from model.language_model + lm_head (dropping vision_tower and
        # multi_modal_projector), i.e. exactly the class used by medgemma-27b-text-it.
        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
        mm = Gemma3ForConditionalGeneration.from_pretrained(a.model_path, dtype=torch.bfloat16)
        text_cfg = mm.config.text_config
        text_cfg.architectures = ["Gemma3ForCausalLM"]
        model = Gemma3ForCausalLM._from_config(text_cfg, dtype=torch.bfloat16)
        model.model.load_state_dict(mm.model.language_model.state_dict(), strict=True)
        model.lm_head.weight = mm.lm_head.weight
        model.tie_weights()
        model.generation_config = mm.generation_config
        assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight), "lm_head must be tied"
        print("[quant] text-only Gemma3ForCausalLM built from language_model (dropped vision_tower + "
              "multi_modal_projector); strict load ok, tied lm_head", flush=True)
        del mm
        processor = AutoTokenizer.from_pretrained(a.model_path)
        tokenizer = processor
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
        processor = AutoTokenizer.from_pretrained(a.model_path)
        tokenizer = processor
    print(f"[quant] model loaded on {next(model.parameters()).device} in {time.time()-t0:.0f}s",
          flush=True)

    # ---- calibration data: IDENTICAL to ARM A (src/accel4bit_gptq_quant.py): llm-compressor's
    # built-in `ultrachat_200k` loader, first N samples of train_sft (unshuffled), chat-templated with the
    # model's tokenizer template, truncated to max_seq_len. (torch.manual_seed(42) is set above.)
    ds = "ultrachat_200k"
    splits = {"calibration": f"{a.split}[:{a.num_samples}]"}
    print(f"[quant] calib set: {ds} {splits} x {a.max_seq_len} tokens", flush=True)

    ignore = ["lm_head", "re:.*embed_tokens.*"]
    if multimodal:
        ignore += ["re:.*vision_tower.*", "re:.*multi_modal_projector.*"]
    recipe = QuantizationModifier(targets="Linear", scheme=a.scheme, ignore=ignore)
    print(f"[quant] recipe: targets=Linear scheme={a.scheme} ignore={ignore}", flush=True)

    kw = {}
    if a.pipeline:
        kw["pipeline"] = a.pipeline
    if a.sequential_targets:
        kw["sequential_targets"] = a.sequential_targets.split(",")
    t1 = time.time()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        splits=splits,
        recipe=recipe,
        max_seq_length=a.max_seq_len,
        num_calibration_samples=a.num_samples,
        shuffle_calibration_samples=False,  # deterministic first-N, as ARM A
        batch_size=1,
        sequential_offload_device=a.offload_device,
        **kw,
    )
    print(f"[quant] oneshot done in {time.time()-t1:.0f}s; "
          f"peak cuda mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, save_compressed=True)
    processor.save_pretrained(a.out)
    # keep quant provenance next to the artifact
    with open(os.path.join(a.out, "accel4bit_quant_meta.json"), "w") as f:
        json.dump({"argv": sys.argv, "scheme": a.scheme, "ignore": ignore,
                   "num_samples": a.num_samples, "max_seq_len": a.max_seq_len, "seed": a.seed,
                   "dataset": f"ultrachat_200k:{splits}", "multimodal": multimodal,
                   "llmcompressor": llmcompressor.__version__,
                   "compressed_tensors": compressed_tensors.__version__,
                   "transformers": transformers.__version__, "torch": torch.__version__,
                   "wall_s": time.time() - t0}, f, indent=1)
    src_ct = os.path.join(a.model_path, "chat_template.jinja")
    if os.path.exists(src_ct) and not os.path.exists(os.path.join(a.out, "chat_template.jinja")):
        import shutil
        shutil.copy(src_ct, a.out)
    print(f"[quant] saved to {a.out}; total wall {time.time()-t0:.0f}s", flush=True)
    print("QUANT_OK", flush=True)


if __name__ == "__main__":
    main()
