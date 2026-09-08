"""accel4bit ARM A: GPTQ W4A16 (sym, g128) with llm-compressor GPTQModifier.

Base recipe: temp/llm-compressor/examples/multimodal_vision/{gemma3,medgemma}_example.py
(GPTQModifier targets="Linear", scheme="W4A16", ignore lm_head/vision_tower/multi_modal_projector,
512 x 2048 calibration) with two additions: dampening_frac=0.07 (RedHatAI Gemma-3-27B card),
actorder="static" (llm-compressor default), calibration = HuggingFaceH4/ultrachat_200k train_sft
(512 samples x 2048 tokens, chat-templated, seed 42).

Text-only handling of the multimodal gemma-3-4b checkpoint: load Gemma3ForConditionalGeneration (there is no
text-only 4b checkpoint), calibrate with TEXT-ONLY ultrachat through the tokenizer (not the processor), and ignore
vision_tower / multi_modal_projector. medgemma-27b-text-it is Gemma3ForCausalLM and loads via AutoModelForCausalLM.

The model is loaded on CPU (bf16); llm-compressor's sequential pipeline onloads one decoder layer at a time to the
GPU (CUDA_VISIBLE_DEVICES = one MIG slice), so the 27B (54 GB bf16) fits a 1g.24gb slice.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

MODELS = {
    "gemma-3-4b": dict(
        path="/scratch/ckp908/prism_hf/hub/models--unsloth--gemma-3-4b-it/snapshots/"
        "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
        hf_id="unsloth/gemma-3-4b-it",
        multimodal=True,
    ),
    "medgemma-27b": dict(
        path="/scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/"
        "b780610baf99c087ba3719a77cf0dacec7261a65",
        hf_id="unsloth/medgemma-27b-text-it",
        multimodal=False,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--out", required=True, help="artifact dir (compressed-tensors checkpoint)")
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--dampening-frac", type=float, default=0.07)
    ap.add_argument("--actorder", default="static")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offload-hessians", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
    import llmcompressor
    import compressed_tensors
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    spec = MODELS[args.model]
    print(f"[quant] versions: llmcompressor {llmcompressor.__version__} compressed_tensors "
          f"{compressed_tensors.__version__} transformers {transformers.__version__} torch {torch.__version__}",
          flush=True)
    print(f"[quant] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}",
          flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(spec["path"])
    if spec["multimodal"]:
        # TEXT-ONLY path: the multimodal class's forward has an untraceable loss branch
        # (boolean-mask indexing -> torch.nonzero meta error in llm-compressor's sequential tracer; failed at
        # subgraph 35/35 on the first attempt). Rebuild the text model as Gemma3ForCausalLM from
        # model.language_model + lm_head, i.e. exactly the class/path used by medgemma-27b-text-it.
        from transformers import Gemma3ForCausalLM
        mm = Gemma3ForConditionalGeneration.from_pretrained(spec["path"], dtype=torch.bfloat16)
        text_cfg = mm.config.text_config
        text_cfg.architectures = ["Gemma3ForCausalLM"]
        model = Gemma3ForCausalLM._from_config(text_cfg, dtype=torch.bfloat16)
        missing, unexpected = model.model.load_state_dict(mm.model.language_model.state_dict(), strict=True)
        model.lm_head.weight = mm.lm_head.weight
        model.tie_weights()
        model.generation_config = mm.generation_config
        assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight), "lm_head must be tied"
        print(f"[quant] text-only Gemma3ForCausalLM built from language_model (dropped vision_tower + "
              f"multi_modal_projector); strict load ok, tied lm_head", flush=True)
        del mm
    else:
        model = AutoModelForCausalLM.from_pretrained(spec["path"], dtype=torch.bfloat16)
    print(f"[quant] loaded {type(model).__name__} on CPU in {time.time()-t0:.0f}s; "
          f"n_params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    ignore = ["lm_head", r"re:.*embed_tokens.*", r"re:.*vision_tower.*", r"re:.*multi_modal_projector.*"]
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",  # int4, symmetric, group_size=128, weight-only
        ignore=ignore,
        dampening_frac=args.dampening_frac,
        actorder=args.actorder,
        offload_hessians=args.offload_hessians,
    )
    print(f"[quant] recipe: {recipe.model_dump(exclude_none=True)}", flush=True)

    t1 = time.time()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="ultrachat_200k",
        splits={"calibration": f"train_sft[:{args.num_samples}]"},
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.num_samples,
        shuffle_calibration_samples=False,  # deterministic: first N of train_sft (as gemma3_example.py)
        batch_size=1,
        output_dir=None,
    )
    quant_s = time.time() - t1
    print(f"[quant] oneshot done in {quant_s/60:.1f} min", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)
    # (multimodal source: processor NOT saved; the artifact is a text-only Gemma3ForCausalLM checkpoint)
    # keep the chat template file if the source has one (vLLM/lm_eval chat usage)
    src_ct = os.path.join(spec["path"], "chat_template.jinja")
    if os.path.exists(src_ct) and not os.path.exists(os.path.join(args.out, "chat_template.jinja")):
        import shutil
        shutil.copy(src_ct, args.out)

    meta = dict(
        model=args.model, hf_id=spec["hf_id"], source_snapshot=spec["path"],
        recipe=recipe.model_dump(exclude_none=True, mode="json"),
        calibration=dict(dataset="HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.num_samples}]",
                         num_samples=args.num_samples, max_seq_length=args.seq_len, shuffle=False,
                         chat_templated=True, seed=args.seed),
        versions=dict(llmcompressor=llmcompressor.__version__, compressed_tensors=compressed_tensors.__version__,
                      transformers=transformers.__version__, torch=torch.__version__),
        quant_minutes=quant_s / 60, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        argv=sys.argv,
    )
    with open(os.path.join(args.out, "accel4bit_quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[quant] saved to {args.out}; total {(time.time()-t0)/60:.1f} min", flush=True)
    print("QUANT_OK", flush=True)


if __name__ == "__main__":
    main()
