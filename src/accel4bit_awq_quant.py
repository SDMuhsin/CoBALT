#!/usr/bin/env python
"""accel4bit arm D: AWQ W4A16 (asym, g128) via llm-compressor AWQModifier.

Recipe = temp/llm-compressor/examples/awq/llama_example.py (AWQModifier(duo_scaling="both") +
QuantizationModifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=[lm_head, vision, projector]))
with the calibration set from examples/quantization_w4a16/README.md (ultrachat_200k train_sft,
512 x 2048, shuffle seed 42, chat-templated, add_special_tokens=False).

The model is loaded on CPU (bf16) and llm-compressor's default *sequential* pipeline onloads one
decoder layer at a time onto the single visible GPU (24 GB MIG slice), so a 27B bf16 model never has
to fit on the GPU.  Run with CUDA_VISIBLE_DEVICES=<MIG-UUID>.
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
    p.add_argument("--out", required=True, help="output dir for the compressed checkpoint")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duo-scaling", default="both", choices=["both", "true", "false"])
    p.add_argument("--n-grid", type=int, default=20)
    p.add_argument("--offload-device", default="default",
                   help="AWQ cached-activation offload: default (modifier default) | cpu | none")
    p.add_argument("--scheme", default="W4A16_ASYM")
    p.add_argument("--no-gen", action="store_true", help="skip the HF sample generation at the end")
    return p.parse_args()


def main():
    args = parse()
    t0 = time.time()
    from datasets import load_dataset
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    import llmcompressor
    import compressed_tensors
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.transform.awq.mappings import AWQMapping, AWQ_MAPPING_REGISTRY

    print(f"[versions] llmcompressor={llmcompressor.__version__} compressed_tensors={compressed_tensors.__version__} "
          f"transformers={transformers.__version__} torch={torch.__version__} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)} total={torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

    cfg = AutoConfig.from_pretrained(args.model_path)
    arch = cfg.architectures[0]
    print(f"[model] arch={arch} path={args.model_path}", flush=True)
    multimodal = arch == "Gemma3ForConditionalGeneration"
    if multimodal:
        from transformers import Gemma3ForConditionalGeneration as ModelCls
    else:
        from transformers import AutoModelForCausalLM as ModelCls

    # CPU load in bf16 (no device_map): sequential pipeline onloads layer-by-layer.
    model = ModelCls.from_pretrained(args.model_path, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"[load] {time.time()-t0:.0f}s, n_params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    # ---- calibration data: llm-compressor W4A16 README recipe (ultrachat_200k) ----
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.num_samples}]")
    ds = ds.shuffle(seed=args.seed)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=args.max_seq_len,
                         truncation=True, add_special_tokens=False)

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    lens = [len(x) for x in ds["input_ids"]]
    print(f"[calib] ultrachat_200k train_sft[:{args.num_samples}] shuffle(seed={args.seed}) max_len={args.max_seq_len}: "
          f"n={len(ds)} tokens={sum(lens)} mean_len={sum(lens)/len(lens):.0f} n_at_max={sum(l==args.max_seq_len for l in lens)}",
          flush=True)

    # ---- recipe ----
    ignore = ["lm_head"]
    mappings = None
    if multimodal:
        ignore += [r"re:.*vision_tower.*", r"re:.*multi_modal_projector.*"]
        # Registry Gemma mapping, restricted to the text model. Needed because compressed_tensors'
        # match_modules_set walks named_modules() in order and the SigLIP tower's q/k/v_proj match
        # `re:.*q_proj$` before any decoder input_layernorm is seen, collapsing all decoder layers
        # into one set ("AWQ needs to match a single smoothlayer").
        base = AWQ_MAPPING_REGISTRY[arch]
        mappings = [
            AWQMapping(m.smooth_layer.replace("re:.*", "re:.*language_model.*", 1),
                       [b.replace("re:.*", "re:.*language_model.*", 1) for b in m.balance_layers],
                       m.activation_hook_target)
            for m in base
        ]
    duo = {"both": "both", "true": True, "false": False}[args.duo_scaling]
    awq_kwargs = dict(duo_scaling=duo, n_grid=args.n_grid)
    if mappings is not None:
        awq_kwargs["mappings"] = mappings
    if args.offload_device == "cpu":
        awq_kwargs["offload_device"] = torch.device("cpu")
    elif args.offload_device == "none":
        awq_kwargs["offload_device"] = None
    recipe = [
        AWQModifier(**awq_kwargs),
        QuantizationModifier(ignore=ignore, scheme=args.scheme, targets=["Linear"]),
    ]
    print("[recipe] " + json.dumps({
        "AWQModifier": {**{k: str(v) for k, v in awq_kwargs.items() if k != "mappings"},
                        "mappings": "registry default for %s" % arch if mappings is None else
                        [(m.smooth_layer, m.balance_layers) for m in mappings]},
        "QuantizationModifier": {"scheme": args.scheme, "targets": ["Linear"], "ignore": ignore},
    }, indent=1), flush=True)

    t1 = time.time()
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )
    print(f"[oneshot] done in {(time.time()-t1)/60:.1f} min; peak GPU alloc={torch.cuda.max_memory_allocated()/2**30:.1f} GiB "
          f"reserved={torch.cuda.max_memory_reserved()/2**30:.1f} GiB", flush=True)

    if not args.no_gen:
        from compressed_tensors.offload import dispatch_model
        print("========== SAMPLE GENERATION (HF, fake-quant) ==============", flush=True)
        dispatch_model(model)
        for prompt in ["Hello my name is", "Aspirin is commonly used to treat"]:
            sample = tokenizer(prompt, return_tensors="pt")
            sample = {k: v.to(model.device) for k, v in sample.items()}
            out = model.generate(**sample, max_new_tokens=60, do_sample=False, disable_compile=True)
            print(repr(tokenizer.decode(out[0])), flush=True)
        print("==========================================", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)
    if multimodal:
        AutoProcessor.from_pretrained(args.model_path).save_pretrained(args.out)
    with open(os.path.join(args.out, "accel4bit_awq_recipe.json"), "w") as f:
        json.dump({"args": vars(args), "arch": arch, "ignore": ignore,
                   "mappings": None if mappings is None else [(m.smooth_layer, m.balance_layers) for m in mappings],
                   "versions": {"llmcompressor": llmcompressor.__version__, "compressed_tensors": compressed_tensors.__version__,
                                "transformers": transformers.__version__, "torch": torch.__version__},
                   "calib": {"dataset": "HuggingFaceH4/ultrachat_200k", "split": f"train_sft[:{args.num_samples}]",
                             "shuffle_seed": args.seed, "max_seq_len": args.max_seq_len, "tokens": sum(lens)},
                   "oneshot_minutes": (time.time() - t1) / 60}, f, indent=1)
    print(f"[save] {args.out} total {(time.time()-t0)/60:.1f} min", flush=True)
    print("ACCEL4BIT_AWQ_QUANT_OK", flush=True)


if __name__ == "__main__":
    main()
