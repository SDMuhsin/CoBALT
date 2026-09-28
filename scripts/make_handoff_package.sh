#!/usr/bin/env bash
# Build a self-contained, plug-and-play CoBALT inference package for a partner.
#
#   scripts/make_handoff_package.sh <recipe> <artifact_dir> <hf_snapshot> <out_dir>
#
# The result needs NO HF checkpoint, NO calibration data and NO quantization run at
# the far end: `python -m prod generate <pkg>/model` works from the package alone.
set -euo pipefail

RECIPE=${1:?recipe name, e.g. blk1632_b4_oproj}
ART=${2:?packed CBK1 artifact dir}
SNAP=${3:?HF snapshot dir (config.json + tokenizer)}
OUT=${4:?output package dir}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "==> staging $OUT"
rm -rf "$OUT"; mkdir -p "$OUT/model" "$OUT/src/prod" "$OUT/src/cobaltkernel/csrc" "$OUT/docs"

# ---- 1. weights: the packed artifact, byte-for-byte what the kernel mmaps ----
echo "==> copying artifact ($(du -sh "$ART" | cut -f1))"
cp -L "$ART"/*.bin "$ART"/manifest.json "$OUT/model/"

# ---- 2. make the artifact self-contained: config + tokenizer live beside it ----
# runner.py reads plain config.json keys; load_pretrained() takes the tokenizer from
# the artifact dir when no config_dir is given.  With these present, --config is never
# needed and the package has no external model dependency.
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         tokenizer.model special_tokens_map.json added_tokens.json; do
    [ -e "$SNAP/$f" ] && cp -L "$SNAP/$f" "$OUT/model/"
done

# ---- 3. code: the inference closure only (no quantizer, no research probes) ----
cp "$REPO"/src/prod/*.py "$REPO"/src/prod/README.md      "$OUT/src/prod/"
for f in __init__.py runner.py prefill_runner.py dequant_ref.py ref_gemma3.py; do
    cp "$REPO/src/cobaltkernel/$f"                        "$OUT/src/cobaltkernel/"
done
for f in bindings.cpp megakernel.cu megakernel.cuh prefill_bindings.cpp \
         prefill_kernel.cu prefill_kernel.cuh cobalt_gemm.cuh cobalt_gemv.cuh \
         gemv_api.cuh cobalt_format.h prmt_lut.inc; do
    cp "$REPO/src/cobaltkernel/csrc/$f"                   "$OUT/src/cobaltkernel/csrc/"
done
cp "$REPO"/docs/*.md "$OUT/docs/"

# ---- 4. the one command they run ----
cat > "$OUT/RUN.md" <<EOF
# CoBALT MedGemma-27B — $RECIPE

Self-contained. No Hugging Face checkpoint, no calibration data, no quantization step.

    export PYTHONPATH=\$PWD/src
    python -m prod doctor                  # toolchain check
    python -m prod verify  model           # ~2 s, no GPU, no build
    python -m prod generate model --prompt "A 54-year-old presents with" -n 64
    python -m prod bench   model --prompt 512 --gen 128 --out bench.json

Needs: python>=3.10, torch>=2.4 built against your CUDA, nvcc on PATH, ninja on PATH,
safetensors, numpy, transformers (tokenizer only).

The CUDA extension JIT-builds on first use and is then cached. **Budget 10-20 minutes for
that first build** -- megakernel.cu is a big templated translation unit and nvcc is
single-threaded on it. Nothing is printed while it compiles.

If a build is ever interrupted, torch leaves a zero-byte \\`lock\\` file behind and the next
run blocks on it forever -- alive, silent, and holding no GPU memory, so \\`nvidia-smi\\`
shows nothing. Clear it with:

    find "\$TORCH_EXTENSIONS_DIR" -name lock -delete

\`--config\` is NOT required: config.json and the tokenizer ship inside model/.
Full reference: docs/REPRODUCTION.md (§5 verify, §6 serve, §7 bench, §8 limits).
EOF

# ---- 5. integrity ----
echo "==> checksums"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS )

echo
echo "package: $OUT"
du -sh "$OUT"
echo "files:   $(find "$OUT" -type f | wc -l)"
