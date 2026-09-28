#!/usr/bin/env bash
# Build the "big files" overlay that a git clone of this repo is missing.
#
#   scripts/make_handoff_overlay.sh <out_dir> <hf_snapshot> <recipe>:<artifact_dir> [...]
#
# The partner clones the repo, unpacks this overlay AT THE REPO ROOT, and runs.
# It deliberately contains NO code and NO documentation -- those come from the
# clone, so there is exactly one copy of each and no chance of the two drifting.
#
# What it does contain is everything git does not track:
#   results/accel4bit/calib_*        -- .gitignore excludes results/, and every
#                                       recipe pins this path, so stage 1 cannot
#                                       run without it
#   artifacts/<name>/                -- the packed CBK1 weights, plus config.json
#                                       and the tokenizer, so serving needs no
#                                       Hugging Face download at all
set -euo pipefail

OUT=${1:?output directory}; shift
SNAP=${1:?HF snapshot dir (config.json + tokenizer)}; shift
[ $# -ge 1 ] || { echo "need at least one <recipe>:<artifact_dir>" >&2; exit 2; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CALIB="results/accel4bit/calib_ultrachat_512x2048.txt"

rm -rf "$OUT"; mkdir -p "$OUT/results/accel4bit" "$OUT/artifacts"

# ---- 1. the calibration file, at the exact repo-relative path recipes pin ----
echo "==> calibration data"
cp -L "$REPO/$CALIB"      "$OUT/$CALIB"
cp -L "$REPO/$CALIB.json" "$OUT/$CALIB.json" 2>/dev/null || true

# ---- 2. one directory per arm ----
NAMES=()
for spec in "$@"; do
    recipe=${spec%%:*}; art=${spec#*:}
    [ -d "$art" ] || { echo "no such artifact dir: $art" >&2; exit 2; }
    dest="$OUT/artifacts/medgemma-27b-$recipe"
    echo "==> $recipe  ($(du -sh "$art" | cut -f1))"
    mkdir -p "$dest"
    cp -L "$art"/*.bin "$art"/manifest.json "$dest/"
    # config.json + tokenizer beside the weights: runner.py reads plain config.json
    # keys (not AutoConfig), and load_pretrained() takes the tokenizer from the
    # artifact dir -- so `--config` is never needed and no HF download is required.
    for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
             tokenizer.model special_tokens_map.json added_tokens.json; do
        [ -e "$SNAP/$f" ] && cp -L "$SNAP/$f" "$dest/"
    done
    NAMES+=("medgemma-27b-$recipe")
done

# ---- 3. how to use it ----
DEFAULT=${NAMES[0]}
cat > "$OUT/READ_ME_FIRST.md" <<EOF
# CoBALT — the files the repository does not track

Unpack this **at the root of a clone** of the CoBALT repository. It adds two
things and overwrites nothing:

    results/accel4bit/   the calibration set the recipes pin (git ignores results/)
    artifacts/           the packed CBK1 weights, one directory per arm

## Run it

    git clone <repo> && cd <repo>
    tar -xf cobalt-bigfiles.tar            # or unzip, here, at the repo root
    sha256sum -c SHA256SUMS                # optional; 11 GB transfers do get truncated

    export PYTHONPATH=\$PWD/src
    python -m prod doctor
    python -m prod verify   artifacts/$DEFAULT
    python -m prod generate artifacts/$DEFAULT --prompt "A 54-year-old presents with" -n 64
    python -m prod bench    artifacts/$DEFAULT --prompt 512 --gen 128 --out bench.json

The weights are already built, so **stages 1 and 2 of docs/REPRODUCTION.md are
already done** — you never run the quantizer. \`--config\` is not needed either:
config.json and the tokenizer sit beside the weights, so no Hugging Face download
is required to reproduce the inference numbers.

The calibration file is here only so that stage 1 *can* be re-run if you ever want
to rebuild the weights yourself; reproducing our numbers does not need it.

## Arms included

$(for n in "${NAMES[@]}"; do echo "* \`artifacts/$n\`"; done)

\`python -m prod recipes\` prints what each one measured. Everything else —
setup, the format spec, the kernel internals, the limits — is in \`docs/\` in the
clone, starting with \`docs/REPRODUCTION.md\`.
EOF

echo "==> checksums"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS )

# ---- 4. the tarball, ROOTLESS ----
# Members must be ./results/... and ./artifacts/... with NO wrapping directory, so a
# plain `tar -xf` at the repo root drops the files into place. Taring the directory
# itself yields <repo>/<name>/results/... and the pinned calibration path stays missing.
TAR="$OUT.tar"
echo "==> $TAR"
tar -cf "$TAR" -C "$OUT" .
sha256sum "$TAR" | tee "$TAR.sha256"

echo; echo "overlay: $OUT"; du -sh "$OUT"; echo "files:   $(find "$OUT" -type f | wc -l)"
echo "tarball: $TAR"
echo "rootless check (must NOT start with a directory name):"; tar -tf "$TAR" | head -3
