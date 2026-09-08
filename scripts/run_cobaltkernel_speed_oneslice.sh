#!/usr/bin/env bash
# accel4bit-protocol one-slice speed run for the cobaltkernel megakernel.
#
#   scripts/run_cobaltkernel_speed_oneslice.sh <model> <MIG-uuid|2g|1g> [arm]
#
#     model : medgemma-27b | gemma-3-4b
#     MIG   : slice selector passed to scripts/cobaltkernel_env.sh
#     arm   : cobalt_dense4 (default) | cobalt_bf16 | cobalt_b1632_4 | cobalt_b1632_6
#             | cobalt_b1632_4_ohyb | cobalt_b1632_6_ohyb4   (mixed-layout o_proj hybrids)
#
# The CoBALT-16:32 arms (FORMAT.md sec.13) read the *_cbk1_b1632_{4,6}f artifacts and
# build the megakernel with COBALT_BLK1632={4,6}.  The PREFILL megakernel only reads
# DENSE4, so those arms run the v2 protocol (prompt fed through the decode kernel); take
# the DENSE4 control the same way (COBALT_NO_PREFILL_KERNEL=1) if you compare TTFT.
#
# The DENSE4 arm uses the *_dense4f artifact (fused q/k/v row space, FORMAT.md 2.4b),
# which the v2 megakernel requires.
#
# v3: ONE end-to-end 512->128 run whose TTFT / prefill_tok_s come from the
# PREFILL megakernel (one cooperative launch for the whole prompt) and whose
# decode_tok_s covers tokens 2..128 through the decode megakernel.  Pass
# COBALT_NO_PREFILL_KERNEL=1 to reproduce the v2 protocol (prompt fed one token per
# decode launch); the v2 records are kept as <arm>.v2.json.
#
# Writes results/cobaltkernel/<model>/speed_oneslice/<arm>.json
# (schema: results/cobaltkernel/SPEED_JSON_SCHEMA.md).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: $0 <model> <MIG|2g|1g> [arm]}"
SLICE="${2:?usage: $0 <model> <MIG|2g|1g> [arm]}"
ARM="${3:-cobalt_dense4}"

# shellcheck disable=SC1090
source "$REPO/scripts/cobaltkernel_env.sh" "$SLICE"

AM=/scratch/root/PTQResearch/accel4bit_models
case "$MODEL" in
  medgemma-27b)
    CFG=/scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65
    DENSE4="$AM/medgemma-27b/cobalt_sp0.5_b4_g128_cbk1_dense4f"
    B1632_4="$AM/medgemma-27b/cobalt_sp0.5_b4_g128_blk32_cbk1_b1632_4f"
    B1632_6="$AM/medgemma-27b/cobalt_sp0.5_b6_g128_blk32_cbk1_b1632_6f"
    OHYB4="$AM/medgemma-27b/cobalt_sp0.5_b4_g128_blk32_ohyb_cbk1f"
    OHYB46="$AM/medgemma-27b/cobalt_sp0.5_b6_g128_blk32_ohyb4_cbk1f"
    BF16="$CFG" ;;
  gemma-3-4b)
    CFG="$AM/gemma-3-4b/cobalt_sp0.5_b4_g128_fakequant_hf"
    DENSE4="$AM/gemma-3-4b/cobalt_sp0.5_b4_g128_cbk1_dense4f"
    B1632_4="$AM/gemma-3-4b/cobalt_sp0.5_b4_g128_blk32_cbk1_b1632_4f"
    B1632_6="$AM/gemma-3-4b/cobalt_sp0.5_b6_g128_blk32_cbk1_b1632_6f"
    OHYB4="$AM/gemma-3-4b/cobalt_sp0.5_b4_g128_blk32_ohyb_cbk1f"
    OHYB46="$AM/gemma-3-4b/cobalt_sp0.5_b6_g128_blk32_ohyb4_cbk1f"
    BF16="$AM/gemma-3-4b/text_bf16" ;;
  *) echo "unknown model '$MODEL'"; exit 2 ;;
esac

case "$ARM" in
  cobalt_dense4)  DIR="$DENSE4";  CFGARG=(--config "$CFG") ;;
  cobalt_bf16)    DIR="$BF16";    CFGARG=() ;;
  # Both arms use the stock gate/up path (fused NX=2, RP from pick_R).  MEASURED on the 2g
  # slice of record, 27B, 512->128: RP=2 37.10 > RP=1 36.85 > split RP=2 36.37 > NX=2 RP=4
  # 33.48 > split RP=4 33.31; b=6: 30.72 stock > 28.93 split.  COBALT_GATEUP_SPLIT / _RP
  # are still available as knobs.  (A 1g screen ranked split-RP=4 FIRST -- it does not
  # transfer: never screen decode variants on a smaller slice than you report on.)
  #
  # The BLK arms build with COBALT_BLK1632_LUT defaulting to 2 (the expand-to-dense
  # selector table staged in __shared__ once per block).  MEASURED on this slice, 27B,
  # 512->128: b=4 36.94 -> 41.36 tok/s, b=6 30.90 -> 34.28.  Export
  # COBALT_BLK1632_LUT=0 for the pre-staging behaviour.
  cobalt_b1632_4) DIR="$B1632_4"; CFGARG=(--config "$CFG"); export COBALT_BLK1632=4 ;;
  cobalt_b1632_6) DIR="$B1632_6"; CFGARG=(--config "$CFG"); export COBALT_BLK1632=6 ;;
  # MIXED-LAYOUT o_proj-hybrid arms (2026-09-07): six matrices BLK16_32,
  # `o_proj` canonical global-top-k = DENSE4.  The decode megakernel dispatches on the
  # per-matrix layout id at runtime, so it needs no extra flag; the prefill megakernel's
  # gemm_tile BLK is compile-time, so COBALT_BLK1632_O=0 selects DENSE4 for the o_proj GEMM.
  cobalt_b1632_4_ohyb)  DIR="$OHYB4";  CFGARG=(--config "$CFG")
                        export COBALT_BLK1632=4 COBALT_BLK1632_O=0 ;;
  cobalt_b1632_6_ohyb4) DIR="$OHYB46"; CFGARG=(--config "$CFG")
                        export COBALT_BLK1632=6 COBALT_BLK1632_O=0 ;;
  *) echo "unknown arm '$ARM'"; exit 2 ;;
esac

SL="${CUDA_VISIBLE_DEVICES:-unknown}"
case "$SL" in
  MIG-1d47bdbe*) ST=2g.48gb ;;
  MIG-*)         ST=1g.24gb ;;
  *)             ST=unknown ;;
esac

# The prefill megakernel is BLK16_32-aware (2026-09-06): its
# cbk::gemm_tile reads BLK1632_4 / BLK1632_6 directly (PF_BLK1632, set from COBALT_BLK1632),
# so the 16:32 arms now run the SAME v3 protocol as DENSE4 and their TTFT / prefill_tok_s
# are real prefill measurements again.  Export COBALT_NO_PREFILL_KERNEL=1 to reproduce the
# v2 protocol (prompt fed one token per decode launch) that CAVEATS 12 describes.
PFK=(--prefill-kernel)
if [ -n "${COBALT_NO_PREFILL_KERNEL:-}" ] || [ "$ARM" = "cobalt_bf16" ]; then
  PFK=(--no-prefill-kernel)
fi

OUT="$REPO/results/cobaltkernel/$MODEL/speed_oneslice/$ARM.json"
mkdir -p "$(dirname "$OUT")"
echo "[cobaltkernel-speed] model=$MODEL arm=$ARM slice=$SL ($ST) -> $OUT"
exec python "$REPO/src/cobaltkernel/speed_oneslice.py" \
  --model-dir "$DIR" "${CFGARG[@]}" --model-name "$MODEL" --arm "$ARM" \
  --slice "$SL" --slice-type "$ST" --prompt 512 --gen 128 --batch-M 2 4 8 \
  "${PFK[@]}" --out "${COBALT_OUT_OVERRIDE:-$OUT}"
