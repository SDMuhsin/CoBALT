#!/bin/bash
# One-slice speed protocol for BioMistral-7B (REQUIREMENTS G7): llama.cpp Q4_K_M bar + the CoBALT arms,
# all SEQUENTIAL on one slice, DENSE4 control repeated at both ends.
# usage: scripts/run_biomistral_speed.sh <MIG-uuid> [arms...]   default arms: gguf dense4 b1632_4 dense4_end
set -u
MIG=${1:?mig}; shift
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(gguf dense4 b1632_4 dense4_end)
ROOT=/workspace/PTQResearch; cd $ROOT
stamp() { echo "[$(date '+%F %T')] $*"; }
for arm in "${ARMS[@]}"; do
  case $arm in
    gguf)
      stamp "ARM gguf (llama.cpp Q4_K_M, MMQ/MMVQ) on $MIG"
      bash scripts/run_accel4bit_speed_oneslice.sh biomistral-7b $MIG gguf 2>&1 | grep -E "^\[|exit=|tok/s|\|" | tail -30 ;;
    dense4|dense4_end)
      stamp "ARM cobalt_dense4 control (COBALT_KPB=256) [$arm]"
      COBALT_KPB=256 COBALT_OUT_OVERRIDE=$ROOT/results/cobaltkernel/biomistral-7b/speed_oneslice/cobalt_dense4${arm#dense4}.json \
        bash scripts/run_cobaltkernel_speed_oneslice.sh biomistral-7b $MIG cobalt_dense4 2>&1 | grep -vE "^\s*$" | tail -30 ;;
    b1632_4)
      stamp "ARM cobalt_b1632_4 (COBALT_BLK1632=4)"
      bash scripts/run_cobaltkernel_speed_oneslice.sh biomistral-7b $MIG cobalt_b1632_4 2>&1 | grep -vE "^\s*$" | tail -30 ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
done
stamp "SPEED_DONE"
