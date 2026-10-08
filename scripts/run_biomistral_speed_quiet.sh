#!/bin/bash
# Quiet-window speed protocol (REQUIREMENTS G7 + 17:00 amendment) on ONE slice: waits until nothing else runs on the slice and
# the host load is < LOADMAX, then runs the passes SEQUENTIALLY with symmetric pinning (nice -n -5 taskset -c 0-7):
#   llama.cpp Q4_K_M -> <arm> -> cobalt_b1632_4 (shipped 16:32 control) -> llama.cpp -> <arm>
# with the shipped MODEL_TUNING knobs (BLOCKS 160 / KPB 128 / KUNROLL 4 / PVB 2 / MMA qkv|o) exported for every CoBALT arm.
# usage: $0 <MIG> <tag> <arm>      results: results/accel4bit/biomistral-7b/speed_oneslice_<tag>{1,2}/gguf.json,
#                                           results/cobaltkernel/biomistral-7b/speed_oneslice/<arm>_<tag>{1,2}.json, ..._ctl.json
set -u
MIG=${1:?mig}; TAG=${2:?tag}; ARM=${3:?arm}; LOADMAX=${LOADMAX:-4.0}
ROOT=/workspace/PTQResearch; cd $ROOT
stamp() { echo "[$(date '+%F %T')] $*"; }
export COBALT_BLOCKS=160 COBALT_KPB=128 COBALT_ATTN_KUNROLL=4 COBALT_PVB=2 COBALT_BLK1632_MMA=1 COBALT_MMA_PHASES=3 COBALT_MMA_PF_L2=1 COBALT_MMA_UPW=2 COBALT_MMA_UPW_O=1
short=${MIG#MIG-}; short=${short%%-*}
while :; do
  # command substitution forks a copy of this script: exclude by name, not by $$
  others=$(ps -eo pid,cmd | grep "$short" | grep -v "speed_quiet" | grep -v grep); busy=$(echo -n "$others" | grep -c .); load=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$busy" = 0 ] && awk -v l=$load -v m=$LOADMAX 'BEGIN{exit !(l<m)}'; then break; fi
  stamp "waiting: slice procs=$busy load=$load $(echo "$others" | head -2 | cut -c1-90 | tr '\n' ' ')"; sleep 60
done
stamp "QUIET WINDOW START load=$(cut -d' ' -f1 /proc/loadavg) mig=$MIG arm=$ARM"
CO=$ROOT/results/cobaltkernel/biomistral-7b/speed_oneslice
run_gguf() { OUT_SUBDIR=speed_oneslice_$1 nice -n -5 taskset -c 0-7 bash scripts/run_accel4bit_speed_oneslice.sh biomistral-7b $MIG gguf 2>&1 | grep -E "tok/s|decode|exit=" | tail -4; }
run_cob() { COBALT_OUT_OVERRIDE=$CO/$2.json nice -n -5 taskset -c 0-7 bash scripts/run_cobaltkernel_speed_oneslice.sh biomistral-7b $MIG $1 2>&1 | grep -E "tok/s|step|exit|Error" | tail -6; }
stamp "pass 1 llama.cpp";   run_gguf ${TAG}1
stamp "pass 1 $ARM";        run_cob $ARM ${ARM}_${TAG}1
stamp "control cobalt_b1632_4"; run_cob cobalt_b1632_4 cobalt_b1632_4_${TAG}_ctl
stamp "pass 2 llama.cpp";   run_gguf ${TAG}2
stamp "pass 2 $ARM";        run_cob $ARM ${ARM}_${TAG}2
stamp "load now $(cut -d' ' -f1 /proc/loadavg)"
stamp "SPEED_QUIET_DONE"
