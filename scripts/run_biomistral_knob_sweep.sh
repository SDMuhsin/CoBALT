#!/bin/bash
# Runtime-knob sweep for the BioMistral 16:32 arm, judged by the jitter-free GPU-side kernel step
# (phase_breakdown_ms.total_measured), not by host-inclusive tok/s.  usage: $0 <MIG> [arm]
set -u
MIG=${1:?mig}; ARM=${2:-cobalt_b1632_4}
ROOT=/workspace/PTQResearch; cd $ROOT
R=results/cobaltkernel/biomistral-7b/knob_sweep; mkdir -p $R
stamp() { echo "[$(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg)] $*"; }
for KPB in 128 64 48 32; do for BL in 0 141 94; do
  tag=${ARM}_kpb${KPB}_bl${BL}
  [ -f $R/$tag.json ] && { stamp "skip $tag"; continue; }
  stamp "run $tag"
  env COBALT_KPB=$KPB $( [ $BL -gt 0 ] && echo COBALT_BLOCKS=$BL ) COBALT_OUT_OVERRIDE=$R/$tag.json \
    nice -n -5 taskset -c 0-7 bash scripts/run_cobaltkernel_speed_oneslice.sh biomistral-7b $MIG $ARM > $R/$tag.log 2>&1
  python3 - "$R/$tag.json" << 'PY'
import json,sys
j=json.load(open(sys.argv[1])); pb=j.get("phase_breakdown_ms",{}); ss=j.get("single_stream",{})
print(f"   step {pb.get('total_measured',float('nan')):.3f} ms  attn {pb.get('attn',0):.3f} qkv {pb.get('qkv',0):.3f} o {pb.get('o_proj',0):.3f} gu {pb.get('gateup_geglu',0):.3f} down {pb.get('down_proj',0):.3f} | host decode {ss.get('decode_tok_s')} tok/s  status {j.get('status')}")
PY
done; done
stamp "KNOB_SWEEP_DONE"
