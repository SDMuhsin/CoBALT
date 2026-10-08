#!/bin/bash
# G4 bit-identity gate: the PRE-port kernel (git HEAD, pristine worktree) and the POST-port kernel (working
# tree) must produce torch.equal logits on gemma-3-4b DENSE4 for the same token sequence.
# usage: $0 <MIG>
set -u
MIG=${1:?mig}; ROOT=/workspace/PTQResearch; cd $ROOT
source scripts/cobaltkernel_env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1 COBALT_KPB=256
PY=/scratch/root/PTQResearch/env/bin/python
G=/scratch/root/PTQResearch/accel4bit_models/gemma-3-4b
WT=/scratch/root/PTQResearch/wt_pre_port
OUT=$ROOT/results/biomistral/regress; mkdir -p $OUT
stamp() { echo "[$(date '+%F %T')] $*"; }
[ -d $WT ] || git worktree add --detach $WT HEAD >/dev/null 2>&1
cp src/cobaltkernel/dump_logits.py $WT/src/cobaltkernel/dump_logits.py
stamp "OLD build (git HEAD $(git rev-parse --short HEAD)) -> separate extension cache"
( cd $WT && TORCH_EXTENSIONS_DIR=/scratch/root/PTQResearch/cache/torch_ext_preport $PY -u src/cobaltkernel/dump_logits.py \
    --model $G/cobalt_sp0.5_b4_g128_cbk1_dense4f --config $G/cobalt_sp0.5_b4_g128_fakequant_hf --steps 48 --out $OUT/logits_pre_port.pt 2>&1 | tail -3 )
stamp "NEW build (working tree)"
$PY -u src/cobaltkernel/dump_logits.py --model $G/cobalt_sp0.5_b4_g128_cbk1_dense4f --config $G/cobalt_sp0.5_b4_g128_fakequant_hf --steps 48 --out $OUT/logits_post_port.pt 2>&1 | tail -3
$PY - << 'PY'
import torch
a=torch.load("/workspace/PTQResearch/results/biomistral/regress/logits_pre_port.pt"); b=torch.load("/workspace/PTQResearch/results/biomistral/regress/logits_post_port.pt")
eq=torch.equal(a["logits"],b["logits"]); d=(a["logits"]-b["logits"]).abs().max().item()
print(f"BITIDENT {'PASS' if eq else 'FAIL'} torch.equal={eq} max|diff|={d:.3e} steps={a['logits'].shape[0]} blocks pre/post={a['blocks']}/{b['blocks']}")
PY
stamp "BITIDENT_DONE"
