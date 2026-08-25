#!/bin/bash
# Supervisor for the bias-correction ablation grid. Auto-discovers all MIG slices,
# launches one shard per slice, and keeps relaunching until every config has a
# done-marker — riding out the box's periodic glitches (clock jumps / rc=126).
#
# Launch DETACHED so it survives a session teardown:
#   cd /workspace/PRISM
#   ABL_DATASET=wikitext2 setsid nohup bash scripts/supervise_ablation.sh \
#       >> logs/abl_supervisor_wt2.log 2>&1 </dev/null &
#
# ENV: ABL_DATASET (wikitext2|c4), ABL_DS (set=run downstream too), ABL_TECHS,
#      ABL_MODEL, plus the run_ablation_grid.sh knobs. NSLICES caps slices used.
set -u
cd /workspace/PRISM

PY=/workspace/PRISM/env/bin/python
export ABL_MODEL="${ABL_MODEL:-qwen-0.5b}"
export ABL_DATASET="${ABL_DATASET:-wikitext2}"
export ABL_TAG="${ABL_TAG:-ablation_${ABL_MODEL}_${ABL_DATASET}}"
DEFAULT_TECHS="abl-base abl-corr abl-extras abl-full prism wanda-sinq \
sparsegpt sparsegpt-corr slim slim-corr jsq-wo jsq-wo-corr"
export ABL_TECHS="${ABL_TECHS:-$DEFAULT_TECHS}"

DONEDIR=results/$ABL_TAG/done
NTECH=$(echo $ABL_TECHS | wc -w)
NSPARS=$(echo "${ABL_SPARS:-0.05 0.25 0.50}" | wc -w)
NBITS=$(echo "${ABL_BITS:-3 4 5}" | wc -w)
TARGET=$((NTECH * NSPARS * NBITS))
MAXROUNDS="${MAXROUNDS:-80}"

ndone() { ls -1 "$DONEDIR" 2>/dev/null | wc -l | tr -d ' '; }
# MIG_EXCLUDE = comma-separated UUIDs to skip (e.g. a slice running another job).
mig_list() {
    nvidia-smi -L 2>/dev/null | grep -oE 'MIG-[0-9a-f-]+' | while read -r u; do
        case ",${MIG_EXCLUDE:-}," in *",$u,"*) ;; *) echo "$u";; esac
    done
}

echo "[abl-sup] START $(date) tag=$ABL_TAG target=$TARGET techs=$NTECH ds=${ABL_DS:-off}"

for r in $(seq 1 "$MAXROUNDS"); do
    d=$(ndone)
    if [ "$d" -ge "$TARGET" ]; then
        echo "[abl-sup] ALL $TARGET CONFIGS DONE at round $r $(date)"; break
    fi
    echo "[abl-sup] round $r begin $(date) done=$d/$TARGET"

    if ! nvidia-smi -L >/dev/null 2>&1 || \
       ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        echo "[abl-sup] GPU/torch not ready — waiting 120s"; sleep 120; continue
    fi

    mapfile -t SLICES < <(mig_list)
    N=${#SLICES[@]}
    [ -n "${NSLICES:-}" ] && [ "$N" -gt "$NSLICES" ] && N="$NSLICES"
    if [ "$N" -lt 1 ]; then echo "[abl-sup] no MIG slices — waiting 120s"; sleep 120; continue; fi
    echo "[abl-sup] launching $N shards"

    PIDS=()
    for i in $(seq 0 $((N-1))); do
        PRISM_MIG="${SLICES[$i]}" DS_SHARD="$i/$N" \
            bash scripts/run_ablation_grid.sh >> "logs/abl_${ABL_TAG}_shard${i}.log" 2>&1 &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"

    after=$(ndone)
    echo "[abl-sup] round $r end $(date) done=$after/$TARGET (+$((after-d)))"
    if [ "$after" -le "$d" ]; then echo "[abl-sup] no progress — backing off 90s"; sleep 90; else sleep 10; fi
done
echo "[abl-sup] EXIT $(date) done=$(ndone)/$TARGET"
