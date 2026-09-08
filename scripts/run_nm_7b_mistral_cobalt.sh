#!/usr/bin/env bash
# mistral-7b cobalt / cobalt-awclip 2:4 cells OOM on a 24GB MIG slice (nosink OBS path holds full H_inv);
# rerun them on the 48GB slice once the llama-7b stream there finishes. FAILED rows are not cached, so
# camera_bench recomputes them.
set -u; cd /workspace/PTQResearch
until grep -q ALLDONE results/nm24/b7/run_llama-7b.log; do sleep 120; done
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
source env.sh >/dev/null 2>&1
OUT=results/nm24/b7; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for B in 4 3; do
  for MTH in cobalt cobalt-awclip; do
    $PY src/camera_bench.py --model mistral-7b --method "$MTH" --nm 2:4 --bits $B --force-true-bits \
        --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        > "$OUT/mistral-7b_${MTH}_b${B}.48g.log" 2>&1
    echo "[done] mistral-7b $MTH b$B (48g rerun)"
  done
done
echo "ALLDONE_MISTRAL_COBALT_RERUN"
