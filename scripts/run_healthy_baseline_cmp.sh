#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
CSV=results/sens_alloc/baseline_cmp/results.csv
for MODEL in tinyllama qwen-1.5b; do
  echo "########## $MODEL ##########"
  python scripts/camera_dispatch.py --model $MODEL \
    --methods cobalt,wanda-awq,wanda-sinq,sparsegpt \
    --sparsities 0.5,0.6 --bits 3 --force-true-bits \
    --cobalt-group-size 128 --col-balance-exps 0.5 --sgpt-percdamps 0.01,0.1 \
    --ds-tasks arc_easy,piqa --ppl-tasks "" --limit 1000 \
    --csv $CSV --log-dir results/sens_alloc/baseline_cmp/logs 2>&1 | tail -3
done
echo "########## BASELINE CMP DONE ##########"
