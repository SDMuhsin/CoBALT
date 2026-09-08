set -e
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
CSV=/workspace/PTQResearch/results/awclip_smoke/smoke.csv
COM="python -u src/camera_bench.py --model gemma-2b --sparsity 0.5 --bits 3 --cobalt-group-size 128 --cobalt-beta 0.5 --force-true-bits --csv $CSV --ppl-tasks wikitext2 --ds-tasks arc_easy,piqa --limit 600"
echo "=== cobalt (RTN reference) ==="; $COM --method cobalt --hp "smoke"
echo "=== cobalt-awclip ==="; $COM --method cobalt-awclip --hp "smoke"
echo "ALL_SMOKE_DONE"
