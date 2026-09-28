#!/bin/bash
# Run all three tuned suites back to back, one at a time so they do not fight over the
# shared MIG slices. Each stage is resume-safe, so the chain can be killed and relaunched.
#   setsid nohup bash scripts/run_table1_all.sh > <log> 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(STAMP) CHAIN_START"
for suite in gemma vit glue; do
  echo "$(STAMP) SUITE_BEGIN $suite"
  bash scripts/run_table1_sweep.sh "$suite"
  echo "$(STAMP) SUITE_END $suite rc=$?"
done
echo "$(STAMP) CHAIN_DONE"
