#!/usr/bin/env bash
# Run the medgemma-27b AWQ QUALITY eval on the 48 GB slice when it is idle
# (24 GB slice gives ~1 KV sequence -> hours). Polls every 2 min (nvidia-smi MIG row + /proc/*/environ),
# then runs the runner's eval stage there, kills the 24 GB quality runner/chain by PID once the 48 GB engine
# is past model load, and summarizes.
MIG48=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
ROOT=/workspace/PTQResearch; R=$ROOT/results/accel4bit/medgemma-27b/awq
PY=/scratch/root/PTQResearch/env_accel_awq/bin/python
stamp() { date '+%F %T'; }
busy() {  # 0 = busy, 1 = idle
  local mem; mem=$(nvidia-smi 2>/dev/null | awk '/MIG devices/,/^$/' | grep -E "^\|  +1 +1 " | grep -oE "[0-9]+MiB" | head -1 | tr -d MiB)  # GPU1 GI1 = the 2g.48gb slice (nvidia-smi --id=<MIG> is not supported here)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -q "$MIG48" && { echo "[$(stamp)] busy: pid $p on $MIG48 (mem ${mem} MiB)"; return 0; }
  done
  [ "${mem:-0}" -gt 1024 ] && { echo "[$(stamp)] busy: ${mem} MiB used on $MIG48 (no environ match)"; return 0; }
  return 1
}
echo "[$(stamp)] eval48 poller start"
while busy; do sleep 120; done
echo "[$(stamp)] $MIG48 idle -> launching 27B eval there"
mkdir -p $R/eval24_abandoned; for f in eval.log eval_summary.json; do [ -f $R/$f ] && mv $R/$f $R/eval24_abandoned/; done; rm -f $R/eval_*.json
( bash $ROOT/scripts/run_accel4bit_awq.sh medgemma-27b $MIG48 eval ) & EV=$!
# once the 48 GB engine is past model load, kill the 24 GB quality runner + chain by PID
for i in $(seq 1 60); do sleep 20; grep -qE "Running|\[accel4bit_lmeval\] task|Processed prompts|load_s" $R/eval.log 2>/dev/null && break; done
for p in $(pgrep -f '[r]un_accel4bit_awq_chain.sh') $(pgrep -f "[r]un_accel4bit_awq.sh medgemma-27b MIG-12daede5"); do echo "[$(stamp)] killing 24GB job pid $p ($(tr '\0' ' ' </proc/$p/cmdline | cut -c1-80))"; pkill -P $p; kill $p; done
for p in $(pgrep -f '[a]ccel4bit_lmeval.py' ) $(pgrep -f '[v]llm.entrypoints.cli.main bench'); do tr '\0' ' ' </proc/$p/environ 2>/dev/null | grep -q MIG-12daede5 && { echo "[$(stamp)] killing 24GB pid $p"; kill $p; }; done
wait $EV; echo "[$(stamp)] eval48 runner exit $?"
$PY $ROOT/src/accel4bit_awq_summarize.py $R $MIG48
echo "[$(stamp)] EVAL48_DONE"
