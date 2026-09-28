#!/usr/bin/env bash
# Launch the three transferred-config GLUE families, one MIG slice each.
# A family's whole grid stays on one slice: placement moves scores, so a split family
# would confound the arms against each other.
set -u
source "$(dirname "$0")/../env.sh" >/dev/null 2>&1
PY="$(command -v python)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
declare -A SLICE=(
  [electra]=MIG-bef7a31e-4317-582c-a97d-75e9e429441c
  [roberta-large]=MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a
  [deberta-v3-base]=MIG-475dbed1-782f-5c45-ae0c-1d2507268638
)
for fam in "${!SLICE[@]}"; do
  out="$ROOT/results/glue_transfer_$fam"; mkdir -p "$out"
  setsid nohup "$PY" "$ROOT/scripts/dispatch_glue_transfer.py" \
      --family "$fam" --migs "${SLICE[$fam]}" \
      > "$out/dispatch.log" 2>&1 < /dev/null &
  disown
  echo "launched $fam on ${SLICE[$fam]} (pid $!) -> $out/dispatch.log"
done
