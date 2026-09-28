#!/usr/bin/env bash
# Launch the transferred-configuration ViT suite, one MIG slice per model.
# A model's whole grid stays on one slice: placement moves scores.
set -u
source "$(dirname "$0")/../env.sh" >/dev/null 2>&1
PY="$(command -v python)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
out="$ROOT/results/vit_transfer"; mkdir -p "$out"
declare -A SLICE=(
  [patch16-224]=MIG-bef7a31e-4317-582c-a97d-75e9e429441c
  [patch16-384]=MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a
)
for m in "${!SLICE[@]}"; do
  PYTHONUNBUFFERED=1 setsid nohup "$PY" -u "$ROOT/scripts/dispatch_vit_transfer.py" \
      --models "$m" --migs "${SLICE[$m]}" \
      >> "$out/dispatch_$m.log" 2>&1 < /dev/null &
  disown
  echo "launched $m on ${SLICE[$m]} (pid $!)"
done
