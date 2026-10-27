#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-25: errors-only-last-line on .16 after errors-only seed 1). Wait for the seed-1 run
# to start and then to exit, free the GPUs, and launch errors-only-last-line
# (configs/serve-appworld-qwen35-tp2-echo-errors-last-line.yaml) with the same image, budget, seed 0 and sample cap.
#
#   after_seed1_launch_last_line_dot16.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; REPRO="$(cd "$HERE/.." && pwd)"
SEED1=/works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors-seed1
LOG=/works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors/ops/after_seed1.log
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
log "waiting for the seed-1 launcher to start"
until [ -s "$SEED1/ops/launcher.pid" ]; do sleep 60; done
PID=$(cat "$SEED1/ops/launcher.pid")
log "waiting for the seed-1 launcher $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
log "seed-1 launcher exited: $(grep -ohE 'driver exited rc=[0-9]+' "$SEED1/ops/launcher.log" | tail -1), $(grep -ohE 'trained: [0-9]+/[0-9]+ steps' "$SEED1/ops/launcher.log" | tail -1)"
# The driver starts each AppWorld server in its own session; kill the training ones by their root.
ps -eo pid=,args= | awk '$0 ~ /appworld serve environment/ && $0 ~ /--root \/home\/yanan\/appworld-root( |$)/ {print $1}' \
  | xargs -r kill -9
docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [[ $busy == 0 ]] && break; sleep 10
done
(( busy == 0 )) || { log "GPUs still busy; errors-only-last-line NOT launched"; exit 1; }
log "launching errors-only-last-line"
cd "$REPRO" && SAO_ECHO=1 SAO_ECHO_VARIANT=errors-last-line LAYOUT=tp2 SAO_MAX_SAMPLE_TOKENS=32768 \
  bash appworld/launch_16_qwen35_2gpu.sh formal >> "$LOG" 2>&1
