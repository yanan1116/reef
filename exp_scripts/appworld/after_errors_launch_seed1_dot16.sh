#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-25: "评测只在 .29 进行；.16 只负责训练"). When the errors-only ECHO run exits, free the
# GPUs and launch its replicate: the same config (configs/serve-appworld-qwen35-tp2-echo-errors.yaml), image, budget
# and sample cap, with SAO_SEED=1 (the driver's task order; the paired SAO-vs-errors gap is within noise so far).
#
#   after_errors_launch_seed1_dot16.sh ERRORS_LAUNCHER_PID
set -u
PID=${1:?usage: $0 ERRORS_LAUNCHER_PID}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPRO="$(cd "$HERE/.." && pwd)"
LOG=/works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors/ops/after_errors.log
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
log "waiting for the errors-only launcher $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
log "errors-only launcher exited: $(grep -ohE 'driver exited rc=[0-9]+' /works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors/ops/launcher.log | tail -1), $(grep -ohE 'trained: [0-9]+/[0-9]+ steps' /works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors/ops/launcher.log | tail -1)"
# The driver starts each AppWorld server in its own session; kill the training ones by their root.
ps -eo pid=,args= | awk '$0 ~ /appworld serve environment/ && $0 ~ /--root \/home\/yanan\/appworld-root( |$)/ {print $1}' \
  | xargs -r kill -9
docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [[ $busy == 0 ]] && break; sleep 10
done
(( busy == 0 )) || { log "GPUs still busy; seed-1 run NOT launched"; exit 1; }
log "launching errors-only ECHO seed 1"
cd "$REPRO" && SAO_ECHO=1 SAO_ECHO_VARIANT=errors SAO_SEED=1 LAYOUT=tp2 SAO_MAX_SAMPLE_TOKENS=32768 \
  RUN_ROOT=/works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors-seed1 \
  TAG=appworld-qwen35-tp2-echo-errors-seed1-b15-$(date +%Y%m%dT%H%M%S) \
  bash appworld/launch_16_qwen35_2gpu.sh formal >> "$LOG" 2>&1
