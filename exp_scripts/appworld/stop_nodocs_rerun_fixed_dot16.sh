#!/usr/bin/env bash
# Run ON .16 only after the user's rule fires (2026-10-24: "如果 from epoch 1 to 5 一直下降 那么 就不需要继续了";
# then "改配置重跑：固定一个小系数（如 0.0025）"): stop the echo-nodocs run once its epoch-5 adapter (step 30) is kept,
# and launch the fixed-weight rerun (configs/serve-appworld-qwen35-tp2-echo-nodocs-fixed.yaml) with the same layout,
# image and sample cap.
#   1. Stop the after_echo_nodocs_dot16.sh chain first, so it does not start the .16 evaluation on the freed GPUs.
#   2. Require the kept step-30 adapter, then stop the nodocs launcher's process group (driver, sidecar, AppWorld
#      servers) and remove its stack.
#   3. Launch SAO_ECHO=1 SAO_ECHO_VARIANT=nodocs-fixed.
#
#   stop_nodocs_rerun_fixed_dot16.sh CHAIN_PID
set -u
CHAIN=${1:?usage: $0 CHAIN_PID}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPRO="$(cd "$HERE/.." && pwd)"
RUN=/works/yanan/reef-sao-appworld-qwen35-tp2-echo-nodocs
LOG=$RUN/ops/stopped.log
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" | tee -a "$LOG"; }
[[ $(cat "/proc/$CHAIN/cmdline" 2>/dev/null | tr '\0' ' ') == *after_echo_nodocs_dot16.sh* ]] \
  || { echo "PID $CHAIN is not after_echo_nodocs_dot16.sh" >&2; exit 1; }
[[ -f $RUN/kept-checkpoints/adapters/hf_rollout_00029/hf/adapter_model.safetensors ]] \
  || { echo "the epoch-5 adapter (hf_rollout_00029) is not kept yet" >&2; exit 1; }
PID=$(cat "$RUN/ops/launcher.pid")
kill "$CHAIN" && log "stopped the evaluation chain $CHAIN (user rule: nodocs declined from epoch 1 to 5)"
kill -TERM -- "-$PID" && log "stopped the nodocs launcher process group $PID after the kept step-30 adapter"
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -0 "$PID" 2>/dev/null && { kill -KILL -- "-$PID"; log "process group $PID killed"; }
# The driver starts each AppWorld server in its own session, so the group kill above misses them. Left alive, they
# hold ports 7800-7814 while the next driver starts: its readiness probe reaches the old server, /initialize goes there
# and every later call 400s on the new one (2026-10-24 17:28, the first rerun wave). Kill them by their training root.
ps -eo pid=,args= | awk '$0 ~ /appworld serve environment/ && $0 ~ /--root \/home\/yanan\/appworld-root( |$)/ {print $1}' \
  | xargs -r kill -9 && log "killed the stopped run's AppWorld servers"
docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [[ $busy == 0 ]] && break; sleep 10
done
(( busy == 0 )) || { log "GPUs still busy; rerun NOT launched"; exit 1; }
log "launching the fixed-weight rerun (nodocs-fixed, coef 0.0025)"
cd "$REPRO" && SAO_ECHO=1 SAO_ECHO_VARIANT=nodocs-fixed LAYOUT=tp2 SAO_MAX_SAMPLE_TOKENS=32768 \
  bash appworld/launch_16_qwen35_2gpu.sh formal 2>&1 | tee -a "$LOG"
