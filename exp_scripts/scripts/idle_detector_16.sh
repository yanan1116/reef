#!/usr/bin/env bash
# Fallback launcher (asked 2026-09-29): every 10 minutes, check whether .16 is idle; after two
# consecutive idle checks (20 min, so a stack restart is not mistaken for idleness), start the
# AppWorld SAO run on .16 (appworld/smoke_then_formal.sh: smoke, then the formal run if it passes)
# once, and exit. It never stops or kills anything; it only removes an *exited* reef-sao-stack
# container, which smoke_then_formal.sh would otherwise refuse to start next to.
#
# .16 is idle when all hold:
#   no running reef-sao-stack container,
#   no SAO launcher (bash .../{finqa_multitable,appworld,finqa_singletable,finqa}/{smoke_then_formal,run_*}.sh)
#     and no SAO driver / checkpoint copier (.venv-finqa python -u .../stream_*.py | sidecar.py),
#   both GPUs below 1000 MiB.
# An ssh failure counts as busy (never launch on missing information).
#
# AppWorld run: Qwen3-4B-Instruct-2507, react_code, 30 tasks/step, 90 steps, T=1.0, fp32 logits,
# length-filtered (assembled samples > 16,384 tokens not trained), data under /works/yanan.
#
# Runs on .29: nohup setsid bash scripts/idle_detector_16.sh; log results/appworld/idle-detector-16.log
set -uo pipefail
X=/home/yanan/agents/reef/exp_scripts
LOG=$X/results/appworld/idle-detector-16.log
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 10.225.68.16)
INTERVAL_S=${INTERVAL_S:-600}
log() { echo "[idle-detector $(date '+%F %T %Z')] $*" >> "$LOG"; }

exec 9<>"$X/results/appworld/idle-detector-16.lock"
flock -n 9 || { log "another detector holds the lock; exiting"; exit 1; }
echo $$ > "$X/results/appworld/idle-detector-16.pid"

REMOTE_STATE='c=$(docker ps --format "{{.Names}}" | grep -cx reef-sao-stack)
p=$(ps -eo args | grep -cE "^bash /home/yanan/agents/reef/exp_scripts/(finqa_multitable|appworld|finqa_singletable|finqa)/(smoke_then_formal|run_[a-z_]+)\.sh|^/home/yanan/agents/reef/exp_scripts/\.venv-finqa/bin/python -u /home/yanan/agents/reef/exp_scripts/[a-z_/]*(stream_[a-z]+|sidecar)\.py")
g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1 > 1000" | wc -l)
echo "$c $p $g"'

log "start: interval ${INTERVAL_S}s; launches AppWorld SAO on .16 after 2 consecutive idle checks"
streak=0
while :; do
  if state=$("${SSH[@]}" "$REMOTE_STATE" 2>/dev/null) && read -r containers procs gpus <<< "$state" && [ -n "${gpus:-}" ]; then
    if [ "$containers" = 0 ] && [ "$procs" = 0 ] && [ "$gpus" = 0 ]; then
      streak=$((streak + 1))
      log "idle check $streak/2 (stack containers=$containers sao processes=$procs busy GPUs=$gpus)"
    else
      [ "$streak" -gt 0 ] && log "busy again (containers=$containers processes=$procs GPUs=$gpus); idle streak reset"
      streak=0
    fi
  else
    log "ssh/state check failed ('${state:-}'); counted as busy"
    streak=0
  fi
  if [ "$streak" -ge 2 ]; then
    exited=$("${SSH[@]}" "docker ps -a --filter name=^reef-sao-stack$ --filter status=exited --format '{{.Names}}'")
    if [ -n "$exited" ]; then
      "${SSH[@]}" "docker rm reef-sao-stack" >> "$LOG" 2>&1 && log "removed the exited reef-sao-stack container"
    fi
    TS=$(date +%Y%m%dT%H%M%S)
    L=$X/results/appworld/smoke_then_formal-16-$TS.log
    echo "$L" > "$X/results/appworld/LATEST_STF16"
    timeout 60 "${SSH[@]}" "nohup setsid env DOCKER_RUNTIME=nvidia REEF_BF16_LOGITS=0 RUN_BASE=/works/yanan/reef-sao-appworld \
      APPWORLD_ROOT=/home/yanan/appworld-root APPWORLD_VENV=/home/yanan/.venvs-local/appworld \
      bash $X/appworld/smoke_then_formal.sh > $L 2>&1 < /dev/null &"
    sleep 90
    if grep -q "smoke: tag=" "$L" 2>/dev/null; then
      log "LAUNCHED AppWorld SAO smoke_then_formal on .16 (log $L); exiting"
    else
      log "LAUNCH FAILED: $(tail -n 3 "$L" 2>/dev/null | tr '\n' ' '); exiting"
    fi
    exit 0
  fi
  sleep "$INTERVAL_S"
done
