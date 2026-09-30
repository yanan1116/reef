#!/usr/bin/env bash
# One-off (asked for 2026-09-29): on .16, stop the FinQA SAO run once its step-279 adapter is
# kept, then start the AppWorld SAO formal run (2507, 30 tasks/step, 90 steps, context 32768).
# Runs on .29 and drives .16 over ssh. Every step checks its precondition and stops with the
# reason when one fails.
set -uo pipefail
X=/home/yanan/agents/reef/exp_scripts
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 10.225.68.16)
log() { echo "[switch $(date '+%F %T %Z')] $*"; }
die() { log "STOPPED: $*"; exit 1; }

log "1. waiting for FinQA step 279 (hf_rollout_00278) to be kept on .16"
until "${SSH[@]}" "test -f /home/yanan/reef-sao-finqa/kept-checkpoints/adapters/hf_rollout_00278/hf/adapter_model.safetensors"; do
  sleep 60
done
log "   step 279 adapter kept"

"${SSH[@]}" "bash $X/finqa/stop_finqa.sh" 2>&1 | tail -5
sleep 10
"${SSH[@]}" "docker ps -a --format '{{.Names}}'" | grep -qx reef-sao-stack && die "reef-sao-stack still exists on .16 after stop_finqa.sh"
busy=$("${SSH[@]}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" | awk -F', ' '$2 > 1000 {print $1}')
[ -z "$busy" ] || die ".16 GPU(s) $busy still hold memory after stop_finqa.sh"
log "2. FinQA training on .16 stopped (step $(cat $X/results/finqa/finqa-b64-20260927T004448/progress.txt)); GPUs free"

TS=$(date +%Y%m%dT%H%M%S)
TAG=appworld-b30-$TS
LAUNCH=$X/results/appworld/launch-16-$TS.log
timeout 60 "${SSH[@]}" "nohup setsid env DOCKER_RUNTIME=nvidia RUN_ROOT=/home/yanan/reef-sao-appworld \
  APPWORLD_ROOT=/home/yanan/appworld-root APPWORLD_VENV=/home/yanan/.venvs-local/appworld \
  CFG=/repro/configs/serve-appworld-2507.yaml TAG=$TAG bash $X/appworld/run_appworld.sh > $LAUNCH 2>&1 < /dev/null &" || true
for i in $(seq 1 60); do grep -qE "driver: batch|exited|never became healthy|expected" "$LAUNCH" 2>/dev/null && break; sleep 20; done
grep -q "driver: batch" "$LAUNCH" || die "AppWorld run did not reach its driver; see $LAUNCH"
log "3. AppWorld formal run started on .16: $TAG (log $LAUNCH)"

[ -f $X/results/finqa/gpu_monitor_16.tsv.pid ] && kill "$(cat $X/results/finqa/gpu_monitor_16.tsv.pid)" 2>/dev/null
OUT=$X/results/appworld/gpu_monitor_16.tsv RUN_GLOB="$X/results/appworld/appworld-b30-*" \
  nohup setsid bash $X/scripts/gpu_monitor.sh 10.225.68.16 > /dev/null 2>&1 < /dev/null &
log "4. .16 GPU monitor now follows $TAG (results/appworld/gpu_monitor_16.tsv)"
