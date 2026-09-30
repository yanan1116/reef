#!/usr/bin/env bash
# One-off (2026-09-29): once stop_finqa_after_279.sh reports DONE, run the AppWorld SAO
# smoke-then-formal on .16 with the bf16-logits image (reef:sao-59b3c50b, REEF_BF16_LOGITS=1),
# the host-local AppWorld env, and the .16 configs (context 32768, max-tokens-per-gpu 16384).
set -uo pipefail
X=/home/yanan/agents/reef/exp_scripts
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 10.225.68.16)
STOP_LOG=$X/$(cat $X/results/appworld/LATEST_STOP279)
log() { echo "[appworld16 $(date '+%F %T %Z')] $*"; }
log "waiting for FinQA to stop ($STOP_LOG)"
until grep -qE "DONE|STOPPED" "$STOP_LOG"; do sleep 60; done
grep -q DONE "$STOP_LOG" || { log "STOPPED: $(tail -1 "$STOP_LOG")"; exit 1; }
busy=$("${SSH[@]}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" | awk -F', ' '$2 > 1000 {print $1}')
[ -z "$busy" ] || { log "STOPPED: .16 GPU(s) $busy hold memory"; exit 1; }
STF=$X/results/appworld/smoke_then_formal-16-$(date +%Y%m%dT%H%M%S).log
timeout 60 "${SSH[@]}" "nohup setsid env DOCKER_RUNTIME=nvidia RUN_BASE=/home/yanan/reef-sao-appworld \
  APPWORLD_ROOT=/home/yanan/appworld-root APPWORLD_VENV=/home/yanan/.venvs-local/appworld REEF_BF16_LOGITS=1 \
  bash $X/appworld/smoke_then_formal.sh > $STF 2>&1 < /dev/null &" || true
echo "$STF" > $X/results/appworld/LATEST_STF16
sleep 30
grep -q "smoke: tag=" "$STF" || { log "STOPPED: smoke did not start; see $STF"; exit 1; }
OUT=$X/results/appworld/gpu_monitor_16.tsv RUN_GLOB="$X/results/appworld/appworld-*" \
  nohup setsid bash $X/scripts/gpu_monitor.sh 10.225.68.16 > /dev/null 2>&1 < /dev/null &
log "DONE: AppWorld smoke started on .16 ($STF); formal follows if it passes; .16 GPU monitor on appworld runs"
