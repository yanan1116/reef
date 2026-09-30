#!/usr/bin/env bash
# One-off (2026-09-29): stop the FinQA SAO run on .16 once its step-279 adapter is kept.
# AppWorld starts separately, after the bf16-logits image passes its smoke on .16.
set -uo pipefail
X=/home/yanan/agents/reef/exp_scripts
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 10.225.68.16)
log() { echo "[stop279 $(date '+%F %T %Z')] $*"; }
log "waiting for FinQA step 279 (hf_rollout_00278) to be kept on .16"
until "${SSH[@]}" "test -f /home/yanan/reef-sao-finqa/kept-checkpoints/adapters/hf_rollout_00278/hf/adapter_model.safetensors"; do sleep 60; done
log "step 279 adapter kept; stopping FinQA"
"${SSH[@]}" "bash $X/finqa/stop_finqa.sh" 2>&1 | tail -5
sleep 10
"${SSH[@]}" "docker ps -a --format '{{.Names}}'" | grep -qx reef-sao-stack && { log "STOPPED: reef-sao-stack still exists"; exit 1; }
[ -f $X/results/finqa/gpu_monitor_16.tsv.pid ] && kill "$(cat $X/results/finqa/gpu_monitor_16.tsv.pid)" 2>/dev/null
log "DONE: FinQA stopped at step $(cat $X/results/finqa/finqa-b64-20260927T004448/progress.txt); .16 GPU monitor stopped"
