#!/usr/bin/env bash
# One-off (2026-09-29): on .29, once the FinQA single-table evaluation queue is drained (the training
# stopped at step 279, so its job set is final), halt its two workers and start the multi-table
# workers (finqa_multitable/eval_queue_worker.sh) on GPU 0 and 1. Every step checks its
# precondition and stops with the reason when one fails.
set -uo pipefail
X=/home/yanan/agents/reef/exp_scripts
SINGLE_RUN=finqa-b64-20260927T004448
SINGLE_OUT=$X/results/finqa-eval/$SINGLE_RUN
SINGLE_CLAIMS=/mnt/disk1t/sao-finqa-eval/claims/$SINGLE_RUN
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 10.225.68.16)
log() { echo "[handoff $(date '+%F %T %Z')] $*"; }
die() { log "STOPPED: $*"; exit 1; }

rids=$("${SSH[@]}" "ls /home/yanan/reef-sao-finqa/kept-checkpoints/adapters" | sed -n 's/^hf_rollout_//p')
[ -n "$rids" ] || die "expected the single-table adapters on .16; none listed"
expected=()
for rid in $rids; do
  for job in lora_t0k4 lora_t07k4 merged_greedy merged_t07k4; do expected+=("$(printf 'step_%03d_%s' $((10#$rid + 1)) "$job")"); done
done
log "1. waiting for ${#expected[@]} single-table jobs to finish (each needs \$OUT/<tag>/test.json)"
while :; do
  left=0
  for tag in "${expected[@]}"; do [ -s "$SINGLE_OUT/$tag/test.json" ] || left=$((left + 1)); done
  running=$(pgrep -fc "^bash $X/finqa/eval_finqa_checkpoint.sh" || true)
  [ "$left" -eq 0 ] && [ "$running" -eq 0 ] && break
  # a job that FAILED never writes test.json; stop waiting on it once nothing runs and nothing is unclaimed
  if [ "$running" -eq 0 ]; then
    unclaimed=0
    for tag in "${expected[@]}"; do [ -d "$SINGLE_CLAIMS/$tag" ] || unclaimed=$((unclaimed + 1)); done
    [ "$unclaimed" -eq 0 ] && { log "   $left job(s) failed (see the worker logs); the queue is drained"; break; }
  fi
  sleep 120
done
log "   single-table queue drained"

touch "$SINGLE_CLAIMS/HALT"
workers=$(pgrep -f "^bash $X/finqa/eval_queue_worker.sh" || true)
for i in $(seq 1 60); do
  workers=$(pgrep -f "^bash $X/finqa/eval_queue_worker.sh" || true)
  [ -z "$workers" ] && break
  sleep 30
done
[ -z "$workers" ] || die "single-table workers $workers still running 30 min after HALT"
log "2. single-table workers exited"

for gpu in 0 1; do
  nohup setsid bash "$X/finqa_multitable/eval_queue_worker.sh" "$gpu" "1807$((gpu + 3))" \
    >> "$X/results/finqa_multitable/eval_worker_gpu$gpu.log" 2>&1 < /dev/null &
  echo $! > "$X/results/finqa_multitable/eval_worker_gpu$gpu.pid"
done
sleep 5
for gpu in 0 1; do kill -0 "$(cat "$X/results/finqa_multitable/eval_worker_gpu$gpu.pid")" 2>/dev/null || die "multi-table worker gpu$gpu exited at start"; done
log "3. DONE: multi-table workers started (results/finqa_multitable/eval_worker_gpu{0,1}.log)"
