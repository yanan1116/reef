#!/usr/bin/env bash
# .29 interim tracking of the echo-nodocs arm (user, 2026-10-24: "no docs 的评估目前结果如何"): link each new
# whole-epoch adapter (global_step 6k, k >= 2) of the .16 nodocs run for the .29 queue followers started by
# echo_eval_follow_29.sh (its queue.txt carries the nodocs lines). Exits when $EVAL/queue/STOP exists.
set -u
SRC=/home/yanan/mnt/dot16-works/reef-sao-appworld-qwen35-tp2-echo-nodocs/kept-checkpoints/adapters
EVAL=/mnt/disk1t/sao-echo-appworld-eval-29
DST=$EVAL/nodocs/adapters
mkdir -p "$DST" "$EVAL/nodocs/dev" "$EVAL/nodocs/test_normal_scen1"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL/follow.log"; }
while [ ! -e "$EVAL/queue/STOP" ]; do
  for dir in "$SRC"/hf_rollout_*; do
    [ -f "$dir/hf/adapter_model.safetensors" ] || continue
    step=$(( 10#${dir##*_} + 1 ))
    (( step % 6 == 0 && step > 6 )) || continue
    [ -e "$DST/global_step_$step" ] && continue
    sleep 30   # let the sidecar finish copying the adapter
    ln -sfn "$dir/hf" "$DST/global_step_$step"
    log "linked nodocs epoch $((step / 6)) (global_step_$step)"
  done
  sleep 120
done
