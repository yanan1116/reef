#!/usr/bin/env bash
# .29 interim tracking of one .16 ECHO arm (user, 2026-10-24: evaluate each arm promptly): link each new whole-epoch
# adapter (global_step 6k, k >= 2) of RUN_ROOT for the .29 queue followers started by echo_eval_follow_29.sh, whose
# queue.txt carries this arm's lines. Exits when $EVAL/queue/STOP exists.
#
#   arm_link_29.sh ARM RUN_ROOT_NAME     e.g. nodocs-fixed reef-sao-appworld-qwen35-tp2-echo-nodocs-fixed
set -u
ARM=${1:?arm} RUN=${2:?run root name under /works/yanan}
SRC=/home/yanan/mnt/dot16-works/$RUN/kept-checkpoints/adapters
EVAL=/mnt/disk1t/sao-echo-appworld-eval-29
DST=$EVAL/$ARM/adapters
mkdir -p "$DST" "$EVAL/$ARM/dev" "$EVAL/$ARM/test_normal_scen1"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL/follow.log"; }
while [ ! -e "$EVAL/queue/STOP" ]; do
  for dir in "$SRC"/hf_rollout_*; do
    [ -f "$dir/hf/adapter_model.safetensors" ] || continue
    step=$(( 10#${dir##*_} + 1 ))
    (( step % 6 == 0 && step > 6 )) || continue
    [ -e "$DST/global_step_$step" ] && continue
    sleep 30   # let the sidecar finish copying the adapter
    ln -sfn "$dir/hf" "$DST/global_step_$step"
    log "linked $ARM epoch $((step / 6)) (global_step_$step)"
  done
  sleep 120
done
