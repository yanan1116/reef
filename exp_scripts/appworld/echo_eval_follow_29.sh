#!/usr/bin/env bash
# Interim, same-host tracking of SAO+ECHO on .29 while it trains on .16 (user, 2026-10-23: "持续跟进表现"). Whenever
# the ECHO run keeps a whole-epoch adapter (global_step 6k, k >= 2; step 6 is the critic-only zero adapter), link it
# and the previous SAO run's adapter of the same step, so .29 evaluates both on the AppWorld dev split, plus the base
# once (to compare .29 with .16's base). Same episode protocol as the SAO table (rllm eval_gpu_pool.py: greedy, LoRA
# unmerged, 8 episodes per GPU, 1800 s per call, bf16). The comparable numbers for the SAO table still come from .16
# after training (the user compares each split only within one host); these are the paired .29 preview.
# .29 reads .16 through its read-only sshfs mount. Exits when $EVAL/queue/STOP exists.
#
#   echo_eval_follow_29.sh
set -u
DOT16=/home/yanan/mnt/dot16-works
ECHO_ADAPTERS=$DOT16/reef-sao-appworld-qwen35-tp2-echo/kept-checkpoints/adapters
SAO_ADAPTERS=$DOT16/reef-sao-appworld-qwen35-tp2/kept-checkpoints/adapters
EVAL=/mnt/disk1t/sao-echo-appworld-eval-29
CTRL=$EVAL/queue
RUN_DIR=/home/yanan/agents/rllm/exp_scripts/appworld-prpo-run
VENV=/home/yanan/agents/rllm/.venv
STEPS_PER_EPOCH=6
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
mkdir -p "$EVAL/echo/adapters" "$EVAL/sao/adapters" "$EVAL/echo/dev" "$EVAL/sao/dev" "$CTRL"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL/follow.log"; }
# ECHO's epochs first, each followed by its SAO twin; the base once, last.
cat > "$CTRL/queue.txt" <<EOF
# SPLIT OUT ADAPTER_SOURCE SELECT
dev $EVAL/echo/dev $EVAL/echo/adapters whole:$STEPS_PER_EPOCH
dev $EVAL/sao/dev $EVAL/sao/adapters whole:$STEPS_PER_EPOCH+base
EOF
for g in 0 1; do
  setsid nohup env EVAL_PYTHON=$VENV/bin/python APPWORLD_ROOT=/mnt/disk1t/appworld-root-eval PYTHONPATH=/home/yanan/agents/rllm \
    LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib APPWORLD_LLM_TIMEOUT_S=1800 \
    "$RUN_DIR/eval_queue_follow.sh" $g 1864$g "$CTRL" "$CTRL/queue.txt" 300 -- \
    --model-path /mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix --episodes-per-gpu 8 \
    > /dev/null 2>&1 < /dev/null 9<&- &
done
log "queue followers started on GPU 0 and 1 ($CTRL)"
while [ ! -e "$CTRL/STOP" ]; do
  for dir in "$ECHO_ADAPTERS"/hf_rollout_*; do
    [ -f "$dir/hf/adapter_model.safetensors" ] || continue
    step=$(( 10#${dir##*_} + 1 ))
    (( step % STEPS_PER_EPOCH == 0 && step > STEPS_PER_EPOCH )) || continue
    [ -e "$EVAL/echo/adapters/global_step_$step" ] && continue
    sao="$SAO_ADAPTERS/hf_rollout_$(printf %05d $((step - 1)))/hf"
    [ -f "$sao/adapter_model.safetensors" ] || { log "no SAO adapter for step $step"; continue; }
    sleep 30   # let the sidecar finish copying the ECHO adapter
    ln -sfn "$dir/hf" "$EVAL/echo/adapters/global_step_$step"
    ln -sfn "$sao" "$EVAL/sao/adapters/global_step_$step"
    log "linked epoch $((step / STEPS_PER_EPOCH)) (global_step_$step) for ECHO and SAO"
  done
  sleep 300
done
log "STOP found"
