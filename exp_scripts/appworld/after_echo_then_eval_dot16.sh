#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-23: SAO+ECHO is compared directly with the previous SAO AppWorld table): wait for the
# SAO+ECHO run to exit; if it trained every step, remove its stack, expose its whole-epoch adapters (global_step 6k,
# k >= 2; step 6 is the critic-only zero adapter) and evaluate them exactly as the SAO run was evaluated: .16, bf16,
# rllm eval_queue_follow.sh + eval_gpu_pool.py (greedy, LoRA unmerged, 8 episodes per GPU, 1800 s per call), dev in
# full, test_normal_scen1 and test_challenge_scen1. The base rows are the SAO evaluation's own (same host, protocol and
# weights), linked rather than re-run. Results: /works/yanan/sao-appworld-eval/appworld-qwen35-tp2-echo-b15.
#
#   after_echo_then_eval_dot16.sh ECHO_RUN_ROOT
set -u
RUN=${1:?usage: $0 ECHO_RUN_ROOT}
RLLM=/home/yanan/agents/rllm/exp_scripts/appworld-prpo-run
SAO_EVAL=/works/yanan/sao-appworld-eval/appworld-qwen35-tp2-b15
EVAL=/works/yanan/sao-appworld-eval/appworld-qwen35-tp2-echo-b15
SRC=$EVAL/adapters
CTRL=$EVAL/queue
STEPS_PER_EPOCH=6
[ -d /works/yanan ] || { echo "run this on .16" >&2; exit 1; }
mkdir -p "$EVAL" "$SRC" "$CTRL"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL/after_echo.log"; }
PID=$(cat "$RUN/ops/launcher.pid")
log "waiting for the SAO+ECHO launcher $PID ($RUN)"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
rc=$(grep -ohE "driver exited rc=[0-9]+" "$RUN/ops/launcher.log" | tail -1)
trained=$(grep -ohE "trained: [0-9]+/[0-9]+ steps" "$RUN/ops/launcher.log" | tail -1)
log "launcher exited: '$rc', '$trained'"
if [[ $rc != "driver exited rc=0" || ! $trained =~ ^trained:\ ([0-9]+)/([0-9]+) || ${BASH_REMATCH[1]} != "${BASH_REMATCH[2]}" ]]; then
  log "SAO+ECHO did not finish every step; no evaluation started"
  exit 1
fi
docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [[ $busy == 0 ]] && break; sleep 10
done
(( busy == 0 )) || { log "GPUs still busy after removing the stack; evaluation NOT started"; exit 1; }
n=0
for dir in "$RUN"/kept-checkpoints/adapters/hf_rollout_*; do
  step=$(( 10#${dir##*_} + 1 ))
  (( step % STEPS_PER_EPOCH == 0 && step > STEPS_PER_EPOCH )) || continue
  [[ -f $dir/hf/adapter_model.safetensors ]] && ln -sfn "$dir/hf" "$SRC/global_step_$step" && n=$((n + 1))
done
log "linked $n whole-epoch adapters as $SRC/global_step_N"
for split in dev test_normal_scen1 test_challenge_scen1; do
  mkdir -p "$EVAL/$split"
  ln -sfn "$SAO_EVAL/$split/base" "$EVAL/$split/base"
done
cat > "$CTRL/queue.txt" <<EOF
# SPLIT OUT ADAPTER_SOURCE SELECT
dev $EVAL/dev $SRC whole:$STEPS_PER_EPOCH
test_normal_scen1 $EVAL/test_normal_scen1 $SRC whole:$STEPS_PER_EPOCH
test_challenge_scen1 $EVAL/test_challenge_scen1 $SRC whole:$STEPS_PER_EPOCH
EOF
VENV=/home/yanan/agents/rllm/.venv
for g in 0 1; do
  setsid nohup env EVAL_PYTHON=$VENV/bin/python APPWORLD_ROOT=/works/yanan/appworld-root-eval PYTHONPATH=/home/yanan/agents/rllm \
    LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib APPWORLD_LLM_TIMEOUT_S=1800 \
    "$RLLM/eval_queue_follow.sh" $g 1861$g "$CTRL" "$CTRL/queue.txt" 300 -- \
    --model-path /works/yanan/models/qwen35-4b-851bf6e8-eosfix --episodes-per-gpu 8 \
    > /dev/null 2>&1 < /dev/null 9<&- &
done
sleep 3
log "started the .16 evaluation (dev, test_normal_scen1, test_challenge_scen1 of every whole epoch; base linked from SAO)"
