#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-25: keep .16 fully used). When the errors-only ECHO launcher exits (stopped or
# finished), free the GPUs and resume the official .16 evaluation of the four ECHO arms (paused at 03:24 for this run),
# as the SAO run was evaluated: rllm eval_queue_follow.sh +
#      eval_gpu_pool.py (bf16, greedy, LoRA unmerged, 8 episodes per GPU, 1800 s per call), dev, test_normal_scen1,
#      test_challenge_scen1 of every kept whole epoch (global_step 6k, k >= 2). Base rows link to the SAO evaluation's.
#
#   after_errors_resume_eval_dot16.sh ERRORS_LAUNCHER_PID
set -u
PID=${1:?usage: $0 ERRORS_LAUNCHER_PID}
EVAL_ROOT=/works/yanan/sao-appworld-eval
CTRL=$EVAL_ROOT/echo-source-queue
SAO_EVAL=$EVAL_ROOT/appworld-qwen35-tp2-b15
MODEL=/works/yanan/models/qwen35-4b-851bf6e8-eosfix
VENV=/home/yanan/agents/rllm/.venv
RLLM=/home/yanan/agents/rllm/exp_scripts/appworld-prpo-run
STEPS_PER_EPOCH=6
[ -d /works/yanan ] || { echo "run this on .16" >&2; exit 1; }
mkdir -p "$CTRL"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL_ROOT/echo-source.log"; }

log "waiting for the errors-only launcher $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
log "errors-only launcher exited: $(grep -ohE 'driver exited rc=[0-9]+' /works/yanan/reef-sao-appworld-qwen35-tp2-echo-errors/ops/launcher.log | tail -1)"
# The driver starts each AppWorld server in its own session; kill the training ones by their root.
ps -eo pid=,args= | awk '$0 ~ /appworld serve environment/ && $0 ~ /--root \/home\/yanan\/appworld-root( |$)/ {print $1}' \
  | xargs -r kill -9
docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [[ $busy == 0 ]] && break; sleep 10
done
(( busy == 0 )) || { log "GPUs still busy; evaluation NOT resumed"; exit 1; }

# The official evaluation of the four ECHO arms.
arms=(echo echo-nodocs echo-nodocs-fixed echo-errors)
echo "# SPLIT OUT ADAPTER_SOURCE SELECT" > "$CTRL/queue.txt.new"
for split in dev test_normal_scen1 test_challenge_scen1; do
  for arm in "${arms[@]}"; do
    eval_dir=$EVAL_ROOT/appworld-qwen35-tp2-$arm-b15
    mkdir -p "$eval_dir/$split"
    ln -sfn "$SAO_EVAL/$split/base" "$eval_dir/$split/base"
    echo "$split $eval_dir/$split $eval_dir/adapters whole:$STEPS_PER_EPOCH" >> "$CTRL/queue.txt.new"
  done
done
for arm in "${arms[@]}"; do
  src=$EVAL_ROOT/appworld-qwen35-tp2-$arm-b15/adapters
  mkdir -p "$src"; n=0
  for dir in /works/yanan/reef-sao-appworld-qwen35-tp2-$arm/kept-checkpoints/adapters/hf_rollout_*; do
    step=$(( 10#${dir##*_} + 1 ))
    (( step % STEPS_PER_EPOCH == 0 && step > STEPS_PER_EPOCH )) || continue
    [[ -f $dir/hf/adapter_model.safetensors ]] && ln -sfn "$dir/hf" "$src/global_step_$step" && n=$((n + 1))
  done
  log "linked $n whole-epoch adapters of $arm as $src/global_step_N"
done
mv "$CTRL/queue.txt.new" "$CTRL/queue.txt"
for g in 0 1; do
  setsid nohup env EVAL_PYTHON=$VENV/bin/python APPWORLD_ROOT=/works/yanan/appworld-root-eval PYTHONPATH=/home/yanan/agents/rllm \
    LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib APPWORLD_LLM_TIMEOUT_S=1800 \
    "$RLLM/eval_queue_follow.sh" $g 1861$g "$CTRL" "$CTRL/queue.txt" 300 -- \
    --model-path "$MODEL" --episodes-per-gpu 8 \
    > /dev/null 2>&1 < /dev/null 9<&- &
done
sleep 3
log "started the .16 evaluation of ${arms[*]} (dev, test_normal_scen1, test_challenge_scen1; base linked from SAO)"
