#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-23): after the SAO+ECHO run, train the ECHO-source arm, then evaluate both.
# The pair compares where ECHO's cross-entropy comes from ("实验对照就是 ce loss 的来源之对比"):
#   echo         configs/serve-appworld-qwen35-tp2-echo.yaml         paper env_only, API documentation kept
#   echo-nodocs  configs/serve-appworld-qwen35-tp2-echo-nodocs.yaml  the same, API documentation skipped
# Code, image, seed, budget (10 epochs = 60 steps) and sample cap (32768) are the same; only sao-echo-rules differs.
#
# 1. Wait for the ECHO launcher. If it trained every step, remove its stack and launch the nodocs arm
#    (launch_16_qwen35_2gpu.sh formal, SAO_ECHO=1 SAO_ECHO_VARIANT=nodocs LAYOUT=tp2). Otherwise stop, stack kept.
# 2. Wait for the nodocs launcher, remove its stack, and evaluate the whole-epoch adapters (global_step 6k, k >= 2;
#    step 6 is the critic-only zero adapter) of both arms exactly as the SAO run was evaluated: .16, bf16, rllm
#    eval_queue_follow.sh + eval_gpu_pool.py (greedy, LoRA unmerged, 8 episodes per GPU, 1800 s per call), dev in
#    full, test_normal_scen1, test_challenge_scen1. Base rows link to the SAO evaluation's (same host and protocol).
#    An arm that did not train every step is left out of the evaluation.
#
#   after_echo_nodocs_dot16.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; REPRO="$(cd "$HERE/.." && pwd)"
ECHO_RUN=/works/yanan/reef-sao-appworld-qwen35-tp2-echo
NODOCS_RUN=/works/yanan/reef-sao-appworld-qwen35-tp2-echo-nodocs
EVAL_ROOT=/works/yanan/sao-appworld-eval
SAO_EVAL=$EVAL_ROOT/appworld-qwen35-tp2-b15
CTRL=$EVAL_ROOT/echo-source-queue
RLLM=/home/yanan/agents/rllm/exp_scripts/appworld-prpo-run
VENV=/home/yanan/agents/rllm/.venv
STEPS_PER_EPOCH=6
[ -d /works/yanan ] || { echo "run this on .16" >&2; exit 1; }
mkdir -p "$CTRL"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL_ROOT/echo-source.log"; }

# Wait for RUN's launcher; succeed only if the driver exited 0 after training every step.
finished() {
  local run=$1 pid rc trained
  pid=$(cat "$run/ops/launcher.pid")
  log "waiting for launcher $pid ($run)"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  rc=$(grep -ohE "driver exited rc=[0-9]+" "$run/ops/launcher.log" | tail -1)
  trained=$(grep -ohE "trained: [0-9]+/[0-9]+ steps" "$run/ops/launcher.log" | tail -1)
  log "launcher exited ($run): '$rc', '$trained'"
  [[ $rc == "driver exited rc=0" && $trained =~ ^trained:\ ([0-9]+)/([0-9]+) && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]]
}

free_gpus() {
  local busy
  docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
  for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
    [[ $busy == 0 ]] && return 0; sleep 10
  done
  log "GPUs still busy after removing the stack"; return 1
}

finished "$ECHO_RUN" || { log "SAO+ECHO did not finish every step; nodocs NOT launched, stack kept"; exit 1; }
free_gpus || { log "nodocs NOT launched"; exit 1; }
log "launching the nodocs arm"
(cd "$REPRO" && SAO_ECHO=1 SAO_ECHO_VARIANT=nodocs LAYOUT=tp2 SAO_MAX_SAMPLE_TOKENS=32768 \
  bash appworld/launch_16_qwen35_2gpu.sh formal) >> "$EVAL_ROOT/echo-source.log" 2>&1 \
  || { log "nodocs launch failed; no evaluation started"; exit 1; }

arms=(echo)
if finished "$NODOCS_RUN"; then arms+=(echo-nodocs); else log "nodocs did not finish every step; evaluating echo only"; fi
free_gpus || { log "evaluation NOT started"; exit 1; }
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
    --model-path /works/yanan/models/qwen35-4b-851bf6e8-eosfix --episodes-per-gpu 8 \
    > /dev/null 2>&1 < /dev/null 9<&- &
done
sleep 3
log "started the .16 evaluation of ${arms[*]} (dev, test_normal_scen1, test_challenge_scen1; base linked from SAO)"
