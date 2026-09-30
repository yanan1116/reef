#!/usr/bin/env bash
# One evaluation worker per .29 GPU for the FinQA multi-table SAO formal run: repeatedly take the
# highest-priority unevaluated job for the adapters kept on the training host, read the adapter in place
# here, and run eval_multitable_checkpoint.sh on it. finqa/eval_queue_worker.sh with the benchmark
# swapped (same job classes and order); the base job covers both sampling settings up front.
#
# Jobs, in priority order (step = rollout_id + 1):
#   0. base_t0k4, base_t07k4 (MODE=base) -- once each, into results/finqa_multitable-eval/; then the same two
#      for Qwen3.5-4B (base_qwen35_*: eos-fixed snapshot, qwen3_coder parser, thinking off, and its template
#      with later system messages rendered as system turns, templates/qwen35_relaxed_system.jinja)
#   1. lora_t0k4     (MODE=lora   SAMPLING=t0k4)    -- greedy x4, avg@4: the headline; newest step first
#   2. lora_t07k4    (MODE=lora   SAMPLING=t07k4)
#   3. merged_greedy (MODE=merged SAMPLING=greedy)  -- PRPO's bf16-merged protocol (one greedy attempt)
#   4. merged_t07k4  (MODE=merged SAMPLING=t07k4)
# Within classes 2-4, lower steps go first.
# Before starting a job the worker waits for its GPU to be free (< 1000 MiB).
# A job is claimed atomically by mkdir in $CLAIMS; a failed job keeps its claim (not retried)
# and is logged with FAILED. Stop: touch $CLAIMS/HALT (workers exit between jobs).
#
# Adapters are read in place from the training host's kept-checkpoints through the read-only
# sshfs mount of .16's /works/yanan (scripts/mount_dot16_works.sh), never copied here (2026-09-30).
#
# usage: eval_queue_worker.sh GPU PORT   (env RUN_TAG, SRC, POLL_S)
set -uo pipefail
GPU=${1:?usage: $0 GPU PORT}
PORT=${2:?usage: $0 GPU PORT}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
RUN_TAG=${RUN_TAG:-finqa-multitable-formal}
MOUNT=/home/yanan/mnt/dot16-works
SRC=${SRC:-$MOUNT/reef-sao-finqa-multitable/kept-checkpoints/adapters}
BASEDIR=/mnt/disk1t/sao-finqa-multitable-eval
CLAIMS=$BASEDIR/claims/$RUN_TAG
OUT_ROOT=$REPRO/results/finqa_multitable-eval/$RUN_TAG
BASE_OUT_ROOT=$REPRO/results/finqa_multitable-eval
POLL_S=${POLL_S:-300}
mkdir -p "$CLAIMS" "$OUT_ROOT"
case "$SRC" in
  "$MOUNT"/*) mountpoint -q "$MOUNT" && timeout 20 ls "$MOUNT" >/dev/null 2>&1 || {
    echo "[worker gpu$GPU] expected .16's /works/yanan mounted read-only at $MOUNT to read adapters in place;" \
         "it is not (or is stale). Run scripts/mount_dot16_works.sh, then restart this worker." >&2; exit 1; } ;;
esac
log() { echo "[worker gpu$GPU $(date '+%F %T %Z')] $*"; }

next_job() {  # prints "tag step mode sampling" of the best unclaimed job, or nothing
  local rids
  rids=$(ls "$SRC" 2>/dev/null | grep -E '^hf_rollout_[0-9]{5}$' | sed 's/hf_rollout_//' | sort -n) || rids=""
  local steps=() s
  for r in $rids; do steps+=($((10#$r + 1))); done
  local order=("0 base t0k4" "0 base t07k4" "0 base35 t0k4" "0 base35 t07k4")
  for s in $(printf '%s\n' "${steps[@]}" | sort -rn); do order+=("$s lora t0k4"); done   # newest first
  for s in "${steps[@]}"; do order+=("$s lora t07k4"); done
  for s in "${steps[@]}"; do order+=("$s merged greedy"); done
  for s in "${steps[@]}"; do order+=("$s merged t07k4"); done
  local job step mode sampling tag
  for job in "${order[@]}"; do
    read -r step mode sampling <<< "$job"
    if [ "$mode" = base ]; then tag=base_$sampling; [ -e "$BASE_OUT_ROOT/$tag" ] && continue
    elif [ "$mode" = base35 ]; then tag=base_qwen35_$sampling; [ -e "$BASE_OUT_ROOT/$tag" ] && continue
    else tag=$(printf 'step_%03d_%s_%s' "$step" "$mode" "$sampling"); [ -e "$OUT_ROOT/$tag" ] && continue; fi
    mkdir "$CLAIMS/$tag" 2>/dev/null || continue   # atomic claim
    echo "$tag $step $mode $sampling"; return 0
  done
}

log "start: run=$RUN_TAG out=$OUT_ROOT port=$PORT"
while :; do
  [ -e "$CLAIMS/HALT" ] && { log "HALT present, exiting"; break; }
  job=$(next_job)
  if [ -z "$job" ]; then sleep "$POLL_S"; continue; fi
  read -r tag step mode sampling <<< "$job"
  until [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done
  if [ "$mode" = base ] || [ "$mode" = base35 ]; then
    log "begin $tag"
    extra=()
    [ "$mode" = base35 ] && extra=(BASE=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix TOOL_PARSER=qwen3_coder
                                   CHAT_TEMPLATE="$HERE/templates/qwen35_relaxed_system.jinja"
                                   'CHAT_TEMPLATE_KWARGS={"enable_thinking": false}')
    if env "${extra[@]}" GPU=$GPU PORT=$PORT MODE=base SAMPLING=$sampling OUT_ROOT=$BASE_OUT_ROOT WORK=$BASEDIR/work \
         bash "$HERE/eval_multitable_checkpoint.sh" "$tag" - > "$BASEDIR/work-$tag.log" 2>&1; then
      log "done $tag: $(grep -h 'COMPLETE' "$BASEDIR/work-$tag.log" | sed 's/.*COMPLETE //' | tr '\n' ' ')"
    else
      log "FAILED $tag (rc=$?): see $BASEDIR/work-$tag.log"
    fi
    continue
  fi
  rid=$(printf '%05d' $((step - 1)))
  dest=$SRC/hf_rollout_$rid  # read in place (no copy)
  if [ ! -f "$dest/hf/adapter_model.safetensors" ] || [ ! -f "$dest/hf/adapter_config.json" ]; then
    log "FAILED $tag: expected $dest/hf/{adapter_model.safetensors,adapter_config.json}; not readable (mount stale?)"; continue
  fi
  log "begin $tag"
  if GPU=$GPU PORT=$PORT MODE=$mode SAMPLING=$sampling OUT_ROOT=$OUT_ROOT WORK=$BASEDIR/work \
       bash "$HERE/eval_multitable_checkpoint.sh" "$tag" "$dest/hf" > "$BASEDIR/work-$tag.log" 2>&1; then
    log "done $tag: $(grep -h 'COMPLETE' "$BASEDIR/work-$tag.log" | sed 's/.*COMPLETE //' | tr '\n' ' ')"
  else
    log "FAILED $tag (rc=$?): see $BASEDIR/work-$tag.log"
  fi
done
