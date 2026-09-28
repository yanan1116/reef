#!/usr/bin/env bash
# One evaluation worker per .29 GPU for the FinQA SAO formal run: repeatedly take the
# highest-priority unevaluated job for the adapters kept on the training host, copy the
# adapter here, and run eval_finqa_checkpoint.sh on it. Never idles while work exists;
# polls for new adapters every POLL_S when there is none.
#
# Jobs, in priority order (step = rollout_id + 1):
#   0. base_t0k4     (MODE=base   SAMPLING=t0k4)    -- once, into results/finqa-eval/base_t0k4
#   1. lora_t0k4     (MODE=lora   SAMPLING=t0k4)    -- greedy x4, avg@4: the headline; newest step first
#   2. lora_t07k4    (MODE=lora   SAMPLING=t07k4)
#   3. merged_greedy (MODE=merged SAMPLING=greedy)  -- PRPO's bf16-merged protocol (one greedy attempt)
#   4. merged_t07k4  (MODE=merged SAMPLING=t07k4)
# Within classes 2-4, lower steps go first. (Before 2026-09-28 11:20 class 1 was a single
# greedy attempt, lora_greedy; those results stay, lora_t0k4 replaces it from then on.)
# Before starting a job the worker waits for its GPU to be free (< 1000 MiB).
# A job is claimed atomically by mkdir in $CLAIMS; a failed job keeps its claim (not retried)
# and is logged with FAILED. Stop: touch $CLAIMS/HALT (workers exit between jobs).
#
# usage: eval_queue_worker.sh GPU PORT   (env RUN_TAG, TRAIN_HOST, POLL_S)
set -uo pipefail
GPU=${1:?usage: $0 GPU PORT}
PORT=${2:?usage: $0 GPU PORT}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
RUN_TAG=${RUN_TAG:-finqa-b64-20260927T004448}
TRAIN_HOST=${TRAIN_HOST:-10.225.68.16}
SRC=/home/yanan/reef-sao-finqa/kept-checkpoints/adapters
BASEDIR=/mnt/disk1t/sao-finqa-eval
ADAPTERS=$BASEDIR/adapters/$RUN_TAG
CLAIMS=$BASEDIR/claims/$RUN_TAG
OUT_ROOT=$REPRO/results/finqa-eval/$RUN_TAG
BASE_OUT_ROOT=$REPRO/results/finqa-eval
POLL_S=${POLL_S:-300}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o Compression=no -c aes128-gcm@openssh.com "$TRAIN_HOST")
mkdir -p "$ADAPTERS" "$CLAIMS" "$OUT_ROOT"
log() { echo "[worker gpu$GPU $(date '+%F %T %Z')] $*"; }

next_job() {  # prints "tag step mode sampling" of the best unclaimed job, or nothing
  local rids
  rids=$("${SSH[@]}" "ls $SRC 2>/dev/null | grep -E '^hf_rollout_[0-9]{5}$'" 2>/dev/null | sed 's/hf_rollout_//' | sort -n) || return 0
  [ -n "$rids" ] || return 0
  local steps=() s
  for r in $rids; do steps+=($((10#$r + 1))); done
  local order=("0 base t0k4")
  for s in $(printf '%s\n' "${steps[@]}" | sort -rn); do order+=("$s lora t0k4"); done   # newest first
  for s in "${steps[@]}"; do order+=("$s lora t07k4"); done
  for s in "${steps[@]}"; do order+=("$s merged greedy"); done
  for s in "${steps[@]}"; do order+=("$s merged t07k4"); done
  local job step mode sampling tag
  for job in "${order[@]}"; do
    read -r step mode sampling <<< "$job"
    if [ "$mode" = base ]; then tag=base_$sampling; [ -e "$BASE_OUT_ROOT/$tag" ] && continue
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
  if [ "$mode" = base ]; then
    log "begin $tag"
    if GPU=$GPU PORT=$PORT MODE=base SAMPLING=$sampling OUT_ROOT=$BASE_OUT_ROOT WORK=$BASEDIR/work \
         bash "$HERE/eval_finqa_checkpoint.sh" "$tag" - > "$BASEDIR/work-$tag.log" 2>&1; then
      log "done $tag: $(grep -h 'COMPLETE' "$BASEDIR/work-$tag.log" | sed 's/.*COMPLETE //' | tr '\n' ' ')"
    else
      log "FAILED $tag (rc=$?): see $BASEDIR/work-$tag.log"
    fi
    continue
  fi
  rid=$(printf '%05d' $((step - 1)))
  dest=$ADAPTERS/hf_rollout_$rid
  if [ ! -f "$dest/hf/adapter_model.safetensors" ]; then
    mkdir -p "$dest.incoming"
    if ! "${SSH[@]}" "tar -C $SRC/hf_rollout_$rid -cf - hf" | tar -C "$dest.incoming" -xf -; then
      log "FAILED $tag: adapter copy from $TRAIN_HOST"; rm -rf "$dest.incoming"; continue
    fi
    rm -rf "$dest"; mv "$dest.incoming" "$dest"
  fi
  log "begin $tag"
  if GPU=$GPU PORT=$PORT MODE=$mode SAMPLING=$sampling OUT_ROOT=$OUT_ROOT WORK=$BASEDIR/work \
       bash "$HERE/eval_finqa_checkpoint.sh" "$tag" "$dest/hf" > "$BASEDIR/work-$tag.log" 2>&1; then
    log "done $tag: $(grep -h 'COMPLETE' "$BASEDIR/work-$tag.log" | sed 's/.*COMPLETE //' | tr '\n' ' ')"
  else
    log "FAILED $tag (rc=$?): see $BASEDIR/work-$tag.log"
  fi
done
