#!/usr/bin/env bash
# Poll the .16 SAO run's kept adapters and enqueue each new one for evaluation,
# ahead of any filler jobs (new checkpoints are the priority). Runs on .29.
#
# The sidecar publishes adapters atomically (copy to *.incoming, then mv), so a
# hf_rollout_NNNNN directory that exists on .16 is complete. Each adapter is
# copied to local disk and its sha256 checked against the source before it is
# enqueued as lr5x_step_<NNN+1>, twice (a second copy tagged _r2), so the two GPU
# workers evaluate it in parallel: same latency, two measurements per checkpoint.
#
# usage: feed_new_checkpoints.sh QUEUE_FILE      (stop: touch QUEUE_FILE.halt; FEED_POLL_S, default 60)
set -uo pipefail
Q=${1:?usage: $0 QUEUE_FILE}
SRC_HOST=10.225.68.16
SRC=/home/yanan/reef-sao-deepcoder/kept-checkpoints-b128/adapters
DST=/mnt/disk1t/sao-lr5x-eval/adapters
NFS=${NFS:-$(cd "$(dirname "$0")/.." && pwd)/results/deepcoder/eval-c32-lora-lr5x}
SSH=(ssh -o ConnectTimeout=15 -o Compression=no "$SRC_HOST")
LOCK="$Q.lock"; CLAIMED="$Q.claimed"; STOP="$Q.halt"   # same stop file as eval_priority_worker.sh
mkdir -p "$DST"

enqueue_front() {   # $1 = line
  exec 9<>"$LOCK"; flock -x 9
  if ! grep -qxF -- "$1" "$Q" 2>/dev/null; then { echo "$1"; cat "$Q" 2>/dev/null; } > "$Q.tmp" && mv "$Q.tmp" "$Q"; fi
  flock -u 9; exec 9>&-
}

while :; do
  [ -e "$STOP" ] && { echo "[feeder] stop file present, exiting $(date '+%F %T %Z')"; break; }
  for name in $("${SSH[@]}" "ls $SRC 2>/dev/null | grep -E '^hf_rollout_[0-9]{5}$'" 2>/dev/null); do
    rid=$((10#${name#hf_rollout_})); tag=$(printf 'lr5x_step_%03d' $((rid + 1)))
    line="$tag=$DST/$name/hf"
    # already evaluated, queued, or claimed: nothing to do
    [ -e "$NFS/$tag" ] && continue
    grep -qxF -- "$line" "$Q" 2>/dev/null && continue
    grep -qF -- " $line" "$CLAIMED" 2>/dev/null && continue
    if [ ! -f "$DST/$name/hf/adapter_model.safetensors" ]; then
      rm -rf "$DST/$name.incoming"
      scp -q -r -o Compression=no "$SRC_HOST:$SRC/$name" "$DST/$name.incoming" || { echo "[feeder] copy failed: $name"; continue; }
      want=$("${SSH[@]}" "sha256sum $SRC/$name/hf/adapter_model.safetensors" | cut -d' ' -f1)
      got=$(sha256sum "$DST/$name.incoming/hf/adapter_model.safetensors" | cut -d' ' -f1)
      [ -n "$want" ] && [ "$want" = "$got" ] || { echo "[feeder] sha256 mismatch for $name: src=$want dst=$got"; rm -rf "$DST/$name.incoming"; continue; }
      mv "$DST/$name.incoming" "$DST/$name"
    fi
    enqueue_front "${tag}_r2=$DST/$name/hf"
    enqueue_front "$line"
    echo "[feeder] $(date '+%F %T %Z') enqueued $tag and ${tag}_r2 (front)"
  done
  sleep "${FEED_POLL_S:-60}"
done
