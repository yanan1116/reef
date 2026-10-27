#!/usr/bin/env bash
# Build every prepared task's environment image once on .16 before the full 20-task SAO run (user, 2026-10-19:
# "Prebuild + smoke, then auto formal"). Harbor builds each trial's image with docker compose; a prior plain
# `docker build` of the same context fills the BuildKit layer cache, so the first training batch does not wait on
# arXiv downloads, apt, or fix-security-bug's maven build. Tags slb-prebuild/<instance> keep the layers from being
# pruned. Failures are listed in $TASKS/prebuild.failed and stop nothing else.
#
#   prebuild_images_16.sh [TASKS=/works/yanan/slb-sao-tasks-all] [JOBS=4]
set -u
TASKS=${1:-/works/yanan/slb-sao-tasks-all}
JOBS=${2:-4}
LOG="$TASKS/prebuild.log"
: > "$TASKS/prebuild.failed"
echo "$(TZ=America/New_York date '+%F %T %Z') prebuild start ($JOBS parallel)" >> "$LOG"
ls -d "$TASKS"/*/environment | xargs -P "$JOBS" -I{} bash -c '
  dir=$(dirname "$1"); name=$(basename "$dir")
  if docker build -q -t "slb-prebuild/$name" "$1" > "$dir/prebuild.out" 2>&1; then
    echo "$(TZ=America/New_York date "+%F %T %Z") ok $name" >> "$2"
  else
    echo "$(TZ=America/New_York date "+%F %T %Z") FAILED $name" >> "$2"; echo "$name" >> "$3"
  fi' _ {} "$LOG" "$TASKS/prebuild.failed"
echo "$(TZ=America/New_York date '+%F %T %Z') prebuild done: $(grep -c ' ok ' "$LOG") ok, $(wc -l < "$TASKS/prebuild.failed") failed" >> "$LOG"
