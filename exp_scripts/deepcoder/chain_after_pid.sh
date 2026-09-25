#!/usr/bin/env bash
# Wait for an explicit PID to exit and GPU 0 to be released, then run a command.
# usage: chain_after_pid.sh PID -- CMD...
set -uo pipefail
WAIT_PID=$1; shift; [ "$1" = -- ] && shift
echo "[chain] waiting for pid $WAIT_PID $(date '+%F %T %Z')"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
for i in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 1000 ] && break; sleep 10
done
echo "[chain] pid $WAIT_PID gone, GPU0 used=${used} MiB; starting $(date '+%F %T %Z')"
exec "$@"
