#!/usr/bin/env bash
# Stop the formal SAO run. PIDs are resolved from cmdlines anchored at the
# executable, never from a pattern that could also match this script's caller.
set -u
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=$(ls -dt "$REPRO"/results/deepcoder/b128-* | head -1)
SELF=$$
pids_for() {  # $1 = exact argv[0..1] prefix
  for d in /proc/[0-9]*; do
    p=${d#/proc/}; [ "$p" = "$SELF" ] && continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in "$1"*) echo "$p";; esac
  done
}
L=$(pids_for "bash $REPRO/deepcoder/run_formal.sh")
D=$(pids_for "/home/yanan/agents/rllm/.venv/bin/python -u $REPRO/deepcoder/stream.py")
S=$(cat "$OUT/sidecar.pid" 2>/dev/null)
echo "launcher=[$L] driver=[$D] sidecar=[$S]"
for p in $L $D $S; do kill -TERM "$p" 2>/dev/null && echo "TERM $p"; done
sleep 6
for p in $L $D $S; do kill -0 "$p" 2>/dev/null && { kill -9 "$p"; echo "KILL $p"; }; done
docker stop -t 120 reef-sao-stack >/dev/null 2>&1; docker rm reef-sao-stack >/dev/null 2>&1 && echo "stack removed"
sleep 2
FS=$(pids_for "/home/yanan/agents/rllm/.venv/bin/python -c from multiprocessing.forkserver")
[ -n "$FS" ] && { kill -9 $FS 2>/dev/null; echo "killed $(echo $FS | wc -w) grader forkserver processes"; }
echo "remaining driver-side: [$(pids_for "bash $REPRO/deepcoder/run_formal.sh") $(pids_for "/home/yanan/agents/rllm/.venv/bin/python -u $REPRO/deepcoder")]"
docker ps -a --format '{{.Names}} {{.Status}}'
free -g | sed -n 2p
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
