#!/usr/bin/env bash
# Stop the FinQA SAO run on this host: the launcher, the driver, the checkpoint copier, then the stack container. PIDs come
# from cmdlines anchored at the executable, never from a pattern that could also
# match this script's caller (as scripts/stop_formal.sh).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
PY=${FINQA_VENV:-$REPRO/.venv-finqa}/bin/python
SELF=$$
pids_for() {  # $1 = exact argv prefix
  for d in /proc/[0-9]*; do
    p=${d#/proc/}; [ "$p" = "$SELF" ] && continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in "$1"*) echo "$p";; esac
  done
}
L=$(pids_for "bash $HERE/run_finqa.sh")
D=$(pids_for "$PY -u $HERE/stream_finqa.py")
S=$(pids_for "$PY -u $REPRO/deepcoder/sidecar.py")
echo "launcher=[$L] driver=[$D] copier=[$S]"
for p in $L $D $S; do kill -TERM "$p" 2>/dev/null && echo "TERM $p"; done
sleep 6
for p in $L $D $S; do kill -0 "$p" 2>/dev/null && { kill -9 "$p"; echo "KILL $p"; }; done
docker stop -t 120 reef-sao-stack >/dev/null 2>&1; docker rm reef-sao-stack >/dev/null 2>&1 && echo "stack removed"
docker ps -a --format '{{.Names}} {{.Status}}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
