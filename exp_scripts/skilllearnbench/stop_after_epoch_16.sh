#!/usr/bin/env bash
# Stop the SkillLearnBench SAO run on .16 once epoch EPOCH has ended (user, 2026-10-19: "epoch 8 结束之后，训练结束，
# 不继续了"). Waits for run/epochs.jsonl's "end" record of EPOCH and for sidecar.py to keep that step's adapter
# (kept/adapters/hf_rollout_{R-1}, the checkpoint the .29 follower evaluates), then stops by explicit PID: the
# driver (run.py train) first so no new episode starts, then the sidecar and the launcher, then the stack container
# and the Harbor trial containers and networks the killed driver leaves behind. Run ON .16.
#
#   stop_after_epoch_16.sh RUN_ROOT EPOCH DRIVER_PID SIDECAR_PID LAUNCHER_PIDS...
set -u
ROOT=$1 EPOCH=$2 DRIVER=$3 SIDECAR=$4; shift 4
LOG="$ROOT/stop.log"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }

log "waiting for epoch $EPOCH to end (driver $DRIVER, sidecar $SIDECAR, launchers $*)"
releases=""
until [ -n "$releases" ]; do
  releases=$(python3 - "$ROOT/run/epochs.jsonl" "$EPOCH" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("event") == "end" and r.get("epoch") == int(sys.argv[2]):
        print(r["training_releases"])
EOF
)
  [ -n "$releases" ] || sleep 30
done
adapter="$ROOT/kept/adapters/hf_rollout_$(printf %05d $((releases - 1)))/hf/adapter_model.safetensors"
log "epoch $EPOCH ended at release $releases; waiting for $adapter"
until [ -s "$adapter" ]; do sleep 10; done
sleep 30   # let the sidecar finish copying the adapter directory

kill "$DRIVER" && log "driver $DRIVER stopped"
sleep 5
kill "$SIDECAR" && log "sidecar $SIDECAR stopped"
for pid in "$@"; do kill "$pid" 2>/dev/null && log "launcher $pid stopped"; done
docker stop -t 120 reef-slb-stack >/dev/null && log "stack container reef-slb-stack stopped"
trials=$(docker ps -aq --filter name=__env-main)
[ -n "$trials" ] && docker rm -f $trials >/dev/null && log "removed $(echo "$trials" | wc -w) trial containers"
nets=$(docker network ls -q --filter name=__env)
[ -n "$nets" ] && docker network rm $nets >/dev/null && log "removed $(echo "$nets" | wc -w) trial networks"
log "done; GPUs: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
