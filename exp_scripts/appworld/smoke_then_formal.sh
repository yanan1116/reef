#!/usr/bin/env bash
# On the training host: a 4-step AppWorld SAO smoke run (configs/serve-appworld-2507-smoke.yaml,
# 120 episodes), checked by smoke_check.py; only if every check passes, the smoke's Reef
# checkpoint tree is deleted and the formal run (run_appworld.sh, 90 steps = 30 epochs) starts.
# The FinQA smoke_then_formal.sh with the benchmark swapped.
#
# Refuses to start while any reef-sao-stack container exists or a GPU holds memory:
# start_stack.sh would otherwise remove the running stack by name.
#
# usage: smoke_then_formal.sh   (env DOCKER_RUNTIME, default nvidia; SMOKE_CFG / FORMAL_CFG pick host variants,
#        e.g. the -dot29 configs on .29; SKIP_FORMAL=1 stops after the check)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
export DOCKER_RUNTIME=${DOCKER_RUNTIME-nvidia}
STAMP=$(date +%Y%m%dT%H%M%S)
SMOKE_TAG=appworld-smoke-$STAMP
RUN_BASE=${RUN_BASE:-/works/yanan/reef-sao-appworld}
SMOKE_ROOT=$RUN_BASE/smoke-$STAMP
STEPS=4
BUDGET=$((STEPS * 30))
PY=${FINQA_VENV:-$REPRO/.venv-finqa}/bin/python
SRC_COMMIT=$(git -C "$REPRO/.." -c safe.directory='*' log -1 --format=%h -- . ':(exclude)exp_scripts')
IMAGE=${IMAGE:-reef:sao-$SRC_COMMIT}
log() { echo "[smoke_then_formal $(date '+%F %T %Z')] $*"; }

if docker ps -a --format '{{.Names}}' | grep -qx reef-sao-stack; then
  log "expected no reef-sao-stack container; one exists (another run holds the stack). Stop it first."; exit 1
fi
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1000 {print $1}')
[ -z "$busy" ] || { log "expected both GPUs free (< 1000 MiB); GPU(s) $busy hold memory"; exit 1; }
[ -x "$PY" ] || { log "expected $PY; run finqa/setup_venv.sh"; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { log "expected image $IMAGE; run scripts/build_image_head.sh"; exit 1; }

log "smoke: tag=$SMOKE_TAG root=$SMOKE_ROOT image=$IMAGE steps=$STEPS budget=$BUDGET"
rc=0
TAG=$SMOKE_TAG RUN_ROOT=$SMOKE_ROOT IMAGE=$IMAGE CFG=${SMOKE_CFG:-/repro/configs/serve-appworld-2507-smoke.yaml} SAO_BUDGET=$BUDGET \
  SIDECAR_ADAPTER_EVERY=1 SIDECAR_FULL_EVERY=$STEPS SIDECAR_FINAL_STEP=$STEPS SAO_TRAIN_DRAIN_TIMEOUT_S=3600 \
  timeout 5h bash "$HERE/run_appworld.sh" || rc=$?
log "smoke driver exited rc=$rc"
# After a timeout the driver and copier may outlive their launcher: end them by exact cmdline.
for d in /proc/[0-9]*; do
  c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$c" in "$PY -u $HERE/stream_appworld.py"*|"$PY -u $REPRO/deepcoder/sidecar.py"*) kill -9 "${d#/proc/}" 2>/dev/null && log "killed leftover ${d#/proc/}: $c";; esac
done
cp "$SMOKE_ROOT/state/reef.log" "$REPRO/results/appworld/$SMOKE_TAG/reef.log" 2>/dev/null || true
docker stop -t 120 reef-sao-stack >/dev/null 2>&1 || true
docker rm reef-sao-stack >/dev/null 2>&1 || true

check_rc=0
"$PY" "$HERE/smoke_check.py" "$REPRO/results/appworld/$SMOKE_TAG" "$SMOKE_ROOT/kept-checkpoints" "$STEPS" "$BUDGET" \
  "${APPWORLD_ROOT:-/home/yanan/agents/gitlab/tail/appworld}" "$SMOKE_TAG" "${APPWORLD_PORT_BASE:-7800}" "${SAO_IN_FLIGHT:-30}" \
  | tee "$REPRO/results/appworld/$SMOKE_TAG/smoke_check.txt" || check_rc=$?   # pipefail: python's status
if [ "$rc" != 0 ] || [ "$check_rc" != 0 ]; then
  log "smoke did not pass (driver rc=$rc, check rc=$check_rc); formal run NOT started. See $REPRO/results/appworld/$SMOKE_TAG"
  exit 1
fi
# Checkpoint files are root-owned: delete the smoke's Reef tree from inside a container.
docker run --rm --entrypoint rm -v "$SMOKE_ROOT/state":/state "$IMAGE" -rf /state/checkpoints
log "smoke passed; deleted $SMOKE_ROOT/state/checkpoints"
[ "${SKIP_FORMAL:-0}" = 1 ] && { log "SKIP_FORMAL=1: stopping here"; exit 0; }

for i in $(seq 1 30); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1000 {print $1}')
  [ -z "$busy" ] && break
  sleep 10
done
[ -z "$busy" ] || { log "expected GPUs free after the smoke; GPU(s) $busy still hold memory"; exit 1; }
log "formal: starting run_appworld.sh"
exec env TAG=appworld-b30-$(date +%Y%m%dT%H%M%S) IMAGE="$IMAGE" RUN_ROOT="$RUN_BASE" CFG="${FORMAL_CFG:-/repro/configs/serve-appworld-2507.yaml}" bash "$HERE/run_appworld.sh"
