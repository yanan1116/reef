#!/usr/bin/env bash
# SAO on AppWorld (react_code protocol), formal run: Qwen3-4B-Instruct-2507, 30 tasks per
# step with one rollout each, one sample per episode, 30 epochs of train90 = 90 optimizer
# steps (3 per epoch). The FinQA setup (finqa/run_finqa.sh) with the benchmark swapped: same
# image, recipe (finqa/sao_multiturn.py, loaded from /repro/finqa), sidecar and driver shape.
#
# Every step's adapter is kept for evaluation (SIDECAR_ADAPTER_EVERY=1), plus full
# actor+critic bundles every 30 steps (10 epochs) and at the final step.
#
# Host specifics: RUN_ROOT is local disk (default /mnt/disk1t/reef-sao-appworld on .29);
# DOCKER_RUNTIME=nvidia as on .16. APPWORLD_ROOT must be a local, non-NFS checkout on the
# training host (grpo_vanilla saw NFS staleness SIGBUS-kill AppWorld servers on .24).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
TAG=${TAG:-appworld-b30-$(date +%Y%m%dT%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/works/yanan/reef-sao-appworld}   # .16 NVMe data disk (run data only, no code)
STATE_DIR=${STATE_DIR:-$RUN_ROOT/state}
KEEP_DIR=${KEEP_DIR:-$RUN_ROOT/kept-checkpoints}
SRC_COMMIT=$(git -C "$REPRO/.." -c safe.directory='*' log -1 --format=%h -- . ':(exclude)exp_scripts')
export IMAGE=${IMAGE:-reef:sao-$SRC_COMMIT}
export DOCKER_RUNTIME=${DOCKER_RUNTIME-nvidia}
# fp32 policy logits (REEF_BF16_LOGITS=0): the bf16-logits patch (docker/patch/mcore_bf16_logits.py)
# trips Slime's fp32 assertion on the first actor step (multi-table smoke 2026-09-29). Long samples
# are handled by the driver instead: assembled samples > SAO_MAX_SAMPLE_TOKENS (16384) are not
# trained (length-filtered SAO; report the filter rate).
export REEF_BF16_LOGITS=${REEF_BF16_LOGITS:-0}   # 1 trips Slime's fp32-logits assertion on the first actor step
CFG=${CFG:-/repro/configs/serve-appworld-2507.yaml}
VENV=${FINQA_VENV:-$REPRO/.venv-finqa}
PY="$VENV/bin/python"
OUT="$REPRO/results/appworld/$TAG"

[ -x "$PY" ] || { echo "[appworld] expected $PY (finqa/setup_venv.sh)" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { echo "[appworld] expected image $IMAGE on this host" >&2; exit 1; }
if [ "$REEF_BF16_LOGITS" = 1 ] && ! docker run --rm --entrypoint grep "$IMAGE" -q _reef_keep_bf16_logits /root/Megatron-LM/megatron/core/transformer/module.py; then
  echo "[appworld] REEF_BF16_LOGITS=1 but $IMAGE lacks docker/patch/mcore_bf16_logits.py; rebuild it (scripts/build_image_head.sh)" >&2; exit 1
fi
[ -f /home/yanan/reef-sao/models/Qwen3-4B-Instruct-2507/config.json ] || { echo "[appworld] expected the model under /home/yanan/reef-sao/models (start_stack.sh mounts it)" >&2; exit 1; }
(cd "$HERE/appworld_react" && sha256sum --quiet -c SHA256SUMS) || { echo "[appworld] appworld_react differs from its SHA256SUMS (the copy shared with rllm)" >&2; exit 1; }
if [ -e "$STATE_DIR/checkpoints" ]; then
  echo "[appworld] $STATE_DIR already holds checkpoints; a fresh run needs a fresh STATE_DIR" >&2
  exit 1
fi
mkdir -p "$OUT" "$KEEP_DIR"

echo "[appworld] $(date '+%F %T %Z') tag=$TAG image=$IMAGE state=$STATE_DIR keep=$KEEP_DIR cfg=$CFG"
STATE_DIR="$STATE_DIR" KEEP_DIR="$KEEP_DIR" CFG="$CFG" EXTRA_PYTHONPATH=/repro/finqa \
  bash "$REPRO/scripts/start_stack.sh"

for i in $(seq 1 160); do
  if curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null 2>&1; then echo "[appworld] reef healthy $(date +%T)"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx reef-sao-stack; then
    echo "[appworld] stack exited during startup; see $STATE_DIR/stack/*.log" >&2
    grep -hE "Error:|Exception:" "$STATE_DIR"/stack/*.log 2>/dev/null | tail -3 >&2
    exit 1
  fi
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null || { echo "[appworld] reef never became healthy" >&2; exit 1; }

export PYTHONPATH="$HERE:$REPRO/vendor"
export SAO_SCENARIO=${SAO_SCENARIO:-sao-appworld}
export SAO_RUN_TAG="$TAG"
export SAO_BATCH=30                               # = recipe.config.batch-size in $CFG
export SAO_IN_FLIGHT=${SAO_IN_FLIGHT:-30}
export SAO_BUDGET=${SAO_BUDGET:-2700}             # 90 steps x 30 = 30 epochs of train90 (3 steps each)
export SAO_TEMPERATURE=1.0                        # = rollout-temperature in $CFG (DIS needs them equal)
export SAO_TOP_P=1.0
export SAO_CONTEXT_TOKENS=32768                   # = context-length in $CFG
export SAO_SEED=${SAO_SEED:-0}
export SAO_RECORDS_PATH="$OUT/records.jsonl"
export SAO_EPISODES_PATH="$OUT/episodes.jsonl"
export SAO_PROGRESS_FILE="$OUT/progress.txt"
export SIDECAR_HF_DIR="$STATE_DIR/checkpoints/hf"
export SIDECAR_KEEP_DIR="$KEEP_DIR"
export SIDECAR_LOG="$OUT/sidecar.log"
export SIDECAR_ADAPTER_EVERY=${SIDECAR_ADAPTER_EVERY:-1}    # every step: each is evaluated later
export SIDECAR_FULL_EVERY=${SIDECAR_FULL_EVERY:-30}         # full actor+critic bundle every 10 epochs
export SIDECAR_FINAL_STEP=${SIDECAR_FINAL_STEP:-90}

"$PY" -u "$REPRO/deepcoder/sidecar.py" > "$OUT/sidecar.stdout" 2>&1 &
SIDECAR=$!
echo "$SIDECAR" > "$OUT/sidecar.pid"
trap 'kill "$SIDECAR" 2>/dev/null || true' EXIT

echo "[appworld] driver: batch=$SAO_BATCH in_flight=$SAO_IN_FLIGHT budget=$SAO_BUDGET T=$SAO_TEMPERATURE sidecar=$SIDECAR"
"$PY" -u "$HERE/stream_appworld.py" 2>&1 | tee "$OUT/driver.log"
echo "[appworld] driver exited rc=${PIPESTATUS[0]} $(date '+%F %T %Z')"
