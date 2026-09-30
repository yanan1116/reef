#!/usr/bin/env bash
# SAO on DeepCoder, formal run: Qwen3-4B-Instruct-2507, batch 128, one rollout
# per task, 190 optimizer steps (= one pass over the 24,287-task train split),
# no evaluation during training. Config: $CFG (default the lr5x config).
#
# Image: reef:sao-<last commit touching Reef source in this fork checkout>,
# built by scripts/build_image_head.sh. Fresh state directory on the training
# host's local disk. Checkpoint copies: see sidecar.py (SIDECAR_* env).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
TAG=${TAG:-b128-$(date +%Y%m%dT%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/home/yanan/reef-sao-deepcoder}
STATE_DIR=${STATE_DIR:-$RUN_ROOT/state-b128}
KEEP_DIR=${KEEP_DIR:-$RUN_ROOT/kept-checkpoints-b128}
# Default: the image built from this checkout's Reef source (see scripts/build_image_head.sh).
SRC_COMMIT=$(git -C "$REPRO/.." -c safe.directory='*' log -1 --format=%h -- . ':(exclude)exp_scripts')
export IMAGE=${IMAGE:-reef:sao-$SRC_COMMIT}
CFG=${CFG:-/repro/configs/serve-deepcoder-2507-b128-lr5x.yaml}
OUT="$REPRO/results/deepcoder/$TAG"
mkdir -p "$OUT" "$KEEP_DIR"

if [ -e "$STATE_DIR/checkpoints" ]; then
  echo "[formal] $STATE_DIR already holds checkpoints; a fresh run needs a fresh STATE_DIR" >&2
  exit 1
fi

echo "[formal] $(date '+%F %T %Z') tag=$TAG image=$IMAGE state=$STATE_DIR keep=$KEEP_DIR cfg=$CFG"
STATE_DIR="$STATE_DIR" KEEP_DIR="$KEEP_DIR" CFG="$CFG" bash "$REPRO/scripts/start_stack.sh"

for i in $(seq 1 160); do
  if curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null 2>&1; then echo "[formal] reef healthy $(date +%T)"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx reef-sao-stack; then
    echo "[formal] stack exited during startup; see $STATE_DIR/stack/*.log" >&2
    grep -hE "Error:|Exception:" "$STATE_DIR"/stack/*.log 2>/dev/null | tail -3 >&2
    exit 1
  fi
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null || { echo "[formal] reef never became healthy" >&2; exit 1; }

export RLLM_HOME=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/runtime
export PYTHONPATH="$HERE:$REPRO/vendor:/home/yanan/agents/rllm/exp_scripts/deepcoder-run:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SAO_PROBLEMS="$REPRO/data/deepcoder_train.jsonl"
export SAO_SCENARIO=${SAO_SCENARIO:-sao-deepcoder}
export SAO_BATCH=128                             # = recipe.config.batch-size in $CFG
# Grading concurrency cap: uncapped, 128 simultaneous hidden-test parses took the
# driver to 221 GB and Ray OOM-killed the trainer (2026-09-23 17:03). 16 peaks at 34 GB.
export DEEPCODER_GRADE_CONCURRENCY=${DEEPCODER_GRADE_CONCURRENCY:-16}
export SAO_IN_FLIGHT=${SAO_IN_FLIGHT:-128}       # paper shape: at least one batch in flight
export SAO_BUDGET=${SAO_BUDGET:-24320}           # 190 steps x 128
export SAO_MAX_TOKENS=${SAO_MAX_TOKENS:-12288}   # = rollout-max-response-len in $CFG
export SAO_SEED=${SAO_SEED:-0}
export SAO_RECORDS_PATH="$OUT/records.jsonl"
export SAO_PROGRESS_FILE="$OUT/progress.txt"
export SIDECAR_HF_DIR="$STATE_DIR/checkpoints/hf"
export SIDECAR_KEEP_DIR="$KEEP_DIR"
export SIDECAR_LOG="$OUT/sidecar.log"

/home/yanan/agents/rllm/.venv/bin/python -u "$HERE/sidecar.py" > "$OUT/sidecar.stdout" 2>&1 &
SIDECAR=$!
echo "$SIDECAR" > "$OUT/sidecar.pid"
trap 'kill "$SIDECAR" 2>/dev/null || true' EXIT

echo "[formal] driver: batch=$SAO_BATCH in_flight=$SAO_IN_FLIGHT budget=$SAO_BUDGET max_tokens=$SAO_MAX_TOKENS sidecar=$SIDECAR"
/home/yanan/agents/rllm/.venv/bin/python -u "$HERE/stream.py" 2>&1 | tee "$OUT/driver.log"
echo "[formal] driver exited rc=${PIPESTATUS[0]} $(date '+%F %T %Z')"
