#!/usr/bin/env bash
# SAO on FinQA single-table, formal run: Qwen3-4B-Instruct-2507, 64 tasks per
# step with one rollout each, one sample per multi-turn episode, compared against
# the PRPO FinQA runs. 10 epochs = 620 optimizer steps (62 per epoch; set
# 2026-09-26, was 20 epochs); stop early from the evaluations if it is not improving.
#
# Same shape as deepcoder/run_formal.sh, without rllm: the driver and the
# checkpoint copier run in exp_scripts/.venv-finqa (finqa/setup_venv.sh), and
# the Reef service loads sao_multiturn.py from /repro/finqa (EXTRA_PYTHONPATH).
#
# Image: reef:sao-<last commit touching Reef source> (scripts/build_image_head.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
TAG=${TAG:-finqa-b64-$(date +%Y%m%dT%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/home/yanan/reef-sao-finqa}
STATE_DIR=${STATE_DIR:-$RUN_ROOT/state}
KEEP_DIR=${KEEP_DIR:-$RUN_ROOT/kept-checkpoints}
SRC_COMMIT=$(git -C "$REPRO/.." -c safe.directory='*' log -1 --format=%h -- . ':(exclude)exp_scripts')
export IMAGE=${IMAGE:-reef:sao-$SRC_COMMIT}
CFG=${CFG:-/repro/configs/serve-finqa-2507.yaml}
VENV=${FINQA_VENV:-$REPRO/.venv-finqa}
PY="$VENV/bin/python"
OUT="$REPRO/results/finqa/$TAG"

[ -x "$PY" ] || { echo "[finqa] $PY missing: run finqa/setup_venv.sh first" >&2; exit 1; }
[ -r "${FINQA_JUDGE_CREDS:-$HERE/.judge_creds}" ] || { echo "[finqa] judge credentials missing (FINQA_JUDGE_CREDS)" >&2; exit 1; }
if [ -e "$STATE_DIR/checkpoints" ]; then
  echo "[finqa] $STATE_DIR already holds checkpoints; a fresh run needs a fresh STATE_DIR" >&2
  exit 1
fi
if [ ! -s "$HERE/data/finqa_train.jsonl" ]; then
  "$PY" "$HERE/export_finqa.py" "$HERE/data"
fi
mkdir -p "$OUT" "$KEEP_DIR"

echo "[finqa] $(date '+%F %T %Z') tag=$TAG image=$IMAGE state=$STATE_DIR keep=$KEEP_DIR cfg=$CFG"
STATE_DIR="$STATE_DIR" KEEP_DIR="$KEEP_DIR" CFG="$CFG" EXTRA_PYTHONPATH=/repro/finqa \
  bash "$REPRO/scripts/start_stack.sh"

for i in $(seq 1 160); do
  if curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null 2>&1; then echo "[finqa] reef healthy $(date +%T)"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx reef-sao-stack; then
    echo "[finqa] stack exited during startup; see $STATE_DIR/stack/*.log" >&2
    grep -hE "Error:|Exception:" "$STATE_DIR"/stack/*.log 2>/dev/null | tail -3 >&2
    exit 1
  fi
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null || { echo "[finqa] reef never became healthy" >&2; exit 1; }

export PYTHONPATH="$HERE:$REPRO/vendor"
export SAO_PROBLEMS="$HERE/data/finqa_train.jsonl"
export SAO_SCENARIO=${SAO_SCENARIO:-sao-finqa}
export SAO_BATCH=64                               # = recipe.config.batch-size in $CFG
export SAO_IN_FLIGHT=${SAO_IN_FLIGHT:-64}
export SAO_BUDGET=${SAO_BUDGET:-39680}            # 620 steps x 64 = 10 epochs of 4030 tasks (62 steps each)
export SAO_TEMPERATURE=0.7                        # = rollout-temperature in $CFG (DIS needs them equal)
export SAO_TOP_P=1.0
export SAO_MAX_TOKENS=2048                        # per turn, PRPO max_response_length
export SAO_MAX_PROMPT_TOKENS=8192                 # per turn prompt, PRPO max_prompt_length
export SAO_SEED=${SAO_SEED:-0}
export SAO_RECORDS_PATH="$OUT/records.jsonl"
export SAO_PROGRESS_FILE="$OUT/progress.txt"
export FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish.tsv"
# Checkpoint copies out of Reef's managed tree (deepcoder/sidecar.py, task-independent).
export SIDECAR_HF_DIR="$STATE_DIR/checkpoints/hf"
export SIDECAR_KEEP_DIR="$KEEP_DIR"
export SIDECAR_LOG="$OUT/sidecar.log"
export SIDECAR_ADAPTER_EVERY=${SIDECAR_ADAPTER_EVERY:-31}   # every half epoch: what the evaluation reads
export SIDECAR_FULL_EVERY=${SIDECAR_FULL_EVERY:-155}        # full actor+critic bundle every 2.5 epochs (155, 310, 465, 620)
export SIDECAR_FINAL_STEP=${SIDECAR_FINAL_STEP:-620}

"$PY" -u "$REPRO/deepcoder/sidecar.py" > "$OUT/sidecar.stdout" 2>&1 &
SIDECAR=$!
echo "$SIDECAR" > "$OUT/sidecar.pid"
trap 'kill "$SIDECAR" 2>/dev/null || true' EXIT

echo "[finqa] driver: batch=$SAO_BATCH in_flight=$SAO_IN_FLIGHT budget=$SAO_BUDGET T=$SAO_TEMPERATURE sidecar=$SIDECAR"
"$PY" -u "$HERE/stream_finqa.py" 2>&1 | tee "$OUT/driver.log"
echo "[finqa] driver exited rc=${PIPESTATUS[0]} $(date '+%F %T %Z')"
