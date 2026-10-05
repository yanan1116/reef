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
# Usage: bash finqa_singletable/run_finqa.sh --image IMAGE [launcher flags] [driver flags]
# Launcher flags:
#   --image IMAGE       Reef image the stack runs (required), e.g. reef:sao-31f75d74
#   --config PATH       recipe config, container path (default /repro/configs/serve-finqa-2507.yaml)
#   --run_root DIR      run state and kept checkpoints (default /home/yanan/reef-sao-finqa)
#   --tag TAG           results/finqa/TAG (default finqa-b64-<time>)
#   --scenario NAME     Reef scenario, for the driver and the checkpoint copier (default sao-finqa)
#   --gpus LIST         host GPUs for the stack, e.g. 0 (default: all)
#   --name NAME         container name (default reef-sao-stack)
#   --port PORT         Reef's port on the host (default 8900)
#   --private_network   own network namespace, for a second stack on this host (config reef.host 0.0.0.0)
#   --memory SIZE       docker --memory cap on host RAM, e.g. 100g
#   --adapter_every N   keep every Nth step's adapter for evaluation (default 31, half an epoch)
#   --full_every N      keep a full actor+critic bundle every N steps (default 155)
#   --final_step N      the run's last step (default 620)
# Every other flag goes to stream_finqa.py (its --help): e.g. --sync, --pvf_enhanced, --budget 640.
#
# Example, enhanced PVF next to another stack on GPU 0 of .16:
#   bash finqa_singletable/run_finqa.sh --image reef:sao-31f75d74 \
#     --config /repro/configs/serve-finqa-2507-dot29-colocate-pvf.yaml \
#     --run_root /works/yanan/reef-sao-finqa-pvf-locexp --gpus 0 --name reef-pvfx-single-stack \
#     --port 8902 --private_network --memory 100g --sync --pvf_enhanced
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
COMMAND="$0 $*"

IMAGE=
CFG=/repro/configs/serve-finqa-2507.yaml
RUN_ROOT=/home/yanan/reef-sao-finqa
TAG=finqa-b64-$(date +%Y%m%dT%H%M%S)
SCENARIO=sao-finqa
GPUS=
NAME=reef-sao-stack
PORT=8900
NETWORK=host
MEMORY=
ADAPTER_EVERY=31
FULL_EVERY=155
FINAL_STEP=620
DRIVER_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --config) CFG=$2; shift 2 ;;
    --run_root) RUN_ROOT=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --scenario) SCENARIO=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --private_network) NETWORK=private; shift ;;
    --memory) MEMORY=$2; shift 2 ;;
    --adapter_every) ADAPTER_EVERY=$2; shift 2 ;;
    --full_every) FULL_EVERY=$2; shift 2 ;;
    --final_step) FINAL_STEP=$2; shift 2 ;;
    *) DRIVER_ARGS+=("$1"); shift ;;
  esac
done
STATE_DIR=$RUN_ROOT/state
KEEP_DIR=$RUN_ROOT/kept-checkpoints
PY="$REPRO/.venv-finqa/bin/python"
OUT="$REPRO/results/finqa/$TAG"
REEF_SERVICE_URL=http://127.0.0.1:$PORT
HOST_CFG="$REPRO/${CFG#/repro/}"

[ -n "$IMAGE" ] || { echo "[finqa] expected --image (e.g. reef:sao-31f75d74)" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { echo "[finqa] expected image $IMAGE on this host" >&2; exit 1; }
[ -r "$HOST_CFG" ] || { echo "[finqa] expected the config at $HOST_CFG (--config $CFG)" >&2; exit 1; }
[ -x "$PY" ] || { echo "[finqa] $PY missing: run finqa/setup_venv.sh first" >&2; exit 1; }
[ -r "${FINQA_JUDGE_CREDS:-$HERE/.judge_creds}" ] || { echo "[finqa] judge credentials missing (FINQA_JUDGE_CREDS)" >&2; exit 1; }
# The switches and the config must agree, or the run fails late (or trains the wrong thing).
PVF=0; SYNC=0
for arg in "${DRIVER_ARGS[@]}"; do
  case "$arg" in --pvf|--pvf_explanation|--pvf_enhanced) PVF=1 ;; --sync) SYNC=1 ;; esac
done
CFG_PVF=0; grep -qE '^\s*privileged-value:\s*true' "$HOST_CFG" && CFG_PVF=1
if [ "$PVF" != "$CFG_PVF" ]; then
  echo "[finqa] a --pvf* flag needs a config with privileged-value: true, and only such a config; got pvf flag=$PVF, $CFG privileged-value=$CFG_PVF" >&2; exit 1
fi
if grep -qE '^\s*colocate:\s*true' "$HOST_CFG" && [ "$SYNC" = 0 ]; then
  echo "[finqa] $CFG is colocated (one GPU takes turns sampling and training): pass --sync" >&2; exit 1
fi
if [ "$NETWORK" = private ] && ! grep -qE '^\s*host:\s*0\.0\.0\.0' "$HOST_CFG"; then
  echo "[finqa] --private_network needs reef.host 0.0.0.0 in $CFG (Reef is reached through the published port)" >&2; exit 1
fi
if [ -e "$STATE_DIR/checkpoints" ]; then
  echo "[finqa] $STATE_DIR already holds checkpoints; a fresh run needs a fresh --run_root" >&2
  exit 1
fi
if [ ! -s "$HERE/data/finqa_train.jsonl" ]; then
  "$PY" "$HERE/export_finqa.py" "$HERE/data"
fi
mkdir -p "$OUT" "$KEEP_DIR"

echo "[finqa] $(date '+%F %T %Z') command: $COMMAND"
echo "[finqa] tag=$TAG image=$IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE")) state=$STATE_DIR keep=$KEEP_DIR cfg=$CFG container=$NAME reef=$REEF_SERVICE_URL"
IMAGE="$IMAGE" NAME="$NAME" STATE_DIR="$STATE_DIR" KEEP_DIR="$KEEP_DIR" CFG="$CFG" EXTRA_PYTHONPATH=/repro/finqa \
  DOCKER_RUNTIME=nvidia STACK_GPUS="$GPUS" STACK_NETWORK="$NETWORK" REEF_PORT="$PORT" STACK_MEMORY="$MEMORY" \
  bash "$REPRO/scripts/start_stack.sh"

for i in $(seq 1 160); do
  if curl -sf -m 5 "$REEF_SERVICE_URL/healthz" >/dev/null 2>&1; then echo "[finqa] reef healthy $(date +%T)"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[finqa] stack exited during startup; see $STATE_DIR/stack/*.log" >&2
    grep -hE "Error:|Exception:" "$STATE_DIR"/stack/*.log 2>/dev/null | tail -3 >&2
    exit 1
  fi
  sleep 15
done
curl -sf -m 5 "$REEF_SERVICE_URL/healthz" >/dev/null || { echo "[finqa] reef never became healthy" >&2; exit 1; }

export PYTHONPATH="$HERE:$REPRO/vendor"
export FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish.tsv"   # read by the judge module (shared with the evaluation)

# Checkpoint copies out of Reef's managed tree (deepcoder/sidecar.py, task-independent; shared with the
# DeepCoder and AppWorld launchers, so it keeps its environment interface).
REEF_SERVICE_URL="$REEF_SERVICE_URL" SAO_SCENARIO="$SCENARIO" SAO_PROGRESS_FILE="$OUT/progress.txt" \
  SIDECAR_HF_DIR="$STATE_DIR/checkpoints/hf" SIDECAR_CONTAINER="$NAME" SIDECAR_KEEP_DIR="$KEEP_DIR" \
  SIDECAR_LOG="$OUT/sidecar.log" SIDECAR_ADAPTER_EVERY="$ADAPTER_EVERY" SIDECAR_FULL_EVERY="$FULL_EVERY" \
  SIDECAR_FINAL_STEP="$FINAL_STEP" \
  "$PY" -u "$REPRO/deepcoder/sidecar.py" > "$OUT/sidecar.stdout" 2>&1 &
SIDECAR=$!
echo "$SIDECAR" > "$OUT/sidecar.pid"
trap 'kill "$SIDECAR" 2>/dev/null || true' EXIT

DRIVER=("$PY" -u "$HERE/stream_finqa.py" --problems "$HERE/data/finqa_train.jsonl" --reef_service_url "$REEF_SERVICE_URL"
        --scenario "$SCENARIO" --records_path "$OUT/records.jsonl" --progress_file "$OUT/progress.txt" "${DRIVER_ARGS[@]}")
echo "[finqa] sidecar=$SIDECAR driver: ${DRIVER[*]:2}"
"${DRIVER[@]}" 2>&1 | tee "$OUT/driver.log"
echo "[finqa] driver exited rc=${PIPESTATUS[0]} $(date '+%F %T %Z')"
