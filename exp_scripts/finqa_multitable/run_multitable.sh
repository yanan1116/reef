#!/usr/bin/env bash
# SAO on FinQA multi-table (rllm's multi-table v2 protocol), formal run: Qwen3-4B-Instruct-2507,
# 64 tasks per step with one rollout each, one sample per multi-turn episode, compared against the
# rllm multi-table lines. 10 epochs of multi_train (991) = 10 x floor(991/64) = 150 optimizer steps (rllm drop_last epoch,
# as single-table 10 x 62 = 620; set 2026-09-30).
#
# finqa/run_finqa.sh with the benchmark swapped: same image, recipe (finqa/sao_multiturn.py, loaded
# from /repro/finqa), venv, judge credentials and checkpoint copier. Every 5th step's adapter is kept
# for evaluation (multi_val 126 + multi_test 131, LoRA-served), plus full actor+critic bundles every
# 31 steps (with the durable Reef versions) and at the final step.
#
# Image: reef:sao-<last commit touching Reef source>; with REEF_BF16_LOGITS=0 its Megatron is byte for byte
# the single-table image's (the bf16-logits patch is opt-in).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
TAG=${TAG:-finqa-multitable-b64-$(date +%Y%m%dT%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/works/yanan/reef-sao-finqa-multitable}   # .16 NVMe data disk: run data only, no code
STATE_DIR=${STATE_DIR:-$RUN_ROOT/state}
KEEP_DIR=${KEEP_DIR:-$RUN_ROOT/kept-checkpoints}
SRC_COMMIT=$(git -C "$REPRO/.." -c safe.directory='*' log -1 --format=%h -- . ':(exclude)exp_scripts')
export IMAGE=${IMAGE:-reef:sao-$SRC_COMMIT}
export DOCKER_RUNTIME=${DOCKER_RUNTIME-nvidia}
# fp32 policy logits, as single-table. REEF_BF16_LOGITS=1 (docker/patch/mcore_bf16_logits.py) breaks the
# first actor step: Slime's get_log_probs_and_entropy asserts fp32 logits (smoke 2026-09-29 13:57).
export REEF_BF16_LOGITS=${REEF_BF16_LOGITS:-0}
CFG=${CFG:-/repro/configs/serve-finqa-multitable-2507.yaml}
VENV=${FINQA_VENV:-$REPRO/.venv-finqa}
PY="$VENV/bin/python"
OUT="$REPRO/results/finqa_multitable/$TAG"

[ -x "$PY" ] || { echo "[multitable] expected $PY (finqa/setup_venv.sh)" >&2; exit 1; }
[ -r "${FINQA_JUDGE_CREDS:-$REPRO/finqa/.judge_creds}" ] || { echo "[multitable] expected judge credentials (FINQA_JUDGE_CREDS)" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { echo "[multitable] expected image $IMAGE on this host" >&2; exit 1; }
if [ "$REEF_BF16_LOGITS" = 1 ] && ! docker run --rm --entrypoint grep "$IMAGE" -q _reef_keep_bf16_logits /root/Megatron-LM/megatron/core/transformer/module.py; then
  echo "[multitable] REEF_BF16_LOGITS=1 but $IMAGE lacks docker/patch/mcore_bf16_logits.py; rebuild it (scripts/build_image_head.sh)" >&2; exit 1
fi
(cd "$HERE/rllm_multitable" && sha256sum --quiet -c SHA256SUMS) || { echo "[multitable] rllm_multitable differs from its SHA256SUMS (rllm's v2 protocol)" >&2; exit 1; }
if [ -e "$STATE_DIR/checkpoints" ]; then
  echo "[multitable] $STATE_DIR already holds checkpoints; a fresh run needs a fresh STATE_DIR" >&2
  exit 1
fi
[ -s "$HERE/data/multi_train.jsonl" ] || "$PY" "$HERE/export_multitable.py"
mkdir -p "$OUT" "$KEEP_DIR"

echo "[multitable] $(date '+%F %T %Z') tag=$TAG image=$IMAGE state=$STATE_DIR keep=$KEEP_DIR cfg=$CFG"
STATE_DIR="$STATE_DIR" KEEP_DIR="$KEEP_DIR" CFG="$CFG" EXTRA_PYTHONPATH=/repro/finqa \
  bash "$REPRO/scripts/start_stack.sh"

for i in $(seq 1 160); do
  if curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null 2>&1; then echo "[multitable] reef healthy $(date +%T)"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx reef-sao-stack; then
    echo "[multitable] stack exited during startup; see $STATE_DIR/stack/*.log" >&2
    grep -hE "Error:|Exception:" "$STATE_DIR"/stack/*.log 2>/dev/null | tail -3 >&2
    exit 1
  fi
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null || { echo "[multitable] reef never became healthy" >&2; exit 1; }

export PYTHONPATH="$HERE:$REPRO/finqa:$REPRO/vendor"
export SAO_PROBLEMS="$HERE/data/multi_train.jsonl"
export SAO_SCENARIO=${SAO_SCENARIO:-sao-finqa-multitable}
export SAO_BATCH=64                               # = recipe.config.batch-size in $CFG
export SAO_IN_FLIGHT=${SAO_IN_FLIGHT:-64}
export SAO_BUDGET=${SAO_BUDGET:-9600}             # 150 steps x 64 = 10 epochs of 991 tasks (15 steps each)
export SAO_TEMPERATURE=0.7                        # = rollout-temperature in $CFG (DIS needs them equal)
export SAO_TOP_P=1.0
export SAO_CONTEXT_TOKENS=49152                   # = context-length in $CFG
export SAO_SEED=${SAO_SEED:-0}
export SAO_RECORDS_PATH="$OUT/records.jsonl"
export SAO_PROGRESS_FILE="$OUT/progress.txt"
export FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish.tsv"
export FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano # = rllm multi-table training and evaluation
export SIDECAR_HF_DIR="$STATE_DIR/checkpoints/hf"
export SIDECAR_KEEP_DIR="$KEEP_DIR"
export SIDECAR_LOG="$OUT/sidecar.log"
export SIDECAR_ADAPTER_EVERY=${SIDECAR_ADAPTER_EVERY:-5}    # ~3 per epoch: what the evaluation reads
export SIDECAR_FULL_EVERY=${SIDECAR_FULL_EVERY:-30}         # every 2 epochs, on durable versions (30, 60, ..., 150)
export SIDECAR_FINAL_STEP=${SIDECAR_FINAL_STEP:-150}

"$PY" -u "$REPRO/deepcoder/sidecar.py" > "$OUT/sidecar.stdout" 2>&1 &
SIDECAR=$!
echo "$SIDECAR" > "$OUT/sidecar.pid"
trap 'kill "$SIDECAR" 2>/dev/null || true' EXIT

echo "[multitable] driver: batch=$SAO_BATCH in_flight=$SAO_IN_FLIGHT budget=$SAO_BUDGET T=$SAO_TEMPERATURE sidecar=$SIDECAR"
"$PY" -u "$HERE/stream_multitable.py" 2>&1 | tee "$OUT/driver.log"
echo "[multitable] driver exited rc=${PIPESTATUS[0]} $(date '+%F %T %Z')"
