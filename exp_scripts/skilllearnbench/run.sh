#!/usr/bin/env bash
# SAO training on the SkillLearnBench split, on .16 (2x RTX 6000 Ada). Run ON .16.
#
#   bash skilllearnbench/run.sh smoke    # 3 train instances x 4 repeats, batch 4 episodes; must reach training step 3
#   bash skilllearnbench/run.sh formal   # 41 train instances x 8 per epoch, 10 epochs, batch 16 episodes
#
# Same stack sequence as appworld/run_appworld.sh: scripts/start_stack.sh with the TP2 colocated Qwen3.5
# config, wait for /healthz, deepcoder/sidecar.py keeps every step's adapter, then the episodes through
# reef-eval (run.py train) in an ephemeral uv env, like recipes/sao/examples/ceobench/run.sh. One episode is
# one sample (exp_scripts/finqa/sao_multiturn.py on the stack's PYTHONPATH), in synchronous batches.
set -euo pipefail
MODE=${1:?smoke|formal}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
[ -d /works/yanan ] || { echo "run this on .16 (/works/yanan is its NVMe)" >&2; exit 1; }
export IMAGE=${IMAGE:-reef:sao-76fb540e}
case "$MODE" in
  smoke)  CFG=/repro/configs/serve-skilllearnbench-qwen35-tp2-smoke.yaml
          RUN_ROOT=${RUN_ROOT:-/works/yanan/slb-sao-smoke-$(date +%Y%m%dT%H%M%S)}
          export SLB_EPOCHS=1 SLB_REPEATS=4 SLB_CONCURRENCY=4
          export SLB_ONLY=${SLB_ONLY:-court-form-filling-2,offer-letter-generator-2,weighted-gdp-calculation-3} ;;
  formal) CFG=/repro/configs/serve-skilllearnbench-qwen35-tp2.yaml
          RUN_ROOT=${RUN_ROOT:-/works/yanan/slb-sao-formal}
          export SLB_EPOCHS=10 SLB_REPEATS=8 SLB_CONCURRENCY=${SLB_CONCURRENCY:-16} ;;
  *) echo "mode must be smoke or formal" >&2; exit 1 ;;
esac
export SLB_TASKS=${SLB_TASKS:-/works/yanan/slb-sao-tasks}
HOST_CFG="$REPRO/${CFG#/repro/}"
export SLB_BATCH=$(grep -m1 -E '^\s*batch-size:' "$HOST_CFG" | awk '{print $2}')   # episodes per step = the recipe's
export SLB_RUN="$RUN_ROOT/run" SLB_MODE=train SLB_MODEL=reef SLB_TEMPERATURE=1.0
export SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 SLB_TRAIN_MAX_TOKENS=32768
export SLB_COMMAND_TIMEOUT_S=180   # one bash command (user, 2026-10-14); the agent default 600 is for the base grid
# user, 2026-10-15: v2 lifts the base (test-pass 0.230 -> 0.306, SR 0.7% -> 8.0%). The full 20-task run (user,
# 2026-10-19) may pass v3 with budget notices (SLB_BUDGET_WARN=0.5,0.7); notices are append-only user turns.
export SLB_PROMPT_VERSION=${SLB_PROMPT_VERSION:-v2} SLB_BUDGET_WARN=${SLB_BUDGET_WARN-}
# github-repo-analytics reads GitHub live; the trials get GH_TOKEN from this process (token file mode 600, never logged).
[ -r "$HOME/.config/slb/gh_token" ] && GH_TOKEN=$(cat "$HOME/.config/slb/gh_token") && export GH_TOKEN
export REEF_SERVICE_URL=http://127.0.0.1:8900 REEF_SCENARIO=slb-sao REEF_TOKEN=reef-local
STATE_DIR="$RUN_ROOT/state" KEEP_DIR="$RUN_ROOT/kept"

[ -f "$SLB_TASKS/split.json" ] || { echo "no $SLB_TASKS/split.json: run prepare_tasks.py --out $SLB_TASKS first" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { echo "image $IMAGE is not on this host" >&2; exit 1; }
[ -e "$STATE_DIR" ] && { echo "$STATE_DIR exists; a fresh run needs a fresh RUN_ROOT" >&2; exit 1; }
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
[ "$busy" = 0 ] || { echo "a GPU on this host is in use; refusing to launch" >&2; exit 1; }
mkdir -p "$SLB_RUN" "$KEEP_DIR"
{
  echo "started_at=$(TZ=America/New_York date '+%FT%T%z') host=$(hostname) mode=$MODE image=$IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE")) cfg=$CFG"
  echo "tasks=$SLB_TASKS epochs=$SLB_EPOCHS repeats=$SLB_REPEATS batch=$SLB_BATCH concurrency=$SLB_CONCURRENCY only=${SLB_ONLY:-all} prompt=$SLB_PROMPT_VERSION notices=${SLB_BUDGET_WARN:-off} gh_token=${GH_TOKEN:+set}"
  (cd "$REPRO" && sha256sum "configs/$(basename "$CFG")" skilllearnbench/run.sh skilllearnbench/run.py skilllearnbench/harness/agent.py scripts/start_stack.sh)
  echo "git: $(git -C "$REPRO/.." -c safe.directory='*' rev-parse --abbrev-ref HEAD) $(git -C "$REPRO/.." -c safe.directory='*' rev-parse --short HEAD) (exp_scripts uncommitted; launch uses the code on disk)"
} | tee "$RUN_ROOT/launch.txt"

NAME=reef-slb-stack STATE_DIR="$STATE_DIR" KEEP_DIR="$KEEP_DIR" CFG="$CFG" STACK_NETWORK=private EXTRA_PYTHONPATH=/repro/finqa \
  STACK_TE_FA2_SM89_HDIM256=1 STACK_MEMORY=${STACK_MEMORY:-160g} DOCKER_RUNTIME=nvidia \
  NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 bash "$REPRO/scripts/start_stack.sh"
for i in $(seq 1 160); do
  curl -sf -m 5 "$REEF_SERVICE_URL/healthz" >/dev/null 2>&1 && { echo "reef healthy $(date +%T)"; break; }
  docker ps --format '{{.Names}}' | grep -qx reef-slb-stack || { echo "stack exited during startup; see $STATE_DIR/stack/*.log" >&2; exit 1; }
  sleep 15
done
curl -sf -m 5 "$REEF_SERVICE_URL/healthz" >/dev/null || { echo "reef never became healthy" >&2; exit 1; }

export SAO_SCENARIO=$REEF_SCENARIO SAO_PROGRESS_FILE="$SLB_RUN/progress.txt" SIDECAR_CONTAINER=reef-slb-stack
export SIDECAR_HF_DIR="$STATE_DIR/checkpoints/hf" SIDECAR_KEEP_DIR="$KEEP_DIR" SIDECAR_LOG="$SLB_RUN/sidecar.log"
export SIDECAR_ADAPTER_EVERY=1 SIDECAR_FULL_EVERY=0 SIDECAR_FINAL_STEP=0
python3 -u "$REPRO/deepcoder/sidecar.py" > "$SLB_RUN/sidecar.stdout" 2>&1 &
SIDECAR=$!
trap 'kill "$SIDECAR" 2>/dev/null || true' EXIT

cd "$HERE"
uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
  run.py train 2>&1 | tee "$SLB_RUN/train.log"
echo "train exited rc=${PIPESTATUS[0]} $(TZ=America/New_York date '+%F %T %Z')"
