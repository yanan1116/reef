#!/usr/bin/env bash
# Launch SAO on AppWorld with Qwen3.5-4B on .16 (2x RTX 6000 Ada): SGLang on one GPU, actor+critic on
# the other (configs/serve-appworld-qwen35-2gpu*.yaml), synchronous batches, thinking off.
# Run ON .16. Everything goes through appworld/run_appworld.sh; this only fixes the host settings.
#
#   bash appworld/launch_16_qwen35_2gpu.sh smoke    # batch 4, budget 12 = 3 steps; must reach step 3
#   bash appworld/launch_16_qwen35_2gpu.sh formal   # batch 15, budget 900 = 10 epochs of train90 = 60 steps
#
# LAYOUT selects the GPU layout (default 2gpu, unchanged):
#   2gpu  configs/serve-appworld-qwen35-2gpu*.yaml: SGLang on GPU 1, actor+critic on GPU 0
#   tp2   configs/serve-appworld-qwen35-tp2*.yaml: actor+critic TP2 over both GPUs, one SGLang engine per
#         GPU, colocated (the 2gpu layout OOMed on a ~26k-token sample, 2026-10-09 22:45)
# SAO_PROBLEMS (optional) is the driver's task-id file (default: train90), e.g. a smoke pool of long tasks.
# SAO_ECHO=1 runs the ECHO arm (user, 2026-10-23): the same layout, image and settings with the config's -echo
# twin (two sao-echo-* lines) and STACK_SAO_ECHO=1, which mounts the ECHO recipe files over the image. Unset: as before.
# SAO_ECHO_VARIANT=NAME (with SAO_ECHO=1) selects the -echo-NAME config, run root and tag instead, e.g. nodocs
# SAO_CFG_VARIANT=NAME (ECHO off) selects the -NAME config, run root and tag of a plain SAO variant, e.g. lr10x
# (configs/serve-appworld-qwen35-tp2-echo-nodocs.yaml: ECHO without API documentation, user 2026-10-23).
# A GPU memory sampler writes RUN_ROOT/ops/gpu-mem.log every 10 s while the launcher runs.
#
# Host settings: STACK_NETWORK=private (own Ray), STACK_TE_FA2_SM89_HDIM256=1 (FlashAttention-2 for
# head_dim 256 on sm89), NVTE_DEBUG=1 (TE logs the attention backend it selected), STACK_MEMORY=160g
# (offloaded actor/critic state; the host has 250 GB). Run data on the local NVMe (/works/yanan).
set -euo pipefail
MODE=${1:?smoke|formal}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
[ "$(hostname)" = tai ] || [ -d /works/yanan ] || { echo "run this on .16 (/works/yanan is its NVMe)" >&2; exit 1; }
export IMAGE=${IMAGE:-reef:sao-76fb540e}
export LAYOUT=${LAYOUT:-2gpu}
case "$LAYOUT" in 2gpu|tp2) ;; *) echo "LAYOUT must be 2gpu or tp2, got $LAYOUT" >&2; exit 1 ;; esac
ECHO_SUFFIX=""
if [ "${SAO_ECHO:-0}" = 1 ]; then ECHO_SUFFIX=-echo${SAO_ECHO_VARIANT:+-$SAO_ECHO_VARIANT}; fi
# SAO_CFG_VARIANT=NAME selects the -NAME config twin (e.g. lr10x: the SAO config with lr x10), its run root and tag,
# with the ECHO switch off. Unset: as before.
if [ -n "${SAO_CFG_VARIANT:-}" ]; then
  [ "${SAO_ECHO:-0}" = 1 ] && { echo "SAO_CFG_VARIANT is for plain SAO; unset SAO_ECHO" >&2; exit 1; }
  ECHO_SUFFIX=-$SAO_CFG_VARIANT
fi
case "$MODE" in
  smoke)  export CFG=/repro/configs/serve-appworld-qwen35-$LAYOUT-smoke$ECHO_SUFFIX.yaml
          export RUN_ROOT=${RUN_ROOT:-/works/yanan/reef-sao-appworld-qwen35-$LAYOUT-smoke$ECHO_SUFFIX}
          export SAO_BUDGET=12 SAO_IN_FLIGHT=4
          export TAG=${TAG:-appworld-qwen35-$LAYOUT-smoke$ECHO_SUFFIX-$(date +%Y%m%dT%H%M%S)} ;;
  formal) export CFG=/repro/configs/serve-appworld-qwen35-$LAYOUT$ECHO_SUFFIX.yaml
          export RUN_ROOT=${RUN_ROOT:-/works/yanan/reef-sao-appworld-qwen35-$LAYOUT$ECHO_SUFFIX}
          export SAO_BUDGET=900 SAO_IN_FLIGHT=15          # 10 epochs x 90 tasks, 15 per step -> 60 steps
          export TAG=${TAG:-appworld-qwen35-$LAYOUT$ECHO_SUFFIX-b15-$(date +%Y%m%dT%H%M%S)} ;;
  *) echo "mode must be smoke or formal" >&2; exit 1 ;;
esac
export STACK_NETWORK=private STACK_TE_FA2_SM89_HDIM256=1 STACK_MEMORY=${STACK_MEMORY:-160g}
export STACK_SAO_ECHO=${SAO_ECHO:-0}
export NVTE_DEBUG=${NVTE_DEBUG:-1} NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-1}
export APPWORLD_ROOT=/home/yanan/appworld-root
# Train every sample that fits the 32k window (user, 2026-10-09): with the actor on its own 48 GB GPU the
# 16384 length filter of the colocated runs is not needed. max-tokens-per-gpu stays 16384 (packing knob);
# a longer sample gets a micro-batch of its own.
export SAO_MAX_SAMPLE_TOKENS=${SAO_MAX_SAMPLE_TOKENS:-32768}
docker image inspect "$IMAGE" >/dev/null || { echo "image $IMAGE is not on this host" >&2; exit 1; }
[ -e "$RUN_ROOT/state" ] && { echo "$RUN_ROOT/state exists; a fresh run needs a fresh RUN_ROOT" >&2; exit 1; }
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
[ "$busy" = 0 ] || { echo "a GPU on this host is in use; refusing to launch" >&2; exit 1; }
mkdir -p "$RUN_ROOT/ops"
{
  echo "started_at=$(TZ=America/New_York date '+%FT%T%z') host=$(hostname) mode=$MODE tag=$TAG"
  echo "image=$IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE")) cfg=$CFG run_root=$RUN_ROOT budget=$SAO_BUDGET in_flight=$SAO_IN_FLIGHT"
  echo "stack: layout=$LAYOUT network=$STACK_NETWORK te_fa2_patch=$STACK_TE_FA2_SM89_HDIM256 memory=$STACK_MEMORY nvte_debug=$NVTE_DEBUG max_sample_tokens=$SAO_MAX_SAMPLE_TOKENS problems=${SAO_PROBLEMS:-train90} sao_echo=$STACK_SAO_ECHO"
  (cd "$REPRO" && sha256sum "configs/$(basename "$CFG")" appworld/run_appworld.sh appworld/stream_appworld.py scripts/start_stack.sh)
  if [ "$STACK_SAO_ECHO" = 1 ]; then
    (cd "$REPRO/.." && sha256sum recipes/sao/slime/__init__.py recipes/sao/slime/objective.py recipes/sao/slime/echo.py)
  fi
  echo "git: $(git -C "$REPRO/.." -c safe.directory='*' rev-parse --abbrev-ref HEAD) $(git -C "$REPRO/.." -c safe.directory='*' rev-parse --short HEAD) (exp_scripts uncommitted; launch uses the code on disk)"
} | tee "$RUN_ROOT/ops/launch.txt"
setsid nohup bash "$HERE/run_appworld.sh" --sync --no_think > "$RUN_ROOT/ops/launcher.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/ops/launcher.pid"
# GPU memory every 10 s while the launcher lives ("HH:MM:SS 0, MiB 1, MiB"), for the peak in the report.
setsid nohup bash -c 'while kill -0 "$1" 2>/dev/null; do echo "$(TZ=America/New_York date +%T) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr "\n" " ")" >> "$2"; sleep 10; done' \
  _ "$(cat "$RUN_ROOT/ops/launcher.pid")" "$RUN_ROOT/ops/gpu-mem.log" > /dev/null 2>&1 < /dev/null &
echo "launcher pid $(cat "$RUN_ROOT/ops/launcher.pid"); driver log: $REPRO/results/appworld/$TAG/driver.log; stack log: $RUN_ROOT/state/stack/slime-driver.log"
