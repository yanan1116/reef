#!/usr/bin/env bash
# AppWorld (react_code, official prompt) on the evaluation host .36, data parallel: one vLLM server per
# free GPU (whole model on every GPU), a random (seed 0) 1/n of the split's tasks to one measure_base.py per
# server, summarize_appworld.py joins the episodes (mean corrected reward, TGC, SGC).
#
# Protocol (memory: appworld-2507-code-mode-no-signal, appworld-discovery-mode-official-budget):
# --modality code, official react_code prompt (GRPO_VANILLA_APPWORLD_REACT_PROMPT), max-steps 50,
# episode token wall 24576, split dev (57) by default, greedy (T=0). vLLM serves with
# --enable-auto-tool-choice --tool-call-parser hermes as .29's base runs did, max-model-len 32768
# (one 16 GB card holds ~34.6k KV tokens; .29 used 65536), plus .36's host-forced differences (see
# eval_sharded.sh): vllm 0.22.1+cu129, --dtype half, Triton attention, torch sampler, utilization 0.80.
#
# The AppWorld server binary runs from the read-only mounted checkout's .venv; its writable root
# (data/ + experiments outputs, graded from disk) is a local copy, as AppWorld needs (NFS staleness
# SIGBUS-killed servers on .24). Experiment outputs under that root are temporary and removed.
#
# usage: appworld_sharded.sh TAG [ADAPTER_DIR]
#   env: MODE (base|lora), SPLIT (dev), TEMPERATURE (0.0), WORKERS per shard (4), GPU_GROUPS,
#        PORT_BASE (18300), APPWORLD_PORT_BASE (7300), BASE, MAX_LEN (32768), OUT_ROOT, WORK,
#        TOOL_PARSER (hermes; qwen3_coder for Qwen3.5), CHAT_TEMPLATE (a template file, optional).
#        The collector already sends chat_template_kwargs {"enable_thinking": false} on every call.
set -euo pipefail
TAG=${1:?usage: $0 TAG [ADAPTER_DIR]}
ADAPTER=${2:-}
MODE=${MODE:-base}
SPLIT=${SPLIT:-dev}
TEMPERATURE=${TEMPERATURE:-0.0}
WORKERS=${WORKERS:-4}
PORT_BASE=${PORT_BASE:-18300}
APPWORLD_PORT_BASE=${APPWORLD_PORT_BASE:-7300}
MAX_LEN=${MAX_LEN:-32768}
TOOL_PARSER=${TOOL_PARSER:-hermes}
TEMPLATE_ARGS=()
[ -n "${CHAT_TEMPLATE:-}" ] && TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
SERVE_VENV=${SERVE_VENV:-/home/yanan/eval36/env/venv-finqa-cu129}
TAIL=/home/yanan/agents/gitlab/tail
CLIENT_PY=$TAIL/appworld/.venv/bin/python
ROOT=${APPWORLD_ROOT_LOCAL:-/home/yanan/eval36/env/appworld_root}
BASE=${BASE:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
WORK=${WORK:-/home/yanan/eval36/work}
OUT_ROOT=${OUT_ROOT:-/home/yanan/eval36/results/appworld}
OUT="$OUT_ROOT/$TAG"
GPU_GROUPS=${GPU_GROUPS:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 256 {printf "%s ", $1}')}
read -r -a GPU_SETS <<< "$GPU_GROUPS"
N=${#GPU_SETS[@]}
[ "$N" -gt 0 ] || { echo "expected at least one free GPU on $(hostname); all are in use" >&2; exit 2; }
[ -x "$CLIENT_PY" ] || { echo "expected $CLIENT_PY (is the read-only mount of .29:/home/yanan/agents up?)" >&2; exit 2; }
[ -s "$ROOT/data/datasets/$SPLIT.txt" ] || { echo "expected the local AppWorld root $ROOT with data/datasets/$SPLIT.txt" >&2; exit 2; }
test -f "$BASE/config.json" || { echo "base model not found: $BASE" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 2; }
SERVE_ARGS=()
MODEL_NAME=base
if [ "$MODE" = lora ]; then
  test -f "$ADAPTER/adapter_model.safetensors" -a -f "$ADAPTER/adapter_config.json" || { echo "not a PEFT adapter: $ADAPTER" >&2; exit 2; }
  R=$("$SERVE_VENV/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['r'])" "$ADAPTER/adapter_config.json")
  SERVE_ARGS=(--enable-lora --max-lora-rank "$R" --max-loras 1 --lora-modules "ckpt=$ADAPTER")
  MODEL_NAME=ckpt
elif [ "$MODE" != base ]; then
  echo "MODE must be base or lora" >&2; exit 2
fi
mkdir -p "$OUT" "$WORK/$TAG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -9 -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  # AppWorld servers start in their own sessions; stop any this run left behind (by port range).
  for ((p=APPWORLD_PORT_BASE; p<APPWORLD_PORT_BASE+N*20; p++)); do
    pid=$(ss -ltnp "sport = :$p" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  return 0
}
trap cleanup EXIT
for i in "${!GPU_SETS[@]}"; do
  port=$((PORT_BASE + i))
  if curl -fsS -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "port $port is in use" >&2; exit 2; fi
  CUDA_VISIBLE_DEVICES=${GPU_SETS[$i]} setsid "$SERVE_VENV/bin/python" -m vllm.entrypoints.cli.main serve "$BASE" \
    --served-model-name base --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "${GPU_UTIL:-0.80}" --dtype half --attention-backend TRITON_ATTN \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" "${TEMPLATE_ARGS[@]}" "${SERVE_ARGS[@]}" > "$OUT/server_$i.log" 2>&1 &
  PIDS+=($!)
done
for i in "${!GPU_SETS[@]}"; do
  port=$((PORT_BASE + i))
  for ((t=0; t<600; t++)); do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "[$TAG] vLLM $i (GPU ${GPU_SETS[$i]}) exited before ready; see $OUT/server_$i.log" >&2; exit 1; }
    curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null
done
echo "[$TAG] appworld $SPLIT mode=$MODE T=$TEMPERATURE $N servers ready (GPUs: $GPU_GROUPS) $(date '+%F %T %Z')"

# Random but reproducible 1/n of the tasks per GPU (seed 0), as eval_finqa.py --shard.
mapfile -t TASKS < <("$SERVE_VENV/bin/python" -c "import random,sys; t=open(sys.argv[1]).read().split(); random.Random(0).shuffle(t); print(chr(10).join(t))" "$ROOT/data/datasets/$SPLIT.txt")
SHARD_PIDS=()
for i in "${!GPU_SETS[@]}"; do
  ids=$(for k in "${!TASKS[@]}"; do if [ $((k % N)) -eq "$i" ]; then printf '%s,' "${TASKS[$k]}"; fi; done)
  ids=${ids%,}
  [ -n "$ids" ] || continue
  GRPO_VANILLA_APPWORLD_REACT_PROMPT=/home/yanan/agents/appworld/experiments/prompts/react_code_agent/instructions.txt \
  GRPO_VANILLA_APPWORLD_AUTH_HINT=1 setsid "$CLIENT_PY" -u "$REPRO/appworld/measure_base.py" \
    --endpoint "http://127.0.0.1:$((PORT_BASE + i))/v1" --model "$MODEL_NAME" --out "$WORK/$TAG/$TAG-shard_$i" \
    --split "$SPLIT" --task-ids "$ids" --workers "$WORKERS" --port-base $((APPWORLD_PORT_BASE + 20 * i)) \
    --temperature "$TEMPERATURE" --modality code --max-steps 50 --episode-token-wall 24576 \
    --appworld-root "$ROOT" --max-context "$MAX_LEN" > "$OUT/measure_$i.log" 2>&1 &
  SHARD_PIDS+=($!)
done
for pid in "${SHARD_PIDS[@]}"; do wait "$pid" || { echo "[$TAG] a shard failed; see $OUT/measure_*.log" >&2; exit 1; }; done
GPU_GROUPS_USED="$GPU_GROUPS" "$SERVE_VENV/bin/python" "$HERE/summarize_appworld.py" "$OUT" "$ROOT/data/datasets/$SPLIT.txt" \
  "$WORK/$TAG"/"$TAG"-shard_* | tee "$OUT/summary.log"
rm -rf -- "${WORK:?}/$TAG" "$ROOT/experiments/outputs/sao-measure/$TAG-shard_"*   # temporary
echo "[$TAG] done $(date '+%F %T %Z'); results in $OUT"
