#!/usr/bin/env bash
# DeepCoder test (687 LiveCodeBench tasks) on the evaluation host .36, data parallel: one vLLM
# server per free GPU (whole model on every GPU), rllm's eval_base.py --num-shards n --shard-index i
# per server, merge_shards.py joins them. Protocol is reef/exp_scripts/deepcoder/eval_sao_checkpoints.sh's
# (max-model-len 32768, max-num-seqs 32 = EVAL_CONCURRENCY, max_tokens 16384; greedy = T0 seed 1234,
# t1k4 = T1.0 x4 no seed) with .36's host-forced differences (see eval_sharded.sh): vllm 0.22.1+cu129,
# --dtype half, Triton attention, torch sampler, gpu-memory-utilization 0.80.
#
# The client runs in rllm's venv read through the read-only mount of .29:/home/yanan/agents; its
# RLLM_HOME is a small local copy whose dataset files link back to the mount, so nothing writes there.
#
# usage: deepcoder_sharded.sh TAG [ADAPTER_DIR]
#   env: MODE (base|lora), SAMPLING (greedy|t1k4), GPU_GROUPS (default: every GPU with < 256 MiB in
#        use), PORT_BASE (18200), BASE, OUT_ROOT (/home/yanan/eval36/results/deepcoder), WORK
set -euo pipefail
TAG=${1:?usage: $0 TAG [ADAPTER_DIR]}
ADAPTER=${2:-}
MODE=${MODE:-base}
SAMPLING=${SAMPLING:-greedy}
PORT_BASE=${PORT_BASE:-18200}
SERVE_VENV=${SERVE_VENV:-/home/yanan/eval36/env/venv-finqa-cu129}
RLLM=/home/yanan/agents/rllm
RUN_DIR=$RLLM/exp_scripts/deepcoder-run
CLIENT_PY=$RLLM/.venv/bin/python
BASE=${BASE:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
WORK=${WORK:-/home/yanan/eval36/work}
OUT_ROOT=${OUT_ROOT:-/home/yanan/eval36/results/deepcoder}
OUT="$OUT_ROOT/$TAG"
case "$SAMPLING" in
  greedy) SAMPLING_ENV=(EVAL_TEMPERATURE=0 EVAL_ATTEMPTS=1 EVAL_SEED=1234) ;;
  t1k4)   SAMPLING_ENV=(EVAL_TEMPERATURE=1.0 EVAL_ATTEMPTS=4 EVAL_SEED=none) ;;
  *) echo "SAMPLING must be greedy or t1k4" >&2; exit 2 ;;
esac
GPU_GROUPS=${GPU_GROUPS:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 256 {printf "%s ", $1}')}
read -r -a GPU_SETS <<< "$GPU_GROUPS"
N=${#GPU_SETS[@]}
[ "$N" -gt 0 ] || { echo "expected at least one free GPU on $(hostname); all are in use" >&2; exit 2; }
[ -x "$CLIENT_PY" ] || { echo "expected $CLIENT_PY (is the read-only mount of .29:/home/yanan/agents up?)" >&2; exit 2; }
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

# Local RLLM_HOME: registry copies, dataset files linked to the read-only mount.
RT=/home/yanan/eval36/env/rllm_runtime
mkdir -p "$RT/datasets/deepcoder"
cp "$RUN_DIR/runtime/datasets/registry.json" "$RT/datasets/registry.json"
cp "$RUN_DIR/runtime/snapshots.json" "$RT/snapshots.json"
ln -sfn "$RUN_DIR/runtime/datasets/deepcoder/test.parquet" "$RT/datasets/deepcoder/test.parquet"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -9 -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  return 0
}
trap cleanup EXIT
for i in "${!GPU_SETS[@]}"; do
  port=$((PORT_BASE + i))
  if curl -fsS -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "port $port is in use" >&2; exit 2; fi
  CUDA_VISIBLE_DEVICES=${GPU_SETS[$i]} setsid "$SERVE_VENV/bin/python" -m vllm.entrypoints.cli.main serve "$BASE" \
    --served-model-name base --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
    --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization "${GPU_UTIL:-0.80}" --dtype half \
    --attention-backend TRITON_ATTN "${SERVE_ARGS[@]}" > "$OUT/server_$i.log" 2>&1 &
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
echo "[$TAG] deepcoder mode=$MODE sampling=$SAMPLING $N servers ready (GPUs: $GPU_GROUPS) $(date '+%F %T %Z')"

SHARD_PIDS=()
for i in "${!GPU_SETS[@]}"; do
  ( cd "$RUN_DIR" && env "${SAMPLING_ENV[@]}" EVAL_CONCURRENCY=32 RLLM_HOME="$RT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="$RUN_DIR:$RLLM/cookbooks/deepcoder:$RLLM" \
      "$CLIENT_PY" -u "$RUN_DIR/eval_base.py" --url "http://127.0.0.1:$((PORT_BASE + i))/v1" --model "$MODEL_NAME" \
      --output "$WORK/$TAG/shard_$i" --num-shards "$N" --shard-index "$i" ) > "$OUT/eval_$i.log" 2>&1 &
  SHARD_PIDS+=($!)
done
for pid in "${SHARD_PIDS[@]}"; do wait "$pid" || { echo "[$TAG] a shard failed; see $OUT/eval_*.log" >&2; exit 1; }; done
merge_args=()
for i in "${!GPU_SETS[@]}"; do merge_args+=(--shard "$WORK/$TAG/shard_$i"); done
( cd "$RUN_DIR" && PYTHONPATH="$RUN_DIR:$RLLM" "$CLIENT_PY" "$RUN_DIR/merge_shards.py" "${merge_args[@]}" --output "$WORK/$TAG/merged" ) | tee "$OUT/merge.log"
cp "$WORK/$TAG/merged/result.json" "$OUT/result.json"
GPU_GROUPS_USED="$GPU_GROUPS" "$SERVE_VENV/bin/python" - "$WORK/$TAG/merged/protocol.json" "$OUT/protocol.json" <<'PY'
import json, os, sys
protocol = json.load(open(sys.argv[1]))
protocol.update(host=".36 Quadro RTX 5000 (sm_75)", dtype="float16", vllm="0.22.1+cu129", attention_backend="TRITON_ATTN",
                gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.80")), gpu_groups=os.environ["GPU_GROUPS_USED"])
json.dump(protocol, open(sys.argv[2], "w"), indent=2)
PY
rm -rf -- "${WORK:?}/$TAG"   # temporary (episodes can reach many GB)
echo "[$TAG] done $(date '+%F %T %Z'); results in $OUT"
