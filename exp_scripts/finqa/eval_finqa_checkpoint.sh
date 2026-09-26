#!/usr/bin/env bash
# Evaluate one model on FinQA val and test with the PRPO evaluation protocol
# (see eval_finqa.py), on one GPU of the evaluation host.
#
#   MODE=lora    serve the base with the adapter mounted (--enable-lora), unmerged
#   MODE=merged  merge the adapter into the base in bf16 first (PRPO's way), serve that
#   MODE=base    serve the base model itself (ADAPTER unused)
#   SAMPLING=greedy  temperature 0, top_p 1.0, seed 1234, 1 attempt (PRPO's protocol)
#   SAMPLING=t07k4   temperature 0.7 (the training temperature), top_p 1.0, 4 attempts, no seed
#
# vLLM flags are the PRPO evaluation's (eval_parallel.sh serve): max-model-len 12288,
# gpu-memory-utilization 0.85, --enable-auto-tool-choice --tool-call-parser hermes.
#
# usage: eval_finqa_checkpoint.sh TAG ADAPTER_DIR|- ; env MODE, SAMPLING, GPU, PORT, BASE, WORK, OUT_ROOT
set -euo pipefail
TAG=${1:?usage: $0 TAG ADAPTER_DIR|-}
ADAPTER=${2:?usage: $0 TAG ADAPTER_DIR|-}
MODE=${MODE:-lora}
SAMPLING=${SAMPLING:-greedy}
GPU=${GPU:-0}
PORT=${PORT:-18051}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
BASE=${BASE:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
WORK=${WORK:-/mnt/disk1t/sao-finqa-eval/work}
OUT_ROOT=${OUT_ROOT:-$REPRO/results/finqa-eval}
OUT="$OUT_ROOT/$TAG"
VENV=${FINQA_VENV:-$REPRO/.venv-finqa}
PY="$VENV/bin/python"

case "$SAMPLING" in
  greedy) SAMPLING_ARGS=(--temperature 0 --top-p 1.0 --seed 1234 --attempts 1) ;;
  t07k4)  SAMPLING_ARGS=(--temperature 0.7 --top-p 1.0 --seed none --attempts 4) ;;
  *) echo "SAMPLING must be greedy or t07k4" >&2; exit 2 ;;
esac
[ -x "$PY" ] || { echo "$PY missing: run setup_venv.sh" >&2; exit 2; }
test -f "$BASE/config.json" || { echo "base model not found: $BASE" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 2; }
if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "port $PORT is in use" >&2; exit 2; fi
if [ "$MODE" != base ]; then
  test -f "$ADAPTER/adapter_model.safetensors" -a -f "$ADAPTER/adapter_config.json" || { echo "not a PEFT adapter: $ADAPTER" >&2; exit 2; }
fi
for split in val test; do
  [ -s "$HERE/data/finqa_$split.jsonl" ] || "$PY" "$HERE/export_finqa.py" "$HERE/data"
done

source "$HERE/vllm_env.sh"
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$WORK/$TAG" "$OUT"
SERVE_MODEL=$BASE
SERVE_ARGS=()
MODEL_NAME=base
case "$MODE" in
  lora)
    R=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['r'])" "$ADAPTER/adapter_config.json")
    SERVE_ARGS=(--enable-lora --max-lora-rank "$R" --max-loras 1 --lora-modules "ckpt=$ADAPTER")
    MODEL_NAME=ckpt ;;
  merged)
    SERVE_MODEL=$WORK/$TAG/merged
    "$PY" "$HERE/merge_adapter.py" "$BASE" "$ADAPTER" "$SERVE_MODEL" > "$OUT/merge.log" 2>&1
    tail -1 "$OUT/merge.log" ;;
  base) ;;
  *) echo "MODE must be lora, merged or base" >&2; exit 2 ;;
esac

cleanup() { [ -n "${SPID:-}" ] && { kill -9 -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }; [ "$MODE" = merged ] && rm -rf -- "$WORK/$TAG/merged"; }
trap cleanup EXIT
setsid "$VENV/bin/vllm" serve "$SERVE_MODEL" --served-model-name base --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --max-model-len 12288 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes "${SERVE_ARGS[@]}" > "$WORK/$TAG/server.log" 2>&1 &
SPID=$!
for ((i=0; i<300; i++)); do
  kill -0 "$SPID" 2>/dev/null || { echo "[$TAG] vLLM exited before ready; see $WORK/$TAG/server.log" >&2; exit 1; }
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null
echo "[$TAG] mode=$MODE sampling=$SAMPLING server ready $(date '+%F %T %Z')"
if [ "$MODE" = lora ]; then
  # The adapter must change the policy: greedy log-probs of one prompt must differ from the base's.
  "$PY" - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
def logprobs(model):
    body = dict(model=model, messages=[{"role": "user", "content": "What was 3M's total revenue in 2023?"}],
                temperature=0, max_tokens=48, logprobs=True, top_logprobs=1, seed=1234)
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return [t["logprob"] for t in json.load(urllib.request.urlopen(request, timeout=300))["choices"][0]["logprobs"]["content"]]
base, ckpt = logprobs("base"), logprobs("ckpt")
n = min(len(base), len(ckpt))
difference = max(abs(x - y) for x, y in zip(base[:n], ckpt[:n]))
print(f"[adapter-check] max |logprob(base) - logprob(ckpt)| over {n} tokens = {difference:.4g}")
if difference == 0:
    raise SystemExit("the adapter request returned the base's log-probs: the adapter is not applied")
PY
fi

for split in val test; do
  FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_$split.tsv" "$PY" -u "$HERE/eval_finqa.py" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" --split "$split" --output "$WORK/$TAG/$split" \
    "${SAMPLING_ARGS[@]}" > "$OUT/eval_$split.log" 2>&1
  cp "$WORK/$TAG/$split/result.json" "$OUT/$split.json"
  cp "$WORK/$TAG/$split/protocol.json" "$OUT/${split}_protocol.json"
  echo "[$TAG] $split $(tail -1 "$OUT/eval_$split.log")"
done
