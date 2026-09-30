#!/usr/bin/env bash
# Evaluate one model on FinQA multi_val and multi_test (see eval_multitable.py), on one GPU of the
# evaluation host. finqa/eval_finqa_checkpoint.sh with the benchmark swapped: same modes, sampling
# settings, adapter check and vLLM flags, except max-model-len 49152 (rllm's multi-table context).
#
#   MODE=lora    serve the base with the adapter mounted (--enable-lora), unmerged
#   MODE=merged  merge the adapter into the base in bf16 first (PRPO's way), serve that
#   MODE=base    serve the base model itself (ADAPTER unused)
#   SAMPLING=greedy  temperature 0, top_p 1.0, seed 1234, 1 attempt (PRPO's protocol)
#   SAMPLING=t0k4    greedy protocol run 4 times per task (independent requests), scored avg@4;
#                    vLLM greedy is not deterministic under concurrent batching, so the 4 differ
#   SAMPLING=t07k4   temperature 0.7 (the training temperature), top_p 1.0, 4 attempts, no seed
#
# vLLM flags: max-model-len 49152, gpu-memory-utilization 0.85, --enable-auto-tool-choice
# --tool-call-parser hermes (single-table's, with rllm multi-table's context).
#
# Another base model (e.g. Qwen3.5-4B) sets TOOL_PARSER (qwen3_coder), CHAT_TEMPLATE (a template file)
# and CHAT_TEMPLATE_KWARGS (e.g. '{"enable_thinking": false}'); defaults are 2507's (hermes, its own
# template, no kwargs).
#
# usage: eval_multitable_checkpoint.sh TAG ADAPTER_DIR|- ; env MODE, SAMPLING, GPU, PORT, BASE, WORK, OUT_ROOT,
#        TOOL_PARSER, CHAT_TEMPLATE, CHAT_TEMPLATE_KWARGS
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
WORK=${WORK:-/mnt/disk1t/sao-finqa-multitable-eval/work}
OUT_ROOT=${OUT_ROOT:-$REPRO/results/finqa_multitable-eval}
OUT="$OUT_ROOT/$TAG"
VENV=${FINQA_VENV:-$REPRO/.venv-finqa}
PY="$VENV/bin/python"

case "$SAMPLING" in
  greedy) SAMPLING_ARGS=(--temperature 0 --top-p 1.0 --seed 1234 --attempts 1) ;;
  t0k4)   SAMPLING_ARGS=(--temperature 0 --top-p 1.0 --seed 1234 --attempts 4) ;;   # greedy x4, avg@4 like t07k4
  t07k4)  SAMPLING_ARGS=(--temperature 0.7 --top-p 1.0 --seed none --attempts 4) ;;
  *) echo "SAMPLING must be greedy, t0k4 or t07k4" >&2; exit 2 ;;
esac
[ -x "$PY" ] || { echo "$PY missing: run setup_venv.sh" >&2; exit 2; }
test -f "$BASE/config.json" || { echo "base model not found: $BASE" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 2; }
if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "port $PORT is in use" >&2; exit 2; fi
if [ "$MODE" != base ]; then
  test -f "$ADAPTER/adapter_model.safetensors" -a -f "$ADAPTER/adapter_config.json" || { echo "not a PEFT adapter: $ADAPTER" >&2; exit 2; }
fi
for split in multi_val multi_test; do
  [ -s "$HERE/data/$split.jsonl" ] || "$PY" "$HERE/export_multitable.py"
done
(cd "$HERE/rllm_multitable" && sha256sum --quiet -c SHA256SUMS) || { echo "rllm_multitable differs from its SHA256SUMS" >&2; exit 2; }

source "$REPRO/finqa_singletable/vllm_env.sh"
TOOL_PARSER=${TOOL_PARSER:-hermes}
TEMPLATE_ARGS=()
[ -n "${CHAT_TEMPLATE:-}" ] && { test -f "$CHAT_TEMPLATE" || { echo "chat template not found: $CHAT_TEMPLATE" >&2; exit 2; }; TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE"); }
KWARGS_ARGS=()
[ -n "${CHAT_TEMPLATE_KWARGS:-}" ] && KWARGS_ARGS=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
# Qwen3.5's kernels JIT-compile with ninja from the venv, and its FlashInfer sampler fails on this host.
export PATH="$VENV/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
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
    "$PY" "$REPRO/finqa_singletable/merge_adapter.py" "$BASE" "$ADAPTER" "$SERVE_MODEL" > "$OUT/merge.log" 2>&1
    tail -1 "$OUT/merge.log" ;;
  base) ;;
  *) echo "MODE must be lora, merged or base" >&2; exit 2 ;;
esac

# The trap's last status becomes the script's exit status, so no bare `[ ... ] &&` may end it
# (with MODE=lora the merged test returned 1 and every successful evaluation exited 1).
cleanup() {
  if [ -n "${SPID:-}" ]; then kill -9 -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; fi
  if [ "$MODE" = merged ]; then rm -rf -- "$WORK/$TAG/merged"; fi
  return 0
}
trap cleanup EXIT
setsid "$VENV/bin/vllm" serve "$SERVE_MODEL" --served-model-name base --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --max-model-len 49152 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" "${TEMPLATE_ARGS[@]}" "${SERVE_ARGS[@]}" > "$WORK/$TAG/server.log" 2>&1 &
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

for split in multi_val multi_test; do
  FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_$split.tsv" FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano "$PY" -u "$HERE/eval_multitable.py" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" --split "$split" --output "$WORK/$TAG/$split" \
    "${SAMPLING_ARGS[@]}" "${KWARGS_ARGS[@]}" > "$OUT/eval_$split.log" 2>&1
  cp "$WORK/$TAG/$split/result.json" "$OUT/$split.json"
  cp "$WORK/$TAG/$split/protocol.json" "$OUT/${split}_protocol.json"
  echo "[$TAG] $split $(tail -1 "$OUT/eval_$split.log")"
done
