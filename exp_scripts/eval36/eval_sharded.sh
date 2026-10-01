#!/usr/bin/env bash
# Evaluate one model on FinQA single-table (val, test) or multi-table (multi_val, multi_test) on the
# evaluation host .36, sharding the tasks across its free GPUs: one vLLM server per GPU group, each
# group runs eval_finqa.py / eval_multitable.py --shard i/n, and merge_eval_shards.py joins the shards.
#
# The protocol is finqa_singletable/eval_finqa_checkpoint.sh's and finqa_multitable/
# eval_multitable_checkpoint.sh's (same sampling settings, vLLM flags, judge, adapter check). Data
# parallel: every GPU holds the whole model and serves 1/n of the tasks. Host-forced differences,
# stated in every result's protocol.json; base and checkpoints are compared within .36 only:
#   --dtype half     .36's Quadro RTX 5000 (Turing, sm_75) has no bf16; every other host serves bf16
#   TRITON_ATTN      no nvcc on .36, so FlashInfer (JIT) is unavailable; sampler is torch's
#   GPU_UTIL 0.80    (others 0.85) Triton's full CUDA-graph capture OOMs a 16 GB card at 0.85
#   vllm 0.22.1+cu129 .36's driver 535 (CUDA 12.2) cannot run the PyPI CUDA-13 build of the same
#                    version (its kernels silently write zeros); the venv is .29's .venv-finqa with
#                    only vllm swapped (FINQA_VENV=/home/yanan/eval36/env/venv-finqa-cu129)
#
# .36 reads code, the venv and adapters read-only from other hosts (scripts/mount_eval_host_36.sh)
# and keeps only base models; this script's work tree is temporary and removed after merging.
#
# usage: eval_sharded.sh single|multi TAG [ADAPTER_DIR]
#   env: MODE (base|lora, default base), SAMPLING (greedy|t0k4|t07k4), GPU_GROUPS (default
#        every GPU with < 256 MiB in use at launch: one GPU per server), PORT_BASE (18100), BASE, TOOL_PARSER,
#        CHAT_TEMPLATE, CHAT_TEMPLATE_KWARGS, OUT_ROOT (/home/yanan/eval36/results/<bench>),
#        WORK (/home/yanan/eval36/work), MAX_LEN (override the benchmark's max-model-len),
#        EVAL36_HOSTS (required: "judge-host=ip", see dns_override/sitecustomize.py),
#        EVAL_SPLITS (override the benchmark's splits, e.g. "val" or "multi_val")
set -euo pipefail
BENCH=${1:?usage: $0 single|multi TAG [ADAPTER_DIR]}
TAG=${2:?usage: $0 single|multi TAG [ADAPTER_DIR]}
ADAPTER=${3:-}
MODE=${MODE:-base}
SAMPLING=${SAMPLING:-greedy}
PORT_BASE=${PORT_BASE:-18100}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
VENV=${FINQA_VENV:-/home/yanan/eval36/env/venv-finqa-cu129}
PY="$VENV/bin/python"
BASE=${BASE:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
WORK=${WORK:-/home/yanan/eval36/work}
case "$BENCH" in
  single) SRC="$REPRO/finqa_singletable"; EVAL="$SRC/eval_finqa.py"; SPLITS=(val test); MAX_LEN=${MAX_LEN:-12288}
          ;;
  multi)  SRC="$REPRO/finqa_multitable"; EVAL="$SRC/eval_multitable.py"; SPLITS=(multi_val multi_test); MAX_LEN=${MAX_LEN:-49152}
          ;;
  *) echo "first argument must be single or multi" >&2; exit 2 ;;
esac
OUT_ROOT=${OUT_ROOT:-/home/yanan/eval36/results/$BENCH}
OUT="$OUT_ROOT/$TAG"
TOOL_PARSER=${TOOL_PARSER:-hermes}
case "$SAMPLING" in
  greedy) SAMPLING_ARGS=(--temperature 0 --top-p 1.0 --seed 1234 --attempts 1) ;;
  t0k4)   SAMPLING_ARGS=(--temperature 0 --top-p 1.0 --seed 1234 --attempts 4) ;;
  t07k4)  SAMPLING_ARGS=(--temperature 0.7 --top-p 1.0 --seed none --attempts 4) ;;
  *) echo "SAMPLING must be greedy, t0k4 or t07k4" >&2; exit 2 ;;
esac
# .36 is shared: take every GPU nobody else is using (other users' jobs come and go).
GPU_GROUPS=${GPU_GROUPS:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 256 {printf "%s ", $1}')}
if [ -n "${EVAL_SPLITS:-}" ]; then read -r -a SPLITS <<< "$EVAL_SPLITS"; fi
read -r -a GPU_SETS <<< "$GPU_GROUPS"  # (GROUPS is a bash special variable)
N=${#GPU_SETS[@]}
[ "$N" -gt 0 ] || { echo "expected at least one free GPU on $(hostname); all are in use" >&2; exit 2; }
[ -x "$PY" ] || { echo "expected $PY (is the read-only mount of .29:/home/yanan/agents up?)" >&2; exit 2; }
test -f "$BASE/config.json" || { echo "base model not found: $BASE" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 2; }
SERVE_ARGS=()
MODEL_NAME=base
if [ "$MODE" = lora ]; then
  test -f "$ADAPTER/adapter_model.safetensors" -a -f "$ADAPTER/adapter_config.json" || { echo "not a PEFT adapter: $ADAPTER" >&2; exit 2; }
  R=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['r'])" "$ADAPTER/adapter_config.json")
  SERVE_ARGS=(--enable-lora --max-lora-rank "$R" --max-loras 1 --lora-modules "ckpt=$ADAPTER")
  MODEL_NAME=ckpt
elif [ "$MODE" != base ]; then
  echo "MODE must be base or lora" >&2; exit 2
fi
TEMPLATE_ARGS=()
[ -n "${CHAT_TEMPLATE:-}" ] && TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
KWARGS_ARGS=()
[ -n "${CHAT_TEMPLATE_KWARGS:-}" ] && KWARGS_ARGS=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
# .36 cannot reach its DNS server; the judge's host is pinned to the address .29 resolves it to.
[ -n "${EVAL36_HOSTS:-}" ] || { echo "expected EVAL36_HOSTS=judge-host=ip (resolve it on .29); .36 has no DNS" >&2; exit 2; }
export EVAL36_HOSTS
export PYTHONPATH="$HERE/dns_override${PYTHONPATH:+:$PYTHONPATH}"
# A judge that cannot be reached scores every answer 0 after its retries: refuse to start instead.
( cd "$REPRO/finqa_singletable" && "$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
from judge_env import load_judge_env
load_judge_env()
import finqa_env, finqa_eval
ok, _ = finqa_eval._call_judge(finqa_eval.CORRECTNESS_PROMPT, "question : What is 2+2?\nmodel response : 4\nlabel : 4", multi_table=False)
if ok is not True:
    raise SystemExit("judge preflight: expected True for a trivially correct answer; the judge is unreachable or misconfigured")
print("[judge preflight] ok")
PY
) || exit 2
mkdir -p "$OUT" "$WORK/$TAG"
FINQA_VENV=$VENV source "$REPRO/finqa_singletable/vllm_env.sh"
# .36 has no CUDA toolkit (nvcc): FlashInfer's JIT attention and sampler cannot build there, so the
# Triton attention backend (its own compiler) and the torch sampler serve instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -9 -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  return 0
}
trap cleanup EXIT
for i in "${!GPU_SETS[@]}"; do
  gpus=${GPU_SETS[$i]}
  tp=$(tr ',' '\n' <<< "$gpus" | wc -l)
  port=$((PORT_BASE + i))
  if curl -fsS -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "port $port is in use" >&2; exit 2; fi
  # python -m, not bin/vllm: the venv was copied to .36, so its console scripts point at .29's venv.
  CUDA_VISIBLE_DEVICES=$gpus setsid "$PY" -m vllm.entrypoints.cli.main serve "$BASE" --served-model-name base --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$tp" --max-model-len "$MAX_LEN" --gpu-memory-utilization "${GPU_UTIL:-0.80}" --dtype half \
    --attention-backend TRITON_ATTN \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" "${TEMPLATE_ARGS[@]}" "${SERVE_ARGS[@]}" \
    > "$OUT/server_$i.log" 2>&1 &
  PIDS+=($!)
done
for i in "${!GPU_SETS[@]}"; do
  port=$((PORT_BASE + i))
  for ((t=0; t<600; t++)); do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "[$TAG] vLLM $i (GPUs ${GPU_SETS[$i]}) exited before ready; see $OUT/server_$i.log" >&2; exit 1; }
    curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null
done
echo "[$TAG] $BENCH mode=$MODE sampling=$SAMPLING splits=${SPLITS[*]} $N servers ready (groups: $GPU_GROUPS) $(date '+%F %T %Z')"
if [ "$MODE" = lora ]; then
  # The adapter must change the policy (as eval_*_checkpoint.sh checks): greedy log-probs of one prompt
  # must differ between the base and the adapter on every server.
  for i in "${!GPU_SETS[@]}"; do
    "$PY" - "$((PORT_BASE + i))" <<'PY'
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
print(f"[adapter-check] port {port}: max |logprob(base) - logprob(ckpt)| over {n} tokens = {difference:.4g}")
if difference == 0:
    raise SystemExit("the adapter request returned the base's log-probs: the adapter is not applied")
PY
  done
fi

for split in "${SPLITS[@]}"; do
  SHARD_PIDS=()
  for i in "${!GPU_SETS[@]}"; do
    FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_${split}_$i.tsv" FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano "$PY" -u "$EVAL" \
      --base-url "http://127.0.0.1:$((PORT_BASE + i))/v1" --model "$MODEL_NAME" --split "$split" \
      --output "$WORK/$TAG/$split/shard_$i" --shard "$i/$N" "${SAMPLING_ARGS[@]}" "${KWARGS_ARGS[@]}" \
      > "$OUT/eval_${split}_$i.log" 2>&1 &
    SHARD_PIDS+=($!)
  done
  for pid in "${SHARD_PIDS[@]}"; do wait "$pid" || { echo "[$TAG] a $split shard failed; see $OUT/eval_${split}_*.log" >&2; exit 1; }; done
  alarms=$(cat "$OUT"/eval_"${split}"_*.log | grep -c "finqa-judge\] ALARM" || true)
  if [ "$alarms" -gt 0 ]; then
    echo "[$TAG] $split: $alarms answers were scored 0 because the judge never answered; the result is invalid" >&2
    exit 1
  fi
  shard_dirs=()
  for i in "${!GPU_SETS[@]}"; do shard_dirs+=("$WORK/$TAG/$split/shard_$i"); done
  "$PY" "$REPRO/finqa_multitable/merge_eval_shards.py" "$WORK/$TAG/$split/merged" "${shard_dirs[@]}" | tee "$OUT/merge_$split.log"
  GPU_GROUPS_USED="$GPU_GROUPS" "$PY" - "$WORK/$TAG/$split/merged/protocol.json" "$OUT/${split}_protocol.json" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
protocol.update(host=".36 Quadro RTX 5000 (sm_75)", dtype="float16", vllm="0.22.1+cu129", attention_backend="TRITON_ATTN",
                gpu_memory_utilization=float(__import__("os").environ.get("GPU_UTIL", "0.80")),
                gpu_groups=__import__("os").environ.get("GPU_GROUPS_USED"))
json.dump(protocol, open(sys.argv[2], "w"), indent=2)
PY
  cp "$WORK/$TAG/$split/merged/result.json" "$OUT/$split.json"
done
rm -rf -- "${WORK:?}/$TAG"   # temporary: only the merged results and logs under $OUT are kept
echo "[$TAG] done $(date '+%F %T %Z'); results in $OUT"
