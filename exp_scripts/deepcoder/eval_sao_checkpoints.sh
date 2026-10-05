#!/usr/bin/env bash
# Evaluate several SAO LoRA adapters on DeepCoder test (687 tasks) from ONE vLLM
# server, sequentially, with the c32 protocol of every earlier evaluation: same
# rllm venv (vllm 0.22.1), same eval_base.py (greedy temp 0, top_p 1, seed 1234,
# max_tokens 16384, max_model_len 32768), same serve flags, EVAL_CONCURRENCY=32.
# Defaults: .29 GPU 0, as eval_sao_merged.sh (.24 is no longer used). Every host-specific value can be
# overridden: EXPECT_IP, GPU, BASE, WORK, NFS (results root), PORT.
#
# Multi-adapter successor of eval_sao_checkpoint.sh (which also ran a paired base
# arm). Written as a separate file because that script was executing when this
# was needed; once it has exited, this is the one entry point.
#
# Why LoRA serving and NOT a merge: SAO's adapter delta (~7e-6 per element) is
# below bf16 resolution against base weights (~1.8e-2); a bf16 merge keeps only
# 5-12% of it (measured 2026-09-24), so a merged model is effectively the base.
#
# usage: eval_sao_checkpoints.sh TAG=ADAPTER_DIR [TAG=ADAPTER_DIR ...]
#        add base=base to also evaluate the base on the same server.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 TAG=ADAPTER_DIR [...]" >&2; exit 2; }
GPU=${GPU:-0}
PORT=${PORT:-18041}
export EVAL_CONCURRENCY=32

DC=/home/yanan/agents/rllm/exp_scripts/deepcoder-run
EXPECT_IP=${EXPECT_IP:-10.225.68.29}
BASE=${BASE:-/home/yanan/reef-sao/models/Qwen3-4B-Instruct-2507}   # sha256-identical to the .29 base snapshot
WORK=${WORK:-/home/yanan/reef-sao-deepcoder/eval}                   # host-local disk: 16 GiB episodes files
NFS=${NFS:-$(cd "$(dirname "$0")/.." && pwd)/results/deepcoder/eval-c32-lora}

[ "$(hostname -I | grep -cw "$EXPECT_IP")" = 1 ] || { echo "expected host $EXPECT_IP, got $(hostname -I)" >&2; exit 2; }
test -f "$BASE/config.json" || { echo "base model not found: $BASE" >&2; exit 2; }
if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "port $PORT already serves something" >&2; exit 2; fi

TAGS=(); LORA_ARGS=(); R=0
for spec in "$@"; do
  tag=${spec%%=*}; dir=${spec#*=}
  [[ "$tag" =~ ^[a-z0-9_]+$ && "$tag" != "$spec" ]] || { echo "bad spec $spec (want TAG=DIR)" >&2; exit 2; }
  if [ -e "$NFS/$tag" ]; then echo "$NFS/$tag exists; refusing to overwrite" >&2; exit 2; fi
  if [ "$dir" != base ]; then
    test -f "$dir/adapter_model.safetensors" -a -f "$dir/adapter_config.json" || { echo "not a PEFT adapter dir: $dir" >&2; exit 2; }
    r=$(python3 -c "import json;print(json.load(open('$dir/adapter_config.json'))['r'])")
    [ "$r" -gt "$R" ] && R=$r
    LORA_ARGS+=("$tag=$dir")
  fi
  TAGS+=("$tag")
done
[ "$R" -gt 0 ] || R=32
SERVE_LORA=(); [ ${#LORA_ARGS[@]} -gt 0 ] && SERVE_LORA=(--lora-modules "${LORA_ARGS[@]}")

source "$DC/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU
export RLLM_HOME="$DC/runtime"
export PYTHONPATH="$DC:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LOG=$WORK/multi-$(date +%Y%m%dT%H%M%S)
mkdir -p "$LOG"
cleanup() { [ -n "${SPID:-}" ] && { kill -9 -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }; }
trap cleanup EXIT
cd /home/yanan/agents/rllm
setsid vllm serve "$BASE" --served-model-name base \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs "$EVAL_CONCURRENCY" \
  --enable-lora --max-lora-rank "$R" --max-loras 1 "${SERVE_LORA[@]}" \
  >"$LOG/server.log" 2>&1 &
SPID=$!
for ((i=0; i<300; i++)); do
  kill -0 "$SPID" 2>/dev/null || { echo "vLLM exited before ready; see $LOG/server.log" >&2; exit 1; }
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys;print('[models]',[m['id'] for m in json.load(sys.stdin)['data']])"

for tag in "${TAGS[@]}"; do
  model=$tag
  for spec in "$@"; do if [ "${spec%%=*}" = "$tag" ] && [ "${spec#*=}" = base ]; then model=base; fi; done
  if [ "$model" != base ]; then
    # The adapter must change the policy relative to the base on this server.
    python3 - "$PORT" "$model" <<'PY'
import json, sys, urllib.request
port, name = sys.argv[1:]
def lp(model):
    body = dict(model=model, messages=[{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
                temperature=0, max_tokens=64, logprobs=True, top_logprobs=1, seed=1234)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return [t["logprob"] for t in json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["logprobs"]["content"]]
b, s = lp("base"), lp(name)
n = min(len(b), len(s)); diff = max(abs(x - y) for x, y in zip(b[:n], s[:n]))
print(f"[adapter-check] {name}: max |logprob(base) - logprob({name})| over {n} tokens = {diff:.4g}")
assert diff > 0, f"{name} produced identical log-probs to base: adapter not applied"
PY
  fi
  OUT=$WORK/$tag/eval
  rm -rf -- "$OUT"; mkdir -p "$WORK/$tag"
  echo "[$tag] eval start $(date '+%F %T %Z')"
  python -u "$DC/eval_base.py" --url "http://127.0.0.1:$PORT/v1" --model "$model" --output "$OUT" >"$WORK/$tag/eval.log" 2>&1
  test -s "$OUT/result.json"
  mkdir -p "$NFS/$tag"; cp "$OUT/result.json" "$OUT/protocol.json" "$NFS/$tag/"
  rm -f "$OUT/episodes.jsonl"
  echo "[$tag] eval COMPLETE $(date '+%F %T %Z') $(python3 -c "import json;d=json.load(open('$NFS/$tag/result.json'));print(d.get('correct'),'/',d.get('total'),d.get('score'))")"
done
