#!/usr/bin/env bash
# Budget-notice test on .29 (user, 2026-10-15: "加上这个 在 .29 测试一下 看看是否有效果 是否能显著提前收尾"):
# Qwen3.5-4B base, the formal run's agent and settings (prompt v2, 8000-char cut, thinking off, T=1, 4096 completion
# tokens, 32k context, 60 turns, 1800 s, 180 s per command), all 50 instances, K runs each, two arms side by side:
#   GPU 0: v2           (no notice; the formal run's agent)
#   GPU 1: v2-warn      (SLB_BUDGET_WARN=0.6,0.8: notices at 60% and 80% of the context, plus 5 turns before the end)
# 10 episodes per GPU (20 trials at most). Results: /mnt/disk1t/slb-budget-test/<arm>/episodes.jsonl
#
# Round 2 (user, 2026-10-15: "提醒提前到 50% ... system prompt 里面加粗一个提醒"): ROUND=2 runs prompt v3 (v2 + a bold
# standing budget reminder) with and without notices at 50% and 70%; round 1's v2 / v2-warn arms are the baseline.
#
#   budget_test_29.sh [K=3]            # round 1: v2, v2-warn (0.6,0.8)
#   ROUND=2 budget_test_29.sh [K=3]    # round 2: v3, v3-warn (0.5,0.7)
set -u
K=${1:-3}
ROUND=${ROUND:-1}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/mnt/disk1t/slb-budget-test
V=/home/yanan/agents/rllm/.venv
MODEL=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/run.log"; }

serve() {  # GPU PORT
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/nvidia/cu13/lib \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port "$2" \
    --max-model-len 32768 --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > "$OUT/vllm-gpu$1.log" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$2/v1/models" >/dev/null && return 0; sleep 10; done
  log "vLLM on GPU $1 not ready"; return 1
}
arm() {  # NAME GPU WARN PROMPT
  local name=$1 gpu=$2 warn=$3 prompt=$4 port=$((18760 + $2))
  serve "$gpu" "$port" || return 1
  mkdir -p "$OUT/$name"
  log "arm $name start (GPU $gpu, prompt $prompt, SLB_BUDGET_WARN='$warn', K $K)"
  (cd "$HERE" && SLB_TASKS=/mnt/disk1t/slb-sao-tasks SLB_RUN="$OUT/$name" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$port/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=1.0 SLB_ENABLE_THINKING=0 \
    SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=$prompt SLB_OUTPUT_CHARS=8000 SLB_COMMAND_TIMEOUT_S=180 \
    SLB_BUDGET_WARN="$warn" SLB_REPEATS=$K SLB_CONCURRENCY=10 SLB_EVAL_TAG=$name SLB_EVAL_SPLIT=all \
    SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
    run.py eval > "$OUT/$name/eval.log" 2>&1)
  log "arm $name done: $(wc -l < "$OUT/$name/episodes.jsonl") episodes"
  P=$(ps -eo pid,args | grep "[v]llm serve.*--port $port" | awk '{print $1}' | head -1)
  [ -n "$P" ] && kill "$P"
}
log "start round $ROUND"
if [ "$ROUND" = 1 ]; then
  arm v2 0 "" v2 &
  arm v2-warn 1 "0.6,0.8" v2 &
else
  arm v3 0 "" v3 &
  arm v3-warn 1 "0.5,0.7" v3 &
fi
wait
log "done; vLLM servers stopped"
