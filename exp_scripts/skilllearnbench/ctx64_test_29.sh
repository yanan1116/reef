#!/usr/bin/env bash
# 64K-context test on .29 (user, 2026-10-15: "如果上下文变为 64K，最终全局 success rate 有没有提升"). Qwen3.5-4B base
# (native context 262144), the formal run's agent (prompt v2, 8000-char cut, thinking off, T=1, 4096 completion
# tokens, 60 turns, 1800 s, 180 s per command) with only the context raised to 65536 (vLLM --max-model-len and
# SLB_CONTEXT_TOKENS). All 50 instances x K; one vLLM per GPU, each GPU runs half of the instances.
# Baseline: budget_test_29.sh round 1's v2 arm (same host and settings at 32768).
# Results: /mnt/disk1t/slb-budget-test/v2-<CTX/1024>k/gpu{0,1}/episodes.jsonl
# 100K run (user, 2026-10-15: ".29 再测试一个 100K 的"): CTX=102400 CONC=8. vLLM holds ~580k KV tokens per GPU,
# so fewer episodes per GPU than at 64K (10) keep preemption low; concurrency does not change the agent.
#
# GPUS picks the lanes (default "0 1"); the instances are split evenly across them. The 100K run uses GPUS=1 so GPU 0
# stays free for eval_ckpt_follow_29.sh (user, 2026-10-15: ".16 一旦保存了 checkpoint 立刻评测").
#
# PROMPT (default v2) and WARN (SLB_BUDGET_WARN, default empty = off) pick the agent variant; the result directory
# is named after them, e.g. v3-warn-100k (user, 2026-10-19: "100K 上下文 + v3-warn（提醒放在 50%/70%）").
#
#   [CTX=65536] [CONC=10] [GPUS="0 1"] [PROMPT=v2] [WARN=] ctx64_test_29.sh [K=3]
set -u
K=${1:-3}
CTX=${CTX:-65536}
CONC=${CONC:-10}
PROMPT=${PROMPT:-v2}
WARN=${WARN-}
NAME=$PROMPT${WARN:+-warn}-$((CTX / 1024))k
GPUS=${GPUS:-"0 1"}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/mnt/disk1t/slb-budget-test
V=/home/yanan/agents/rllm/.venv
MODEL=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/run.log"; }

for _ in $(seq 1 60); do
  busy=$(for g in $GPUS; do nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits; done | awk '$1 > 1000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
[ "$busy" = 0 ] || { log "$NAME test: GPUs busy; not started"; exit 1; }
mkdir -p "$OUT/$NAME"
python3 - /mnt/disk1t/slb-sao-tasks/split.json "$OUT/$NAME" $GPUS <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); names = s["test"] + s["train"]; gpus = sys.argv[3:]
for i, g in enumerate(gpus):
    open(f"{sys.argv[2]}/half{g}.txt", "w").write(",".join(names[i::len(gpus)]))
EOF

lane() {  # GPU
  local g=$1 port=$((18760 + $1)) dir="$OUT/$NAME/gpu$1"
  mkdir -p "$dir"
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$g VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/nvidia/cu13/lib \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port "$port" \
    --max-model-len $CTX --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > "$dir/vllm.log" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 10; done
  curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null || { log "$NAME: vLLM on GPU $g not ready"; return 1; }
  log "$NAME lane GPU $g start (context $CTX, K $K)"
  (cd "$HERE" && SLB_TASKS=/mnt/disk1t/slb-sao-tasks SLB_RUN="$dir" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$port/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=1.0 SLB_ENABLE_THINKING=0 \
    SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=$PROMPT SLB_BUDGET_WARN="$WARN" SLB_OUTPUT_CHARS=8000 SLB_COMMAND_TIMEOUT_S=180 \
    SLB_REPEATS=$K SLB_CONCURRENCY=$CONC SLB_EVAL_TAG=$NAME SLB_EVAL_SPLIT=all SLB_ONLY="$(cat "$OUT/$NAME/half$g.txt")" \
    SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=$CTX \
    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
    run.py eval > "$dir/eval.log" 2>&1)
  log "$NAME lane GPU $g done: $(wc -l < "$dir/episodes.jsonl") episodes"
  P=$(ps -eo pid,args | grep "[v]llm serve.*--port $port" | awk '{print $1}' | head -1)
  [ -n "$P" ] && kill "$P"
}
log "$NAME test start (context $CTX, concurrency $CONC per GPU, GPUs $GPUS, K $K)"
for g in $GPUS; do lane "$g" & done
wait
log "$NAME test done; vLLM servers stopped"
