#!/usr/bin/env bash
# Prompt x tool-output-cut ablation on .29 (user, 2026-10-14: approved after the .36 run failed on docker network
# pools). bf16, the same vLLM (rllm venv) and settings as base_grid_29.sh's nothink-T1 arm, which is the v1-8000
# cell (K=3), so only the other three cells run here, one after the other on ONE GPU (the last free one), so
# the other GPU stays with eval_ckpt_follow_29.sh (checkpoint evaluation has priority):
#   v2-4000, then v1-4000, then v2-8000
# Thinking off, T=1, all 50 instances, K runs per instance, 10 episodes at a time (with the follower's 12, below
# docker's ~30 default networks). Waits until the vLLM/SGLang comparison on this host is done.
# Results: /mnt/disk1t/slb-base-grid/<arm>/episodes.jsonl (next to the grid's arms).
#
#   ablation_29.sh [K=3]
set -u
K=${1:-3}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/mnt/disk1t/slb-base-grid
V=/home/yanan/agents/rllm/.venv
MODEL=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/grid.log"; }

log "ablation waiting for the engine comparison (/mnt/disk1t/slb-engine-compare/compare.log: done)"
until grep -qE "done; servers removed|not ready|not started" /mnt/disk1t/slb-engine-compare/compare.log 2>/dev/null; do sleep 60; done
gpu=""
until [ -n "$gpu" ]; do
  sleep 120
  # the LAST free GPU: eval_ckpt_follow_29.sh takes the first, so the two never pick the same one
  gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {g=$1} END {print g}')
done

serve() {  # GPU PORT
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/nvidia/cu13/lib \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port "$2" \
    --max-model-len 32768 --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > "$OUT/vllm-ablation-gpu$1.log" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$2/v1/models" >/dev/null && return 0; sleep 10; done
  log "ablation: vLLM on GPU $1 not ready"; return 1
}
arm() {  # NAME PORT PROMPT CHARS
  mkdir -p "$OUT/$1"
  log "arm $1 start (port $2, prompt $3, output cut $4, think 0, T 1.0, K $K)"
  (cd "$HERE" && SLB_TASKS=/mnt/disk1t/slb-sao-tasks SLB_RUN="$OUT/$1" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$2/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=1.0 SLB_ENABLE_THINKING=0 \
    SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=$3 SLB_OUTPUT_CHARS=$4 SLB_REPEATS=$K SLB_CONCURRENCY=10 \
    SLB_EVAL_TAG=$1 SLB_EVAL_SPLIT=all SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
    run.py eval > "$OUT/$1/eval.log" 2>&1)
  log "arm $1 done: $(wc -l < "$OUT/$1/episodes.jsonl") episodes"
}
port=$((18770 + gpu))
log "ablation on GPU $gpu (port $port)"
serve "$gpu" "$port" && { arm v2-4000 "$port" v2 4000; arm v1-4000 "$port" v1 4000; arm v2-8000 "$port" v2 8000; }
P=$(ps -eo pid,args | grep "[v]llm serve.*--port $port" | awk '{print $1}' | head -1)
[ -n "$P" ] && kill "$P"
log "ablation done; vLLM servers stopped"
