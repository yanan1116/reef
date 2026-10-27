#!/usr/bin/env bash
# Base-model grid on .29 (user, 2026-10-14: "多跑几次 greedy versus T=1, thinking mode on versus off"):
# Qwen3.5-4B, the training agent, all 50 SkillLearnBench instances of the split (41 train + 9 test),
# K runs per arm. One vLLM per GPU; each GPU runs its arms one after the other.
#   GPU 0: think off, T=1 and T=0     GPU 1: think on, T=1 and T=0
# Thinking-on arms get 8192 completion tokens per call (the thinking text needs room); off arms keep the
# training setting 4096. Results: $OUT/<arm>/episodes.jsonl.
#
#   base_grid_29.sh [K=3] [LANES="0 1"]   (start one lane now and the other later: LANES=1, then LANES=0)
set -u
K=${1:-3}
LANES=${2:-0 1}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/mnt/disk1t/slb-base-grid
V=/home/yanan/agents/rllm/.venv
MODEL=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/grid.log"; }

serve() {  # GPU PORT -> starts vLLM, waits until ready
  local gpu=$1 port=$2
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/nvidia/cu13/lib \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port "$port" \
    --max-model-len 32768 --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > "$OUT/vllm-gpu$gpu.log" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null && return 0; sleep 10; done
  log "vLLM on GPU $gpu did not become ready"; return 1
}

arm() {  # NAME PORT THINK TEMPERATURE MAX_COMPLETION
  local name=$1 port=$2 think=$3 temp=$4 maxc=$5
  mkdir -p "$OUT/$name"
  log "arm $name start (port $port, think $think, T $temp, max completion $maxc, K $K)"
  (cd "$HERE" && SLB_TASKS=/mnt/disk1t/slb-sao-tasks SLB_RUN="$OUT/$name" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$port/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=$temp SLB_ENABLE_THINKING=$think \
    SLB_MAX_COMPLETION_TOKENS=$maxc SLB_REPEATS=$K SLB_CONCURRENCY=10 SLB_EVAL_TAG=$name SLB_EVAL_SPLIT=all \
    SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
    run.py eval > "$OUT/$name/eval.log" 2>&1)
  log "arm $name done: $(wc -l < "$OUT/$name/episodes.jsonl") episodes"
}

lane0() { serve 0 18770 && { arm nothink-T1 18770 0 1.0 4096; arm nothink-T0 18770 0 0.0 4096; }; }
lane1() { serve 1 18771 && { arm think-T1 18771 1 1.0 8192; arm think-T0 18771 1 0.0 8192; }; }
log "grid start K=$K lanes=$LANES"
for lane in $LANES; do "lane$lane" & done
wait
log "grid done"
