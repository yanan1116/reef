#!/usr/bin/env bash
# Full-benchmark check on .29 (user, 2026-10-19: "扩展到全量任务 ... 检查 smoke test 比如运行这 100 instances 看看缺少
# 哪些文件，哪些配置、网络限制等等"). All 20 SkillLearnBench tasks, 99 instances (weighted-gdp-calculation-2 has no
# tests), prepared by `prepare_tasks.py --out /mnt/disk1t/slb-all-tasks --all-families --keep-solution`.
#   phase 1 oracle: Harbor's oracle agent runs each solution/solve.sh once. A pass means the image builds, the
#                   network the build and the tests need is reachable, and the verifier can score 1.
#   phase 2 base:   Qwen3.5-4B base with the formal run's agent (prompt v2, no budget notice, 32k context, 60 turns,
#                   T=1, 4096 completion tokens, 1800 s, 180 s per command), once per instance, vLLM on GPU $GPU.
# The two phases run side by side (user, 2026-10-19: "把这 100 instances 都跑一遍，使用目前的方案，base model 看看是否顺利"),
# 8 trials each (docker address pools; the checkpoint follower may run 10 more).
# Results: /mnt/disk1t/slb-all-check/{oracle,base}/episodes.jsonl.
#
# ONLY (comma-separated instances, default all) and OUT re-check a subset, e.g. the rebuilt
# dependency-vulnerability-check (user, 2026-10-19).
#
# PHASES (default "oracle base"), REPEATS (base runs per instance, default 1) and BASE_NAME (default base) add more
# base runs without the oracle, e.g. PHASES=base REPEATS=2 BASE_NAME=base-r2 (user, 2026-10-19: "99 instances 的初步评估继续下去").
#
# PROMPT (default v2) and WARN (SLB_BUDGET_WARN, default off) pick the agent's prompt variant for the base phase
# (user, 2026-10-19: "找到一个 32K 上下文前提下，最好的 prompt 版本" for the 20-task training).
# ADAPTER (a LoRA adapter dir, served as "ckpt") and TASKS (default /mnt/disk1t/slb-all-tasks) evaluate a trained
# checkpoint on chosen instances, e.g. the unseen-task transfer test (user, 2026-10-20).
# Never edit this file while a run is executing it: bash reads it as it goes (a 2026-10-19 edit killed one).
#
#   [ONLY=a,b] [OUT=...] [PHASES="oracle base"] [REPEATS=1] [BASE_NAME=base] [PROMPT=v2] [WARN=] slb100_check_29.sh [GPU=1]
set -u
GPU=${1:-1}
HERE="$(cd "$(dirname "$0")" && pwd)"
TASKS=${TASKS:-/mnt/disk1t/slb-all-tasks}
OUT=${OUT:-/mnt/disk1t/slb-all-check}
ONLY=${ONLY-}
PHASES=${PHASES:-oracle base}
REPEATS=${REPEATS:-1}
BASE_NAME=${BASE_NAME:-base}
PROMPT=${PROMPT:-v2}
WARN=${WARN-}
ADAPTER=${ADAPTER-}
served=qwen35-4b; lora=()
[ -n "$ADAPTER" ] && { served=ckpt; lora=(--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules "ckpt=$ADAPTER"); }
V=/home/yanan/agents/rllm/.venv
MODEL=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
[ -e "$TASKS/split.json" ] || { echo "no $TASKS/split.json; run prepare_tasks.py first" >&2; exit 1; }
mkdir -p "$OUT"
# github-repo-analytics reads GitHub live; the token stays in this file and the environment only.
[ -r "$HOME/.config/slb/gh_token" ] && GH_TOKEN=$(cat "$HOME/.config/slb/gh_token") && export GH_TOKEN
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/run.log"; }

evaluate() {  # NAME, then extra env assignments
  local name=$1; shift
  mkdir -p "$OUT/$name"
  (cd "$HERE" && env SLB_TASKS="$TASKS" SLB_RUN="$OUT/$name" SLB_MODE=eval SLB_REPEATS=$REPEATS SLB_CONCURRENCY=8 \
    SLB_EVAL_TAG="$name" SLB_EVAL_SPLIT=all SLB_ONLY="$ONLY" "$@" \
    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
    run.py eval > "$OUT/$name/eval.log" 2>&1)
  log "$name done: $(wc -l < "$OUT/$name/episodes.jsonl" 2>/dev/null || echo 0) episodes"
}

if [[ " $PHASES " == *" oracle "* ]]; then
  log "phase 1 oracle start"
  REPEATS=1 evaluate oracle SLB_AGENT=oracle &
fi
[[ " $PHASES " == *" base "* ]] || { wait; log "done"; exit 0; }

port=$((18760 + GPU))
until [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 60; done
PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
  LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/nvidia/cu13/lib \
  setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port "$port" \
  --max-model-len 32768 --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' "${lora[@]}" \
  > "$OUT/vllm-gpu$GPU.log" 2>&1 < /dev/null &
for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 10; done
curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null || { log "vLLM on GPU $GPU not ready"; exit 1; }
log "phase 2 $BASE_NAME start (GPU $GPU, $REPEATS runs per instance, prompt $PROMPT, notices '${WARN:-off}', model $served ${ADAPTER})"
evaluate "$BASE_NAME" SLB_BASE_URL="http://127.0.0.1:$port/v1" SLB_MODEL=$served SLB_TEMPERATURE=1.0 SLB_ENABLE_THINKING=0 \
  SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=$PROMPT SLB_BUDGET_WARN="$WARN" SLB_OUTPUT_CHARS=8000 SLB_COMMAND_TIMEOUT_S=180 \
  SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768
P=$(ps -eo pid,args | grep "[v]llm serve.*--port $port" | awk '{print $1}' | head -1)
[ -n "$P" ] && kill "$P"
wait
log "done; vLLM stopped"
