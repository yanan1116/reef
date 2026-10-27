#!/usr/bin/env bash
# Follower on .29: evaluate the formal SkillLearnBench SAO run (.16 /works/yanan/slb-sao-formal) on the 9 test
# instances, base first and then every finished epoch (user, 2026-10-14). Each item: T=1 x K with the training
# agent exactly as trained (prompt v2, 8000-char cut, thinking off, 4096 completion tokens, 32k context, 60 turns,
# 1800 s, 180 s per command), served by vLLM (rllm venv, bf16) with the LoRA unmerged.
# Greedy (T=0) is cancelled (user, 2026-10-18: "取消 greedy 评测，只用 T=1 ... T=1 运行 8 次"): greedy decoding
# loops on one command until max_turns, and its K runs are near-identical. The old T=1 x4 + T=0 x4 results stay
# in /mnt/disk1t/slb-sao-formal-eval; this protocol writes to $OUT and re-evaluates base and every epoch.
# The epoch-N adapter is the step published at epoch N's end: run/epochs.jsonl "end" records the training release
# count R, and sidecar.py kept that step as kept/adapters/hf_rollout_{R-1}. .16 is read through .29's read-only
# sshfs mount. One follower per GPU; an item is claimed with an atomic mkdir of $OUT/ITEM/claim, so two
# followers never take the same item. Concurrency 10 per GPU keeps the host at 20 trials (docker address pools).
# Exits when $OUT/STOP exists.
#
# The full 20-task run (user, 2026-10-20) uses the same follower through SRC, OUT, TASKS and the training agent's
# prompt (PROMPT, WARN): SRC=/home/yanan/mnt/dot16-works/slb-sao-full20 OUT=/mnt/disk1t/slb-sao-full20-eval
# TASKS=/mnt/disk1t/slb-sao-tasks-all PROMPT=v3 WARN=0.5,0.7 (notices in the tool result, the agent default).
#
#   [SRC=...] [OUT=...] [TASKS=...] [PROMPT=v2] [WARN=] eval_ckpt_follow_29.sh GPU [K=8]
set -u
GPU=$1
K=${2:-8}
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=${SRC:-/home/yanan/mnt/dot16-works/slb-sao-formal}
OUT=${OUT:-/mnt/disk1t/slb-sao-formal-eval-t1}
TASKS=${TASKS:-/mnt/disk1t/slb-sao-tasks}
PROMPT=${PROMPT:-v2}
WARN=${WARN-}
V=/home/yanan/agents/rllm/.venv
MODEL=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
mkdir -p "$OUT"
# github-repo-analytics (full 20-task set) reads GitHub live; the token stays in this file and the environment only.
[ -r "$HOME/.config/slb/gh_token" ] && GH_TOKEN=$(cat "$HOME/.config/slb/gh_token") && export GH_TOKEN
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/follow.log"; }

items() {  # prints "ITEM ADAPTER_DIR" for every item whose adapter exists ("base -" for the base model)
  echo "base -"
  python3 - "$SRC/run/epochs.jsonl" "$SRC/kept/adapters" <<'EOF'
import json, os, sys
path, adapters = sys.argv[1:]
if not os.path.exists(path):
    sys.exit()
for line in open(path):
    r = json.loads(line)
    if r.get("event") != "end" or not r.get("training_releases"):
        continue
    adapter = f"{adapters}/hf_rollout_{r['training_releases'] - 1:05d}/hf"
    if os.path.exists(f"{adapter}/adapter_model.safetensors"):
        print(f"epoch{r['epoch']}", adapter)
EOF
}

claim_next() {  # claims the first open item; prints "ITEM ADAPTER_DIR"
  local item adapter
  while read -r item adapter; do
    [ -e "$OUT/$item/DONE" ] || [ -e "$OUT/$item/FAILED" ] && continue
    mkdir -p "$OUT/$item"
    mkdir "$OUT/$item/claim" 2>/dev/null || continue
    echo "$item $adapter"; return
  done < <(items)
}

port=$((18790 + GPU))
log "follower started (pid $$, GPU $GPU, K $K, src $SRC, tasks $TASKS, prompt $PROMPT, notices '${WARN:-off}')"
while [ ! -e "$OUT/STOP" ]; do
  item=""; adapter=""
  read -r item adapter < <(claim_next)
  if [ -z "${item:-}" ]; then sleep 60; continue; fi
  until [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 60; done
  served=qwen35-4b; lora=()
  if [ "$adapter" != "-" ]; then served=ckpt; lora=(--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules "ckpt=$adapter"); fi
  echo "$adapter" > "$OUT/$item/adapter.txt"
  log "$item: GPU $GPU, adapter $adapter"
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/nvidia/cu13/lib \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port "$port" \
    --max-model-len 32768 --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' "${lora[@]}" \
    > "$OUT/$item/vllm.log" 2>&1 < /dev/null &
  ready=0
  for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null && { ready=1; break; }; sleep 10; done
  vpid=$(ps -eo pid,args | grep "[v]llm serve.*--port $port" | awk '{print $1}' | head -1)
  if [ "$ready" = 1 ]; then
    (cd "$HERE" && SLB_TASKS="$TASKS" SLB_RUN="$OUT/$item/T1" SLB_MODE=eval \
      SLB_BASE_URL="http://127.0.0.1:$port/v1" SLB_MODEL=$served SLB_TEMPERATURE=1 SLB_ENABLE_THINKING=0 \
      SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=$PROMPT SLB_BUDGET_WARN="$WARN" SLB_OUTPUT_CHARS=8000 SLB_COMMAND_TIMEOUT_S=180 \
      SLB_REPEATS=$K SLB_CONCURRENCY=10 SLB_EVAL_TAG="$item-T1" SLB_EVAL_SPLIT=test \
      SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
      uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
      run.py eval > "$OUT/$item/eval-T1.log" 2>&1)
    log "$item T=1: $(wc -l < "$OUT/$item/T1/episodes.jsonl" 2>/dev/null || echo 0) episodes"
    touch "$OUT/$item/DONE"
  else
    echo x >> "$OUT/$item/attempts"
    if [ "$(wc -l < "$OUT/$item/attempts")" -ge 2 ]; then
      touch "$OUT/$item/FAILED"; log "$item: vLLM not ready twice (see $OUT/$item/vllm.log); marked FAILED"
    else
      log "$item: vLLM on GPU $GPU not ready (see $OUT/$item/vllm.log); retry later"
      rmdir "$OUT/$item/claim"
    fi
  fi
  [ -n "$vpid" ] && kill "$vpid"
  sleep 30
done
log "STOP found, exiting (GPU $GPU)"
