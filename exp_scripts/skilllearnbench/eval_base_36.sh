#!/usr/bin/env bash
# fp16-vs-bf16 check on .36 (user, 2026-10-15: "在 .36 上用 fp16 把 v2 的 base 评估跑一遍，和 .29 的 bf16 结果对比。
# 多跑几次"; docker on .36 authorized the same day). Qwen3.5-4B base, the formal run's eval protocol exactly as
# eval_ckpt_follow_29.sh runs it (9 test instances, prompt v2, 8000-char cut, thinking off, 4096 completion
# tokens, 32k context, 60 turns, 1800 s, 180 s per command), K runs per instance at T=1 and at T=0.
# .36 is Turing: vLLM fp16 with TRITON_ATTN (as the AppWorld .36 evals). Four GPUs, one vLLM each:
#   GPU 0/1: T=1, half of the instances each        GPU 2/3: T=0, half each
# 4 episodes per GPU = 16 trials at most (docker's default address pools allow ~30 networks).
# .36 only reads .29 (read-only mounts); harness and tasks are copied to $ROOT and results stay there.
#
#   eval_base_36.sh [K=8]
set -u
K=${1:-8}
ROOT=/home/yanan/eval36/slb-base36
SRC=/home/yanan/agents/reef/exp_scripts/skilllearnbench          # read-only sshfs of .29
TASKS_SRC=/home/yanan/mnt/dot29-disk1t/slb-sao-tasks
MODEL=/home/yanan/models/qwen35-4b-851bf6e8-eosfix
V=/home/yanan/eval36/env/venv-finqa-cu129
[ "$(hostname)" = tai-SYS-4029GP-TRT2 ] || { echo "run this on .36" >&2; exit 1; }
[ -e "$ROOT/out" ] && { echo "$ROOT/out exists; use a fresh ROOT" >&2; exit 1; }
mkdir -p "$ROOT/out" "$ROOT/harness" "$ROOT/tasks"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$ROOT/out/run.log"; }
cp "$SRC/run.py" "$SRC/pyproject.toml" "$ROOT/harness/" && cp -r "$SRC/harness" "$ROOT/harness/"
cp -rL "$TASKS_SRC/." "$ROOT/tasks/"
python3 - "$ROOT/tasks/split.json" "$ROOT/out" <<'EOF'
import json, sys
test = json.load(open(sys.argv[1]))["test"]
open(f"{sys.argv[2]}/half0.txt", "w").write(",".join(test[0::2]))
open(f"{sys.argv[2]}/half1.txt", "w").write(",".join(test[1::2]))
EOF
log "copied harness (agent.py $(sha256sum "$ROOT/harness/harness/agent.py" | cut -c1-12)) and tasks; K=$K"

for g in 0 1 2 3; do
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$g VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port $((18810 + g)) \
    --dtype float16 --attention-backend TRITON_ATTN --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' > "$ROOT/out/vllm-gpu$g.log" 2>&1 < /dev/null &
done
for g in 0 1 2 3; do
  for _ in $(seq 1 120); do curl -sf -m 3 "http://127.0.0.1:$((18810 + g))/v1/models" >/dev/null && break; sleep 10; done
  curl -sf -m 3 "http://127.0.0.1:$((18810 + g))/v1/models" >/dev/null || { log "vLLM on GPU $g not ready"; exit 1; }
done
log "4 vLLM servers ready (fp16, TRITON_ATTN)"

run() {  # GPU HALF TEMPERATURE
  local g=$1 half=$2 t=$3 dir="$ROOT/out/T$3/gpu$1"
  mkdir -p "$dir"
  (cd "$ROOT/harness" && SLB_TASKS="$ROOT/tasks" SLB_RUN="$dir" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$((18810 + g))/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=$t SLB_ENABLE_THINKING=0 \
    SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=v2 SLB_OUTPUT_CHARS=8000 SLB_COMMAND_TIMEOUT_S=180 \
    SLB_REPEATS=$K SLB_CONCURRENCY=4 SLB_EVAL_TAG="base36-T$t" SLB_EVAL_SPLIT=test SLB_ONLY="$(cat "$ROOT/out/half$half.txt")" \
    SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
    uv run --no-project --python 3.12 --python-preference only-managed --with "reef-eval[harbor]" --with reef-client \
    --with-editable "$ROOT/harness" run.py eval > "$dir/eval.log" 2>&1)
  log "T=$t gpu $g done: $(wc -l < "$dir/episodes.jsonl" 2>/dev/null || echo 0) episodes"
}
run 0 0 1 & run 1 1 1 & run 2 0 0 & run 3 1 0 &
wait
for g in 0 1 2 3; do
  P=$(ps -eo pid,args | grep "[v]llm serve.*--port $((18810 + g))" | awk '{print $1}' | head -1)
  [ -n "$P" ] && kill "$P"
done
log "done; vLLM servers stopped"
