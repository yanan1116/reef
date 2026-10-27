#!/usr/bin/env bash
# Prompt x tool-output-cut ablation on .36 (user, 2026-10-14: "把 prompt/截断的消融挪到 .36 跑", "充分利用 .36").
# Qwen3.5-4B base, the training agent, thinking off, T=1, all 50 instances of the split, K runs per instance.
#   arms: v1-8000 (training default), v1-4000, v2-8000, v2-4000   (SLB_PROMPT_VERSION x SLB_OUTPUT_CHARS)
# .36 has 8 Turing GPUs (fp16 only, TRITON_ATTN as the AppWorld .36 evals): one vLLM per GPU, two GPUs per
# arm, each GPU running half of the arm's instances. All four arms run here, so they compare with each other;
# fp16 results are not mixed with the bf16 runs on .29/.16.
# .36 reads .29 only: the harness and tasks are copied from the read-only mounts to .36's local disk, and
# results stay in $ROOT. uv uses its managed CPython: .36's /usr/local python3.12 has no _sqlite3, which
# reef-eval's result store needs.
#
#   ablation_36.sh [K=5] [PER_GPU=4]
set -u
K=${1:-5}
PER_GPU=${2:-4}
ROOT=/home/yanan/eval36/slb
SRC=/home/yanan/agents/reef/exp_scripts/skilllearnbench          # read-only sshfs of .29
TASKS_SRC=/home/yanan/mnt/dot29-disk1t/slb-sao-tasks             # read-only mount of .29's /mnt/disk1t
MODEL=/home/yanan/models/qwen35-4b-851bf6e8-eosfix
V=/home/yanan/eval36/env/venv-finqa-cu129
[ "$(hostname)" = tai-SYS-4029GP-TRT2 ] || { echo "run this on .36" >&2; exit 1; }
mkdir -p "$ROOT/out"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$ROOT/out/ablation.log"; }

rm -rf "$ROOT/harness" "$ROOT/tasks"
mkdir -p "$ROOT/harness" "$ROOT/tasks"
cp "$SRC/run.py" "$SRC/pyproject.toml" "$ROOT/harness/" && cp -r "$SRC/harness" "$ROOT/harness/"
cp -rL "$TASKS_SRC/." "$ROOT/tasks/"
log "copied harness ($(sha256sum "$ROOT/harness/harness/agent.py" | cut -c1-12)) and $(ls -d "$ROOT"/tasks/*/ | wc -l) tasks; K=$K per_gpu=$PER_GPU"

# Half of the 50 instances per GPU of an arm, split alternately so both halves mix families.
python3 - "$ROOT/tasks/split.json" "$ROOT/out" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); names = s["test"] + s["train"]
open(f"{sys.argv[2]}/half0.txt", "w").write(",".join(names[0::2]))
open(f"{sys.argv[2]}/half1.txt", "w").write(",".join(names[1::2]))
EOF

for g in 0 1 2 3 4 5 6 7; do
  curl -sf -m 3 "http://127.0.0.1:$((18800 + g))/v1/models" >/dev/null && continue   # already serving (a rerun)
  PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$g VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
    setsid nohup "$V/bin/vllm" serve "$MODEL" --served-model-name qwen35-4b --host 127.0.0.1 --port $((18800 + g)) \
    --dtype float16 --attention-backend TRITON_ATTN --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' > "$ROOT/out/vllm-gpu$g.log" 2>&1 < /dev/null &
done
for g in 0 1 2 3 4 5 6 7; do
  for _ in $(seq 1 120); do curl -sf -m 3 "http://127.0.0.1:$((18800 + g))/v1/models" >/dev/null && break; sleep 10; done
  curl -sf -m 3 "http://127.0.0.1:$((18800 + g))/v1/models" >/dev/null || { log "vLLM on GPU $g not ready; see vllm-gpu$g.log"; exit 1; }
done
log "8 vLLM servers ready (fp16, TRITON_ATTN)"

run() {  # ARM GPU HALF PROMPT CHARS
  local arm=$1 g=$2 half=$3 prompt=$4 chars=$5 dir="$ROOT/out/$1/gpu$2"
  mkdir -p "$dir"
  (cd "$ROOT/harness" && SLB_TASKS="$ROOT/tasks" SLB_RUN="$dir" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$((18800 + g))/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=1.0 SLB_ENABLE_THINKING=0 \
    SLB_MAX_COMPLETION_TOKENS=4096 SLB_PROMPT_VERSION=$prompt SLB_OUTPUT_CHARS=$chars SLB_REPEATS=$K \
    SLB_CONCURRENCY=$PER_GPU SLB_EVAL_TAG=$arm SLB_EVAL_SPLIT=all SLB_ONLY="$(cat "$ROOT/out/half$half.txt")" \
    SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
    uv run --no-project --python 3.12 --python-preference only-managed --with "reef-eval[harbor]" --with reef-client --with-editable "$ROOT/harness" \
    run.py eval > "$dir/eval.log" 2>&1)
  log "arm $arm gpu $g done: $(wc -l < "$dir/episodes.jsonl") episodes"
}
log "start"
run v1-8000 0 0 v1 8000 & run v1-8000 1 1 v1 8000 &
run v1-4000 2 0 v1 4000 & run v1-4000 3 1 v1 4000 &
run v2-8000 4 0 v2 8000 & run v2-8000 5 1 v2 8000 &
run v2-4000 6 0 v2 4000 & run v2-4000 7 1 v2 4000 &
wait
log "all arms done"
