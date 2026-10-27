#!/usr/bin/env bash
# vLLM vs SGLang for the same evaluation (user, 2026-10-14: "vllm versus SGLang 对比一下"), both engines on one
# host and GPU type. Planned for .16; run on .29 (2x RTX 5000 Ada, bf16) once .16 went to the formal SAO run
# (user, 2026-10-14: "本机目前双卡空闲"). Host paths come from the environment (defaults are .16's). Qwen3.5-4B base, the training agent (prompt v1, 8000-char
# output cut, thinking off), all 50 instances of the split, K runs per arm, T=1 and T=0.
#   GPU 0: SGLang from the training image (the engine Reef samples training rollouts with)
#   GPU 1: vLLM v0.22.1 (docker), the evaluation engine
# On .16 it waits until the SAO smoke stack is gone. Results: $OUT/<engine>-<arm>/episodes.jsonl
#
#   engine_compare_16.sh [K=3]
#   OUT=/mnt/disk1t/slb-engine-compare MODEL_DIR=/mnt/disk1t/models/qwen35-4b-851bf6e8-eosfix \
#     TASKS=/mnt/disk1t/slb-sao-tasks VLLM_IMAGE=vllm/vllm-openai:latest engine_compare_16.sh 3   # on .29
set -u
K=${1:-3}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-/works/yanan/slb-engine-compare}
MODEL_DIR=${MODEL_DIR:-/home/yanan/reef-sao/models/qwen35-4b-851bf6e8-eosfix}
TASKS=${TASKS:-/works/yanan/slb-sao-tasks}
IMAGE=${IMAGE:-reef:sao-76fb540e}                  # the training image's SGLang
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.22.1}  # on .29 :latest is the same image id (0204ba447a71)
[ -f "$MODEL_DIR/config.json" ] && [ -f "$TASKS/split.json" ] || { echo "MODEL_DIR or TASKS not found on this host" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$OUT/compare.log"; }

log "waiting for the GPUs (SAO smoke stack reef-slb-stack)"
while docker ps --format '{{.Names}}' | grep -qx reef-slb-stack; do sleep 60; done
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
[ "$busy" = 0 ] || { log "GPUs still busy; not started"; exit 1; }

docker rm -f slb-sglang slb-vllm >/dev/null 2>&1
docker run -d --name slb-sglang --runtime nvidia --gpus '"device=0"' -p 127.0.0.1:18780:30000 \
  -v "$MODEL_DIR":/model:ro --ipc host --entrypoint python3 "$IMAGE" -m sglang.launch_server \
  --model-path /model --served-model-name qwen35-4b --host 0.0.0.0 --port 30000 --context-length 32768 \
  --mem-fraction-static 0.85 --max-running-requests 16 --tool-call-parser qwen3_coder > /dev/null
docker run -d --name slb-vllm --runtime nvidia --gpus '"device=1"' -p 127.0.0.1:18781:8000 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 -v "$MODEL_DIR":/model:ro "$VLLM_IMAGE" --model /model \
  --served-model-name qwen35-4b --max-model-len 32768 --gpu-memory-utilization 0.88 --max-num-seqs 16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 0, "video": 0}' > /dev/null
for port in 18780 18781; do
  for _ in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 10; done
  curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null || { log "server on $port not ready"; docker logs --tail 20 slb-sglang >> "$OUT/compare.log" 2>&1; exit 1; }
done
log "servers ready: SGLang :18780 (GPU 0), vLLM :18781 (GPU 1)"

arm() {  # NAME PORT TEMPERATURE
  local name=$1 port=$2 temp=$3
  mkdir -p "$OUT/$name"
  log "arm $name start (T $temp, K $K)"
  (cd "$HERE" && SLB_TASKS="$TASKS" SLB_RUN="$OUT/$name" SLB_MODE=eval \
    SLB_BASE_URL="http://127.0.0.1:$port/v1" SLB_MODEL=qwen35-4b SLB_TEMPERATURE=$temp SLB_ENABLE_THINKING=0 \
    SLB_MAX_COMPLETION_TOKENS=4096 SLB_REPEATS=$K SLB_CONCURRENCY=10 SLB_EVAL_TAG=$name SLB_EVAL_SPLIT=all \
    SLB_EPISODE_S=1800 SLB_MAX_TURNS=60 SLB_CONTEXT_TOKENS=32768 \
    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client --with-editable "$HERE" \
    run.py eval > "$OUT/$name/eval.log" 2>&1)
  log "arm $name done: $(wc -l < "$OUT/$name/episodes.jsonl") episodes"
}
{ arm sglang-T1 18780 1.0; arm sglang-T0 18780 0.0; } &
{ arm vllm-T1 18781 1.0; arm vllm-T0 18781 0.0; } &
wait
docker rm -f slb-sglang slb-vllm > /dev/null
log "done; servers removed"
