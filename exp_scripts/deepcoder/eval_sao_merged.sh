#!/usr/bin/env bash
# Evaluate one SAO adapter MERGED into the base in bf16 (the way PRPO/SFT checkpoints
# were evaluated), with everything else identical to eval_sao_checkpoints.sh (the
# LoRA-served path): same rllm venv vLLM 0.22.1, same serve flags, same eval_base.py,
# EVAL_CONCURRENCY=32. Only the adapter loading differs, so a gap between the two is
# the bf16 merge's effect.
#
# Sampling via env, as eval_priority_worker.sh passes it:
#   T0K4: EVAL_TEMPERATURE=0 EVAL_ATTEMPTS=4 (seed 1234)   T1K4: EVAL_TEMPERATURE=1.0 EVAL_ATTEMPTS=4 EVAL_SEED=none
#
# usage: eval_sao_merged.sh TAG ADAPTER_DIR   (env GPU, PORT, BASE, WORK, NFS, EXPECT_IP as eval_sao_checkpoints.sh)
set -euo pipefail
TAG=${1:?usage: $0 TAG ADAPTER_DIR}
ADAPTER=${2:?usage: $0 TAG ADAPTER_DIR}
GPU=${GPU:-0}
PORT=${PORT:-18043}
export EVAL_CONCURRENCY=32
DC=/home/yanan/agents/gitlab/tail/rllm/deepcoder-run
EXPECT_IP=${EXPECT_IP:-10.225.68.29}
BASE=${BASE:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
WORK=${WORK:-/mnt/disk1t/sao-lr5x-eval/work}
NFS=${NFS:-$(cd "$(dirname "$0")/.." && pwd)/results/deepcoder/eval-c32-lora-lr5x}

[ "$(hostname -I | grep -cw "$EXPECT_IP")" = 1 ] || { echo "expected host $EXPECT_IP" >&2; exit 2; }
[ -e "$NFS/$TAG" ] && { echo "$NFS/$TAG exists; refusing to overwrite" >&2; exit 2; }
test -f "$ADAPTER/adapter_model.safetensors" || { echo "not a PEFT adapter: $ADAPTER" >&2; exit 2; }
if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "port $PORT in use" >&2; exit 2; fi

source "$DC/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU
export RLLM_HOME="$DC/runtime"
export PYTHONPATH="$DC:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$WORK/$TAG"
MODEL=$WORK/$TAG/merged
rm -rf -- "$MODEL"
CUDA_VISIBLE_DEVICES= python - "$BASE" "$ADAPTER" "$MODEL" > "$WORK/$TAG/merge.log" 2>&1 <<'PY'
import shutil, sys, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM
base, adapter, out = sys.argv[1:]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
ref = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("mlp.down_proj.weight")}
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
changed = [(model.get_parameter(n) != w).float().mean().item() for n, w in ref.items()]
print(f"merged {adapter}: {100 * sum(changed) / len(changed):.2f}% of down_proj elements changed in bf16")
model.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
    shutil.copy2(Path(base) / f, Path(out) / f)
PY
tail -1 "$WORK/$TAG/merge.log"
cmp "$BASE/tokenizer.json" "$MODEL/tokenizer.json"

cleanup() { [ -n "${SPID:-}" ] && { kill -9 -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }; rm -rf -- "$MODEL"; }
trap cleanup EXIT
cd /home/yanan/agents/rllm
setsid vllm serve "$MODEL" --served-model-name merged --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs "$EVAL_CONCURRENCY" > "$WORK/$TAG/server.log" 2>&1 &
SPID=$!
for ((i=0; i<300; i++)); do
  kill -0 "$SPID" 2>/dev/null || { echo "[$TAG] vLLM exited before ready" >&2; exit 1; }
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null
echo "[$TAG] eval start $(date '+%F %T %Z') T=${EVAL_TEMPERATURE:-0} attempts=${EVAL_ATTEMPTS:-1} seed=${EVAL_SEED:-1234}"
python -u "$DC/eval_base.py" --url "http://127.0.0.1:$PORT/v1" --model merged --output "$WORK/$TAG/eval" > "$WORK/$TAG/eval.log" 2>&1
mkdir -p "$NFS/$TAG"; cp "$WORK/$TAG/eval/result.json" "$WORK/$TAG/eval/protocol.json" "$WORK/$TAG/merge.log" "$NFS/$TAG/"
rm -f "$WORK/$TAG/eval/episodes.jsonl"
echo "[$TAG] eval COMPLETE $(date '+%F %T %Z') $(tail -1 "$WORK/$TAG/eval.log")"
