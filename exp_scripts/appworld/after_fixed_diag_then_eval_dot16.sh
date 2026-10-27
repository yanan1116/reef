#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-24: diagnose ECHO first, keep .16 fully used). When the nodocs-fixed launcher exits
# (stopped by the epoch-5 rule or finished), free the GPUs and:
#   1. Held-out observation CE (echo_heldout_ce.py) on both GPUs in parallel: base, SAO, ECHO, nodocs and nodocs-fixed
#      whole-epoch adapters, on base-played dev episodes (held out) and training-task episodes. Writes DIAG_DONE.
#   2. The official .16 evaluation of the three ECHO arms, as the SAO run was evaluated: rllm eval_queue_follow.sh +
#      eval_gpu_pool.py (bf16, greedy, LoRA unmerged, 8 episodes per GPU, 1800 s per call), dev, test_normal_scen1,
#      test_challenge_scen1 of every kept whole epoch (global_step 6k, k >= 2). Base rows link to the SAO evaluation's.
#
#   after_fixed_diag_then_eval_dot16.sh FIXED_LAUNCHER_PID
set -u
PID=${1:?usage: $0 FIXED_LAUNCHER_PID}
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_ROOT=/works/yanan/sao-appworld-eval
DIAG=$EVAL_ROOT/echo-heldout-ce
CTRL=$EVAL_ROOT/echo-source-queue
SAO_EVAL=$EVAL_ROOT/appworld-qwen35-tp2-b15
MODEL=/works/yanan/models/qwen35-4b-851bf6e8-eosfix
VENV=/home/yanan/agents/rllm/.venv
RLLM=/home/yanan/agents/rllm/exp_scripts/appworld-prpo-run
STEPS_PER_EPOCH=6
[ -d /works/yanan ] || { echo "run this on .16" >&2; exit 1; }
mkdir -p "$DIAG" "$CTRL"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$EVAL_ROOT/echo-source.log"; }

log "waiting for the nodocs-fixed launcher $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
log "nodocs-fixed launcher exited: $(grep -ohE 'driver exited rc=[0-9]+' /works/yanan/reef-sao-appworld-qwen35-tp2-echo-nodocs-fixed/ops/launcher.log | tail -1)"
# The driver starts each AppWorld server in its own session; kill the training ones by their root.
ps -eo pid=,args= | awk '$0 ~ /appworld serve environment/ && $0 ~ /--root \/home\/yanan\/appworld-root( |$)/ {print $1}' \
  | xargs -r kill -9
docker rm -f reef-sao-stack > /dev/null 2>&1 && log "removed the reef-sao-stack container"
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [[ $busy == 0 ]] && break; sleep 10
done
(( busy == 0 )) || { log "GPUs still busy; diagnosis and evaluation NOT started"; exit 1; }

# 1. Held-out CE. Adapters: SAO 30/60; each ECHO arm's kept whole epochs 3, 5 and 10.
adapters=()
for spec in sao:reef-sao-appworld-qwen35-tp2 echo:reef-sao-appworld-qwen35-tp2-echo \
            nodocs:reef-sao-appworld-qwen35-tp2-echo-nodocs fixed:reef-sao-appworld-qwen35-tp2-echo-nodocs-fixed; do
  name=${spec%%:*} run=/works/yanan/${spec#*:}
  for step in 18 30 60; do
    dir=$run/kept-checkpoints/adapters/hf_rollout_$(printf %05d $((step - 1)))/hf
    [[ $name == sao && $step == 18 ]] && continue
    [[ -f $dir/adapter_model.safetensors ]] && adapters+=("$name@$step=$dir")
  done
done
log "held-out CE: base + ${#adapters[@]} adapters (${adapters[*]%%=*})"
half=$(( (${#adapters[@]} + 1) / 2 ))
pids=()
for g in 0 1; do
  args=()
  if [[ $g == 0 ]]; then part=("${adapters[@]:0:$half}"); else part=("${adapters[@]:$half}"); fi
  for a in "${part[@]}"; do args+=(--adapter "$a"); done
  CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib \
    "$VENV/bin/python" "$HERE/echo_heldout_ce.py" --model "$MODEL" --out "$DIAG/gpu$g.jsonl" \
    --episodes dev="$SAO_EVAL/dev/base/episodes.jsonl" \
    --episodes train=/home/yanan/agents/reef/exp_scripts/results/appworld/appworld-qwen35-tp2-b15-20261009T233225/episodes.jsonl \
    --rules paper="$HERE/echo_rules.json" --rules nodocs="$HERE/echo_rules_no_api_docs.json" \
    --rules errors="$HERE/echo_rules_errors_only.json" --max-episodes 40 "${args[@]}" \
    > "$DIAG/gpu$g.log" 2>&1 < /dev/null &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
python3 - "$DIAG" > "$DIAG/summary.txt" <<'EOF'
import json, statistics, sys, glob
rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1] + "/gpu*.jsonl")) for l in open(f)]
names = ["paper", "nodocs", "errors", "action"]
groups = {}
for r in rows:
    groups.setdefault((r["model"], r["set"]), []).append(r)
print(f"{'model':<12} {'set':<6} " + " ".join(f"{n + ' CE':>10}" for n in names) + "  episodes")
for (model, split), rs in sorted(groups.items(), key=lambda k: (k[0][1], k[0][0])):
    cells = []
    for n in names:
        v = [r[f"{n}_ce"] for r in rs if r.get(f"{n}_ce") is not None]
        cells.append(f"{statistics.mean(v):10.3f}" if v else f"{'-':>10}")
    print(f"{model:<12} {split:<6} " + " ".join(cells) + f"  {len(rs)}")
EOF
echo "rc=$rc" > "$DIAG/DIAG_DONE"
log "held-out CE done (rc=$rc): $DIAG/summary.txt"

# 2. The official evaluation of the three ECHO arms.
arms=(echo echo-nodocs echo-nodocs-fixed)
echo "# SPLIT OUT ADAPTER_SOURCE SELECT" > "$CTRL/queue.txt.new"
for split in dev test_normal_scen1 test_challenge_scen1; do
  for arm in "${arms[@]}"; do
    eval_dir=$EVAL_ROOT/appworld-qwen35-tp2-$arm-b15
    mkdir -p "$eval_dir/$split"
    ln -sfn "$SAO_EVAL/$split/base" "$eval_dir/$split/base"
    echo "$split $eval_dir/$split $eval_dir/adapters whole:$STEPS_PER_EPOCH" >> "$CTRL/queue.txt.new"
  done
done
for arm in "${arms[@]}"; do
  src=$EVAL_ROOT/appworld-qwen35-tp2-$arm-b15/adapters
  mkdir -p "$src"; n=0
  for dir in /works/yanan/reef-sao-appworld-qwen35-tp2-$arm/kept-checkpoints/adapters/hf_rollout_*; do
    step=$(( 10#${dir##*_} + 1 ))
    (( step % STEPS_PER_EPOCH == 0 && step > STEPS_PER_EPOCH )) || continue
    [[ -f $dir/hf/adapter_model.safetensors ]] && ln -sfn "$dir/hf" "$src/global_step_$step" && n=$((n + 1))
  done
  log "linked $n whole-epoch adapters of $arm as $src/global_step_N"
done
mv "$CTRL/queue.txt.new" "$CTRL/queue.txt"
for g in 0 1; do
  setsid nohup env EVAL_PYTHON=$VENV/bin/python APPWORLD_ROOT=/works/yanan/appworld-root-eval PYTHONPATH=/home/yanan/agents/rllm \
    LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib APPWORLD_LLM_TIMEOUT_S=1800 \
    "$RLLM/eval_queue_follow.sh" $g 1861$g "$CTRL" "$CTRL/queue.txt" 300 -- \
    --model-path "$MODEL" --episodes-per-gpu 8 \
    > /dev/null 2>&1 < /dev/null 9<&- &
done
sleep 3
log "started the .16 evaluation of ${arms[*]} (dev, test_normal_scen1, test_challenge_scen1; base linked from SAO)"
