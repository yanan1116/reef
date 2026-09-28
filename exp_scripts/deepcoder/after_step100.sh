#!/usr/bin/env bash
# Runs on .29 (2026-09-26 plan, agreed with the user): wait for step 100's T=0 x4 and
# T=1.0 x4 evaluations, apply step100_decision.py, and if the verdict is STOP:
#   1. halt the DeepCoder evaluation queue here (workers and feeder)
#   2. stop the lr5x DeepCoder training on .16 (scripts/stop_formal.sh)
#   3. delete its Reef checkpoint tree (state-b128/checkpoints), after checking that the
#      kept step-100 full bundle is complete; kept-checkpoints-b128 stays
#   4. start the FinQA SAO smoke run on .16, which starts the formal run if it passes
#   5. evaluate the FinQA base model on both .29 GPUs (greedy / T=0.7 x4)
# CONTINUE leaves everything running. Every step checks its precondition and stops the
# script, with the reason, when one fails.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NFS="$HERE/../results/deepcoder/eval-c32-lora-lr5x"
Q=/mnt/disk1t/sao-lr5x-eval/queue.txt
FQ=/home/yanan/agents/reef/exp_scripts/finqa
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 10.225.68.16)
DEADLINE=$(date -d '2026-09-27 14:00' +%s)
log() { echo "[after_step100 $(date '+%F %T %Z')] $*"; }
die() { log "STOPPED: $*"; exit 1; }
gpus_free() {  # $@ = nvidia-smi command prefix; true when every GPU holds < 1000 MiB
  [ -z "$("$@" --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1000 {print $1}')" ]
}

log "waiting for $NFS/lr5x_step_100_t0k4 and _t1k4"
until [ -f "$NFS/lr5x_step_100_t0k4/result.json" ] && [ -f "$NFS/lr5x_step_100_t1k4/result.json" ]; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || die "step 100 results not present by 2026-09-27 14:00"
  sleep 60
done
cd "$HERE"
python3 step100_decision.py lr5x_step_100 | tee "$NFS/step100_decision.txt"
verdict=${PIPESTATUS[0]}
case "$verdict" in
  10) log "CONTINUE: DeepCoder training keeps running; nothing else changed"; exit 0 ;;
  0) log "STOP: proceeding" ;;
  *) die "decision script could not decide (exit $verdict)" ;;
esac

touch "$Q.halt"
log "1. DeepCoder evaluation queue halted ($Q.halt)"

"${SSH[@]}" bash "$(cd "$HERE/.." && pwd)/scripts/stop_formal.sh"
sleep 10
"${SSH[@]}" "docker ps -a --format '{{.Names}}'" | grep -qx reef-sao-stack && die "reef-sao-stack still exists on .16 after stop_formal.sh"
gpus_free "${SSH[@]}" nvidia-smi || die "a .16 GPU still holds memory after stop_formal.sh"
log "2. DeepCoder training on .16 stopped; container gone, GPUs free"

"${SSH[@]}" 'set -e
K=/home/yanan/reef-sao-deepcoder/kept-checkpoints-b128
for p in hf actor critic; do test -d "$K/full/step_100/$p" || { echo "missing $K/full/step_100/$p"; exit 1; }; done
test -f "$K/adapters/hf_rollout_00099/hf/adapter_model.safetensors" || { echo "missing step-100 adapter"; exit 1; }
docker run --rm --entrypoint rm -v /home/yanan/reef-sao-deepcoder/state-b128:/state reef:sao-head-fix -rf /state/checkpoints
df -h /home | tail -1' || die "kept step-100 bundle incomplete or deletion failed; Reef tree left in place"
log "3. deleted .16 state-b128/checkpoints (kept step-100 bundle verified)"

STF_LOG=$FQ/../results/finqa/smoke_then_formal-$(date +%Y%m%dT%H%M%S).log
mkdir -p "$(dirname "$STF_LOG")"
timeout 60 "${SSH[@]}" "nohup setsid bash $FQ/smoke_then_formal.sh > $STF_LOG 2>&1 < /dev/null &" || true
sleep 30
grep -q "smoke: tag=" "$STF_LOG" 2>/dev/null || die "smoke_then_formal.sh did not start on .16; see $STF_LOG"
log "4. FinQA smoke started on .16; log $STF_LOG"

[ -x "$FQ/../.venv-finqa/bin/python" ] || die "FinQA venv missing on .29; base evaluations not started"
for i in $(seq 1 360); do gpus_free nvidia-smi && break; sleep 60; done
gpus_free nvidia-smi || die ".29 GPUs still busy after 6 h; FinQA base evaluations not started"
GPU=0 PORT=18051 MODE=base SAMPLING=greedy bash "$FQ/eval_finqa_checkpoint.sh" base_greedy - > "$FQ/../results/finqa-base_greedy.log" 2>&1 &
GPU=1 PORT=18052 MODE=base SAMPLING=t07k4 bash "$FQ/eval_finqa_checkpoint.sh" base_t07k4 - > "$FQ/../results/finqa-base_t07k4.log" 2>&1 &
log "5. FinQA base evaluations started on .29 GPU0 (greedy) and GPU1 (T=0.7 x4)"
wait
log "done: $(tail -n 2 "$FQ/../results/finqa-base_greedy.log" "$FQ/../results/finqa-base_t07k4.log" | tr '\n' ' ')"
