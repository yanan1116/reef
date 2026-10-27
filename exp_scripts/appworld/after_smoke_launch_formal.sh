#!/usr/bin/env bash
# Run ON .16 after launch_16_qwen35_2gpu.sh smoke: wait for the smoke to end, check it, and if it passed,
# remove its stack and launch the formal run (user, 2026-10-09: "如果 .16 smoke test 正常 那么直接进入正式的训练").
#
#   bash appworld/after_smoke_launch_formal.sh SMOKE_RUN_ROOT SMOKE_TAG
#
# Export the smoke's LAYOUT (2gpu or tp2) so the formal run uses the same layout.
#
# "Normal" means: the driver exited 0 after training every expected step (progress.txt = budget/batch, no
# drain WARNING), and the stack log has no OutOfMemoryError, training_failed or failed health check. The
# attention backend TE selected and the peak GPU memory are recorded for the report; they do not block.
set -uo pipefail
RUN_ROOT=${1:?smoke run root}; TAG=${2:?smoke tag}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPRO="$(cd "$HERE/.." && pwd)"
OUT="$REPRO/results/appworld/$TAG"; LOG="$RUN_ROOT/ops/after_smoke.log"; STACK="$RUN_ROOT/state/stack/slime-driver.log"
ts() { TZ=America/New_York date '+%F %T %Z'; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }
PID=$(cat "$RUN_ROOT/ops/launcher.pid")
log "waiting for smoke launcher $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 20
# run_appworld.sh writes "driver exited rc=N" to the launcher log, not to the driver log.
rc=$(grep -ohE "driver exited rc=[0-9]+" "$OUT/driver.log" "$RUN_ROOT/ops/launcher.log" 2>/dev/null | tail -1)
# The driver's closing "trained: N/M steps committed" line; progress.txt can lag the last commit by one step.
trained=$(grep -oE "^trained: [0-9]+/" "$OUT/driver.log" | tail -1 | grep -oE "[0-9]+" || echo 0)
trained=${trained:-0}
published=$(grep -cE "^sync: step [0-9]+ published" "$OUT/driver.log")
warn=$(grep -cE "WARNING: only|too long, not retried" "$OUT/driver.log")
oom=$(grep -cE "OutOfMemoryError|training_failed|failed its health check" "$STACK" 2>/dev/null)
fa=$(grep -ciE "FlashAttention" "$STACK" 2>/dev/null)
peak=$(awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/ && $i+0 > m) m=$i} END {print m+0}' "$RUN_ROOT/ops/gpu-mem.log" 2>/dev/null)
# The ECHO arm (SAO_ECHO=1) must also show its term in the actor's training metrics.
echo_ok=1
if [ "${SAO_ECHO:-0}" = 1 ]; then
  echo_lines=$(cat "$STACK" "$RUN_ROOT/state/reef.log" 2>/dev/null | grep -c "echo_ce")
  [ "$echo_lines" -ge 1 ] || echo_ok=0
  log "echo_ce lines in the stack logs: $echo_lines"
fi
log "smoke result: $rc trained_steps=$trained published=$published warnings=$warn oom_or_failed=$oom flashattention_mentions=$fa peak_gpu_mem_mib=$peak"
if [ "$rc" = "driver exited rc=0" ] && [ "$trained" -ge 3 ] && [ "$published" -ge 2 ] && [ "$warn" = 0 ] && [ "$oom" = 0 ] && [ "$echo_ok" = 1 ]; then
  log "smoke passed; removing the smoke stack and launching the formal run"
  docker rm -f reef-sao-stack >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
    [ "$busy" = 0 ] && break; sleep 5
  done
  pgrep -f "appworld serve" | xargs -r kill -9 2>/dev/null; sleep 2
  # Sample cap for the formal run, from what the smoke measured. Qwen3.5 has a 248,320-entry vocabulary and
  # Slime's actor loss keeps fp32 logits, so the actor step's memory grows about linearly with the sample
  # length. Scale the longest sample the smoke trained by the measured peak GPU0 memory against a 46,000 MiB
  # budget (49,140 MiB card). Fixed costs do not grow, so this scaling is conservative. Rounded down to 1024,
  # never above the requested 32768. A longer episode is dropped "too_long" (length-filtered SAO); the
  # driver reports every drop.
  longest=$(python3 -c "import json,sys; print(max([json.loads(l)['sample_tokens'] for l in open(sys.argv[1])] or [0]))" "$OUT/records.jsonl")
  # The actor's GPU: GPU 0 in the 2gpu layout; either GPU under LAYOUT=tp2 (actor+critic over both).
  if [ "${LAYOUT:-2gpu}" = tp2 ]; then
    peak=$(awk '{for (i = 3; i <= NF; i += 2) if ($i+0 > m) m = $i+0} END {print m+0}' "$RUN_ROOT/ops/gpu-mem.log")
  else
    peak=$(awk '{if ($3+0 > m) m=$3+0} END {print m+0}' "$RUN_ROOT/ops/gpu-mem.log")
  fi
  cap=$(python3 -c "
longest, peak = int('$longest'), int('$peak')
cap = int(longest * 46000 / max(peak, 1)) // 1024 * 1024
print(max(8192, min(32768, cap)))")
  log "sample cap for the formal run: longest smoke sample ${longest} tokens, peak GPU0 ${peak} MiB -> SAO_MAX_SAMPLE_TOKENS=${cap}"
  # FORMAL_MAX_SAMPLE_TOKENS pins the cap instead, e.g. 32768 for the ECHO arm, so it matches the SAO run it is
  # compared with (user, 2026-10-23); the measured value above is still logged.
  if [ -n "${FORMAL_MAX_SAMPLE_TOKENS:-}" ]; then cap=$FORMAL_MAX_SAMPLE_TOKENS; log "cap pinned by FORMAL_MAX_SAMPLE_TOKENS=${cap}"; fi
  cd "$REPRO" && SAO_MAX_SAMPLE_TOKENS=$cap bash appworld/launch_16_qwen35_2gpu.sh formal 2>&1 | tee -a "$LOG"
  log "formal launch command finished (see launch record above)"
else
  log "smoke NOT passed; formal run not launched. Stack left running for inspection."
  exit 1
fi
