#!/usr/bin/env bash
# Every INTERVAL_S (default 600), sample HOST's GPU utilization once a second for 10 s
# and append one TSV row per check: time, per-GPU mean utilization and memory, and the
# latest FinQA run's trained-step count. In an SAO step the actor GPU and the rollout GPU
# take turns, so both are never busy at once; both near zero means nothing is running.
# Two consecutive checks with every GPU under IDLE_PCT append an ALERT line.
#
# usage: gpu_monitor.sh [HOST]   (env INTERVAL_S, IDLE_PCT, OUT); stop: kill the pid in $OUT.pid
set -uo pipefail
HOST=${1:-10.225.68.16}
INTERVAL_S=${INTERVAL_S:-600}
IDLE_PCT=${IDLE_PCT:-10}
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${OUT:-$REPRO/results/finqa/gpu_monitor_${HOST##*.}.tsv}
echo $$ > "$OUT.pid"
[ -s "$OUT" ] || printf 'time\tgpu0_util\tgpu1_util\tgpu0_mem_mib\tgpu1_mem_mib\ttrained_steps\trun\n' > "$OUT"
idle_streak=0
while :; do
  sample=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
    "for i in \$(seq 10); do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 1; done" 2>/dev/null)
  run=$(ls -dt "$REPRO"/results/finqa/finqa-* 2>/dev/null | head -1)
  steps=$(cat "$run/progress.txt" 2>/dev/null || echo NA)
  if [ -z "$sample" ]; then
    printf '%s\tNA\tNA\tNA\tNA\t%s\t%s\n' "$(date '+%F %T %Z')" "$steps" "${run##*/}" >> "$OUT"
  else
    row=$(echo "$sample" | awk -F', ' '{u[$1]+=$2; m[$1]=$3; n[$1]++} END {printf "%.0f\t%.0f\t%s\t%s", u[0]/n[0], u[1]/n[1], m[0], m[1]}')
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T %Z')" "$row" "$steps" "${run##*/}" >> "$OUT"
    if echo "$row" | awk -F'\t' -v t="$IDLE_PCT" '{exit !($1 < t && $2 < t)}'; then idle_streak=$((idle_streak + 1)); else idle_streak=0; fi
    [ "$idle_streak" = 2 ] && echo "ALERT $(date '+%F %T %Z') both GPUs under ${IDLE_PCT}% for 2 consecutive checks (trained steps $steps)" >> "$OUT"
  fi
  sleep "$INTERVAL_S"
done
