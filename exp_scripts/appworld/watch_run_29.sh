#!/usr/bin/env bash
# Unattended watch of a .16 Reef SAO AppWorld run from .29 (user, 2026-10-23: "继续监控 .16 上面的训练"), for the SAO+ECHO
# run and its kind. Every 5 minutes one ssh poll; every hour the step, the latest actor metrics (pg_loss, echo_ce,
# echo_coef, echo_term_ratio, train_rollout_logprob_abs_diff, pg_clipfrac) and the GPU peak go to the log. Exits (so
# the session that started it is notified) on the first condition that needs a person:
#   EXITED      the driver finished or died (driver log "driver exited" or launcher gone)
#   STACK_DOWN  the reef-sao-stack container is not running
#   STALL       no new published step for STALL_MIN minutes
#   OOM         an out-of-memory or training failure line in the stack log
#   DROPS       the driver reports more than DROPS_ALLOW (default 0) too-long or failed episodes
#
#   watch_run_29.sh RUN_ROOT TAG [STALL_MIN=90]
set -u
ROOT=$1 TAG=$2 STALL_MIN=${3:-90}
DRIVER=/home/yanan/agents/reef/exp_scripts/results/appworld/$TAG/driver.log
STACK=$ROOT/state/stack/slime-driver.log
LOG=${LOG:-/mnt/disk1t/appworld-echo-monitor.log}
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
remote() { timeout 120 ssh -o ConnectTimeout=10 10.225.68.16 "$1" < /dev/null; }

log "watch start: $ROOT ($TAG), stall ${STALL_MIN} min"
last_step=-1 last_change=$(date +%s) last_summary=0
while true; do
  state=$(remote "
    step=\$(grep -oE '^sync: step [0-9]+ published' $DRIVER | tail -1 | grep -oE '[0-9]+'); echo STEP \${step:-0}
    grep -q 'driver exited' $ROOT/ops/launcher.log && echo \"EXITED \$(grep -o 'driver exited rc=[0-9]*' $ROOT/ops/launcher.log | tail -1)\"
    kill -0 \$(cat $ROOT/ops/launcher.pid) 2>/dev/null || echo 'EXITED launcher gone'
    docker ps --format '{{.Names}}' | grep -qx reef-sao-stack || echo STACK_DOWN
    grep -aqiE 'OutOfMemoryError|CUDA out of memory|training_failed' $STACK && echo OOM
    n=\$(grep -cE 'WARNING: only|too long, not retried|failed' $DRIVER); [ \"\$n\" -gt ${DROPS_ALLOW:-0} ] && echo \"DROPS \$n\"
    echo METRICS \$(grep -h 'pg_loss' $STACK | tail -1 | grep -oE \"(pg_loss|echo_ce|echo_coef|echo_term_ratio|train_rollout_logprob_abs_diff|pg_clipfrac)': [-0-9.e]+\" | tr '\n' ' ')
    echo PEAK \$(awk '{for(i=3;i<=NF;i+=2){v=\$i+0; if(v>m)m=v}} END{print m+0}' $ROOT/ops/gpu-mem.log)")
  [ -z "$state" ] && { log "poll failed (ssh)"; sleep 300; continue; }
  step=$(echo "$state" | awk '/^STEP/{print $2}')
  if [ "$step" != "$last_step" ]; then last_step=$step; last_change=$(date +%s); fi
  if [ $(( $(date +%s) - last_summary )) -ge 3600 ]; then
    last_summary=$(date +%s)
    log "hourly: step $step | $(echo "$state" | grep '^METRICS' | cut -c9-) | peak $(echo "$state" | awk '/^PEAK/{print $2}') MiB"
  fi
  problem=$(echo "$state" | grep -E '^(EXITED|STACK_DOWN|OOM|DROPS)' | head -3)
  [ $(( $(date +%s) - last_change )) -gt $(( STALL_MIN * 60 )) ] && problem="STALL at step $step for ${STALL_MIN}+ min $problem"
  if [ -n "$problem" ]; then log "PROBLEM: $(echo $problem) | step $step"; echo "$problem"; exit 1; fi
  sleep 300
done
