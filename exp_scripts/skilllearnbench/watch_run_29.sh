#!/usr/bin/env bash
# Unattended watch of a .16 SkillLearnBench SAO run from .29 (user, 2026-10-20: "持续记录训练当中异常的或很慢的任务，
# 进入无人巡航模式"). Every 5 minutes one ssh poll; every hour the per-task rollout timing summary is appended to the
# log. Exits (so the session that started it is notified) on the first condition that needs a person:
#   SMOKE_FAILED   after_smoke_formal_16.sh did not launch the formal run
#   EXITED         run.py train exited
#   STACK_DOWN     the reef-slb-stack container is not running
#   STALL          no new training step (kept adapter) for STALL_MIN minutes
#   OOM            a CUDA out-of-memory line in the stack log
#   UNREPORTED     more than 3 of the last 48 episodes did not reach the update
#   DROPPED        the driver's assembly check dropped an episode
# Slow episodes (over SLOW_MIN minutes) are logged, not fatal. Waits for the formal run before the run checks.
#
#   watch_run_29.sh [RUN_ROOT=/works/yanan/slb-sao-full20] [SMOKE_LOG=/works/yanan/slb-full20b-after-smoke.log]
set -u
ROOT=${1:-/works/yanan/slb-sao-full20}
SMOKE_LOG=${2:-/works/yanan/slb-full20b-after-smoke.log}
STALL_MIN=${STALL_MIN:-150}
SLOW_MIN=${SLOW_MIN:-30}
LOG=${LOG:-/mnt/disk1t/slb-full20-monitor.log}
ANALYSIS=/works/yanan/slb-rollout-time.py   # scratch copy of the per-task timing script on .16
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
remote() { timeout 120 ssh -o ConnectTimeout=10 10.225.68.16 "$1" < /dev/null; }

log "watch start: $ROOT (stall ${STALL_MIN} min, slow ${SLOW_MIN} min)"
until remote "grep -qE 'NOT launched|formal launched' $SMOKE_LOG"; do sleep 300; done
if remote "grep -q 'NOT launched' $SMOKE_LOG"; then
  log "SMOKE_FAILED: $(remote "grep 'NOT launched' $SMOKE_LOG | tail -1")"; echo SMOKE_FAILED; exit 1
fi
log "formal launched: $(remote "grep 'formal launched' $SMOKE_LOG | tail -1")"
seen_slow=""
last_summary=0
while true; do
  state=$(remote "
    cd $ROOT 2>/dev/null || { echo NOROOT; exit; }
    grep -q 'train exited' launcher.log && echo EXITED
    docker ps --format '{{.Names}}' | grep -qx reef-slb-stack || echo STACK_DOWN
    newest=\$(ls -t kept/adapters 2>/dev/null | head -1)
    if [ -n \"\$newest\" ]; then age=\$(( (\$(date +%s) - \$(stat -c %Y kept/adapters/\$newest)) / 60 )); else age=\$(( (\$(date +%s) - \$(stat -c %Y launch.txt)) / 60 )); fi
    [ \"\$age\" -gt $STALL_MIN ] && echo \"STALL \$newest \${age}min\"
    grep -aqiE 'CUDA out of memory|OutOfMemoryError' state/reef.log && echo OOM
    [ -s run/dropped.jsonl ] && echo \"DROPPED \$(wc -l < run/dropped.jsonl)\"
    python3 - <<'EOF'
import json
rs = [json.loads(l) for l in open('run/episodes.jsonl')] if __import__('os').path.exists('run/episodes.jsonl') else []
tail = rs[-48:]
bad = [r['instance'] for r in tail if not r.get('reported_turns')]
if len(bad) > 3: print('UNREPORTED', len(bad), bad[:6])
print('STEP', max([r.get('batch_step') or 0 for r in rs] or [0]), 'EPISODES', len(rs))
EOF
    python3 - <<'EOF'
import json, glob, os
from datetime import datetime
ts = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
for t in glob.glob('run/lab/trials/*/result.json'):
    try: r = json.load(open(t))
    except Exception: continue
    if r.get('finished_at') and ts(r['finished_at']) - ts(r['started_at']) > $SLOW_MIN * 60:
        slb = ((r.get('agent_result') or {}).get('metadata') or {}).get('slb') or {}
        v = r.get('verifier') or {}
        vs = ts(v['finished_at']) - ts(v['started_at']) if v.get('finished_at') else 0
        print('SLOW', os.path.basename(os.path.dirname(t)), round((ts(r['finished_at']) - ts(r['started_at'])) / 60, 1), 'min model', round((slb.get('model_seconds') or 0) / 60, 1), 'verify', round(vs / 60, 1), slb.get('end', '')[:20])
EOF")
  [ -z "$state" ] && { log "poll failed (ssh)"; sleep 300; continue; }
  progress=$(echo "$state" | grep '^STEP')
  while read -r line; do
    name=$(echo "$line" | awk '{print $2}')
    case " $seen_slow " in *" $name "*) ;; *) seen_slow="$seen_slow $name"; log "slow episode: ${line#SLOW }";; esac
  done < <(echo "$state" | grep '^SLOW')
  if [ $(( $(date +%s) - last_summary )) -ge 3600 ]; then
    last_summary=$(date +%s)
    log "hourly: $progress"
    remote "python3 $ANALYSIS $ROOT 2>/dev/null | sed -n '1,25p'" | sed 's/^/    /' >> "$LOG"
  fi
  problem=$(echo "$state" | grep -E '^(NOROOT|EXITED|STACK_DOWN|STALL|OOM|UNREPORTED|DROPPED)' | head -3)
  if [ -n "$problem" ]; then log "PROBLEM: $(echo $problem) | $progress"; echo "$problem"; exit 1; fi
  sleep 300
done
