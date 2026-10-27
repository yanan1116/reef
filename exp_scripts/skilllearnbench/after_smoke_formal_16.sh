#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-14: "如果一切正常 那么自动启动正式训练"): wait for the whole-episode SAO smoke to
# finish, check it, remove its stack, and launch the formal run (run.sh formal). Nothing is launched unless
# every check passes:
#   - run.sh smoke exited with "train exited rc=0"
#   - 3 training releases on scenario slb-sao
#   - no OOM / Traceback in the stack log or the train log
#   - peak GPU memory below 46,000 MiB per GPU (gpu-mem.log)
#   - the last train_rollout_logprob_abs_diff below 0.05
#   - at most 2 episodes dropped by the driver's assembly check (run/dropped.jsonl)
# run.sh sets SLB_COMMAND_TIMEOUT_S=180 (user, 2026-10-14) for both the smoke and the formal run.
#
# The full 20-task run (user, 2026-10-19) reuses it: FORMAL names the formal run root, and the formal run inherits
# this script's environment (SLB_TASKS, SLB_PROMPT_VERSION, SLB_BUDGET_WARN).
#
#   [FORMAL=/works/yanan/slb-sao-formal] after_smoke_formal_16.sh SMOKE_RUN_ROOT
set -u
SMOKE=${1:?usage: $0 SMOKE_RUN_ROOT}
HERE="$(cd "$(dirname "$0")" && pwd)"
FORMAL=${FORMAL:-/works/yanan/slb-sao-formal}
LOG=${LOG:-/works/yanan/slb-after-smoke-formal.log}
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
fail() { log "NOT launched: $*"; exit 1; }

log "waiting for the smoke in $SMOKE"
until grep -q "train exited rc=" "$SMOKE/launcher.log" 2>/dev/null; do sleep 30; done
grep -q "train exited rc=0" "$SMOKE/launcher.log" || fail "smoke run.py exit: $(grep 'train exited' "$SMOKE/launcher.log")"
steps=$(curl -s -m 30 -H "Authorization: Bearer reef-local" http://127.0.0.1:8900/reef/scenarios/slb-sao/releases \
  | python3 -c "import sys,json;print(sum(1 for r in json.load(sys.stdin)['releases'] if r.get('operation')=='training'))")
[ "${steps:-0}" -ge 3 ] || fail "only ${steps:-?} training steps"
if grep -aqiE "OutOfMemory|CUDA out of memory|Traceback" "$SMOKE/state/reef.log" "$SMOKE/run/train.log"; then
  fail "error in logs: $(grep -ahiE 'OutOfMemory|CUDA out of memory|Traceback' "$SMOKE/state/reef.log" "$SMOKE/run/train.log" | tail -n 1 | cut -c1-200)"
fi
peak=$(awk '{a=$3; gsub(",","",a); b=$6; gsub(",","",b); if(a+0>m)m=a+0; if(b+0>m)m=b+0} END{print m+0}' "$SMOKE/gpu-mem.log")
[ "$peak" -lt 46000 ] || fail "peak GPU memory $peak MiB"
diff=$(grep -aoE "train_rollout_logprob_abs_diff': [0-9.e-]+" "$SMOKE/state/reef.log" | tail -n 1 | awk '{print $2}')
python3 -c "import sys; sys.exit(0 if 0 < float('${diff:-1}') < 0.05 else 1)" || fail "logprob abs diff ${diff:-missing}"
dropped=$(wc -l < "$SMOKE/run/dropped.jsonl" 2>/dev/null || echo 0)
[ "$dropped" -le 2 ] || fail "$dropped episodes dropped by the assembly check"
# Every smoke episode must reach the update (user, 2026-10-20: "确保所有训练任务的结果都能用于模型更新"): none may be
# left unreported (no score: verifier failure or timeout; or over the token limit).
unreported=$(python3 -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if not json.loads(l).get('reported_turns')))" "$SMOKE/run/episodes.jsonl")
[ "$unreported" = 0 ] || fail "$unreported smoke episodes not reported"
log "smoke passed: steps $steps, peak $peak MiB, logprob abs diff $diff, dropped $dropped, unreported $unreported"

docker rm -f reef-slb-stack > /dev/null 2>&1
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
[ "$busy" = 0 ] || fail "GPUs still busy after removing the smoke stack"
[ -e "$FORMAL" ] && fail "$FORMAL exists"
mkdir -p "$FORMAL"
echo "launched by after_smoke_formal_16.sh; smoke $SMOKE passed: steps $steps, peak $peak MiB, logprob diff $diff, dropped $dropped; SLB_TASKS=${SLB_TASKS:-default} SLB_PROMPT_VERSION=${SLB_PROMPT_VERSION:-default} SLB_BUDGET_WARN=${SLB_BUDGET_WARN:-off}" > "$FORMAL/launch-env.txt"
cd "$HERE/.." && RUN_ROOT=$FORMAL setsid nohup bash skilllearnbench/run.sh formal \
  > "$FORMAL/launcher.log" 2>&1 < /dev/null &
sleep 10
log "formal launched: $FORMAL (launcher pid $!)"
setsid nohup bash -c "while docker ps --format {{.Names}} | grep -qx reef-slb-stack || [ \$SECONDS -lt 300 ]; do echo \"\$(TZ=America/New_York date +%T) \$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')\" >> $FORMAL/gpu-mem.log; sleep 10; done" > /dev/null 2>&1 < /dev/null &
