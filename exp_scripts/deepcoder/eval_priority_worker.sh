#!/usr/bin/env bash
# GPU worker with two priority levels, for keeping both .29 GPUs busy without
# ever delaying a new checkpoint:
#
#   1. priority: lines in QUEUE (new checkpoints, enqueued by feed_new_checkpoints.sh).
#   2. filler:   when QUEUE is empty, a filler job is generated for the items
#                (base, and every checkpoint already evaluated at least once):
#                a. first, one sampled evaluation per item at the training
#                   temperature: T=1.0, top_p 1.0, k=4 samples per task, no seed
#                   (tag <item>_t1k4; score = mean over the 4x687 rollouts = avg@4);
#                   base first, then newest checkpoint first;
#                b. then greedy repeats for the item with the fewest greedy evals,
#                   ties -> base, then newest checkpoint, capped at MAX_EVALS.
#                Both reduce the variance of each item's score.
#
# A running filler job is preempted (killed, recorded, dropped) as soon as QUEUE
# has a line, so a new checkpoint waits at most one poll (POLL_S) for a GPU.
# A dropped filler is simply regenerated later, since it never completed.
#
# Successor of eval_queue_worker.sh; separate file because workers were running
# that script when this was written. Stop: touch QUEUE.halt.
#
# usage: eval_priority_worker.sh GPU PORT QUEUE_FILE   (host overrides via env, as eval_sao_checkpoints.sh)
set -uo pipefail
GPU=${1:?usage: $0 GPU PORT QUEUE_FILE}; PORT=${2:?}; Q=${3:?}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCK="$Q.lock"; CLAIMED="$Q.claimed"; HALT="$Q.halt"
NFS=${NFS:?NFS results root required}
ADAPTERS=${ADAPTERS:-/mnt/disk1t/sao-lr5x-eval/adapters}
MAX_EVALS=${MAX_EVALS:-6}
POLL_S=${POLL_S:-20}
touch "$Q" "$CLAIMED"

log() { echo "[worker gpu$GPU $(date '+%F %T %Z')] $*"; }

lock()   { exec 9<>"$LOCK"; flock -x 9; }
unlock() { flock -u 9; exec 9>&-; }

claim_priority() {   # pop the first QUEUE line (caller holds the lock)
  local line; line=$(grep -m1 -v '^[[:space:]]*$' "$Q" || true)
  if [ -n "$line" ]; then
    grep -v -x -F -- "$line" "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"
    echo "$(date '+%F %T %Z') gpu$GPU claimed $line" >> "$CLAIMED"
  fi
  printf '%s' "$line"
}

gen_filler() {       # choose and claim one repeat evaluation (caller holds the lock)
  python3 - "$NFS" "$CLAIMED" "$ADAPTERS" "$MAX_EVALS" "$GPU" <<'PY'
import os, re, sys, time
nfs, claimed_path, adapters, max_evals, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
def stem_of(tag):
    return re.sub(r"_r\d+$", "", tag)
done = {}      # stem -> set of completed tags
for d in os.listdir(nfs) if os.path.isdir(nfs) else []:
    if os.path.isfile(os.path.join(nfs, d, "result.json")):
        done.setdefault(stem_of(d), set()).add(d)
running, used = {}, {}   # tags claimed but not finished / every tag ever claimed
for line in open(claimed_path):
    m = re.search(r" gpu\d (claimed|done|FAILED rc=\d+|preempted) (\S+?)=", line)
    if not m:
        continue
    kind, tag = m.group(1), m.group(2)
    used.setdefault(stem_of(tag), set()).add(tag)
    if kind == "claimed":
        running[tag] = True
    else:
        running.pop(tag, None)
stems = [s for s in done if s == "base_29" or re.fullmatch(r"lr5x_step_\d{3}", s)]
def spec_of(s):
    return "base" if s == "base_29" else f"{adapters}/hf_rollout_{int(s[-3:]) - 1:05d}/hf"
def emit(tag, spec, kind):
    line = f"{tag}={spec}"
    with open(claimed_path, "a") as f:
        f.write(f"{time.strftime('%F %T %Z')} gpu{gpu} claimed {line} ({kind})\n")
    print(line, end="")
    sys.exit(0)
# a. one T=1.0 k=4 evaluation per item: base first, then newest checkpoint first
for s in sorted(stems, key=lambda s: (0 if s == "base_29" else 1, -(int(s[-3:]) if s != "base_29" else 0))):
    t = f"{s}_t1k4"
    if t not in done.get(t, set()) and t not in running:
        emit(t, spec_of(s), "filler t1k4")
def count(s):
    return len(done.get(s, set())) + sum(1 for t in running if stem_of(t) == s)
cands = [s for s in stems if count(s) < max_evals]
if not cands:
    sys.exit(0)
def key(s):
    step = -1 if s == "base_29" else int(s[-3:])
    return (count(s), 0 if s == "base_29" else 1, -step)
s = min(cands, key=key)
taken = done.get(s, set()) | used.get(s, set())
k = 2
while f"{s}_r{k}" in taken:
    k += 1
tag = f"{s}_r{k}"
emit(tag, spec_of(s), "filler")
PY
}

kill_job() {   # $1 = job pid (its own process group)
  kill -TERM -- "-$1" 2>/dev/null; sleep 5; kill -9 -- "-$1" 2>/dev/null
  # the job's vLLM server runs in a process group of its own (setsid): find it by this worker's port
  for d in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in "/home/yanan/agents/rllm/.venv/bin/python /home/yanan/agents/rllm/.venv/bin/vllm serve "*" --port $PORT "*)
      kill -9 -- "-$(ps -o pgid= -p "${d#/proc/}" | tr -d ' ')" 2>/dev/null;; esac
  done 2>/dev/null
  wait "$1" 2>/dev/null
}

idle_logged=""
while :; do
  [ -e "$HALT" ] && { log "halt file present, exiting"; break; }
  lock
  JOB=$(claim_priority); KIND=priority
  if [ -z "$JOB" ]; then JOB=$(gen_filler); KIND=filler; fi
  unlock
  if [ -z "$JOB" ]; then
    [ -z "$idle_logged" ] && { log "nothing to do (queue empty, every item at $MAX_EVALS evals); waiting"; idle_logged=1; }
    sleep "$POLL_S"; continue
  fi
  idle_logged=""
  log "start $KIND $JOB"
  case "${JOB%%=*}" in
    *_t1k4|*_t1k4_r*) SAMPLING=(EVAL_TEMPERATURE=1.0 EVAL_ATTEMPTS=4 EVAL_SEED=none) ;;
    *) SAMPLING=() ;;
  esac
  setsid env GPU="$GPU" PORT="$PORT" "${SAMPLING[@]}" bash "$HERE/eval_sao_checkpoints.sh" "$JOB" &
  JPID=$!
  preempted=""
  while kill -0 "$JPID" 2>/dev/null; do
    if [ "$KIND" = filler ] && grep -q -v '^[[:space:]]*$' "$Q" 2>/dev/null; then
      log "preempting filler $JOB: a priority job is waiting"
      kill_job "$JPID"; preempted=1
      echo "$(date '+%F %T %Z') gpu$GPU preempted $JOB" >> "$CLAIMED"
      rm -rf "${WORK:?}/${JOB%%=*}"
      break
    fi
    sleep "$POLL_S"
  done
  [ -n "$preempted" ] && continue
  wait "$JPID"; rc=$?
  if [ "$rc" = 0 ]; then
    echo "$(date '+%F %T %Z') gpu$GPU done $JOB" >> "$CLAIMED"
  else
    echo "$(date '+%F %T %Z') gpu$GPU FAILED rc=$rc $JOB" >> "$CLAIMED"
    log "job failed rc=$rc: $JOB; exiting"
    exit "$rc"
  fi
done
