#!/usr/bin/env bash
# One shot on .16 (user, 2026-10-25: "test challenge split 也要用于评测", errors-only vs SAO first). Wait until
# after_errors_resume_eval_dot16.sh has written the official queue and started its followers, then move the
# errors-only arm's lines (dev, test_normal_scen1, test_challenge_scen1) to the front. The followers re-read the queue
# at every poll, so the order changes without a restart. SAO's own .16 rows (all three splits) already exist.
#
#   errors_first_queue_dot16.sh
set -u
EVAL_ROOT=/works/yanan/sao-appworld-eval
QUEUE=$EVAL_ROOT/echo-source-queue/queue.txt
LOG=$EVAL_ROOT/echo-source.log
until grep -q "started the .16 evaluation of echo echo-nodocs echo-nodocs-fixed echo-errors" "$LOG"; do sleep 60; done
python3 - "$QUEUE" <<'EOF'
import sys
path = sys.argv[1]
lines = open(path).read().splitlines()
header = [line for line in lines if line.startswith("#")]
body = [line for line in lines if line.strip() and not line.startswith("#")]
first = [line for line in body if "-echo-errors-b15/" in line]
rest = [line for line in body if line not in first]
open(path + ".new", "w").write("\n".join(header + first + rest) + "\n")
EOF
mv "$QUEUE.new" "$QUEUE"
echo "$(TZ=America/New_York date '+%F %T %Z') moved the errors-only arm (dev, test_normal_scen1, test_challenge_scen1) to the front of the queue" >> "$LOG"
