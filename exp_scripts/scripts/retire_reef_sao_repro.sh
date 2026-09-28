#!/usr/bin/env bash
# One-off (asked for 2026-09-26): once DeepCoder step 100's evaluations have finished, move what is
# left of the old experiment directory ~/agents/reef_sao_repro into this fork's
# exp_scripts, verify every file, then delete the old directory.
#
#   1. wait for DeepCoder step 100's two evaluations (T=0 x4, T=1.0 x4) to write result.json
#   2. wait until nothing on .29, .16 or .24 uses the old directory: no process with it in
#      its cmdline, cwd or open files, and no container mounting it
#   3. copy results/ and data/ into exp_scripts/results and exp_scripts/data; copy every
#      other file that has no byte-identical twin in exp_scripts (the old-path script
#      variants, docker/ from the pre-fork build) to results/legacy-reef_sao_repro/
#   4. verify: every old file (except __pycache__) has a byte-identical copy in its new place
#   5. delete ~/agents/reef_sao_repro
# Any failed check stops the script before step 5 and says why.
set -uo pipefail
OLD=/home/yanan/agents/reef_sao_repro
NEW=/home/yanan/agents/reef/exp_scripts
LEGACY=$NEW/results/legacy-reef_sao_repro
DEADLINE=$(date -d '2026-09-28 12:00' +%s)
HOSTS=(10.225.68.16 10.225.68.24)
log() { echo "[retire $(date '+%F %T %Z')] $*"; }
die() { log "STOPPED, nothing deleted: $*"; exit 1; }
[ -d "$OLD" ] || die "$OLD does not exist"

EVAL=$OLD/results/deepcoder/eval-c32-lora-lr5x
log "1. waiting for DeepCoder step 100's evaluations (lr5x_step_100_t0k4 and _t1k4)"
until [ -s "$EVAL/lr5x_step_100_t0k4/result.json" ] && [ -s "$EVAL/lr5x_step_100_t1k4/result.json" ]; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || die "step 100 evaluations not finished by 2026-09-28 12:00"
  sleep 120
done
log "   step 100 evaluations finished"

users_here() {  # processes on this host whose cmdline, cwd or open files are under $OLD
  for d in /proc/[0-9]*; do
    p=${d#/proc/}; [ "$p" = "$$" ] && continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in *retire_reef_sao_repro*) continue;; esac
    { case "$c" in *"$OLD"*) true;; *) false;; esac; } || [[ "$(readlink "$d/cwd" 2>/dev/null)" == "$OLD"* ]] \
      || ls -l "$d/fd" 2>/dev/null | grep -q -- "-> $OLD" && echo "$p ${c:0:120}"
  done
}
remote_users() {  # $1 = host: processes as above, plus containers mounting $OLD
  # The probe's own shells carry $OLD in their cmdline too; they are skipped by the marker.
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "true retire-probe
    for d in /proc/[0-9]*; do c=\$(tr '\\0\\n' '  ' < \$d/cmdline 2>/dev/null) || continue
      case \"\$c\" in *retire-probe*) continue;; esac
      { case \"\$c\" in *$OLD*) true;; *) false;; esac; } || [[ \"\$(readlink \$d/cwd 2>/dev/null)\" == $OLD* ]] \\
        || ls -l \$d/fd 2>/dev/null | grep -q -- '-> $OLD' && echo \"\${d#/proc/} \${c:0:120}\"
    done
    for n in \$(docker ps -aq 2>/dev/null); do docker inspect \$n --format '{{.Name}} {{range .Mounts}}{{.Source}} {{end}}' | grep -- '$OLD'; done
    true" 2>&1 || echo "ssh $1 failed"
}
log "2. waiting until nothing uses $OLD"
while :; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || die "still in use at 2026-09-28 12:00: $busy"
  busy="$(users_here)"
  for h in "${HOSTS[@]}"; do r="$(remote_users "$h")"; [ -n "$r" ] && busy+=$'\n'"$h: $r"; done
  busy=$(echo "$busy" | sed '/^$/d')
  [ -z "$busy" ] && break
  [ "$busy" != "${last_busy:-}" ] && { log "   in use by $(echo "$busy" | wc -l):"$'\n'"$(echo "$busy" | sed 's/^/     /')"; last_busy=$busy; }
  sleep 300
done
log "   no users left"

log "3. copying"
mkdir -p "$NEW/results" "$NEW/data" "$LEGACY"
rsync -a --exclude __pycache__ "$OLD/results/" "$NEW/results/" || die "rsync results failed"
rsync -a "$OLD/data/" "$NEW/data/" || die "rsync data failed"
moved=0
while IFS= read -r -d '' f; do
  rel=${f#$OLD/}
  case "$rel" in results/*|data/*) continue;; esac
  cmp -s "$f" "$NEW/$rel" && continue
  mkdir -p "$LEGACY/$(dirname "$rel")"; cp -p "$f" "$LEGACY/$rel" || die "copy $rel failed"; moved=$((moved + 1))
done < <(find "$OLD" -type f ! -path '*/__pycache__/*' -print0)
log "   results/ and data/ synced; $moved old-only file versions kept in $LEGACY"

log "4. verifying"
checked=0; bad=0
while IFS= read -r -d '' f; do
  rel=${f#$OLD/}
  if cmp -s "$f" "$NEW/$rel" || cmp -s "$f" "$LEGACY/$rel"; then checked=$((checked + 1)); else bad=$((bad + 1)); log "   NO COPY: $rel"; fi
done < <(find "$OLD" -type f ! -path '*/__pycache__/*' -print0)
[ "$bad" = 0 ] || die "$bad files have no byte-identical copy"
[ -z "$(find "$OLD" -type l)" ] || die "symlinks present; not handled"
log "   $checked files verified byte-identical in $NEW"

log "5. deleting $OLD"
rm -rf -- "$OLD" || die "rm failed"
[ -e "$OLD" ] && die "$OLD still exists after rm"
log "DONE: $OLD deleted; its results and data live in $NEW, old-only script versions in $LEGACY"
