#!/usr/bin/env bash
# One read-only status snapshot of the formal SAO DeepCoder run on .24.
RUN_ROOT=/home/yanan/reef-sao-deepcoder
STATE=$RUN_ROOT/state-b128
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=$(ls -dt "$REPRO"/results/deepcoder/b128-* 2>/dev/null | head -1)
echo "T=$(date '+%F %T %Z')  run=$(basename "$OUT")"
docker ps --format '{{.Names}}' | grep -qx reef-sao-stack && echo "stack=alive" || echo "stack=GONE"
curl -sf -m 5 http://127.0.0.1:8900/healthz >/dev/null 2>&1 && echo "reef=healthy" || echo "reef=unreachable"
pgrep -f "deepcoder/stream.py" >/dev/null && echo "driver=alive" || echo "driver=GONE"
echo "trained_steps=$(cat "$OUT/progress.txt" 2>/dev/null || echo 0) / 190"
echo "--- step commits (sidecar) ---"; grep "trained steps" "$OUT/sidecar.log" 2>/dev/null | tail -6
echo "--- rollouts ---"
python3 - "$OUT/records.jsonl" <<'PY'
import json, sys
try:
    rs = [json.loads(l) for l in open(sys.argv[1])]
except FileNotFoundError:
    print("  no records yet"); sys.exit()
n = len(rs)
if not n:
    print("  no records yet"); sys.exit()
c = sum(r["score"] for r in rs)
to = sum(bool(r.get("grader_timeout")) for r in rs)
rg = sum(r.get("timeout_regrades", 0) for r in rs)
tr = sum(r.get("finish_reason") == "length" for r in rs)
tok = sorted(r["completion_tokens"] for r in rs)
print(f"  scored={n} correct={c:.0f} ({100*c/n:.1f}%) grader_timeouts={to} regrades={rg} truncated={tr} ({100*tr/n:.1f}%) median_tokens={tok[n//2]}")
last = rs[-200:]
lc = sum(r["score"] for r in last)
print(f"  last {len(last)}: {100*lc/len(last):.1f}% correct")
PY
echo "--- disk ---"
du -sh "$STATE/checkpoints" 2>/dev/null | sed 's/^/  checkpoints: /'
K=$RUN_ROOT/kept-checkpoints-b128
echo "  kept adapters: $(ls -1d "$K"/adapters/hf_rollout_* 2>/dev/null | wc -l)  full bundles: $(ls -1d "$K"/full/step_??? 2>/dev/null | xargs -rn1 basename | tr '\n' ' ')"
du -sh "$K" 2>/dev/null | sed 's/^/  kept total: /'
grep -E "failed" "$(dirname "$OUT")/$(basename "$OUT")/sidecar.log" 2>/dev/null | tail -2 | sed 's/^/  sidecar: /'
df -h "$RUN_ROOT" | tail -1 | awk '{print "  filesystem free: "$4" ("$6")"}'
echo "--- host memory ---"
free -g | awk 'NR==2{print "  used "$3" / "$2" GB, available "$7" GB"}'
D=$(pgrep -f "deepcoder/stream.py" | head -1); [ -n "$D" ] && awk '/VmRSS/{printf "  driver RSS %.1f GB\n", $2/1048576}' /proc/$D/status
echo "--- health ---"
L=$STATE/stack/slime-driver.log
echo "  OOM=$(grep -c 'CUDA out of memory' "$L" 2>/dev/null) tracebacks=$(grep -c '^Traceback' "$L" 2>/dev/null)"
grep -oE "staleness/samples_admitted_stale[^,]*|staleness/samples_fresh[^,]*" "$STATE/checkpoints/hf/.reef-latest-job.json" 2>/dev/null | sed 's/^/  /'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  GPU /'
