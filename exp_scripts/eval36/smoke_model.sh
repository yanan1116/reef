#!/usr/bin/env bash
# Smoke-test a base model on one .36 GPU before evaluating it there (fp16 on Turing, Triton attention,
# vllm 0.22.1+cu129): the server must start, greedy arithmetic must be right with finite log-probs
# (fp16 overflow shows up as NaN/inf or garbage), and a tool call must parse with TOOL_PARSER.
# Tries max-model-len 49152 first, then 28672 if the KV cache cannot hold one such sequence; writes
# the length that worked to $OUT/max_model_len for the evaluations that follow.
#
# usage: smoke_model.sh OUT_DIR ; env BASE, TOOL_PARSER (hermes), CHAT_TEMPLATE, CHAT_TEMPLATE_KWARGS,
#        GPU (0), PORT (18400), GPU_UTIL (0.80)
set -euo pipefail
# EXTRA_SERVE_ARGS: extra vLLM serve flags, space-separated, no spaces inside a value
# (Qwen3.5 on .36: --limit-mm-per-prompt {"image":0,"video":0}; text-only evals, no vision profiling).
read -r -a EXTRA_SERVE <<< "${EXTRA_SERVE_ARGS:-}"
OUT=${1:?usage: $0 OUT_DIR}
GPU=${GPU:-0}
PORT=${PORT:-18400}
SERVE_VENV=${SERVE_VENV:-/home/yanan/eval36/env/venv-finqa-cu129}
PY="$SERVE_VENV/bin/python"
BASE=${BASE:?BASE (model directory) is required}
TOOL_PARSER=${TOOL_PARSER:-hermes}
TEMPLATE_ARGS=()
[ -n "${CHAT_TEMPLATE:-}" ] && TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0
SPID=
cleanup() { [ -n "$SPID" ] && kill -9 -- "-$SPID" 2>/dev/null; [ -n "$SPID" ] && wait "$SPID" 2>/dev/null; return 0; }
trap cleanup EXIT
for len in 49152 28672; do
  CUDA_VISIBLE_DEVICES=$GPU setsid "$PY" -m vllm.entrypoints.cli.main serve "$BASE" --served-model-name base \
    --host 127.0.0.1 --port "$PORT" --max-model-len "$len" --gpu-memory-utilization "${GPU_UTIL:-0.80}" --dtype half \
    --attention-backend TRITON_ATTN "${EXTRA_SERVE[@]}" --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" "${TEMPLATE_ARGS[@]}" \
    > "$OUT/server_$len.log" 2>&1 &
  SPID=$!
  ready=0
  for ((t=0; t<600; t++)); do
    kill -0 "$SPID" 2>/dev/null || break
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 2
  done
  [ "$ready" = 1 ] && { echo "$len" > "$OUT/max_model_len"; break; }
  cleanup; SPID=
  grep -qE "KV cache is needed|larger than the available KV cache|OutOfMemoryError|CUDA out of memory" "$OUT/server_$len.log" \
    || { echo "[smoke] server failed to start at max-model-len $len (not a KV-size limit); see $OUT/server_$len.log" >&2; exit 1; }
  echo "[smoke] max-model-len $len does not fit one GPU (KV cache or startup OOM); trying a shorter one"
done
[ -s "$OUT/max_model_len" ] || { echo "[smoke] no max-model-len fits on one GPU" >&2; exit 1; }
echo "[smoke] server up at max-model-len $(cat "$OUT/max_model_len"); $(grep -hoE 'GPU KV cache size: [0-9,]+ tokens' "$OUT"/server_*.log | tail -1)"
"$PY" - "$PORT" "${CHAT_TEMPLATE_KWARGS:-}" <<'PY' | tee "$OUT/checks.log"
import json, math, sys, urllib.request
port, kwargs = sys.argv[1], (json.loads(sys.argv[2]) if sys.argv[2] else None)
def call(body):
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))
failures = []
r = call(dict(model="base", messages=[{"role": "user", "content": "What is 17 * 23? Reply with just the number."}],
              temperature=0, max_tokens=64, logprobs=True, seed=1234))
text = r["choices"][0]["message"].get("content") or ""
lps = [t["logprob"] for t in r["choices"][0]["logprobs"]["content"]]
print(f"[arith] {text!r}; {len(lps)} tokens, min logprob {min(lps):.3f}")
if "391" not in text:
    failures.append(f"arithmetic: expected 391, got {text!r}")
if not lps or not all(math.isfinite(x) for x in lps):
    failures.append("non-finite log-probs (fp16 overflow?)")
r = call(dict(model="base", messages=[{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
              temperature=0, max_tokens=120))
text = r["choices"][0]["message"].get("content") or ""
ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
print(f"[text] {text[:200]!r} (ascii ratio {ascii_ratio:.2f})")
if len(text.split()) < 10 or ascii_ratio < 0.9 or "blue" not in text.lower():
    failures.append("free text looks garbled")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = call(dict(model="base", messages=[{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
              tools=tools, temperature=0, max_tokens=256))
calls = r["choices"][0]["message"].get("tool_calls") or []
print(f"[tool] {[(c['function']['name'], c['function']['arguments']) for c in calls]}")
if not calls or calls[0]["function"]["name"] != "get_weather" or "paris" not in calls[0]["function"]["arguments"].lower():
    failures.append("tool call did not parse")
if failures:
    print("[smoke] FAIL: " + "; ".join(failures))
    raise SystemExit(1)
print("[smoke] PASS")
PY
