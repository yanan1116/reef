#!/usr/bin/env bash
# Create exp_scripts/.venv-finqa: the SAO FinQA driver, checkpoint copier and
# evaluator run here, never in rllm's venv. Versions are pinned by
# venv_constraints.txt, installed verbatim (a freeze of the rllm venv the PRPO FinQA runs used, minus
# rllm itself), so the judge client (openai) and the evaluation server (vLLM
# 0.22.1 on torch 2.11.0+cu129) are the same builds PRPO was scored with.
#
# vLLM 0.22.1's compiled extension is built against CUDA 13 while this torch is
# cu129; libcudart.so.13 ships in nvidia-cutlass-dsl-libs-cu13 and must be on the
# loader path (vllm_env.sh exports it), as in rllm's finqa-grpo-run/env.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=${FINQA_VENV:-$(cd "$HERE/.." && pwd)/.venv-finqa}
[ -e "$VENV" ] && { echo "$VENV exists; remove it first to rebuild" >&2; exit 1; }
uv venv --python 3.11.13 "$VENV"
# --no-deps installs the freeze verbatim instead of re-resolving it: the rllm venv is not
# self-consistent under a resolver (numpy 2.4.6 next to mistral-common 1.11.7, which
# declares numpy<2.4), so resolving would refuse the very builds PRPO was scored with.
# flash-attn is left out: it has no wheel (the rllm venv built it from source against its
# torch), and nothing here imports it -- the driver and judge do not, and vLLM serves
# Qwen3-4B through its own bundled vllm_flash_attn.
grep -v '^flash-attn==' "$HERE/venv_constraints.txt" > "$VENV/requirements-verbatim.txt"
uv pip install --python "$VENV/bin/python" --no-deps -r "$VENV/requirements-verbatim.txt" \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for name in ("pandas", "asteval", "openai", "vllm", "torch", "transformers", "peft"):
    print(f"{name}=={m.version(name)}")
import openai, pandas, asteval, peft, safetensors, vllm  # noqa: F401  (imports resolve with the verbatim freeze)
PY
echo "ready: $VENV"
