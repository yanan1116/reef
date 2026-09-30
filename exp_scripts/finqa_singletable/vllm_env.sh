# Source before starting vLLM from exp_scripts/.venv-finqa (see setup_venv.sh).
VENV=${FINQA_VENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv-finqa}
export LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
