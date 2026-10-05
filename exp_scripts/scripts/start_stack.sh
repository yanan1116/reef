#!/bin/bash
# Start the Reef SAO training stack inside the reef image built from this checkout
# (scripts/build_image_head.sh; tag reef:sao-<last commit touching Reef source>).
#
# Mounts (host -> container):
#   ~/reef-sao/models    -> /root/models        base model weights
#   $STATE_DIR           -> /var/lib/reef       checkpoints, artifacts, agent records
#   ~/reef-sao/data      -> /root/data          data the stack reads
#   exp_scripts/ (here)  -> /repro              our configs, driver and results
#   $KEEP_DIR (optional) -> /var/lib/reef-kept  checkpoint copies kept outside Reef's tree
#
# NCCL_P2P_DISABLE=1 is the default: peer-to-peer NCCL hung .24, where these runs began (the
# verl/FSDP launchers carry the same export). Both LoRA publication attempts on
# 2026-09-23 stopped at the identical line -- "LoRA adapter loading from
# distributed starts" -- with the training batch left undrained for 1153 s. The
# distributed transport broadcasts the adapter over NCCL; the colocated transport
# exists precisely to avoid "creating an NCCL peer on the same GPU". Same symptom
# with full LoRA and with MLP-only LoRA, so the transport is implicated, not the
# adapter surface.
#
# The container runs with --network host so the driver on the host can reach
# Reef on 127.0.0.1:8900, and --ipc host --shm-size 32g as the training stack needs.
# STACK_NETWORK=private instead gives it its own network and IPC namespaces, for a
# second stack on the same host (e.g. .29: sao on GPU 0, pvf on GPU 1), where
# SGLang, Ray and torch ports would otherwise collide; only Reef's port is
# published, on the host's 127.0.0.1:$REEF_PORT (the config must bind reef.host
# 0.0.0.0 inside). Unset keeps the exact earlier command.
set -euo pipefail
NAME=${NAME:-reef-sao-stack}
REPRO="$(cd "$(dirname "$0")/.." && pwd)"   # exp_scripts/
REEF_ROOT="$(cd "$REPRO/.." && pwd)"          # the reef fork checkout
# Default image: the one build_image_head.sh builds from this checkout's Reef source.
SRC_COMMIT=$(git -C "$REEF_ROOT" -c safe.directory='*' log -1 --format=%h -- . ':(exclude)exp_scripts')
IMAGE=${IMAGE:-reef:sao-$SRC_COMMIT}
CFG=${CFG:-/repro/configs/serve-deepcoder-2507-b128-lr5x.yaml}
# Separate state per experiment: a stack restores its scenario version chains and
# checkpoints from here, so two base models must never share one.
STATE_DIR=${STATE_DIR:-/home/yanan/reef-sao/state}
mkdir -p "$STATE_DIR"
# Optional: a directory OUTSIDE the managed checkpoint tree where copies of
# checkpoints are kept. Checkpoint files are root-owned and not world-readable,
# so the copy has to run inside the container (see deepcoder/sidecar.py).
KEEP_MOUNT=()
if [ -n "${KEEP_DIR:-}" ]; then mkdir -p "$KEEP_DIR"; KEEP_MOUNT=(-v "$KEEP_DIR":/var/lib/reef-kept); fi

# Optional OCI runtime. .16 needs DOCKER_RUNTIME=nvidia: its nvidia-container
# config has no-cgroups = true, so with the default runc the GPU device nodes are
# injected but not permitted and NVML fails ("Failed to initialize NVML: Unknown
# Error"; Ray registers 0 GPUs); .29 needs it too, and the FinQA launchers always pass it.
# Unset keeps docker's default runtime.
RUNTIME_ARGS=()
if [ -n "${DOCKER_RUNTIME:-}" ]; then RUNTIME_ARGS=(--runtime "$DOCKER_RUNTIME"); fi

# Optional: extra module directories for the Reef service, e.g. EXTRA_PYTHONPATH=/repro/finqa so
# recipe.implementation can name a recipe that lives in exp_scripts (container paths). It is put
# in front of the image's own PYTHONPATH (SGLang / Megatron entries) inside the container; unset
# keeps the exact command every earlier run used.
SERVE_CMD="python3 -m reef serve -c $CFG 2>&1 | tee /var/lib/reef/reef.log"
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
  SERVE_CMD="export PYTHONPATH=$EXTRA_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH}; $SERVE_CMD"
fi

# Optional: STACK_GPUS=0 (a comma list of host GPU indices) gives the container only those GPUs,
# e.g. a colocated single-GPU stack on .29 that leaves GPU 1 to evaluation. Unset keeps the exact
# earlier command (all GPUs, CUDA_VISIBLE_DEVICES=0,1).
if [ -n "${STACK_GPUS:-}" ]; then
  GPU_ARGS=(--gpus "\"device=$STACK_GPUS\"")
  VISIBLE=$(seq -s, 0 $(( $(tr ',' '\n' <<< "$STACK_GPUS" | wc -l) - 1 )))  # renumbered from 0 inside
else
  GPU_ARGS=(--gpus all)
  VISIBLE=0,1
fi

if [ "${STACK_NETWORK:-host}" = private ]; then
  NET_ARGS=(-p "127.0.0.1:${REEF_PORT:-8900}:8900" --ipc private --shm-size 32g)
elif [ "${STACK_NETWORK:-host}" = host ]; then
  NET_ARGS=(--network host --ipc host --shm-size 32g)
else
  echo "STACK_NETWORK must be host or private, got ${STACK_NETWORK}" >&2; exit 1
fi

# Optional: STACK_MEMORY (docker --memory, e.g. 60g) caps the container's host RAM, so a second stack
# that outgrows it is OOM-killed inside its own cgroup instead of the kernel picking a process of the
# stack beside it (colocated training offloads optimizer state to host RAM: ~45 GB per 4B stack on .29).
MEM_ARGS=()
if [ -n "${STACK_MEMORY:-}" ]; then MEM_ARGS=(--memory "$STACK_MEMORY"); fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${RUNTIME_ARGS[@]}" \
  "${GPU_ARGS[@]}" "${NET_ARGS[@]}" "${MEM_ARGS[@]}" \
  -v /home/yanan/reef-sao/models:/root/models \
  -v "$STATE_DIR":/var/lib/reef \
  "${KEEP_MOUNT[@]}" \
  -v /home/yanan/reef-sao/data:/root/data \
  -v "$REPRO":/repro \
  -e CUDA_VISIBLE_DEVICES=$VISIBLE \
  -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} \
  ${NVTE_DEBUG:+-e NVTE_DEBUG=$NVTE_DEBUG -e NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-2}} \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e REEF_SGLANG_HEALTH_TIMEOUT_S=${REEF_SGLANG_HEALTH_TIMEOUT_S:-1200} \
  -e REEF_BF16_LOGITS=${REEF_BF16_LOGITS:-0} \
  -w /workspace/Reef \
  "$IMAGE" \
  bash -c "$SERVE_CMD"
# (reef.log lands in $STATE_DIR on the host)
# REEF_BF16_LOGITS=1 keeps the policy's vocabulary logits bf16 instead of Megatron's fp32 upcast
# (docker/patch/mcore_bf16_logits.py; images from that commit on). Default 0 = unchanged.

echo "started $NAME from $IMAGE; log: docker logs -f $NAME  (also $STATE_DIR/reef.log)"
