#!/usr/bin/env bash
# Submit a Qwen3-4B training job on the running vime KubeRay cluster.
# Run this from inside the head pod after the cluster is ready.
#
# Usage:
#   bash /etc/llmd-configs/run-qwen3-4B.sh --native
#       vime's built-in vLLM router; 1 engine with TP=2
#   bash /etc/llmd-configs/run-qwen3-4B.sh --llmd
#       llm-d EPP + Envoy routing; 2 engines with TP=1 (EPP routes between them)
set -euo pipefail

MODE=""
FORCE_DOWNLOAD=false
for arg in "$@"; do
  case "$arg" in
    --native)         MODE=native ;;
    --llmd)           MODE=llmd ;;
    --force-download) FORCE_DOWNLOAD=true ;;
  esac
done
if [ -z "$MODE" ]; then
  echo "Usage: $0 --native | --llmd [--force-download]" >&2
  exit 1
fi

MODEL_DIR="/tmp/vime/models/${MODEL_NAME:-Qwen3-4B}"
MEGATRON_DIR="/tmp/vime/models/${MODEL_NAME:-Qwen3-4B}_megatron"
DATASET_DIR="/tmp/vime/data/${DATASET_NAME:-dapo-math-17k}"

cd /tmp/vime
source scripts/models/qwen3-4B.sh

# --- 1. Download model + dataset (skipped if already present) ---
if [ ! -d "$MODEL_DIR" ] || [ "$FORCE_DOWNLOAD" = true ]; then
  echo "=== Downloading model ==="
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID:-Qwen/Qwen3-4B}',
    local_dir='$MODEL_DIR', local_dir_use_symlinks=False)
"
else
  echo "=== Model already present, skipping download ==="
fi

if [ ! -d "$DATASET_DIR" ] || [ "$FORCE_DOWNLOAD" = true ]; then
  echo "=== Downloading dataset ==="
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${DATASET_ID:-zhuzilin/dapo-math-17k}', repo_type='dataset',
    local_dir='$DATASET_DIR', local_dir_use_symlinks=False)
"
else
  echo "=== Dataset already present, skipping download ==="
fi

# --- 2. Convert HF weights to Megatron format (skipped if already done) ---
if [ ! -d "$MEGATRON_DIR" ]; then
  echo "=== Converting weights to Megatron format ==="
  PYTHONPATH=/tmp/pyfix:/tmp/Megatron-LM python3 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODEL_DIR" \
    --save "$MEGATRON_DIR"
else
  echo "=== Megatron weights already exist, skipping conversion ==="
fi

# --- 3. Submit training ---
echo "=== Submitting training job (mode: $MODE) ==="

EXTRA_ARGS=()
if [ "$MODE" = "llmd" ]; then
  EXTRA_ARGS=(
    --vllm-router-ip "${MY_POD_IP}"
    --vllm-router-port 8081
  )
fi

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/tmp/pyfix:/tmp/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1","TORCHINDUCTOR_CACHE_DIR":"/tmp/torch_inductor_cache","VLLM_TARGET_DEVICE":"cuda","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}}' \
  -- python3 /tmp/vime/train.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODEL_DIR" \
    --load "$MEGATRON_DIR" \
    --prompt-data "$DATASET_DIR/${DATASET_NAME:-dapo-math-17k}.jsonl" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rm-type deepscaler \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --recompute-activations \
    --rollout-num-gpus 2 \
    --rollout-num-gpus-per-engine 1 \
    --rollout-batch-size 16 \
    --num-rollout 3000 \
    --n-samples-per-prompt 8 \
    --num-steps-per-rollout 1 \
    --global-batch-size 128 \
    --rollout-max-response-len 4096 \
    --rollout-temperature 1 \
    --balance-data \
    --advantage-estimator grpo \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --vllm-gpu-memory-utilization 0.9 \
    "${EXTRA_ARGS[@]}"
