#!/usr/bin/env bash
# Submit a Qwen3-4B training job on the running vime KubeRay cluster.
# scripts/run_on_head.sh copies this to the head as run_test.sh.
#
# Usage:
#   FRAMEWORK=vime scripts/run_on_head.sh --mode native
#   FRAMEWORK=vime scripts/run_on_head.sh --mode llm-d --steps 6 --tp 2 --n 4
#
# --mode epp is not supported (use native or llm-d). --task is ignored with a warning.
#
# Both modes run the same engine layout - 4 engines at TP=1
# (--rollout-num-gpus 4 --rollout-num-gpus-per-engine 1, set unconditionally
# below) - so the only difference between the two arms is who picks the endpoint.
set -euo pipefail

MODE=""
STEPS=500
TP=2
N=4
FORCE_DOWNLOAD=false
TASK_SKIP="WARNING: --task is skipped; this driver always runs Qwen3-4B"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)           MODE="$2"; shift 2 ;;
    --mode=*)         MODE="${1#--mode=}"; shift ;;
    --steps)          STEPS="$2"; shift 2 ;;
    --steps=*)        STEPS="${1#--steps=}"; shift ;;
    --tp)             TP="$2"; shift 2 ;;
    --tp=*)           TP="${1#--tp=}"; shift ;;
    --n)              N="$2"; shift 2 ;;
    --n=*)            N="${1#--n=}"; shift ;;
    --force-download) FORCE_DOWNLOAD=true; shift ;;
    --task)
      echo "$TASK_SKIP (got '$2')" >&2
      shift 2 ;;
    --task=*)
      echo "$TASK_SKIP (got '${1#--task=}')" >&2
      shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ]; then
  echo "Usage: $0 --mode native|llm-d [--steps N] [--tp TP] [--n N] [--force-download]" >&2
  exit 1
fi
case "$MODE" in
  epp|epp-*)
    echo "ERROR: --mode $MODE is not supported for vime (use native or llm-d)" >&2
    exit 2 ;;
  native|llm-d) ;;
  *) echo "Unknown --mode: $MODE (use native or llm-d)" >&2; exit 2 ;;
esac

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
echo "=== Submitting training job (mode: $MODE, steps: $STEPS) ==="

EXTRA_ARGS=()
if [ "$MODE" = "llm-d" ]; then
  EXTRA_ARGS=(
    --vllm-router-ip "${MY_POD_IP}"
    --vllm-router-port 8081
  )
else
  # vime default is cache_aware with balance_abs_threshold=10, which can
  # send every generate to one worker. vime docs: set this to 0 to force
  # even spread. llm-d does not use vllm-router, so this is native-only.
  EXTRA_ARGS=(
    --router-balance-abs-threshold 0
  )
fi

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/tmp/pyfix:/tmp/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1","TORCHINDUCTOR_CACHE_DIR":"/tmp/torch_inductor_cache","VLLM_TARGET_DEVICE":"cuda","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}}' \
  -- python3 /tmp/vime/train.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODEL_DIR" \
    --load "$MEGATRON_DIR" \
    --ref-load "$MEGATRON_DIR" \
    --prompt-data "$DATASET_DIR/${DATASET_NAME:-dapo-math-17k}.jsonl" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rm-type deepscaler \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$TP" \
    --tensor-model-parallel-size "$TP" \
    --sequence-parallel \
    --recompute-activations \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 1 \
    --rollout-batch-size 32 \
    --num-rollout "$STEPS" \
    --n-samples-per-prompt "$N" \
    --global-batch-size 128 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1 \
    --balance-data \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 9216 \
    --vllm-gpu-memory-utilization 0.7 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    "${EXTRA_ARGS[@]}"
