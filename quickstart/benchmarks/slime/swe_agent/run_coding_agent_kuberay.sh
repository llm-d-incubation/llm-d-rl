#!/usr/bin/env bash
# coding_agent_rl launcher for KubeRay single-pod layout.
#
# Adapted from slime/examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh:
#   - Ray cluster is managed by KubeRay; no ray start / SSH fanout.
#   - MASTER_ADDR and ADAPTER_PUBLIC_HOST come from MY_POD_IP.
#   - Tarball paths resolved from the PVC tools dir.
#   - No --colocate (hangs / errors on this KubeRay layout). Actor and
#     rollout are on separate GPU sets.
#
# Usage:
#   bash run_coding_agent_kuberay.sh                   slime's built-in sglang-router (default)
#   bash run_coding_agent_kuberay.sh --mode native     same
#   bash run_coding_agent_kuberay.sh --mode llm-d      route through Envoy+EPP (llm-d router)
#
# Run from inside the pod after `deploy.sh provision --framework slime`:
#   bash /tmp/slime-benchmarks/run_coding_agent_kuberay.sh
set -euo pipefail

MODE=native
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#--mode=}"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  llm-d|native) ;;
  *) echo "Unknown --mode: $MODE (use llm-d or native)" >&2; exit 2 ;;
esac

SLIME_DIR="${SLIME_DIR:-/tmp/slime-src}"
FW_HOME="${FW_HOME:-/tmp/slime}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark.env"
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

#  model parallelism
# Non-colocate: actor and rollout own separate GPU sets (--colocate hangs on
# this KubeRay layout). The sizes themselves live in benchmark.env.
export PP_SIZE=1
export CP_SIZE=1
export EP_SIZE=1
export ETP_SIZE=1

# Fail before Ray does: asking for more GPUs than the pod has shows up as a
# scheduling hang rather than an error.
_GPUS_WANTED=$(( ACTOR_NUM_GPUS + ROLLOUT_NUM_GPUS ))
_GPUS_VISIBLE="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
if [[ "${_GPUS_VISIBLE:-0}" -gt 0 && "$_GPUS_WANTED" -gt "$_GPUS_VISIBLE" ]]; then
  echo "ERROR: ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS} + ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}" \
       "= ${_GPUS_WANTED} GPUs, but this pod has ${_GPUS_VISIBLE}." >&2
  echo "       Reconcile benchmark.env with FW_slime_swe_agent_HEAD_GPUS in" \
       "kuberay/frameworks.env." >&2
  exit 1
fi

[[ -f "$PROMPT_DATA" ]] || {
  echo "ERROR: PROMPT_DATA not found at $PROMPT_DATA"
  echo "       Run setup_sandbox.sh locally first (it writes swe_train.jsonl to the PVC)"
  exit 1
}

# --hf-checkpoint: local HuggingFace snapshot (config.json + tokenizer + weights).
# Megatron and SGLang both read it. --ref-load is the same weights converted to
# Megatron torch_dist (required for the actor). Downloaded to the PVC once.
MODEL_DIR="${MODEL_DIR:-${FW_HOME}/models/${MODEL_NAME:-Qwen3-4B}}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${FW_HOME}/models/${MODEL_NAME:-Qwen3-4B}_torch_dist}"
export HF_HOME="${HF_HOME:-${FW_HOME}/hf_cache}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "=== Downloading ${MODEL_ID:-Qwen/Qwen3-4B} -> ${MODEL_DIR} ==="
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID:-Qwen/Qwen3-4B}',
    local_dir='${MODEL_DIR}', local_dir_use_symlinks=False)
"
else
  echo "=== HF checkpoint already present at ${MODEL_DIR} ==="
fi

if [[ ! -d "${TORCH_DIST_DIR}" ]] || [[ -z "$(ls -A "${TORCH_DIST_DIR}" 2>/dev/null)" ]]; then
  echo "=== Converting HF weights to torch_dist -> ${TORCH_DIST_DIR} ==="
  PYTHONPATH=/tmp/pyfix:"${SLIME_DIR}":/tmp/Megatron-LM \
    python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
      --hf-checkpoint "$MODEL_DIR" \
      --save "$TORCH_DIST_DIR" \
      "${MODEL_ARGS[@]}" \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0
else
  echo "=== torch_dist checkpoint already present at ${TORCH_DIST_DIR} ==="
fi

CKPT_ARGS=(
   --hf-checkpoint "$MODEL_DIR"
   --ref-load "$TORCH_DIST_DIR"
)

EXP_TAG="${EXP_TAG:-coding_agent_rl_${MODE}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
mkdir -p "${RUN_ROOT}/rollout_dumps"
# Per-run log directory on the PVC: <mode>-run-YYYYmmdd_HHMMSS.
RUN_DIR="${FW_HOME}/data/${MODE}-run-${STAMP}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/driver.log}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "RUN_DIR=${RUN_DIR}"

# For llm-d mode, EPP and Envoy are already running (started by provision/slime.sh).
# Forward their logs into the run dir for later inspection.
_TAIL_PIDS=()
if [[ "$MODE" == "llm-d" ]]; then
  for _src in /tmp/epp.log /tmp/envoy.log; do
    _dst="${RUN_DIR}/$(basename "${_src}")"
    touch "${_src}" "${_dst}"
    tail -F "${_src}" >> "${_dst}" &
    _TAIL_PIDS+=($!)
  done
fi
_cleanup() {
  [[ ${#_TAIL_PIDS[@]} -gt 0 ]] && kill "${_TAIL_PIDS[@]}" 2>/dev/null || true
}
trap _cleanup EXIT

# tarballs (resolved from PVC tools dir
TOOLS_DIR="${FW_HOME}/tools"
SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_NODE_TARBALL:-$(ls "${TOOLS_DIR}"/node-v22*.tar.xz 2>/dev/null | head -1)}"
SLIME_AGENT_CC_TARBALL="${SLIME_AGENT_CC_TARBALL:-$(ls "${TOOLS_DIR}"/anthropic-ai-claude-code-*.tgz 2>/dev/null | head -1)}"
[[ -n "$SLIME_AGENT_NODE_TARBALL" ]] || { echo "ERROR: node tarball not found in ${TOOLS_DIR}"; exit 1; }
[[ -n "$SLIME_AGENT_CC_TARBALL"   ]] || { echo "ERROR: claude-code tarball not found in ${TOOLS_DIR}"; exit 1; }
export SLIME_AGENT_NODE_TARBALL SLIME_AGENT_CC_TARBALL

# network
export MASTER_ADDR="${MASTER_ADDR:-${MY_POD_IP:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-6379}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# SWE / claude-code knobs 
_RUNNERS_FILE="${FW_HOME}/data/sandbox_runners.txt"
[[ -f "$_RUNNERS_FILE" ]] || {
  echo "error: $_RUNNERS_FILE missing — run setup_sandbox.sh first" >&2
  exit 1
}
export DIND_SANDBOX_SERVICE_URLS="$(cat "$_RUNNERS_FILE")"
[[ -n "$DIND_SANDBOX_SERVICE_URLS" ]] || {
  echo "error: $_RUNNERS_FILE is empty" >&2
  exit 1
}
echo "=== Sandbox runners: ${DIND_SANDBOX_SERVICE_URLS} ==="
_N_RUNNERS="$(awk -F, '{print NF}' <<< "$DIND_SANDBOX_SERVICE_URLS")"
# dind_sandbox.py reads this on the Ray workers, so it has to reach them via
# RUNTIME_ENV_JSON below.
export SANDBOX_ASSIGNMENT_FILE="${SANDBOX_ASSIGNMENT_FILE:-${FW_HOME}/data/sandbox_assignment.json}"

export SWE_AGENT="${SWE_AGENT:-claude_code}"
# 172.17.0.1 is the Docker bridge gateway inside the sandbox-runner pod's
# network namespace. DinD containers reach it directly; sandbox-api relays
# those requests to the real adapter via cluster DNS.
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-172.17.0.1}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"
# Slime's own cap on simultaneous sandbox boots (generate.py, default 16). It is
# fleet-wide, so the right value is the per-pod cap times the runner count;
# slime's default would serialise boots across the whole pool.
export SWE_BOOT_CONCURRENCY=$(( _N_RUNNERS * SANDBOX_MAX_PER_POD ))

SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":80000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"

export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

# rollout args
# Sizes come from benchmark.env. 
# NUM_ROLLOUT = training steps
# number of replicas = ROLLOUT_NUM_GPUS / ROLLOUT_TP_SIZE.
# global-batch-size = rollout_batch_size * n_samples (sequences per training step)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( ROLLOUT_BATCH_SIZE * N_SAMPLES ))}"

ROLLOUT_ARGS=(
   --custom-generate-function-path dind_generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-temperature 1.0
   --rollout-stop-token-ids 151645 151643
   --num-steps-per-rollout 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu "${MAX_CONTEXT_LEN}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}"
   --sglang-cpu-offload-gb "${ROLLOUT_CPU_OFFLOAD_GB}"
   --sglang-dp-size "${ROLLOUT_DP_SIZE}"
   --sglang-tool-call-parser qwen25
   --sglang-reasoning-parser qwen3
   --sglang-enable-metrics
)
if (( ROLLOUT_HICACHE_SIZE > 0 )); then
  SGLANG_ARGS+=(
    --sglang-enable-hierarchical-cache
    --sglang-hicache-size "${ROLLOUT_HICACHE_SIZE}"
    --sglang-hicache-io-backend kernel
    --sglang-hicache-write-policy write_through
  )
fi

ROUTER_ARGS=()
if [[ "$MODE" == "llm-d" ]]; then
  ROUTER_ARGS+=(--sglang-router-ip "${MASTER_ADDR}" --sglang-router-port 8081)
else
  ROUTER_ARGS+=(--router-balance-abs-threshold 0)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --rm-type deepscaler
)

# Ray workers
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_AGENT", "SWE_TRAIN_PROTOCOL",
    "ADAPTER_PUBLIC_HOST",
    "SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_CC_PROMPT",
    "DIND_SANDBOX_SERVICE_URLS", "SANDBOX_ASSIGNMENT_FILE",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
# ConfigMap is read-only, so skip .pyc writes next to dind_*.py.
env["PYTHONPATH"] = "/tmp/pyfix:/etc/llmd-configs:/tmp/Megatron-LM:/tmp/slime-src"
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)


echo "=== Submitting training job (mode: $MODE) ==="
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${ROUTER_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
