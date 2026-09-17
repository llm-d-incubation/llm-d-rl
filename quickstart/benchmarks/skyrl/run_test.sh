#!/usr/bin/env bash
# run_test.sh  --mode <native|epp>  [options]
#
# Usage examples:
#   bash run_test.sh --mode native
#   bash run_test.sh --mode epp
#   bash run_test.sh --mode epp --steps 3
#
# Options:
#   --mode        native | epp                  (required)
#   --steps       total_training_steps          (default: 3)
#   --task        any folder under workloads/   (default: gsm8k)
#   --name        override experiment name      (default: auto-generated)
#   --weight-sync nccl | delta                  (default: nccl; see EPP mode note below)
#
# native: SkyRL's documented GSM8K quick start as-is - colocated
#   (trainer.placement.colocate_all=true), 4 GPUs shared by training and
#   4 in-process vLLM engines (generator.inference_engine.run_engines_locally=
#   true, the default). No llm-d routing.
#
# epp: llm-d EPP + Envoy in the data path. SkyRL forbids colocate_all=true
#   together with external_proxy_url/external_server_urls
#   (skyrl/train/utils/utils.py:_validate_new_inference_cfg), so this mode is
#   necessarily disaggregated: 2 GPUs for FSDP (policy/ref/critic) + 2 GPUs for
#   2 standalone vLLM engines (TP=1) this script starts itself via
#   `skyrl.train.entrypoints.serve` and registers with
#   llm-d-registration-shim. generator.inference_engine.external_proxy_url
#   points at Envoy (127.0.0.1:8081, started by provision/skyrl.sh via
#   llm-d-rl-router); Envoy's ext_proc call to EPP picks the replica,
#   ORIGINAL_DST routes the request there - no vllm-router in this mode, EPP
#   replaces it. external_server_urls (the 2 vLLM engines' own ports) is the
#   control-plane fan-out for pause/resume/sleep/wake/weight-sync.
#
#   --weight-sync nccl (default) hits a reproducible failure in
#   init_weight_sync_state - see integrations/skyrl-plan.md. --weight-sync
#   delta uses disk-based checkpoint deltas instead as a workaround.
set -euo pipefail

MODE=""
STEPS=3
CUSTOM_NAME=""
TASK="gsm8k"
WEIGHT_SYNC="nccl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2";        shift 2 ;;
    --steps)       STEPS="$2";       shift 2 ;;
    --task)        TASK="$2";        shift 2 ;;
    --name)        CUSTOM_NAME="$2"; shift 2 ;;
    --weight-sync) WEIGHT_SYNC="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "$WEIGHT_SYNC" != "nccl" && "$WEIGHT_SYNC" != "delta" ]]; then
  echo "ERROR: --weight-sync must be nccl or delta" >&2
  exit 1
fi

if [[ "$MODE" != "native" && "$MODE" != "epp" ]]; then
  echo "ERROR: --mode is required (native | epp)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- task config: sourced from workloads/<task>/task.env -----------------------
WORKLOADS_DIR="${WORKLOADS_DIR:-}"
if [[ -z "$WORKLOADS_DIR" ]]; then
  if [[ -d "$SCRIPT_DIR/workloads" ]]; then
    WORKLOADS_DIR="$(cd "$SCRIPT_DIR/workloads" && pwd)"
  elif [[ -d /tmp/workloads ]]; then
    WORKLOADS_DIR=/tmp/workloads
  fi
fi
TASK_ENV="$WORKLOADS_DIR/$TASK/task.env"
if [[ ! -f "$TASK_ENV" ]]; then
  echo "ERROR: no task.env for --task '$TASK' (looked at: $TASK_ENV)" >&2
  echo "       available workloads: $(ls -1 "$WORKLOADS_DIR" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$TASK_ENV"

MODEL_RESOLVED="${MODEL_PATH:-$DEF_MODEL}"
TRAIN_DIR_RESOLVED="${TRAIN_DIR:-$DEF_TRAIN_DIR}"
MAXP_RESOLVED="${MAX_PROMPT_LENGTH:-$DEF_MAXP}"
MAXR_RESOLVED="${MAX_RESPONSE_LENGTH:-$DEF_MAXR}"

# -- locate the SkyRL checkout ---------------------------------------------
SKYRL_SRC_DIR="${SKYRL_SRC:-/tmp/skyrl-src}"
[[ -d "$SKYRL_SRC_DIR" ]] || {
  echo "ERROR: no SkyRL checkout at $SKYRL_SRC_DIR." >&2
  echo "       Is the cluster provisioned? kuberay/deploy.sh provision --framework skyrl" >&2
  exit 1
}
cd "$SKYRL_SRC_DIR"

# -- dataset: build once, reuse across runs ---------------------------------
if [[ ! -f "$TRAIN_DIR_RESOLVED/train.parquet" ]]; then
  echo "==> building gsm8k dataset -> $TRAIN_DIR_RESOLVED"
  uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir "$TRAIN_DIR_RESOLVED"
else
  echo "==> gsm8k dataset already present at $TRAIN_DIR_RESOLVED"
fi

EXPERIMENT_NAME="${CUSTOM_NAME:-skyrl_grpo_gsm8k_${MODE}_${STEPS}s}"
LOG_PATH="/tmp/skyrl-logs/${EXPERIMENT_NAME}"

COMMON_ARGS=(
  data.train_data="['$TRAIN_DIR_RESOLVED/train.parquet']"
  data.val_data="['$TRAIN_DIR_RESOLVED/validation.parquet']"
  trainer.algorithm.advantage_estimator="grpo"
  trainer.policy.model.path="$MODEL_RESOLVED"
  trainer.strategy=fsdp
  trainer.epochs=1
  trainer.eval_batch_size=256
  trainer.eval_before_train=false
  trainer.eval_interval=-1
  trainer.update_epochs_per_batch=1
  trainer.train_batch_size=256
  trainer.policy_mini_batch_size=64
  trainer.micro_forward_batch_size_per_gpu=32
  trainer.micro_train_batch_size_per_gpu=32
  trainer.ckpt_interval=-1
  trainer.max_prompt_length="$MAXP_RESOLVED"
  generator.sampling_params.max_generate_length="$MAXR_RESOLVED"
  trainer.policy.optimizer_config.lr=1.0e-6
  trainer.algorithm.use_kl_loss=true
  generator.batched=true
  environment.env_class=gsm8k
  generator.n_samples_per_prompt=8
  generator.inference_engine.gpu_memory_utilization=0.85
  generator.inference_engine.max_num_batched_tokens=16384
  trainer.logger="${LOGGER:-console}"
  trainer.project_name="${DEF_PROJECT}"
  trainer.run_name="$EXPERIMENT_NAME"
  trainer.resume_mode=null
  trainer.log_path="$LOG_PATH"
  trainer.ckpt_path="/tmp/skyrl/ckpts/${EXPERIMENT_NAME}"
  trainer.max_training_steps="$STEPS"
)

# delta weight-sync publishes checkpoint deltas to a shared local directory
# instead of an NCCL broadcast - both training and the vLLM engines run on
# this same pod/disk, so a plain /tmp path is a valid shared sync_dir.
DELTA_ARGS=()
if [[ "$WEIGHT_SYNC" == "delta" ]]; then
  DELTA_ARGS=(generator.inference_engine.delta_weight_sync.sync_dir="/tmp/skyrl/delta_sync/${EXPERIMENT_NAME}")
fi

if [[ "$MODE" == "native" ]]; then
  # SkyRL's documented quick start as-is: colocated, 4 in-process vLLM engines.
  echo "=== Submitting SkyRL GRPO/GSM8K training (mode: native, steps: $STEPS, weight-sync: $WEIGHT_SYNC) ==="
  uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
    "${COMMON_ARGS[@]}" \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.critic_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend="$WEIGHT_SYNC" \
    generator.inference_engine.num_engines=4 \
    generator.inference_engine.tensor_parallel_size=1 \
    "${DELTA_ARGS[@]}"
else
  # EPP mode: disaggregated (2 GPUs training + 2 GPUs inference); Envoy+EPP
  # front 2 standalone vLLM engines started below, registered with the shim
  # so EPP's endpoints YAML picks them up. Envoy listens on 127.0.0.1:8081
  # (started by provision/skyrl.sh via llm-d-rl-router; envoy-shim.yaml's
  # inference listener).
  ENVOY_PORT="${LLMD_ENVOY_PORT:-8081}"
  SHIM_URL="${LLMD_SHIM_URL:-http://127.0.0.1:3001}"
  NUM_EPP_ENGINES=2
  SERVE_LOG="/tmp/skyrl-serve-engines.log"

  # Launch 2 standalone vLLM servers (TP=1) via SkyRL's own
  # skyrl.train.entrypoints.serve. serve.py builds its own internal
  # vllm-router too, but nothing here sends traffic through it; only the
  # per-engine server_urls it prints are used. The training run's data plane
  # goes through Envoy+EPP instead (external_proxy_url below); its control
  # plane (pause/resume/sleep/wake/weight-sync) fans out directly to the
  # same server_urls via external_server_urls.
  echo "=== Starting $NUM_EPP_ENGINES standalone vLLM engine(s) for EPP routing ==="
  rm -f "$SERVE_LOG"
  # weight_sync_backend must match the trainer's below: the engine bakes it
  # into its WeightTransferConfig at startup (build_vllm_cli_args), so a
  # mismatch (e.g. engine=nccl, trainer=delta) surfaces as an obscure
  # collective_rpc failure at init_weight_sync_state time, not a config error.
  nohup uv run --isolated --extra fsdp -m skyrl.train.entrypoints.serve \
    trainer.policy.model.path="$MODEL_RESOLVED" \
    trainer.placement.colocate_all=false \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines="$NUM_EPP_ENGINES" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.gpu_memory_utilization=0.85 \
    generator.inference_engine.max_num_batched_tokens=16384 \
    generator.inference_engine.weight_sync_backend="$WEIGHT_SYNC" \
    trainer.log_path=/tmp/skyrl-serve-logs \
    > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!
  echo "==> serve.py started (pid $SERVE_PID), log: $SERVE_LOG"
  # Tear down the standalone engines whenever this script exits, success or not -
  # otherwise a second run's serve.py fights the first for GPUs/ports.
  trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

  # Wait for serve.py to print its "server_urls" line (engines up + internal
  # router built). Bounded: a stuck vLLM engine init should fail loudly, not
  # hang this script forever.
  DEADLINE=$((SECONDS + 600))
  VLLM_ENGINE_URLS=()
  while (( SECONDS < DEADLINE )); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "ERROR: serve.py (pid $SERVE_PID) exited before printing server_urls; see $SERVE_LOG" >&2
      exit 1
    fi
    line="$(grep -m1 'server_urls (control plane):' "$SERVE_LOG" || true)"
    if [[ -n "$line" ]]; then
      # e.g. "  server_urls (control plane): ['http://10.0.3.1:8000', 'http://10.0.3.1:8100']"
      mapfile -t VLLM_ENGINE_URLS < <(echo "$line" | grep -oE "http://[^'\"]+")
      break
    fi
    sleep 2
  done
  if [[ "${#VLLM_ENGINE_URLS[@]}" -eq 0 ]]; then
    echo "ERROR: timed out waiting for serve.py to report server_urls; see $SERVE_LOG" >&2
    exit 1
  fi
  echo "==> vLLM engine URLs: ${VLLM_ENGINE_URLS[*]}"

  # Register each engine with the shim so EPP's endpoints YAML picks it up.
  # Idempotent on the shim side (dedups by URL).
  for url in "${VLLM_ENGINE_URLS[@]}"; do
    echo "==> registering $url with shim ($SHIM_URL)"
    curl -sf -X POST "$SHIM_URL/workers" \
      -H 'Content-Type: application/json' \
      -d "{\"url\": \"$url\", \"worker_type\": \"regular\"}"
    echo
  done
  echo "==> $NUM_EPP_ENGINES engine(s) registered with EPP"

  echo "=== Submitting SkyRL GRPO/GSM8K training (mode: epp, steps: $STEPS, weight-sync: $WEIGHT_SYNC) ==="
  uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
    "${COMMON_ARGS[@]}" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.critic_num_gpus_per_node=2 \
    trainer.placement.ref_num_gpus_per_node=2 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=false \
    generator.inference_engine.external_proxy_url="http://127.0.0.1:${ENVOY_PORT}" \
    generator.inference_engine.external_server_urls="[$(IFS=,; echo "${VLLM_ENGINE_URLS[*]}")]" \
    generator.inference_engine.weight_sync_backend="$WEIGHT_SYNC" \
    generator.inference_engine.num_engines=2 \
    generator.inference_engine.tensor_parallel_size=1 \
    "${DELTA_ARGS[@]}"
fi
