#!/usr/bin/env bash
# One-shot setup for coding_agent_rl on SWE-rebench.
# Sources benchmark.env, deploys sandbox-runner pods, prepares dataset, prefetches images.
#
# Run after:
#   deploy.sh apply     --framework slime-swe-agent
#   deploy.sh provision --framework slime-swe-agent
#
# Usage:
#   NAMESPACE=<ns> bash setup_sandbox.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBERAY="$(cd "$SCRIPT_DIR/../../../kuberay" && pwd)"
TEMPLATE="$SCRIPT_DIR/manifests/sandbox-runner.yaml.tmpl"

source "$SCRIPT_DIR/benchmark.env"

# CLUSTER_NAME (deploy.env) and IMAGES_PVC_NAME (frameworks.env) go into the
# runner template. Source those files.
set -a
# shellcheck disable=SC1091
. "$KUBERAY/frameworks.env"
# shellcheck disable=SC1091
. "$KUBERAY/deploy.env"
set +a
IMAGES_PVC_NAME="${IMAGES_PVC_NAME:-${FW_slime_swe_agent_IMAGES_PVC_NAME:-}}"

: "${NAMESPACE:?export NAMESPACE=<your-namespace> first}"
: "${ROLLOUT_BATCH_SIZE:?}"
: "${N_SAMPLES:?}"
: "${NUM_ROLLOUT:?}"
: "${SANDBOX_MAX_PER_POD:?}"
: "${DIND_DAEMON_CPUS:?}"
: "${DIND_DAEMON_MEM_GI:?}"
: "${DIND_STORAGE_LIMIT:?}"
: "${CLUSTER_NAME:?not set in ${KUBERAY}/deploy.env}"
: "${IMAGES_PVC_NAME:?FW_slime_swe_agent_IMAGES_PVC_NAME not set in ${KUBERAY}/frameworks.env}"

# Calculate sandbox runner count
# N_SAMPLES rollouts of one problem share one image (GRPO), so they must share a runner.
# images_per_runner = floor(SANDBOX_MAX_PER_POD / N_SAMPLES)
# nrunners          = ceil(ROLLOUT_BATCH_SIZE / images_per_runner)
(( N_SAMPLES > 0 )) || { echo "error: N_SAMPLES must be > 0" >&2; exit 1; }
(( SANDBOX_MAX_PER_POD >= N_SAMPLES )) || {
  echo "error: one GRPO group (${N_SAMPLES} samples) exceeds SANDBOX_MAX_PER_POD=${SANDBOX_MAX_PER_POD}" >&2
  exit 1
}
_images_per_runner=$(( SANDBOX_MAX_PER_POD / N_SAMPLES ))
_nrunners=$(( (ROLLOUT_BATCH_SIZE + _images_per_runner - 1) / _images_per_runner ))
_peak=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES ))
_total_images=$(( NUM_ROLLOUT * ROLLOUT_BATCH_SIZE ))

echo ""
echo "Sandbox runner plan:"
printf "  %s images/step × %s samples/group = %s concurrent containers\n" \
       "$ROLLOUT_BATCH_SIZE" "$N_SAMPLES" "$_peak"
printf "  %s step(s) × %s images/step = %s unique images to prefetch\n" \
       "$NUM_ROLLOUT" "$ROLLOUT_BATCH_SIZE" "$_total_images"
printf "  floor(%s containers/pod ÷ %s samples/group) = %s images/pod\n" \
       "$SANDBOX_MAX_PER_POD" "$N_SAMPLES" "$_images_per_runner"
printf "  ceil(%s images ÷ %s images/pod) = %s sandbox-runner pod(s)\n" \
       "$ROLLOUT_BATCH_SIZE" "$_images_per_runner" "$_nrunners"
echo ""
read -r -p "Deploy ${_nrunners} sandbox-runner pod(s)? [y/N] " _confirm
[[ "$_confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# Deploy sandbox-runner pods
# Delete leftover runners with index >= nrunners (e.g. 8 → 4). prepare_sandbox.sh
# uses every sandbox-runner-* it finds, so extras would join this run's pool.
_stale=()
while IFS= read -r _name; do
  (( "${_name##sandbox-runner-}" >= _nrunners )) && _stale+=("deployment/$_name" "service/$_name")
done < <(kubectl get deployment -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" \
           | grep "^sandbox-runner-[0-9]\+$" || true)
if (( ${#_stale[@]} )); then
  echo "Removing ${#_stale[@]} stale sandbox-runner resource(s) above index $(( _nrunners - 1 ))"
  kubectl delete -n "$NAMESPACE" --ignore-not-found "${_stale[@]}"
fi

# dind sidecar: all sandboxes plus DIND_DAEMON_* from benchmark.env.
# CPU request = limit
export SANDBOX_MEMORY="${SANDBOX_MEM_GI}g"
export DIND_CPU_LIMIT=$(( SANDBOX_CPUS * SANDBOX_MAX_PER_POD + DIND_DAEMON_CPUS ))
export DIND_CPU_REQUEST="$DIND_CPU_LIMIT"
export DIND_MEM_LIMIT="$(( SANDBOX_MEM_GI * SANDBOX_MAX_PER_POD + DIND_DAEMON_MEM_GI ))Gi"
export IMAGES_PVC_NAME DIND_STORAGE_LIMIT

for _i in $(seq 0 $(( _nrunners - 1 ))); do
  RUNNER_INDEX="$_i" envsubst \
    '${NAMESPACE} ${CLUSTER_NAME} ${RUNNER_INDEX} ${IMAGES_PVC_NAME}
     ${SANDBOX_CPUS} ${SANDBOX_MEMORY} ${SANDBOX_PIDS}
     ${DIND_CPU_REQUEST} ${DIND_CPU_LIMIT} ${DIND_MEM_LIMIT} ${DIND_STORAGE_LIMIT}' \
    < "$TEMPLATE" \
  | kubectl apply -f -
done

# Prepare dataset and prefetch images
bash "$SCRIPT_DIR/scripts/prepare_sandbox.sh"
