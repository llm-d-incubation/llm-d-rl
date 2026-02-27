#!/usr/bin/env bash
# run_sweep.sh — Run orchestration overhead benchmarks at 1/2/4 engine scale.
#
# Prerequisites:
#   - kubectl configured for CKS cluster
#   - Namespace llm-d-rl exists with hf-token and ghcr-creds secrets
#   - Rollout controller deployed (for llm-d-rl runs)
#   - vLLM engines deployed via vllm-engine-8b.yaml
#
# Usage:
#   ./run_sweep.sh [--results-dir ./results]

set -euo pipefail

RESULTS_DIR="${1:-./results}"
NAMESPACE="llm-d-rl"
WARMUP=5
MEASURED=20
ENGINE_BASE="vllm-engine"

mkdir -p "$RESULTS_DIR"

log() { echo "$(date '+%H:%M:%S') [sweep] $*"; }

wait_for_engines() {
    local count=$1
    log "Waiting for $count engines to be ready..."
    kubectl -n "$NAMESPACE" wait --for=condition=ready pod \
        -l app=vllm-engine --timeout=600s 2>/dev/null || true
    local ready
    ready=$(kubectl -n "$NAMESPACE" get pods -l app=vllm-engine \
        --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')
    while [ "$ready" -lt "$count" ]; do
        log "  $ready/$count engines ready, waiting..."
        sleep 15
        ready=$(kubectl -n "$NAMESPACE" get pods -l app=vllm-engine \
            --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')
    done
    log "  $ready engines ready"
}

build_engine_urls() {
    local count=$1
    local urls=""
    for i in $(seq 0 $((count - 1))); do
        if [ -n "$urls" ]; then urls="$urls,"; fi
        urls="${urls}http://${ENGINE_BASE}-${i}.${ENGINE_BASE}.${NAMESPACE}.svc.cluster.local:8000"
    done
    echo "$urls"
}

run_llmd_bench() {
    local engines=$1
    local output_file="llmd_${engines}engines.json"
    log "=== llm-d-rl benchmark: $engines engine(s) ==="

    # Delete previous job if exists
    kubectl -n "$NAMESPACE" delete job llmd-bench --ignore-not-found=true 2>/dev/null

    # Update controller engine list
    local engine_list
    engine_list=$(build_engine_urls "$engines")
    log "  Engine URLs: $engine_list"

    # Apply benchmark job
    kubectl apply -f deploy/cks/bench-trainer-job.yaml

    # Wait for job to complete
    log "  Waiting for llmd-bench job..."
    kubectl -n "$NAMESPACE" wait --for=condition=complete job/llmd-bench --timeout=1800s

    # Copy results
    local pod
    pod=$(kubectl -n "$NAMESPACE" get pods -l app=llmd-bench \
        --field-selector=status.phase=Succeeded -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$pod" ]; then
        kubectl cp "$NAMESPACE/$pod:/results/llmd_results.json" "$RESULTS_DIR/$output_file"
        log "  Results saved to $RESULTS_DIR/$output_file"
    else
        log "  WARNING: Could not find completed pod to copy results"
        # Try to get results from logs
        kubectl -n "$NAMESPACE" logs job/llmd-bench > "$RESULTS_DIR/llmd_${engines}engines.log"
    fi

    kubectl -n "$NAMESPACE" delete job llmd-bench --ignore-not-found=true 2>/dev/null
}

run_ray_bench() {
    local engines=$1
    local output_file="ray_${engines}engines.json"
    log "=== Ray benchmark: $engines engine(s) ==="

    # Delete previous job if exists
    kubectl -n "$NAMESPACE" delete job ray-bench --ignore-not-found=true 2>/dev/null

    # Build engine URL list
    local engine_list
    engine_list=$(build_engine_urls "$engines")

    # Patch the ENGINE_URLS env var in the job
    cat deploy/cks/ray-trainer-job.yaml | \
        sed "s|value: \"http://vllm-engine-0.*\"|value: \"$engine_list\"|" | \
        kubectl apply -f -

    # Wait for job to complete
    log "  Waiting for ray-bench job..."
    kubectl -n "$NAMESPACE" wait --for=condition=complete job/ray-bench --timeout=1800s

    # Copy results
    local pod
    pod=$(kubectl -n "$NAMESPACE" get pods -l app=ray-bench \
        --field-selector=status.phase=Succeeded -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$pod" ]; then
        kubectl cp "$NAMESPACE/$pod:/results/ray_results.json" "$RESULTS_DIR/$output_file"
        log "  Results saved to $RESULTS_DIR/$output_file"
    else
        log "  WARNING: Could not find completed pod to copy results"
        kubectl -n "$NAMESPACE" logs job/ray-bench > "$RESULTS_DIR/ray_${engines}engines.log"
    fi

    kubectl -n "$NAMESPACE" delete job ray-bench --ignore-not-found=true 2>/dev/null
}

# --- Main sweep ---

for ENGINE_COUNT in 1 2 4; do
    log ""
    log "=============================================="
    log "Scale: $ENGINE_COUNT engine(s)"
    log "=============================================="

    # Scale vLLM StatefulSet
    kubectl -n "$NAMESPACE" scale statefulset vllm-engine --replicas="$ENGINE_COUNT"
    wait_for_engines "$ENGINE_COUNT"

    # Need to restart controller to pick up new engine list for llm-d-rl runs
    # (controller reads --engines at startup)
    ENGINE_URLS=$(build_engine_urls "$ENGINE_COUNT")
    log "Restarting controller with engines: $ENGINE_URLS"

    # Patch controller deployment with new engine list
    kubectl -n "$NAMESPACE" set env deployment/rollout-controller \
        ENGINE_URLS="$ENGINE_URLS" 2>/dev/null || true

    # Run llm-d-rl benchmark
    run_llmd_bench "$ENGINE_COUNT"

    # Need to restart vLLM engines to clear NCCL state between runs
    log "Restarting vLLM engines to clear NCCL state..."
    kubectl -n "$NAMESPACE" rollout restart statefulset vllm-engine
    wait_for_engines "$ENGINE_COUNT"

    # Run Ray benchmark
    run_ray_bench "$ENGINE_COUNT"

    # Restart engines again for next scale
    if [ "$ENGINE_COUNT" -lt 4 ]; then
        log "Restarting vLLM engines for next scale..."
        kubectl -n "$NAMESPACE" rollout restart statefulset vllm-engine
        sleep 10
    fi
done

log ""
log "=============================================="
log "Sweep complete. Results in $RESULTS_DIR/"
log "=============================================="
ls -la "$RESULTS_DIR/"

log ""
log "Run analysis: python benchmarks/orchestration/analyze.py --results-dir $RESULTS_DIR"
