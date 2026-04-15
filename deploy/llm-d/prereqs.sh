#!/bin/bash
# Install cluster-level prerequisites for the llm-d inference stack.
#
# This script installs:
#   1. Gateway API CRDs (v1.4.0)
#   2. Gateway API Inference Extension CRDs (v1.3.0) — adds InferencePool
#   3. Istio gateway controller with inference extension support
#
# Requires: kubectl, helm
# Run once per cluster. Safe to re-run (idempotent).

set -e
set -o pipefail

GATEWAY_API_CRD_VERSION="${GATEWAY_API_CRD_VERSION:-v1.4.0}"
GAIE_CRD_VERSION="${GAIE_CRD_VERSION:-v1.3.0}"
ISTIO_VERSION="${ISTIO_VERSION:-1.28.1}"

echo "=== Step 1: Gateway API CRDs (${GATEWAY_API_CRD_VERSION}) ==="
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api/config/crd/?ref=${GATEWAY_API_CRD_VERSION}"

echo ""
echo "=== Step 2: Inference Extension CRDs (${GAIE_CRD_VERSION}) ==="
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd/?ref=${GAIE_CRD_VERSION}"

echo ""
echo "=== Step 3: Istio gateway controller (${ISTIO_VERSION}) ==="
helm repo add istio https://istio-release.storage.googleapis.com/charts 2>/dev/null || true
helm repo update istio

helm upgrade --install istio-base istio/base \
  --version "${ISTIO_VERSION}" \
  --namespace istio-system \
  --create-namespace

helm upgrade --install istiod istio/istiod \
  --version "${ISTIO_VERSION}" \
  --namespace istio-system \
  --set meshConfig.defaultConfig.proxyMetadata.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true \
  --set pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true \
  --set tag="${ISTIO_VERSION}" \
  --set hub=docker.io/istio

echo ""
echo "=== Verification ==="
echo "Checking InferencePool CRD..."
kubectl api-resources --api-group=inference.networking.k8s.io 2>/dev/null || echo "WARNING: InferencePool CRD not found"

echo ""
echo "Done. Cluster prerequisites installed."
