#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NAMESPACE:-}" ]]; then
  echo "Error: NAMESPACE is not set. Please export it before running this script:"
  echo "  export NAMESPACE=<your-namespace>"
  exit 1
fi

helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# This may fail if CRDs already exist, which is fine
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v1.5.1&timeout=90s" || true

helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 \
  --namespace "${NAMESPACE}" \
  --set singleNamespaceInstall=true \
  --skip-crds
