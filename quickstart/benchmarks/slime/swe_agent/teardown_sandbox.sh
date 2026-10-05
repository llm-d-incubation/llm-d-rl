#!/usr/bin/env bash
# Delete all sandbox-runner Deployments and Services without touching Ray,
# EPP, Envoy, the shared PVC, or the rest of the training stack.
#
# Usage:
#   NAMESPACE=<namespace> bash teardown_sandbox.sh

set -euo pipefail

: "${NAMESPACE:?export NAMESPACE=<your-namespace> first}"

resources=()
while IFS= read -r resource; do
  [[ "$resource" == */sandbox-runner-* ]] && resources+=("$resource")
done < <(
  kubectl get deployment,service -n "$NAMESPACE" \
    -o name 2>/dev/null
)

if (( ${#resources[@]} == 0 )); then
  echo "No sandbox-runner Deployments or Services found in ${NAMESPACE}."
  exit 0
fi

printf "Deleting %s sandbox-runner resource(s) from %s:\n" \
  "${#resources[@]}" "$NAMESPACE"
printf "  %s\n" "${resources[@]}"
kubectl delete -n "$NAMESPACE" --ignore-not-found "${resources[@]}"
