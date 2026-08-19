#!/usr/bin/env bash
# One KubeRay deployment for every RL framework in this repo.
#
# Usage:
#   ./deploy.sh apply      --framework verl [--engine vllm]   # ConfigMap + cluster
#   ./deploy.sh provision  --framework verl                   # install on every node
#   ./deploy.sh check      --framework verl                   # verify provisioning
#   ./deploy.sh render     --framework verl [--engine sglang]  # print the manifest
#   ./deploy.sh configmap  --framework verl                    # (re)create the ConfigMap
#   ./deploy.sh pvc        --framework verl                    # create the cache PVC
#   ./deploy.sh delete     --framework verl                    # delete the cluster
#                                                             # (never deletes the PVC)
#   ./deploy.sh retriever | retriever-delete | render-retriever # searchr1 helper (verl)
#
# Options:
#   --framework verl|vime|slime   which framework (default verl)
#   --engine vllm|sglang          which rollout engine, from that framework's columns
#   --from-local [DIR]            provision the working tree instead of git main
#
# Apply, then provision. `apply` starts Ray; nothing is installed at pod start, so
# pod Ready means "Ray is up", not "framework installed" - `check` is what tells
# you the latter, and ../benchmarks/scripts/run_on_head.sh refuses to launch
# without it.
#
# All frameworks render the SAME cluster name and are therefore mutually
# exclusive within a namespace: applying one replaces the other. Every script in
# ../benchmarks/scripts resolves pods with `-l ray.io/node-type=head` + items[0],
# which cannot tell two clusters apart.
#
# Requires: envsubst (GNU gettext), kubectl, python3 with pyyaml.
set -euo pipefail

ACTION="apply"
FRAMEWORK="verl"
ENGINE=""
FROM_LOCAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --framework) FRAMEWORK="${2:?--framework needs a value}"; shift 2 ;;
    --engine)    ENGINE="${2:?--engine needs a value}"; shift 2 ;;
    --from-local)
      if [[ "${2:-}" == --* || -z "${2:-}" ]]; then FROM_LOCAL="$(cd ../.. && pwd)"; shift
      else FROM_LOCAL="$(cd "$2" && pwd)"; shift 2; fi ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  ACTION="$1"; shift ;;
  esac
done

cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
INTEGRATIONS="$REPO_ROOT/integrations"
COMMON_CONFIGS="$INTEGRATIONS/common/src/llm_d_rl_common/configs"
COMMON_SRC="$INTEGRATIONS/common/src"

[[ -f "$INTEGRATIONS/$FRAMEWORK/environments.env" ]] || {
  echo "ERROR: unknown framework '$FRAMEWORK' (no integrations/$FRAMEWORK/environments.env)" >&2; exit 2; }

# Sourced widest-to-narrowest, so a later file can override an earlier one:
# routing stack -> framework/engine -> cluster shape -> this cluster.
set -a
# shellcheck disable=SC1091
. "$COMMON_CONFIGS/versions.env"
IMG_EPP="${IMG_EPP:-$LLMD_EPP_IMAGE}"
IMG_ENVOY="${IMG_ENVOY:-$LLMD_ENVOY_IMAGE}"
IMG_SIDECAR="${IMG_SIDECAR:-$LLMD_SIDECAR_IMAGE}"
# shellcheck disable=SC1091
. "$INTEGRATIONS/$FRAMEWORK/environments.env"
# shellcheck disable=SC1091
. ./frameworks.env
# shellcheck disable=SC1091
. ./deploy.env
set +a

: "${NAMESPACE:?not set - export NAMESPACE=<your-namespace>}"

# A framework may have to override the EPP image (slime needs a build carrying
# sglanghttp-parser). Refuse to deploy while that override is still a placeholder,
# rather than rendering an unpullable image and failing confusingly at pod start.
case "$ACTION" in
  render|render-pvc|render-retriever|render-epp) ;;
  *) case "${IMG_EPP:-}" in
       ""|*REPLACE*|*placeholder*|"<"*">")
         echo "ERROR: IMG_EPP is unset or a placeholder ('${IMG_EPP:-}')." >&2
         echo "       Set it in integrations/$FRAMEWORK/environments.env." >&2
         exit 2 ;;
     esac ;;
esac

fw() { local v="FW_${FRAMEWORK}_$1"; echo "${!v-}"; }

# Resolve the framework's cluster shape and its engine column into the names the
# templates use. Fails fast rather than rendering blank values into a manifest
# that would then fail confusingly at pod start.
resolve() {
  export FRAMEWORK
  TOPOLOGY="$(fw TOPOLOGY)"
  [[ -n "$TOPOLOGY" ]] || { echo "ERROR: FW_${FRAMEWORK}_TOPOLOGY not set in frameworks.env" >&2; exit 2; }
  export FW_HOME="$(fw HOME)"
  export FW_PYTHONPATH="$(fw PYTHONPATH)"
  export RAY_VERSION="$(fw RAY_VERSION)"
  export DSHM_SIZE="$(fw DSHM_SIZE)"
  export WORKER_GPUS="$(fw WORKER_GPUS)"
  export HEAD_GPUS="$(fw HEAD_GPUS)"
  export PVC_NAME="${PVC_NAME:-$(fw PVC_NAME)}"
  export CONFIGMAP_NAME="llmd-epp-configs-${FRAMEWORK}"

  local engines; engines="$(fw ENGINES)"
  ENGINE="${ENGINE:-${engines%% *}}"
  case " $engines " in *" $ENGINE "*) ;; *)
    echo "ERROR: $FRAMEWORK does not declare engine '$ENGINE' (has: $engines)" >&2; exit 2 ;;
  esac
  local img="ENGINE_${ENGINE}_IMAGE" py="ENGINE_${ENGINE}_PY_MODULE"
  local cpus="ENGINE_${ENGINE}_HEAD_NUM_CPUS" alloc="ENGINE_${ENGINE}_ALLOC_CONF"
  [[ -n "${!img:-}" ]] || {
    echo "ERROR: no ENGINE_${ENGINE}_IMAGE in integrations/$FRAMEWORK/environments.env" >&2; exit 2; }
  export IMG_TRAINER="${!img}"
  export ENGINE_PY_MODULE="${!py:-$ENGINE}"
  export ENGINE_HEAD_NUM_CPUS="${!cpus:-0}"
  export ENGINE_ALLOC_CONF="${!alloc-}"
  TEMPLATE="ray-cluster.${TOPOLOGY}.yaml.tmpl"
  [[ -f "$TEMPLATE" ]] || { echo "ERROR: no template $TEMPLATE" >&2; exit 2; }
}

render() {
  resolve
  # Explicit var list keeps envsubst away from the container-runtime $LLMD_BINS
  # expansions in the crane args.
  envsubst '${NAMESPACE} ${CLUSTER_NAME} ${FRAMEWORK} ${RAY_VERSION} ${IMG_TRAINER}
            ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY} ${IMG_SIDECAR} ${ENGINE_PY_MODULE}
            ${ENGINE_HEAD_NUM_CPUS} ${ENGINE_ALLOC_CONF} ${FW_HOME} ${FW_PYTHONPATH}
            ${CONFIGMAP_NAME} ${PVC_NAME} ${DSHM_SIZE} ${WORKER_GPUS} ${HEAD_GPUS}' \
    < "$TEMPLATE"
}

render_pvc() {
  resolve
  [[ -n "$PVC_NAME" ]] || { echo "ERROR: $FRAMEWORK declares no FW_${FRAMEWORK}_PVC_NAME" >&2; exit 2; }
  envsubst '${NAMESPACE} ${PVC_NAME}' < pvc.yaml
}

render_retriever() {
  envsubst '${NAMESPACE} ${IMG_RETRIEVER}' \
    < ../benchmarks/verl/workloads/searchr1/retriever/retriever.yaml.tmpl
}

create_configmap() {
  resolve
  # Every EPP variant this framework can use, merged from base.yaml + a profile +
  # modifiers (configs/epp/variants.yaml), then EPP_PARSER substituted. Adding a
  # variant is a line in variants.yaml; this function does not change.
  export EPP_PARSER="${EPP_PARSER:-$LLMD_EPP_PARSER_DEFAULT}"
  local dir; dir="$(mktemp -d)"
  # Expanded now, not at return: a RETURN trap fires after `local dir` is out of
  # scope, so "$dir" would be an unbound variable under set -u, and the trap's
  # failure becomes this function's exit status.
  trap "rm -rf '$dir'" RETURN
  local variants; variants="$(fw EPP_VARIANTS)"
  if [[ -n "$variants" ]]; then
    local v
    for v in $variants; do
      PYTHONPATH="$COMMON_SRC" python3 -m llm_d_rl_common.epp_config render "$v" -o "$dir/$v"
    done
  else
    PYTHONPATH="$COMMON_SRC" python3 -m llm_d_rl_common.epp_config render-all "$dir" >/dev/null
  fi
  local args=() f
  for f in "$dir"/*.yaml; do
    envsubst '${EPP_PARSER}' < "$f" > "$f.sub" && mv "$f.sub" "$f"
    args+=(--from-file="$(basename "$f")=$f")
  done
  args+=(--from-file="envoy.yaml=$COMMON_CONFIGS/$(fw ENVOY_CONFIG)")
  # Workload config a framework's own driver needs on the pod.
  local extra="../benchmarks/$FRAMEWORK/configmap-files.txt"
  if [[ -f "$extra" ]]; then
    while read -r line; do
      [[ -z "$line" || "$line" == \#* ]] && continue
      args+=(--from-file="$line")
    done < "$extra"
  fi
  kubectl create configmap "$CONFIGMAP_NAME" "${args[@]}" \
    --namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
}

# --- provisioning -------------------------------------------------------------
# kermit's konnectivity tunnel returns 502 intermittently and a kubectl exec
# through a broken tunnel HANGS rather than failing, so every call is bounded and
# retried.
K() { timeout "${KUBECTL_TIMEOUT:-120}" kubectl "$@"; }

# The pods have init containers, so kubectl picks one and says so on every call.
# Name it instead: KubeRay calls them ray-head and ray-worker.
container_for() { [[ "$1" == *-head-* ]] && echo ray-head || echo ray-worker; }

pods_of_role() {
  K get pod -n "$NAMESPACE" -l "ray.io/node-type=$1" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true
}

exec_retry() {
  local pod="$1"; shift
  local n=0
  local to="${KUBECTL_PROVISION_TIMEOUT:-900}"
  until timeout "$to" kubectl exec -n "$NAMESPACE" -c "$(container_for "$pod")" "$pod" -- "$@" </dev/null; do
    n=$((n+1)); [[ $n -ge 3 ]] && { echo "ERROR: exec failed 3x on $pod" >&2; return 1; }
    echo "  retrying on $pod ($n/3)" >&2; sleep 5
  done
}

provision() {
  resolve
  local heads workers all
  heads="$(pods_of_role head)"; workers="$(pods_of_role worker)"
  all="$(printf '%s\n%s\n' "$heads" "$workers" | grep -v '^$' || true)"
  [[ -n "$all" ]] || { echo "ERROR: no pods for cluster $CLUSTER_NAME in $NAMESPACE" >&2; exit 1; }

  local env_args=(FRAMEWORK="$FRAMEWORK" ENGINE_PY_MODULE="$ENGINE_PY_MODULE")
  local v
  for v in VERL_COMMIT VIME_REPO VIME_REF SLIME_REPO SLIME_REF MEGATRON_REF \
           LLMD_REPO LLMD_REPO_REF LLMD_LOCAL_SRC "ENGINE_${ENGINE}_MIN_VERSION"; do
    [[ -n "${!v:-}" ]] && env_args+=("$v=${!v}")
  done
  if [[ -n "$FROM_LOCAL" ]]; then
    env_args+=(LLMD_SOURCE=local)
    echo "==> provisioning from the local tree $FROM_LOCAL"
  fi

  local pod
  for pod in $all; do
    echo "==> shipping provision/ to $pod"
    local c; c="$(container_for "$pod")"
    K exec -n "$NAMESPACE" -c "$c" "$pod" -- mkdir -p /tmp/llmd-provision
    K cp -c "$c" provision "$NAMESPACE/$pod:/tmp/" >/dev/null
    if [[ -n "$FROM_LOCAL" ]]; then
      K exec -n "$NAMESPACE" -c "$c" "$pod" -- rm -rf "$LLMD_LOCAL_SRC"
      K exec -n "$NAMESPACE" -c "$c" "$pod" -- mkdir -p "$LLMD_LOCAL_SRC"
      K cp -c "$c" "$FROM_LOCAL/integrations" "$NAMESPACE/$pod:$LLMD_LOCAL_SRC/" >/dev/null
      # The benchmark harness is its own package under quickstart/, so a local
      # provision needs that tree too.
      K exec -n "$NAMESPACE" -c "$c" "$pod" -- mkdir -p "$LLMD_LOCAL_SRC/quickstart/benchmarks"
      K cp -c "$c" "$FROM_LOCAL/quickstart/benchmarks/verl" \
        "$NAMESPACE/$pod:$LLMD_LOCAL_SRC/quickstart/benchmarks/" >/dev/null
    fi
  done

  # The head goes first: on head-workers it owns the shared clone and the model
  # prefetch that the workers then install from.
  for pod in $heads; do
    echo "==> provisioning head $pod"
    exec_retry "$pod" env "${env_args[@]}" bash "/tmp/provision/$FRAMEWORK.sh" head
  done
  for pod in $workers; do
    echo "==> provisioning worker $pod"
    exec_retry "$pod" env "${env_args[@]}" bash "/tmp/provision/$FRAMEWORK.sh" worker
  done
  echo "==> provisioned; verifying"
  check
}

check() {
  resolve
  local all; all="$(printf '%s\n%s\n' "$(pods_of_role head)" "$(pods_of_role worker)" | grep -v '^$' || true)"
  [[ -n "$all" ]] || { echo "ERROR: no pods for cluster $CLUSTER_NAME in $NAMESPACE" >&2; return 1; }
  local pod json tmp; tmp="$(mktemp)"
  # See create_configmap: expand at trap-definition time, not at return.
  trap "rm -f '$tmp'" RETURN
  for pod in $all; do
    # The marker is pretty-printed on the pod for humans, so flatten it: this
    # goes into one tab-separated field per pod, and JSON ignores whitespace.
    if json="$(K exec -n "$NAMESPACE" -c "$(container_for "$pod")" "$pod" -- cat /tmp/llmd-provisioned.json 2>/dev/null | tr -d '\n')"; then
      printf '%s\t%s\n' "$pod" "$json" >> "$tmp"
    else
      printf '%s\t\n' "$pod" >> "$tmp"
    fi
  done
  python3 - "$tmp" "$FRAMEWORK" <<'PY'
import json, sys
rows = [l.rstrip("\n").split("\t", 1) for l in open(sys.argv[1]) if l.strip()]
want = sys.argv[2]
bad, seen = [], {}
for pod, raw in rows:
    if not raw.strip():
        print(f"  {pod}: NOT PROVISIONED"); bad.append(pod); continue
    try:
        d = json.loads(raw)
    except ValueError:
        print(f"  {pod}: UNREADABLE MARKER ({raw[:60]!r})"); bad.append(pod); continue
    print(f"  {pod}: {d['framework']}@{d['framework_ref'][:12]} "
          f"integration={d['integration_source'][:12]} {d['engine']} {d['engine_version']} "
          f"({d['node_role']})")
    if d["framework"] != want:
        print(f"      framework is {d['framework']}, expected {want}"); bad.append(pod)
    for k in ("framework", "framework_ref", "integration_source", "engine"):
        seen.setdefault(k, set()).add(d[k])
for k, vals in seen.items():
    if len(vals) > 1:
        print(f"  DISAGREEMENT on {k}: {sorted(vals)}"); bad.append(k)
if bad:
    print("\nFAIL: run `deploy.sh provision`", file=sys.stderr); sys.exit(1)
print(f"\nOK: {len(rows)} pod(s) provisioned and in agreement")
PY
}

case "$ACTION" in
  render)            render ;;
  configmap)         create_configmap ;;
  apply)             create_configmap; render | kubectl apply -f - ;;
  delete)            render | kubectl delete --ignore-not-found -f - ;;
  provision)         provision ;;
  check)             check ;;
  pvc)               render_pvc | kubectl apply -f - ;;
  render-pvc)        render_pvc ;;
  retriever)         render_retriever | kubectl apply -f - ;;
  retriever-delete)  render_retriever | kubectl delete --ignore-not-found -f - ;;
  render-retriever)  render_retriever ;;
  render-epp)        resolve; PYTHONPATH="$COMMON_SRC" python3 -m llm_d_rl_common.epp_config list ;;
  *) echo "Unknown action: $ACTION (apply | provision | check | delete | render | configmap | pvc | render-pvc | render-epp | retriever*)" >&2; exit 2 ;;
esac
