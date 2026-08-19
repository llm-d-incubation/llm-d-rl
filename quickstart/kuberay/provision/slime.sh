#!/usr/bin/env bash
# Provision slime on one Ray node.   usage: slime.sh <head|worker>
#
# slime's layout is a single pod with the GPUs on the head, so only "head" does
# real work. It also starts the router stack here, because slime has no in-process
# hook to start EPP from - see llm-d-rl-router.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ROLE="${1:?usage: slime.sh <head|worker>}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-main}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.16.0}"
ENGINE_PY_MODULE="${ENGINE_PY_MODULE:-sglang}"
LLMD_CONFIG_DIR="${LLMD_CONFIG_DIR:-/etc/llmd-configs}"
ENDPOINTS_FILE="${ENDPOINTS_FILE:-/tmp/epp-endpoints.yaml}"

llmd_pep668
export PATH="/tmp/.local/bin:$PATH"

# slime is used from PYTHONPATH rather than pip installed.
git clone --quiet --depth=1 -b "$SLIME_REF" "$SLIME_REPO" /tmp/slime-src 2>/dev/null || llmd_log "/tmp/slime-src already present"
git clone --quiet --depth=1 -b "$MEGATRON_REF" https://github.com/NVIDIA/Megatron-LM.git /tmp/Megatron-LM 2>/dev/null || llmd_log "/tmp/Megatron-LM already present"

# Editable-install finders raise PermissionError on some paths under Ray's
# spawned workers; drop the offending finder rather than the whole import.
mkdir -p /tmp/pyfix
printf 'import sys\nclass _S:\n def __init__(self,f):self._f=f\n def find_spec(self,*a,**k):\n  try:return self._f.find_spec(*a,**k)\n  except(PermissionError,OSError):return None\nsys.meta_path[:]=[_S(f) if "editable" in getattr(type(f),"__module__","") else f for f in sys.meta_path]\n' > /tmp/pyfix/sitecustomize.py

# The shim needs an HTTP server, which is the [shim] extra rather than a hard
# dependency of common (verl's in-process path never imports it).
llmd_install common
llmd_require_module llm_d_rl_common
python3 -c "import fastapi, uvicorn" 2>/dev/null || pip install --no-cache-dir fastapi uvicorn >/dev/null
llmd_require_command llm-d-rl-router
llmd_require_command llm-d-registration-shim
llmd_require_module "$ENGINE_PY_MODULE"

# EPP's file-discovery plugin crashes if the file is absent, so create it empty.
printf 'endpoints: []\n' > "$ENDPOINTS_FILE"

if [[ "$ROLE" == "head" ]]; then
  llmd_log "starting EPP + Envoy + registration shim"
  nohup llm-d-rl-router \
    --epp-config "$LLMD_CONFIG_DIR/epp-config.yaml" \
    --envoy-config "$LLMD_CONFIG_DIR/envoy.yaml" \
    </dev/null >> /tmp/router.log 2>&1 &
  nohup llm-d-registration-shim \
    --engine-type sglang --id-field id \
    --host 127.0.0.1 --port 3001 \
    --endpoints-file "$ENDPOINTS_FILE" \
    </dev/null > /tmp/shim.log 2>&1 &
fi

llmd_write_marker slime "$SLIME_REF" "$ENGINE_PY_MODULE" "$ROLE"
llmd_log "slime provisioning complete on $ROLE"
