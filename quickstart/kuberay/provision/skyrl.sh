#!/usr/bin/env bash
# Provision SkyRL on one Ray node.   usage: skyrl.sh <head|worker>
#
# Single-pod layout with the GPUs on the head, so only "head" does real work.
# Also starts the router stack here, since SkyRL has no in-process hook to
# start EPP from - see llm-d-rl-router.
#
# SkyRL is not pip-installed onto a pre-built environment image: its own
# docker/Dockerfile (anyscale/ray:2.57.0-py312-cu130) ships no torch/vllm at
# all. Every dependency (torch, vllm, SkyRL itself) is resolved from SkyRL's
# own uv.lock at `uv run --isolated --extra fsdp ...` invocation time.
# Provisioning clones the repo and warms that uv-managed venv, so a training
# run's first invocation isn't also its first, multi-minute dependency
# resolve.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ROLE="${1:?usage: skyrl.sh <head|worker>}"
SKYRL_REPO="${SKYRL_REPO:-https://github.com/NovaSky-AI/SkyRL.git}"
SKYRL_REF="${SKYRL_REF:-main}"
ENGINE_PY_MODULE="${ENGINE_PY_MODULE:-vllm}"
LLMD_CONFIG_DIR="${LLMD_CONFIG_DIR:-/etc/llmd-configs}"
ENDPOINTS_FILE="${ENDPOINTS_FILE:-/tmp/epp-endpoints.yaml}"
SKYRL_SRC="/tmp/skyrl-src"

llmd_pep668
export PATH="/tmp/.local/bin:$PATH"

command -v uv >/dev/null || {
  llmd_log "installing uv (not on PATH on this image)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
}
command -v uv >/dev/null || llmd_fatal "uv still not on PATH after install"

if [[ -d "$SKYRL_SRC/.git" ]]; then
  llmd_log "$SKYRL_SRC already present"
else
  git init --quiet "$SKYRL_SRC" \
    && git -C "$SKYRL_SRC" remote add origin "$SKYRL_REPO" \
    && git -C "$SKYRL_SRC" fetch --quiet --depth=1 origin "$SKYRL_REF" \
    && git -C "$SKYRL_SRC" checkout --quiet FETCH_HEAD \
    || llmd_fatal "git fetch/checkout of $SKYRL_REPO@$SKYRL_REF into $SKYRL_SRC failed"
fi

# Warm the uv-managed venv now, at provisioning time, rather than on the first
# training invocation - `uv run --isolated` resolves + downloads the full
# fsdp extra (torch, vllm, flash-attn, ...) from SkyRL's own uv.lock on first
# use, which is a multi-minute network-bound step best surfaced here where a
# failure is an exit code, not a silently-slow first training step.
llmd_log "warming SkyRL's uv-managed venv (--extra fsdp; this can take several minutes on first run)"
(cd "$SKYRL_SRC" && uv run --isolated --extra fsdp python3 -c "import torch, vllm; print(f'torch={torch.__version__} vllm={vllm.__version__}')") \
  || llmd_fatal "uv run --isolated --extra fsdp failed to resolve torch/vllm"

# The shim needs an HTTP server, which is the [shim] extra rather than a hard
# dependency of common. Installed on system Python (not SkyRL's uv venv): the
# router/shim processes are launched by this script directly, outside any
# `uv run` invocation.
llmd_log "launching EPP + Envoy + registration shim"
llmd_install common
llmd_require_module llm_d_rl_common
python3 -c "import fastapi, uvicorn" 2>/dev/null \
  || pip install --no-cache-dir fastapi uvicorn >/dev/null \
  || llmd_fatal "pip install fastapi uvicorn failed"
llmd_require_command llm-d-rl-router
llmd_require_command llm-d-registration-shim

# EPP's file-discovery plugin crashes if the file is absent, so create it empty.
printf 'endpoints: []\n' > "$ENDPOINTS_FILE"

if [[ "$ROLE" == "head" ]]; then
  llmd_log "starting EPP + Envoy + registration shim"
  nohup llm-d-rl-router \
    --epp-config "$LLMD_CONFIG_DIR/epp-config.yaml" \
    --envoy-config "$LLMD_CONFIG_DIR/envoy.yaml" \
    >> /tmp/router.log 2>&1 &
  nohup llm-d-registration-shim \
    --engine-type vllm \
    --host 127.0.0.1 --port 3001 \
    --endpoints-file "$ENDPOINTS_FILE" \
    > /tmp/shim.log 2>&1 &
fi

# engine_version: read from the venv uv manages, not system Python - there is
# no engine module importable outside it, so llmd_write_marker's own
# llmd_module_version (system-Python import) can't be reused here. The venv
# warm above already proved this exact command succeeds, so a failure here
# is a real problem, not an expected "module absent" case.
if ! engine_version="$(cd "$SKYRL_SRC" && uv run --isolated --extra fsdp python3 -c \
  "import ${ENGINE_PY_MODULE}; print(getattr(${ENGINE_PY_MODULE}, '__version__', 'unknown'))")"; then
  llmd_fatal "failed to import ${ENGINE_PY_MODULE} in SkyRL's uv venv to read its version"
fi
llmd_log "${ENGINE_PY_MODULE} ${engine_version} (in SkyRL's uv venv)"

python3 - "$LLMD_MARKER" skyrl "$SKYRL_REF" "${LLMD_SOURCE:-git}" \
          "$ENGINE_PY_MODULE" "$engine_version" "$ROLE" <<'PY'
import json, sys, datetime
path, framework, fref, isrc, engine, eversion, role = sys.argv[1:8]
json.dump({
    "framework": framework,
    "framework_ref": fref,
    "integration_source": isrc,
    "engine": engine,
    "engine_version": eversion,
    "node_role": role,
    "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}, open(path, "w"), indent=2, sort_keys=True)
print(f"[provision] wrote {path}")
PY

llmd_log "skyrl provisioning complete on $ROLE"
