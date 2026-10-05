#!/usr/bin/env bash
# Provision coding_agent_rl (SWE-rebench variant) on one Ray node.
# Runs common slime provisioning first, then adds sandbox tooling.
# usage: slime_swe_agent.sh <head|worker>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ROLE="${1:?usage: slime_swe_agent.sh <head|worker>}"
FW_HOME="${FW_HOME:-/tmp/slime}"
SLIME_REF="${SLIME_REF:-main}"
ENGINE_PY_MODULE="${ENGINE_PY_MODULE:-sglang}"

# Common slime provisioning (EPP/Envoy, slime/Megatron clones, router)
FW_HOME="$FW_HOME" bash "$(dirname "${BASH_SOURCE[0]}")/slime.sh" "$@"

# httpx — used by dind/dind_sandbox.py (ConfigMap, /etc/llmd-configs) to talk to
# the sandbox-runner pods.
python3 -c "import httpx" 2>/dev/null || pip install --no-cache-dir httpx >/dev/null

# Tools cached on the PVC; uploaded into every sandbox at boot
TOOLS_DIR="${FW_HOME}/tools"
mkdir -p "$TOOLS_DIR"

# Node 22 runtime tarball — extracted inside each sandbox by install_node22().
NODE_VER="v22.20.0"
NODE_TARBALL="$TOOLS_DIR/node-${NODE_VER}-linux-x64.tar.xz"
if [[ ! -f "$NODE_TARBALL" ]]; then
  llmd_log "downloading Node ${NODE_VER} tarball -> ${TOOLS_DIR}"
  curl -fsSL "https://nodejs.org/dist/${NODE_VER}/node-${NODE_VER}-linux-x64.tar.xz" \
    -o "$NODE_TARBALL"
fi

# Claude Code CLI npm tarball — installed globally inside each sandbox.
if ! ls "$TOOLS_DIR"/anthropic-ai-claude-code-*.tgz 2>/dev/null | grep -q .; then
  llmd_log "downloading @anthropic-ai/claude-code tarball -> ${TOOLS_DIR}"
  CC_VERSION=$(curl -fsSL https://registry.npmjs.org/@anthropic-ai/claude-code/latest \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
  curl -fsSL "https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-${CC_VERSION}.tgz" \
    -o "$TOOLS_DIR/anthropic-ai-claude-code-${CC_VERSION}.tgz"
fi

# SWE-bench grader and Anthropic SDK
# Pin swebench to 3.0.17: v4+ moved make_test_spec out of swebench.harness.test_spec.test_spec.
python3 -c "import swebench.harness.test_spec.test_spec" 2>/dev/null \
  || pip install --no-cache-dir --break-system-packages 'swebench==3.0.17' >/dev/null
python3 -c "import anthropic" 2>/dev/null || pip install --no-cache-dir anthropic >/dev/null

llmd_write_marker slime-swe-agent "$SLIME_REF" "$ENGINE_PY_MODULE" "$ROLE"
llmd_log "slime-swe-agent provisioning complete on $ROLE"
