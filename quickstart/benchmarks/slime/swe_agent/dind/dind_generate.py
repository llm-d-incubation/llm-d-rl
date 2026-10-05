"""slime coding_agent_rl generate, with DinD instead of E2B.

Replaces E2BSandbox in generate.py (agent) and swe.py (eval), then
re-exports generate. Point slime at this with:

    --custom-generate-function-path dind_generate.generate
"""

import examples.coding_agent_rl.generate as _gen
import examples.coding_agent_rl.swe as _swe

from dind_sandbox import DinDSandbox

_gen.E2BSandbox = DinDSandbox
_swe.E2BSandbox = DinDSandbox
generate = _gen.generate
