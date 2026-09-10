"""Bridge between uni_agent's rollout stack and the llm-d-router verl integration.

`AgentFrameworkRolloutAdapter` (`uni_agent/framework/entry.py`) treats the
`llm_client` verl hands it as opaque: it forwards it unmodified into the
Gateway, which only calls `generate(request_id, *, prompt_ids, sampling_params,
...)` on it. Any object that satisfies that shape works, including one that
routes through llm-d's EPP/Envoy stack instead of verl's own
`GlobalRequestLoadBalancer`.

`llm_d_rl_verl_integration`'s `AgentLoopManager` subclasses (e.g.
`LlmdRouterAgentLoopManager`) build exactly such a client: their bare
`__init__` (inherited from `LlmdBaseAgentLoopManager`) discovers the rollout
server addresses, launches EPP/Envoy, and replaces `self.llm_client` with an
`LLMServerClient`-shaped object routed through it. Critically, `__init__`
alone (not `.create()`) never spawns verl's own `AgentLoopWorker` actors, so
constructing one of these classes has exactly one side effect: producing the
routed client.

`LlmdRouterAgentFramework` runs that construction, then delegates everything
else to `AgentFrameworkRolloutAdapter` unchanged, so uni_agent's Task/Agent/
Sandbox rollout stack runs on top of llm-d's routing instead of vanilla verl's.

Wire in via:

    actor_rollout_ref.rollout.agent.agent_loop_manager_class:
        uni_agent.framework.llmd_bridge.LlmdRouterAgentFramework
    actor_rollout_ref.rollout.custom.agent_framework.llmd_manager_fqn:
        llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
"""

from __future__ import annotations

from omegaconf import OmegaConf

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter

_DEFAULT_LLMD_MANAGER_FQN = "llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"


class LlmdRouterAgentFramework:
    """Wraps `AgentFrameworkRolloutAdapter` with an llm-d-routed `llm_client`.

    Satisfies the same `agent_loop_manager_class` contract as
    `AgentFrameworkRolloutAdapter`: verl always calls the `create` classmethod,
    never the constructor directly.
    """

    def __init__(self, *, adapter: AgentFrameworkRolloutAdapter, llmd_manager) -> None:
        self._adapter = adapter
        # Held so the llm-d manager (and whatever EPP/Envoy process handle it
        # owns) is not garbage-collected while the adapter is in use.
        self._llmd_manager = llmd_manager

    @classmethod
    def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        **_,
    ) -> LlmdRouterAgentFramework:
        try:
            from verl.utils.import_utils import load_class_from_fqn
        except ImportError as exc:  # pragma: no cover - verl is always present at runtime
            raise ImportError("uni_agent.framework.llmd_bridge requires verl to be installed.") from exc

        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        manager_fqn = af_cfg.get("llmd_manager_fqn", _DEFAULT_LLMD_MANAGER_FQN)

        try:
            llmd_manager_cls = load_class_from_fqn(manager_fqn, description="llm-d agent loop manager")
        except ImportError as exc:
            raise ImportError(
                f"Could not import {manager_fqn!r}. LlmdRouterAgentFramework requires the "
                "`llm_d_rl_verl_integration` package to be installed on every Ray worker that "
                "runs the rollout. Install it (e.g. `pip install llm-d-rl-verl-integration`) or "
                "point `actor_rollout_ref.rollout.custom.agent_framework.llmd_manager_fqn` at a "
                "reachable class."
            ) from exc

        # Bare construction only: `LlmdBaseAgentLoopManager.__init__` discovers
        # server addresses, launches EPP/Envoy, and swaps in the routed client
        # as a side effect. `.create()` (verl's async classmethod) would also
        # spawn AgentLoopWorker actors we do not want or use here.
        llmd_manager = llmd_manager_cls(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        adapter = AgentFrameworkRolloutAdapter.create(
            config=config,
            llm_client=llmd_manager.llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        return cls(adapter=adapter, llmd_manager=llmd_manager)

    def generate_sequences(self, prompts) -> None:
        return self._adapter.generate_sequences(prompts)

    def generate_sequences_and_wait(self, prompts) -> None:
        return self._adapter.generate_sequences_and_wait(prompts)
