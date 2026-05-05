"""LlmdAgentLoopManager — verl-compatible manager routing through llm-d.

Adopts verl's AgentLoopManager/AgentLoopWorker pattern: spawns N Ray actor
workers (separate processes, separate event loops), each handling a chunk of the batch.

Usage
-----
In your veRL config:

    actor_rollout_ref:
      rollout:
        agent:
          agent_loop_manager_class: "llmd_verl.agent_loop_manager.LlmdAgentLoopManager"
          default_agent_loop: "llmd_single_turn"
          num_workers: 4
        custom:
          llmd:
            controller_url: "http://llmd-controller:8090"
"""

from __future__ import annotations

import asyncio
import logging
import os
from uuid import uuid4

import numpy as np
import ray
from transformers import AutoTokenizer

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await

from .client import AsyncRolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_llmd_config(verl_config) -> LlmdRolloutConfig:
    rollout = verl_config.actor_rollout_ref.rollout
    kw = rollout.get("custom", {}).get("llmd", {})
    return LlmdRolloutConfig(**kw)


class LlmdAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker subclass that injects AsyncRolloutControllerClient as server_manager.

    The guard at AgentLoopWorker.__init__ line 495 (`if not hasattr(self, "server_manager")`)
    skips AsyncLLMServerManager creation when server_manager is already set.
    """

    def __init__(self, config, llmd_config_dict, reward_loop_worker_handles=None):
        import llmd_verl.agent_loop  # noqa: F401 — triggers @register("llmd_single_turn")

        llmd_config = LlmdRolloutConfig(**llmd_config_dict)
        self.server_manager = AsyncRolloutControllerClient(llmd_config)

        super().__init__(
            config=config,
            servers=[],
            load_balancer_handle=None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )


class LlmdAgentLoopManager:
    """verl-compatible AgentLoopManager that routes generation through llm-d.

    Spawns N LlmdAgentLoopWorker Ray actors, splits the batch across them,
    and aggregates results — same pattern as verl's AgentLoopManager.
    """

    def __init__(self, config, reward_loop_worker_handles=None):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.rollout_replicas: list = []
        self.agent_loop_workers: list = []

    @classmethod
    @auto_await
    async def create(
        cls,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,  # noqa: ARG003 — accepted for verl API compat
    ):
        instance = cls(config, reward_loop_worker_handles)
        model_path = config.actor_rollout_ref.model.path
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        instance._init_agent_loop_workers()
        return instance

    def _init_agent_loop_workers(self):
        llmd_config = _extract_llmd_config(self.config)
        llmd_config_dict = {
            "controller_url": llmd_config.controller_url,
            "master_port": llmd_config.master_port,
            "nccl_timeout_s": llmd_config.nccl_timeout_s,
            "weight_sync_backend": llmd_config.weight_sync_backend,
            "http_timeout_s": llmd_config.http_timeout_s,
            "max_retries": llmd_config.max_retries,
            "retry_delay_s": llmd_config.retry_delay_s,
            "use_packed": llmd_config.use_packed,
        }
        num_workers = self.rollout_config.agent.num_workers

        worker_cls = ray.remote(LlmdAgentLoopWorker)
        node_ids = [
            n["NodeID"] for n in ray.nodes()
            if n["Alive"] and n["Resources"].get("CPU", 0) > 0
        ]

        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                worker_cls.options(
                    name=f"llmd_agent_loop_worker_{i}_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, llmd_config_dict, self.reward_loop_worker_handles)
            )

        logger.info("Spawned %d LlmdAgentLoopWorkers", num_workers)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(*[
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        output = DataProto.concat(outputs)

        metrics = [out.meta_info.pop("metrics") for out in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}

        # try:
        #     prompt_text = str(output.non_tensor_batch["raw_prompt"][0])
        #     response_ids = output.batch["responses"][0]
        #     response_ids = response_ids[response_ids != self.tokenizer.pad_token_id].tolist()
        #     response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        #     lps = output.batch["rollout_log_probs"][0][:3].tolist()
        #     logger.info(
        #         "\n[LLMD] ── sample  ──\n"
        #         "  prompt (200): %s\n"
        #         "  response (last 200): %s\n"
        #         "  logprobs[:3]: %s",
        #         prompt_text[:200],
        #         response_text[-200:],
        #         lps,
        #     )
        # except Exception:
        #     pass

        return output

    def _performance_metrics(self, metrics: list[list[dict]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    @auto_await
    async def clear_kv_cache(self) -> None:
        return

    def start_profile(self, **kwargs) -> None:
        pass

    def stop_profile(self) -> None:
        pass
