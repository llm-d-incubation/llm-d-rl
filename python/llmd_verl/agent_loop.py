"""LlmdSingleTurnAgentLoop — single-turn agent loop routing through llm-d.

Subclasses verl's SingleTurnAgentLoop and overrides run() to call
AsyncRolloutControllerClient directly instead of verl's Ray RPC to vLLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("llmd_single_turn")
class LlmdSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that routes generation through llm-d HTTP API.

    self.server_manager is the AsyncRolloutControllerClient instance, injected
    by LlmdAgentLoopWorker via hydra instantiation.
    """

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages, images=images, videos=videos
        )

        prompt_text = apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            resp = await self.server_manager.generate(
                prompt_token_ids=prompt_ids,
                prompt=prompt_text,
                max_tokens=self.response_length,
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 1.0),
                top_k=sampling_params.get("top_k", -1),
            )
        metrics["num_preempted"] = 0

        token_ids = [int(t) for t in (resp.get("output_token_ids") or [])]
        raw_lps = resp.get("logprobs") or []
        log_probs = (
            [float(list(d.values())[0]) if isinstance(d, dict) else float(d) for d in raw_lps]
            if raw_lps else None
        )
        response_mask = [1] * len(token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=log_probs[: self.response_length] if log_probs else None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields={"turn_scores": [], "tool_rewards": []},
        )

        return output
