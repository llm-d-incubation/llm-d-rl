"""veRL BaseRollout adapter for llm-d-rl managed engines.

Implements veRL's BaseRollout interface by proxying calls to the
llm-d-rl rollout controller via HTTP. This lets veRL's training loop
use llm-d-rl for engine lifecycle management without any Ray dependency
on the rollout side.

veRL interface methods:
    resume(tags)             -> POST /v1/engines/wake
    release()                -> POST /v1/engines/sleep
    update_weights(weights)  -> No-op (handled by LlmdCheckpointEngine)
    generate_sequences()     -> POST /v1/generate
"""

from __future__ import annotations

import logging
from typing import Generator

import torch

from .client import RolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)


class LlmdServerAdapter:
    """veRL-compatible rollout adapter backed by llm-d-rl.

    This implements the same interface as veRL's BaseRollout:
        - resume(tags: list[str])
        - release()
        - update_weights(weights: Generator)
        - generate_sequences(prompts)

    But instead of managing Ray actors, it talks to the llm-d-rl
    rollout controller via HTTP.

    Note: This class does NOT inherit from BaseRollout to avoid
    requiring veRL as a hard dependency. It implements the same
    interface so it can be used as a drop-in replacement.
    """

    def __init__(self, config: LlmdRolloutConfig | None = None,
                 controller_url: str | None = None):
        if config is None:
            config = LlmdRolloutConfig(controller_url=controller_url or "http://localhost:8090")
        self.config = config
        self.client = RolloutControllerClient(config)

    async def resume(self, tags: list[str]) -> None:
        """Resume rollout weights or kv cache in GPU memory.

        Maps to: POST /v1/engines/wake {"tags": tags}

        Args:
            tags: Resources to restore, e.g. ["weights"] or ["kv_cache"].
        """
        self.client.wake_up(tags)
        logger.info("Engines resumed (tags=%s)", tags)

    async def release(self) -> None:
        """Release weights and kv cache in GPU memory (sleep engines).

        Maps to: POST /v1/engines/sleep {"level": 2}
        """
        self.client.sleep(level=2)
        logger.info("Engines released (sleep level=2)")

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ) -> None:
        """Update weights on the rollout engines.

        In the llm-d-rl architecture, weight transfer happens directly via
        NCCL between the trainer and the vLLM engines. The Go controller
        orchestrates the lifecycle (pause/resume) via HTTP. The actual NCCL
        broadcast is handled by LlmdCheckpointEngine.sync_weights().

        This method is a no-op: it consumes the weights generator without
        sending anything, because the weights have already been broadcast
        via NCCL by the checkpoint engine.
        """
        # Consume the generator (weights already sent via NCCL)
        for name, tensor in weights:
            pass
        logger.debug("update_weights: generator consumed (weights sent via NCCL)")

    def generate(self, prompt_token_ids: list[int], max_tokens: int = 32,
                 temperature: float = 0.7, **kwargs) -> dict:
        """Generate a rollout via the controller.

        Maps to: POST /v1/generate

        Args:
            prompt_token_ids: Tokenized prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generation response dict with output_token_ids, logprobs, etc.
        """
        return self.client.generate(
            prompt_token_ids, max_tokens=max_tokens,
            temperature=temperature, **kwargs,
        )
