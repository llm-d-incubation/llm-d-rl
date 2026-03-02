"""Checkpoint engine manager for llm-d-rl.

Replaces veRL's CheckpointEngineManager (which is tightly coupled to
RayWorkerGroup) with a standalone manager that coordinates weight
synchronization between a trainer and llm-d-rl managed vLLM engines.

veRL's native flow:
    CheckpointEngineManager.update_weights()
    -> ray.get(trainer.update_weights() + rollout.update_weights())
    -> NCCLCheckpointEngine.send_weights() on trainer
    -> NCCLCheckpointEngine.receive_weights() on rollout workers
    -> ServerAdapter.update_weights() via ZMQ IPC to vLLM

llm-d-rl flow:
    LlmdCheckpointEngineManager.update_weights(model)
    -> LlmdCheckpointEngine.sync_weights(model)
       -> NCCL broadcast directly to vLLM engines (no ZMQ hop)
       -> Go controller orchestrates pause/update_weights/resume via HTTP
"""

from __future__ import annotations

import logging
from typing import Generator

import torch

from .checkpoint_engine import LlmdCheckpointEngine
from .client import RolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)


class LlmdCheckpointEngineManager:
    """Manages weight synchronization between a trainer and llm-d-rl engines.

    This is the main integration point for RL training loops. It provides
    the same lifecycle as veRL's CheckpointEngineManager:

        manager = LlmdCheckpointEngineManager(config)
        manager.init()

        for step in training_loop:
            # Generate rollouts via controller
            responses = manager.client.generate(prompt_ids)

            # Free GPU memory for training
            manager.sleep_replicas()

            # Train (on the same or different GPU)
            train_step(model)

            # Sync weights to engines
            manager.update_weights(model)

            # Restore engines for next generation
            manager.resume_replicas()
    """

    def __init__(self, config: LlmdRolloutConfig | None = None,
                 controller_url: str | None = None):
        if config is None:
            config = LlmdRolloutConfig(controller_url=controller_url or "http://localhost:8090")
        self.config = config
        self.client = RolloutControllerClient(config)
        self.engine = LlmdCheckpointEngine(config)

    def init(self, timeout: float = 600) -> dict:
        """Wait for the controller to be ready and initialize the NCCL group.

        Args:
            timeout: Max seconds to wait for controller readiness.

        Returns:
            Pool status dict from the controller.
        """
        status = self.client.wait_for_ready(timeout=timeout)
        logger.info("Controller ready: %d engines", status.get("total_engines", 0))

        self.engine.init_nccl_group()
        return status

    def sleep_replicas(self, level: int = 2) -> None:
        """Sleep all engines to free GPU memory for training."""
        self.client.sleep(level=level)
        logger.info("Engines sleeping (level=%d)", level)

    def update_weights(self, model: torch.nn.Module,
                       device: torch.device | None = None) -> int:
        """Sync weights from trainer to all engines.

        Performs:
        1. NCCL broadcast of all model parameters to engines
        2. Controller orchestrates pause -> update_weights -> resume on engines
        3. Returns the new weight version

        Args:
            model: The model whose parameters to sync.
            device: CUDA device. Defaults to cuda:0.

        Returns:
            The new weight version.
        """
        return self.engine.sync_weights(model, device)

    def update_weights_from_generator(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        param_names: list[str],
        param_dtypes: list[str],
        param_shapes: list[list[int]],
        device: torch.device | None = None,
    ) -> int:
        """Sync weights from a generator (veRL's pattern).

        For use with veRL's trainer.get_per_tensor_param() which yields
        (name, tensor) pairs.
        """
        import threading

        self.engine.weight_version += 1
        version = self.engine.weight_version

        # Materialize weights for broadcasting
        weight_list = list(weights)

        # NCCL broadcast in background
        broadcast_error: list = [None]
        broadcast_elapsed: list = [0.0]

        def do_broadcast():
            try:
                import time
                time.sleep(0.5)
                broadcast_elapsed[0] = self.engine.broadcast_tensors(
                    iter(weight_list), device)
            except Exception as e:
                broadcast_error[0] = e

        broadcast_thread = threading.Thread(target=do_broadcast, daemon=True)
        broadcast_thread.start()

        # Tell controller to orchestrate engine-side lifecycle
        self.client.update_weights(
            target_version=version,
            param_names=param_names,
            param_dtypes=param_dtypes,
            param_shapes=param_shapes,
        )

        broadcast_thread.join(timeout=self.config.nccl_timeout_s)
        if broadcast_error[0]:
            raise RuntimeError(f"NCCL broadcast failed: {broadcast_error[0]}")

        return version

    def resume_replicas(self, tags: list[str] | None = None) -> None:
        """Wake engines and restore specified resources.

        Args:
            tags: Resources to restore. Defaults to ["weights", "kv_cache"].
        """
        if tags is None:
            tags = ["weights", "kv_cache"]
        self.client.wake_up(tags=tags)
        logger.info("Engines resumed (tags=%s)", tags)

    @property
    def weight_version(self) -> int:
        return self.engine.weight_version

    def finalize(self) -> None:
        """Clean up resources."""
        self.engine.finalize()
