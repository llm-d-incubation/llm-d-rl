"""veRL-compatible checkpoint engine manager for llm-d-rl.

Drop-in replacement for veRL's CheckpointEngineManager that routes weight
synchronization through llm-d's Go controller + vLLM pods instead of
veRL's Ray-managed rollout workers.

Usage
-----
In your veRL config (e.g. config/ppo_llmd.yaml):

    actor_rollout_ref:
      rollout:
        # Tell veRL to use this manager instead of its built-in one
        checkpoint_manager_class: "llmd_verl.checkpoint_engine_manager.LlmdVerlCheckpointEngineManager"
        checkpoint_engine:
          backend: "llmd"
          engine_kwargs:
            llmd:
              controller_url: "http://llmd-controller:8090"
              master_port: 29500

That's the only config change needed in veRL.  Everything else (FSDP
training, GAE, actor/critic update) runs exactly as before.

Architecture
------------

    Ray trainer actors (FSDP)
      rank 0  ──NCCL broadcast──► vLLM pod 0
                               ──► vLLM pod 1  ← llm-d Go controller
                               ──► vLLM pod N    orchestrates lifecycle
      rank 1+  (AllGather only, no broadcast)

Weight sync flow each step:
    1. _init_nccl_group_verl()   [once, lazy]
       - trainer.prepare()           → get rank-0 IP:port
       - trainer.init_process_group()→ rank 0 starts NCCL rendezvous
       - client.init_weight_transfer()→ vLLM pods join rendezvous

    2. update_weights(global_steps)  [every step]
       - trainer.update_weights()    → FSDP AllGather + NCCL send (non-blocking Ray RPC)
       - client.update_weights()     → controller tells vLLM pods to recv (concurrent HTTP)
       - ray.get(trainer_refs)       → wait for broadcast to finish
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import ray
from verl.utils.ray_utils import auto_await

from .client import RolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_llmd_config(checkpoint_engine_config) -> LlmdRolloutConfig:
    """Pull llm-d settings from veRL's CheckpointEngineConfig.engine_kwargs["llmd"]."""
    engine_kwargs = getattr(checkpoint_engine_config, "engine_kwargs", {}) or {}
    kw = engine_kwargs.get("llmd", {})
    return LlmdRolloutConfig(**kw)


class LlmdVerlCheckpointEngineManager:
    """veRL CheckpointEngineManager replacement that routes weights to llm-d.

    veRL's ray_trainer.py constructs this as:

        manager = LlmdVerlCheckpointEngineManager(
            config=checkpoint_engine_config,  # CheckpointEngineConfig
            trainer=actor_rollout_wg,          # RayWorkerGroup
            replicas=rollout_replicas,         # ignored — llm-d manages pods
        )

    and calls:
        manager.sleep_replicas()
        manager.update_weights(global_steps)
        manager.add_replicas(...)   # elastic scaling — no-op here
    """

    def __init__(self, config, trainer, replicas=None):
        self.llmd_config = _extract_llmd_config(config)
        self.client = RolloutControllerClient(self.llmd_config)
        self.trainer = trainer          # RayWorkerGroup
        # replicas intentionally ignored — llm-d controller owns vLLM pods

        # Single source of truth for packed bucket size: veRL's
        # `rollout.checkpoint_engine.update_weights_bucket_megabytes`.
        # Both the trainer (LlmdNcclCheckpointEngine.bucket_size) and vLLM
        # (packed_buffer_size_bytes) must agree, so we read once here and forward.
        self.bucket_size_bytes = int(getattr(config, "update_weights_bucket_megabytes", 2048)) << 20

        self.weight_version = 0
        self._nccl_initialized = False
        self._param_metadata: dict | None = None  # cached after first init

        # Import triggers @CheckpointEngineRegistry.register("llmd") on the driver.
        # On worker processes, custom_backend_module handles this automatically.
        import llmd_verl.checkpoint_engine

    # ------------------------------------------------------------------
    # veRL interface — lifecycle
    # ------------------------------------------------------------------

    @auto_await
    async def sleep_replicas(self) -> None:
        """Sleep all vLLM pods (free GPU memory while training runs)."""
        await asyncio.to_thread(self.client.sleep, 2)
        logger.info("llm-d engines sleeping")

    @auto_await
    async def wake_up_replicas(self) -> None:
        """Resume all vLLM pods: recover KV cache and weights device memory."""
        await asyncio.to_thread(self.client.wake_up, [])  # empty tags = wake all
        logger.info("llm-d engines awake")

    def add_replicas(self, replicas) -> None:
        """No-op: llm-d controller handles elastic scaling."""

    def remove_replicas(self, replicas) -> None:
        """No-op: llm-d controller handles elastic scaling."""

    # ------------------------------------------------------------------
    # veRL interface — weight sync
    # ------------------------------------------------------------------

    @auto_await
    async def update_weights(self, global_steps: int | None = None) -> int:
        """Sync updated weights from trainer FSDP workers to vLLM pods.

        Called by ray_trainer.py after each actor update step.
        """
        if not self._nccl_initialized:
            await self._init_nccl_group()

        self.weight_version += 1
        version = self.weight_version

        # Non-blocking Ray RPC — trainer rank 0 does FSDP AllGather then
        # broadcasts each parameter tensor to vLLM pods via PyNcclCommunicator.
        # Other FSDP ranks participate in AllGather but skip the broadcast.
        trainer_refs = self.trainer.update_weights(global_steps=global_steps)

        # Concurrent HTTP: tell Go controller to orchestrate each vLLM pod
        # (pause → participate in NCCL recv → resume).
        meta = self._param_metadata or {}
        await asyncio.to_thread(
            self.client.update_weights,
            target_version=version,
            param_names=meta.get("param_names"),
            param_dtypes=meta.get("param_dtypes"),
            param_shapes=meta.get("param_shapes"),
            pause_mode="keep",
            reset_kv_cache=True,
            packed_buffer_size_bytes=self.bucket_size_bytes,
        )

        # Wait for trainer broadcast to finish before returning
        await asyncio.to_thread(ray.get, trainer_refs)

        logger.info("Weight sync done: version=%d global_steps=%s", version, global_steps)
        return version

    # ------------------------------------------------------------------
    # Internal — NCCL group setup (called once, lazily)
    # ------------------------------------------------------------------

    async def _init_nccl_group(self) -> None:
        """Connect trainer rank 0 and all vLLM pods into one NCCL group.

        Steps:
          1. trainer.prepare()         → rank 0 reports its IP:port
          2. trainer.init_process_group() → rank 0 starts rendezvous (blocking
                                           until all world_size ranks join)
          3. client.init_weight_transfer() → controller fans out to vLLM pods
                                            so they join the same rendezvous
        """
        # 1. Collect IP:port from each trainer worker (only rank 0 matters)
        metadata = await asyncio.to_thread(
            ray.get,
            self.trainer.execute_checkpoint_engine(
                ["prepare"] * self.trainer.world_size
            ),
        )
        if not metadata:
            raise RuntimeError("trainer.prepare() returned empty metadata")
        master_meta = metadata[0]   # {"master_address": "10.x.x.x", "master_port": 29500}

        # 2. Query controller for how many vLLM pods exist
        # TODO: should update dynamically to support scale up and scale down?
        status = await asyncio.to_thread(self.client.get_pool_status)
        num_engines = status.get("total_engines", 0)
        if num_engines <= 0:
            raise RuntimeError(
                f"llm-d controller reports {num_engines} engines — "
                "cannot initialize NCCL group with no vLLM pods"
            )
        world_size = 1 + num_engines    # trainer rank 0 + N vLLM pods

        logger.info(
            "NCCL init: master=%s:%d world_size=%d (%d pods)",
            master_meta["master_address"], master_meta["master_port"],
            world_size, num_engines,
        )

        # 3. Submit init_process_group on all trainer workers (non-blocking):
        #      rank 0  → joins NCCL group as rank 0, blocks until all join
        #      rank 1+ → set rank=-1, return immediately (skip NCCL)
        ranks = [0] + [-1] * (self.trainer.world_size - 1)
        trainer_refs = self.trainer.execute_checkpoint_engine(
            method=["init_process_group"] * self.trainer.world_size,
            rank=ranks,
            world_size=[world_size] * self.trainer.world_size,
            master_metadata=[master_meta] * self.trainer.world_size,
        )
        logger.debug("submitted trainer init_process_group RPCs")

        # 4. Wait for trainer rank 0's TCPStore to be listening, then tell
        #    the controller to fan out init_weight_transfer to all vLLM pods.
        #    trainer_world_size = full NCCL world_size (trainer rank 0 + N vLLM pods).
        #    The controller passes this directly to each engine as its world_size.
        await asyncio.to_thread(
            self._wait_for_tcpstore,
            master_meta["master_address"],
            master_meta["master_port"],
            30.0,
        )
        logger.debug("TCPStore ready, calling controller init_weight_transfer")
        await asyncio.to_thread(
            self.client.init_weight_transfer,
            master_address=master_meta["master_address"],
            master_port=master_meta["master_port"],
            world_size=world_size,
            backend=self.llmd_config.weight_sync_backend,
        )
        logger.debug("controller init_weight_transfer sent")

        # 5. Wait for rank 0 rendezvous to complete
        await asyncio.to_thread(ray.get, trainer_refs)
        logger.debug("trainer init_process_group completed")

        # 6. Collect param metadata (names/dtypes/shapes) via metadata-only send_weights.
        #
        #    vLLM engines need to know what tensors to expect before the first
        #    broadcast. We collect metadata by draining the FSDP generator WITHOUT
        #    actually broadcasting (metadata_only=True skips NCCL).
        #
        #    This requires 4 round-trips (set_metadata_only, update_weights,
        #    get_cached_metadata, unset_metadata_only) but only happens once
        #    during init, not on every weight sync.
        await asyncio.to_thread(
            ray.get,
            self.trainer.execute_checkpoint_engine(
                ["set_metadata_only"] * self.trainer.world_size,
                enabled=[True] * self.trainer.world_size,
            ),
        )
        await asyncio.to_thread(ray.get, self.trainer.update_weights(global_steps=None))
        logger.debug("update_weights for metadata completed, fetching cached metadata")
        metadata_list = await asyncio.to_thread(
            ray.get,
            self.trainer.execute_checkpoint_engine(
                ["get_cached_metadata"] * self.trainer.world_size,
            ),
        )
        self._param_metadata = metadata_list[0]  # rank 0 has the full metadata
        await asyncio.to_thread(
            ray.get,
            self.trainer.execute_checkpoint_engine(
                ["set_metadata_only"] * self.trainer.world_size,
                enabled=[False] * self.trainer.world_size,
            ),
        )
        logger.info(
            "Collected metadata: %d params",
            len(self._param_metadata["param_names"]),
        )

        self._nccl_initialized = True
        logger.info("NCCL group ready: trainer rank 0 <-> %d vLLM pods", num_engines)

    @staticmethod
    def _wait_for_tcpstore(host: str, port: int, timeout_s: float) -> None:
        import socket as _socket

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((host, port))
                sock.close()
                logger.debug("Trainer TCPStore is listening on %s:%d", host, port)
                return
            except OSError:
                time.sleep(0.5)

        raise RuntimeError(
            f"Trainer rank 0 never started listening on {host}:{port} within {timeout_s:.0f}s"
        )