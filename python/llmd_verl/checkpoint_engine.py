"""Trainer-side NCCL checkpoint engine for llm-d-rl inference pods.

Registers as "llmd" backend in veRL's CheckpointEngineRegistry so that
trainer Ray actors broadcast weights directly to llm-d-managed inference
pods via torch.distributed NCCL, without routing through veRL's rollout
workers.

Uses vLLM's StatelessProcessGroup + PyNcclCommunicator so the trainer and
vLLM use the same NCCL initialization protocol (store-based unique_id
exchange via broadcast_obj).

veRL's flow when backend="llmd":
    engine_workers.py:
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)

    LlmdNcclCheckpointEngine.send_weights():
        rank 0 -> pynccl.broadcast(bucket, src=0) for each bucket
        rank 1..N -> consume the generator (FSDP AllGather still happens)
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import AsyncGenerator, Generator

import torch
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_local_ip() -> str:
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


class _BroadcastOperation:
    """Async NCCL broadcast running in a background thread (like veRL).

    Starts the broadcast immediately in an executor so the event loop
    (and the main packing loop) can keep progressing. Use wait_for_complete()
    before starting the next broadcast to ensure ordering.
    """

    def __init__(self, pynccl, bucket: torch.Tensor, stream):
        self.pynccl = pynccl
        self.bucket = bucket
        self.stream = stream

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        self.pynccl.broadcast(self.bucket, src=0, stream=self.stream)
        self.stream.synchronize()

    async def wait_for_complete(self) -> None:
        await self._task


@CheckpointEngineRegistry.register("llmd")
class LlmdNcclCheckpointEngine(CheckpointEngine):
    """CheckpointEngine that runs on trainer Ray actors and broadcasts to inference pods.

    Only trainer rank 0 participates in the NCCL group with inference pods.
    Other FSDP ranks participate in AllGather (to materialize full params)
    but do not broadcast.

    Matches vLLM's packed_broadcast_producer algorithm exactly:
    - "Add then check" bucket boundary logic
    - torch.cat per bucket (variable size, can exceed bucket_size target)
    - Both sender and receiver must use same bucket_size target
    """

    def __init__(self, bucket_size: int, is_master: bool = False, **kwargs) -> None:
        from .config import LlmdRolloutConfig
        self.is_master = is_master
        cfg = LlmdRolloutConfig(**kwargs)
        self.bucket_size = bucket_size  # in bytes
        self.controller_url = cfg.controller_url
        self.master_port = cfg.master_port
        self.nccl_timeout_s = cfg.nccl_timeout_s

        # Toggle between packed (bucketed) and non-packed (one-per-broadcast) mode
        # Controlled via config: use_packed=False for debugging, True for production
        self.use_packed = cfg.use_packed

        self.rank: int | None = None
        self.world_size: int | None = None
        self._pynccl = None  # PyNcclCommunicator, set in init_process_group

        # Param metadata (names/dtypes/shapes) cached after first send
        self._metadata_only: bool = False
        self._cached_metadata: dict | None = None

    # ------------------------------------------------------------------
    # CheckpointEngine interface
    # ------------------------------------------------------------------

    def prepare(self) -> dict | None:
        """Return local IP:port from every FSDP worker."""
        ip = _get_local_ip()
        return {"master_address": ip, "master_port": self.master_port}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict, dict]:
        """Build NCCL topology: only trainer rank 0 joins the llm-d group."""
        master_meta = metadata[0]
        total_world_size = 1 + rollout_world_size

        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [total_world_size] * trainer_world_size,
            "master_metadata": [master_meta] * trainer_world_size,
        }
        rollout_kwargs: dict = {}
        return trainer_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: dict,
    ) -> None:
        """Initialize NCCL group. rank<0 means this worker is not rank 0."""
        self.rank = rank
        self.world_size = world_size

        if rank < 0:
            return

        master_address = master_metadata["master_address"]
        master_port = master_metadata["master_port"]

        logger.info(
            "[LLMD NCCL] rank=0 rendezvous at %s:%d world_size=%d",
            master_address, master_port, world_size,
        )

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=0,
            world_size=world_size,
            store_timeout=int(self.nccl_timeout_s),
        )
        logger.info("[LLMD NCCL] StatelessProcessGroup created")
        self._pynccl = PyNcclCommunicator(pg, device=torch.cuda.current_device())
        logger.info("[LLMD NCCL] PyNcclCommunicator initialized")

        if self.use_packed:
            logger.info(
                "[LLMD NCCL] packed mode (target bucket size %.1fMB)",
                self.bucket_size / 1e6,
            )
        else:
            logger.info("[LLMD NCCL] non-packed mode (one tensor per broadcast)")

        logger.info("[LLMD NCCL] NCCL group established")

    def finalize(self) -> None:
        """Release NCCL resources."""
        self._pynccl = None
        self.rank = None
        torch.cuda.empty_cache()

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> None:
        """Broadcast all parameters to inference pods via NCCL.

        Supports two modes controlled by self.use_packed:
        - Packed mode (use_packed=True): bucket tensors, double-buffered
        - Non-packed mode (use_packed=False): one tensor per broadcast
        """
        if self.rank is None:
            raise RuntimeError("init_process_group() must be called before send_weights()")

        if self.rank < 0:
            # Non-rank-0: drain generator so FSDP AllGather completes
            for _name, _tensor in weights:
                pass
            logger.debug(
                "[LLMD NCCL] send_weights: non-master worker (rank=%d) drained generator",
                self.rank,
            )
            return

        start = time.perf_counter()
        stream = torch.cuda.current_stream()
        logger.debug(
            "[LLMD NCCL] send_weights: master (rank=0) starting, use_packed=%s",
            self.use_packed,
        )

        # Metadata collection
        names, dtypes, shapes = [], [], []
        count = 0
        mode_str = ""

        if self.use_packed:
            # ===== PACKED MODE =====
            # Matches vLLM's packed_broadcast_producer EXACTLY:
            # - "Add then check": tensor that crosses threshold is INCLUDED in current bucket
            # - Use torch.cat per bucket (variable size, can exceed bucket_size target)
            # - Overlap: pack bucket N+1 while bucket N is broadcasting (like veRL)
            logger.debug(
                "[LLMD NCCL] packed mode: target_bucket_size=%.1fMB",
                self.bucket_size / 1e6,
            )

            iterator = iter(weights)
            num_buckets = 0
            broadcast_op: _BroadcastOperation | None = None

            while True:
                bucket_tensors: list[torch.Tensor] = []
                bucket_size = 0
                done = False

                try:
                    while True:
                        name, tensor = next(iterator)

                        if self._cached_metadata is None:
                            names.append(name)
                            dtypes.append(str(tensor.dtype).replace("torch.", ""))
                            shapes.append(list(tensor.shape))

                        count += 1

                        if self._metadata_only:
                            continue

                        weight = tensor.contiguous()
                        if not weight.is_cuda:
                            weight = weight.cuda()

                        tensor_uint8 = weight.view(torch.uint8).view(-1)
                        bucket_tensors.append(tensor_uint8)
                        bucket_size += tensor_uint8.numel()

                        # "Add then check" - matches vLLM
                        if bucket_size > self.bucket_size:
                            break
                except StopIteration:
                    done = True

                # Broadcast bucket if there's anything to send
                if not self._metadata_only and bucket_tensors:
                    packed = torch.cat(bucket_tensors, dim=0)
                    num_buckets += 1
                    logger.debug(
                        "[LLMD NCCL] packed: bucket #%d, size=%.1fMB, tensors so far=%d",
                        num_buckets, bucket_size / 1e6, count,
                    )
                    # Finish the prior broadcast before launching the next.
                    # Each _BroadcastOperation starts NCCL in a thread; the next outer-loop
                    # iteration can pack the following bucket while that broadcast runs.
                    if broadcast_op is not None:
                        await broadcast_op.wait_for_complete()
                    broadcast_op = _BroadcastOperation(self._pynccl, packed, stream)

                if done:
                    # Wait for final broadcast
                    if broadcast_op is not None:
                        await broadcast_op.wait_for_complete()
                    break

            mode_str = f"in {num_buckets} buckets (packed)"

        else:
            # ===== NON-PACKED MODE: one tensor per broadcast (same overlap pattern) =====
            broadcast_op: _BroadcastOperation | None = None
            for name, tensor in weights:
                if self._cached_metadata is None:
                    names.append(name)
                    dtypes.append(str(tensor.dtype).replace("torch.", ""))
                    shapes.append(list(tensor.shape))

                if self._metadata_only:
                    count += 1
                    continue

                weight = tensor.contiguous()
                if not weight.is_cuda:
                    weight = weight.cuda()

                # Await prior broadcast before starting the next.
                # The new _BroadcastOperation runs in a thread; the next for-loop iteration
                # (generator / AllGather / contiguous / cuda) overlaps with it.
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()
                broadcast_op = _BroadcastOperation(self._pynccl, weight, stream)

                count += 1
                if count % 50 == 0:
                    logger.debug(
                        "[LLMD NCCL] send_weights progress: %d tensors sent", count,
                    )

            # Wait for final broadcast
            if broadcast_op is not None:
                await broadcast_op.wait_for_complete()

            mode_str = "(non-packed)"

        # Cache metadata
        if self._cached_metadata is None and names:
            self._cached_metadata = {
                "param_names": names,
                "param_dtypes": dtypes,
                "param_shapes": shapes,
            }

        elapsed = time.perf_counter() - start
        if self._metadata_only:
            logger.info(
                "[LLMD NCCL] send_weights: collected metadata for %d tensors in %.3fs",
                count, elapsed,
            )
        else:
            logger.info(
                "[LLMD NCCL] send_weights: broadcast %d tensors %s in %.3fs",
                count, mode_str, elapsed,
            )

    def set_metadata_only(self, enabled: bool) -> None:
        """Enable/disable metadata-only mode (skips NCCL broadcast, captures param info).
        
        When enabled, send_weights() drains the FSDP generator and collects
        param names/dtypes/shapes but skips the actual NCCL broadcast.
        This is used once during init so the Go controller knows what tensors
        vLLM engines should expect.
        """
        self._metadata_only = enabled

    def get_cached_metadata(self) -> dict | None:
        """Return cached param metadata collected during metadata-only send_weights()."""
        return self._cached_metadata

    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        raise NotImplementedError(
            "Inference pods receive weights via the Go controller, not this engine"
        )
