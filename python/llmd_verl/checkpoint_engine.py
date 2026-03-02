"""NCCL checkpoint engine for llm-d-rl managed vLLM engines.

Uses vLLM's StatelessProcessGroup + PyNcclCommunicator to broadcast
weights directly from the trainer GPU to vLLM engines. The Go controller
orchestrates the engine-side lifecycle (pause/update_weights/resume)
via HTTP.

Architecture:
    Trainer GPU (rank 0) --> NCCL broadcast --> vLLM Engine 0 (rank 1)
                                           --> vLLM Engine 1 (rank 2)
                                           --> vLLM Engine N (rank N+1)

    Go Controller --> HTTP --> /init_weight_transfer_engine on each engine
                          --> /update_weights on each engine
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Generator

import torch

from .client import RolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)


def _get_pod_ip() -> str:
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


class LlmdCheckpointEngine:
    """Checkpoint engine that broadcasts weights to llm-d-rl managed vLLM engines.

    On the trainer side (is_sender=True):
        - Creates a StatelessProcessGroup as rank 0
        - Creates a PyNcclCommunicator for broadcasting
        - Calls the controller to tell engines to join the NCCL group
        - broadcast_weights() sends all model params via NCCL

    On the rollout side:
        - This class is not used. The vLLM engines receive weights directly
          via their internal WeightTransferEngine, coordinated by the Go controller.
    """

    def __init__(self, config: LlmdRolloutConfig):
        self.config = config
        self.client = RolloutControllerClient(config)
        self.pynccl = None
        self.pg = None
        self.master_address: str | None = None
        self.master_port = config.master_port
        self.world_size: int | None = None
        self.weight_version = 0
        self._initialized = False

    def init_nccl_group(self) -> None:
        """Initialize the NCCL group between trainer and all vLLM engines.

        Must be called before the first broadcast_weights() call.
        This performs the same NCCL rendezvous as llmd_bench.py:
        1. Start NCCL group creation in a background thread (rank 0)
        2. Tell the Go controller to init weight transfer on all engines
        3. Wait for all ranks to join
        """
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        # Determine topology
        self.master_address = _get_pod_ip()
        status = self.client.get_pool_status()
        num_engines = status.get("total_engines", 1)
        self.world_size = 1 + num_engines

        logger.info("NCCL rendezvous: master=%s:%d, world_size=%d",
                     self.master_address, self.master_port, self.world_size)

        # Create NCCL group in background thread (blocks until all ranks connect)
        nccl_result: list = [None, None]
        nccl_error: list = [None]

        def create_nccl():
            try:
                pg = StatelessProcessGroup.create(
                    host=self.master_address,
                    port=self.master_port,
                    rank=0,
                    world_size=self.world_size,
                )
                pynccl = PyNcclCommunicator(pg, device=torch.device("cuda:0"))
                nccl_result[0] = pynccl
                nccl_result[1] = pg
            except Exception as e:
                nccl_error[0] = e

        nccl_thread = threading.Thread(target=create_nccl, daemon=True)
        nccl_thread.start()
        time.sleep(1)  # Let trainer start listening before engines connect

        # Tell controller to init weight transfer on all engines concurrently
        self.client.init_weight_transfer(
            master_address=self.master_address,
            master_port=self.master_port,
            world_size=self.world_size,
            backend=self.config.weight_sync_backend,
        )

        nccl_thread.join(timeout=self.config.nccl_timeout_s)

        if nccl_error[0]:
            raise RuntimeError(f"NCCL group setup failed: {nccl_error[0]}")

        self.pynccl, self.pg = nccl_result
        self._initialized = True
        logger.info("NCCL group established: %d ranks", self.world_size)

    def broadcast_weights(self, model: torch.nn.Module,
                          device: torch.device | None = None) -> float:
        """Broadcast all model parameters via NCCL to all engines.

        This performs the actual GPU-to-GPU tensor transfer. The Go controller
        coordinates the engine-side lifecycle separately.

        Args:
            model: The model whose parameters to broadcast.
            device: CUDA device to use. Defaults to cuda:0.

        Returns:
            Wall-clock time of the NCCL broadcast in seconds.
        """
        if not self._initialized:
            raise RuntimeError("Call init_nccl_group() before broadcast_weights()")

        if device is None:
            device = torch.device("cuda:0")

        start = time.perf_counter()
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            for name, param in model.named_parameters():
                tensor = param.data.contiguous()
                self.pynccl.broadcast(tensor, src=0, stream=stream)
        stream.synchronize()
        elapsed = time.perf_counter() - start

        logger.info("NCCL broadcast complete: %.3fs", elapsed)
        return elapsed

    def broadcast_tensors(self, params: Generator[tuple[str, torch.Tensor], None, None],
                          device: torch.device | None = None) -> float:
        """Broadcast tensors from a generator (veRL's weight generator pattern).

        Args:
            params: Generator yielding (name, tensor) pairs.
            device: CUDA device. Defaults to cuda:0.

        Returns:
            Wall-clock time of the NCCL broadcast in seconds.
        """
        if not self._initialized:
            raise RuntimeError("Call init_nccl_group() before broadcast_tensors()")

        if device is None:
            device = torch.device("cuda:0")

        start = time.perf_counter()
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            for name, tensor in params:
                t = tensor.contiguous()
                self.pynccl.broadcast(t, src=0, stream=stream)
        stream.synchronize()
        elapsed = time.perf_counter() - start

        logger.info("NCCL broadcast complete: %.3fs", elapsed)
        return elapsed

    def sync_weights(self, model: torch.nn.Module,
                     device: torch.device | None = None) -> int:
        """Full weight sync: NCCL broadcast + controller update lifecycle.

        This is the high-level method that performs the complete weight sync:
        1. Start NCCL broadcast in a background thread
        2. Tell the controller to orchestrate pause -> update_weights -> resume
        3. Wait for broadcast to complete
        4. Increment and return the new weight version

        Args:
            model: The model whose parameters to broadcast.
            device: CUDA device. Defaults to cuda:0.

        Returns:
            The new weight version after sync.
        """
        if not self._initialized:
            self.init_nccl_group()

        self.weight_version += 1

        # Start NCCL broadcast in background
        broadcast_error: list = [None]
        broadcast_elapsed: list = [0.0]

        def do_broadcast():
            try:
                time.sleep(0.5)  # Let controller start pause/update lifecycle
                broadcast_elapsed[0] = self.broadcast_weights(model, device)
            except Exception as e:
                broadcast_error[0] = e

        broadcast_thread = threading.Thread(target=do_broadcast, daemon=True)
        broadcast_thread.start()

        # Tell controller to orchestrate the engine-side lifecycle
        self.client.update_weights_from_model(self.weight_version, model)

        broadcast_thread.join(timeout=self.config.nccl_timeout_s)
        if broadcast_error[0]:
            raise RuntimeError(f"NCCL broadcast failed: {broadcast_error[0]}")

        logger.info("Weight sync complete: version=%d, nccl=%.3fs",
                     self.weight_version, broadcast_elapsed[0])
        return self.weight_version

    def finalize(self) -> None:
        """Clean up resources."""
        self.pynccl = None
        self.pg = None
        self._initialized = False
        torch.cuda.empty_cache()
