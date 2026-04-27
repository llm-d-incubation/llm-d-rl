"""HTTP client for the llm-d-rl rollout controller.

Provides both sync and async interfaces for all controller endpoints.
The sync interface uses urllib (no extra dependencies), while the async
interface uses aiohttp when available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RolloutControllerClient:
    """Synchronous HTTP client for the llm-d-rl rollout controller.

    Maps directly to the controller's HTTP API:
        POST /v1/generate          - Generate rollout
        POST /v1/weights/init      - Initialize NCCL/NIXL weight transfer group
        POST /v1/weights/update    - Orchestrate pause -> transfer -> resume
        GET  /v1/weights/version   - Get current weight version
        POST /v1/engines/sleep     - Sleep engines (free GPU memory)
        POST /v1/engines/wake      - Wake engines (restore resources)
        POST /v1/engines/pause     - Pause generation
        POST /v1/engines/resume    - Resume generation
        GET  /v1/pool/status       - Get pool status
        GET  /v1/health            - Health check
    """

    def __init__(self, config: LlmdRolloutConfig | None = None, base_url: str | None = None):
        if config is None:
            config = LlmdRolloutConfig(controller_url=base_url or "http://localhost:8090")
        self.config = config
        self.base_url = config.controller_url.rstrip("/")
        self.timeout = config.http_timeout_s

    def _post(self, path: str, body: dict | None = None) -> dict:
        url = self.base_url + path
        data = json.dumps(body or {}).encode()
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(1 + self.config.max_retries):
            req = Request(url, data=data, method="POST",
                          headers={"Content-Type": "application/json"})
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    content = resp.read()
                    return json.loads(content) if content else {}
            except HTTPError as exc:
                last_exc = exc
            if attempt < self.config.max_retries:
                logger.warning("POST %s failed (attempt %d/%d): %s — retrying in %.1fs",
                               path, attempt + 1, 1 + self.config.max_retries,
                               last_exc, self.config.retry_delay_s)
                time.sleep(self.config.retry_delay_s)
        raise last_exc

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(1 + self.config.max_retries):
            req = Request(url, method="GET")
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    content = resp.read()
                    return json.loads(content) if content else {}
            except HTTPError as exc:
                last_exc = exc
            if attempt < self.config.max_retries:
                logger.warning("GET %s failed (attempt %d/%d): %s — retrying in %.1fs",
                               path, attempt + 1, 1 + self.config.max_retries,
                               last_exc, self.config.retry_delay_s)
                time.sleep(self.config.retry_delay_s)
        raise last_exc

    # --- Health & Status ---

    def health_check(self) -> bool:
        try:
            data = self._get("/v1/health")
            return data.get("status") == "ok"
        except Exception:
            return False

    def wait_for_ready(self, timeout: float = 600) -> dict:
        """Wait until the controller reports at least one ready engine."""
        start = time.time()
        while time.time() - start < timeout:
            if self.health_check():
                status = self.get_pool_status()
                if status.get("ready_engines", 0) > 0:
                    return status
            logger.info("Waiting for controller + engines...")
            time.sleep(10)
        raise TimeoutError(f"Controller not ready after {timeout}s")

    def get_pool_status(self) -> dict:
        return self._get("/v1/pool/status")

    # --- Generation ---

    def generate(self, prompt_token_ids: list[int], max_tokens: int = 32,
                 temperature: float = 0.7, prompt: str | None = None, **kwargs) -> dict:
        body: dict[str, Any] = {
            "prompt_token_ids": prompt_token_ids,
            "return_logprobs": True,
            "return_token_ids": True,
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            },
        }
        if prompt is not None:
            body["prompt"] = prompt
        return self._post("/v1/generate", body)

    # --- Weight Management ---

    def init_weight_transfer(self, master_address: str, master_port: int,
                             world_size: int, backend: str = "nccl") -> dict:
        return self._post("/v1/weights/init", {
            "backend": backend,
            "master_address": master_address,
            "master_port": master_port,
            "trainer_world_size": world_size,
            "packed": self.config.use_packed,
            "is_checkpoint_format": True,
        })

    def update_weights(self, target_version: int,
                       param_names: list[str] | None = None,
                       param_dtypes: list[str] | None = None,
                       param_shapes: list[list[int]] | None = None,
                       pause_mode: str = "keep",
                       reset_kv_cache: bool = True,
                       packed_buffer_size_bytes: int | None = None,
                       packed_num_buffers: int = 2) -> dict:
        """Trigger a weight update. When use_packed is True, the caller MUST pass
        `packed_buffer_size_bytes` — this must equal the trainer's bucket_size so
        both sides of the NCCL broadcast agree on buffer boundaries.
        """
        body: dict[str, Any] = {
            "target_version": target_version,
            "pause_mode": pause_mode,
            "reset_kv_cache": reset_kv_cache,
        }
        if param_names is not None:
            body["param_names"] = param_names
            body["param_dtypes"] = param_dtypes or []
            body["param_shapes"] = param_shapes or []
            body["packed"] = self.config.use_packed
            if self.config.use_packed:
                if packed_buffer_size_bytes is None:
                    raise ValueError(
                        "packed_buffer_size_bytes is required when use_packed=True; "
                        "it must match the trainer checkpoint engine's bucket_size."
                    )
                body["packed_buffer_size_bytes"] = packed_buffer_size_bytes
                body["packed_num_buffers"] = packed_num_buffers
        return self._post("/v1/weights/update", body)

    def update_weights_from_model(self, target_version: int, model) -> dict:
        """Convenience: extract param metadata from a torch model."""
        names, dtypes, shapes = [], [], []
        for name, param in model.named_parameters():
            names.append(name)
            dtypes.append(str(param.dtype).replace("torch.", ""))
            shapes.append(list(param.shape))
        return self.update_weights(target_version, names, dtypes, shapes)

    def get_weight_version(self) -> int:
        data = self._get("/v1/weights/version")
        return data.get("weight_version", 0)

    # --- Engine Lifecycle ---

    def sleep(self, level: int = 2) -> dict:
        return self._post("/v1/engines/sleep", {"level": level})

    def wake_up(self, tags: list[str]) -> dict:
        return self._post("/v1/engines/wake", {"tags": tags})

    def pause(self, mode: str = "keep") -> dict:
        return self._post("/v1/engines/pause", {"mode": mode})

    def resume(self) -> dict:
        return self._post("/v1/engines/resume", {})
