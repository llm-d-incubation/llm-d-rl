"""Configuration for llm-d-rl rollout controller connection."""

from dataclasses import dataclass, field


@dataclass
class LlmdRolloutConfig:
    """Configuration for llm-d-rl as veRL rollout backend."""

    controller_url: str = "http://localhost:8090"
    master_port: int = 29500
    nccl_timeout_s: int = 300
    weight_sync_backend: str = "nccl"
    http_timeout_s: float = 120.0
    max_retries: int = 3
    retry_delay_s: float = 1.0
