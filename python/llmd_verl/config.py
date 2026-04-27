"""Configuration for llm-d-rl rollout controller connection."""

from dataclasses import dataclass


@dataclass
class LlmdRolloutConfig:
    """Configuration for llm-d-rl as veRL rollout backend.

    Note on bucket size: The packed-tensor bucket size is NOT configured here.
    It comes from veRL's `rollout.checkpoint_engine.update_weights_bucket_megabytes`
    (passed to the CheckpointEngine as bucket_size in bytes). The manager reads
    it from the trainer engine at weight-sync time and forwards it to vLLM so
    both sides agree.
    """

    controller_url: str = "http://localhost:8090"
    master_port: int = 29500
    nccl_timeout_s: int = 300
    weight_sync_backend: str = "nccl"
    http_timeout_s: float = 600.0  # generous timeout to support long vLLM generation requests
    max_retries: int = 3
    retry_delay_s: float = 1.0

    # When True, multiple tensors are bucketed into large buffers for efficient
    # NCCL transfer (vLLM 0.17+ "packed tensor broadcasting"). When False, each
    # tensor is broadcast individually.
    use_packed: bool = True
