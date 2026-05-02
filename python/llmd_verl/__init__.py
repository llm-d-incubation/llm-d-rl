"""llmd-verl: veRL integration for llm-d-rl rollout infrastructure.

This package provides veRL-compatible adapters that route training through
llm-d-managed inference pods instead of veRL's Ray-managed vLLM processes.

Components:
    LlmdAgentLoopManager          — generation via HTTP to Go controller
    LlmdVerlCheckpointEngineManager — weight sync lifecycle (HTTP + Ray RPC)
    LlmdNcclCheckpointEngine      — NCCL broadcast on trainer GPU workers
    RolloutControllerClient       — HTTP client for the Go controller
"""

__version__ = "0.1.0"