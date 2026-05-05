"""llmd-verl: veRL integration for llm-d-rl rollout infrastructure.

This package provides veRL-compatible adapters that route training through
llm-d-managed inference pods instead of veRL's Ray-managed vLLM processes.

Components:
    LlmdAgentLoopManager          — spawns AgentLoopWorkers, splits batch, aggregates
    LlmdAgentLoopWorker           — Ray actor injecting HTTP client as server_manager
    LlmdSingleTurnAgentLoop       — agent loop calling llm-d HTTP API directly
    LlmdVerlCheckpointEngineManager — weight sync lifecycle (HTTP + Ray RPC)
    LlmdNcclCheckpointEngine      — NCCL broadcast on trainer GPU workers
    RolloutControllerClient       — sync HTTP client for the Go controller
    AsyncRolloutControllerClient  — async HTTP client for the Go controller
"""

__version__ = "0.1.0"