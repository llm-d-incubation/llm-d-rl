"""llmd-verl: veRL integration for llm-d-rl rollout infrastructure.

This package provides veRL-compatible adapters that let veRL's training loop
use llm-d-rl managed vLLM engines for rollout and weight synchronization,
replacing Ray-managed vLLM processes.

Architecture:
    Trainer GPU --(NCCL via StatelessProcessGroup)--> vLLM Engines directly
    Go Controller --(HTTP lifecycle)--> vLLM Engines (pause/resume/sleep/wake)

Usage:
    from llmd_verl.client import RolloutControllerClient
    from llmd_verl.manager import LlmdCheckpointEngineManager
"""

__version__ = "0.1.0"
