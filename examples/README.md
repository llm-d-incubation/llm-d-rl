# Examples

These examples demonstrate how existing RL training frameworks would use llm-d's rollout controller instead of their current inference orchestration.

Each example is self-contained (zero external dependencies) and maps an existing framework's training loop to llm-d's HTTP API, with inline comments showing the equivalent framework-specific code.

## Available examples

### [Slime — GRPO training loop](slime/)

Maps [Slime's](https://github.com/THUDM/slime) GRPO training loop from Ray actors + SGLang to llm-d HTTP API calls. GRPO uses group-relative advantages (multiple responses per prompt, no critic network).

```bash
python3 examples/slime/grpo_training_loop.py --dry-run
```

7 phases per step: generate, sleep, rewards, train, wake(weights), update_weights, wake(kv_cache).

### [veRL — PPO training loop](verl/)

Maps [veRL's](https://github.com/volcengine/verl) RayPPOTrainer from Ray actors + CUDA IPC/NCCL to llm-d HTTP API calls. PPO uses a separate critic network and GAE for advantage estimation.

```bash
python3 examples/verl/ppo_training_loop.py --dry-run
```

11 phases per step: generate, sleep, rewards, ref_log_probs, values, GAE, actor_train, critic_train, wake(weights), update_weights, wake(kv_cache).

## Key insight

The llm-d API surface is the same across both examples. Only the training-side computation differs (GRPO vs PPO, group-relative vs GAE, with/without critic). The five rollout primitives (weight sync, engine lifecycle, load-aware routing, async generation, partial rollout control) are framework-agnostic.

## Running against a live controller

All examples support three modes:

1. `--dry-run` — logs HTTP calls without sending them (no infrastructure needed)
2. `--simulate-lifecycle` on the controller — real generation, no-op lifecycle (GPU-free)
3. Full mode — real vLLM engines with dev-mode endpoints

See each example's README for detailed instructions.
