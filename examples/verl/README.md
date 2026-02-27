# PPO training loop with llm-d rollout controller

This example demonstrates how [veRL's](https://github.com/volcengine/verl) PPO training loop (RayPPOTrainer) would work using llm-d's rollout controller instead of Ray actors, CUDA IPC weight transfer, and direct vLLM/SGLang server management.

## What it shows

A colocated RL training loop where inference engines and the trainer share the same GPUs. Each training step follows an 11-phase lifecycle that maps to veRL's `RayPPOTrainer.fit()` method:

```
[Engines SERVING — full GPU memory for inference]
  1.  Generate rollouts      (POST /v1/generate, concurrent)
  2.  Sleep engines           (POST /v1/engines/sleep, level=2)

[Engines SLEEPING — GPU memory freed for training]
  3.  Compute rewards         (simulated — reward model scoring)
  4.  Compute ref log probs   (simulated — frozen reference policy)
  5.  Compute values          (simulated — critic forward pass)
  6.  Compute advantages      (GAE with gamma=0.99, lambda=0.95)
  7.  Actor training step     (simulated — clipped PPO loss + KL penalty)
  8.  Critic training step    (simulated — MSE value loss)

[Weight update]
  9.  Wake weights            (POST /v1/engines/wake, tags=["weights"])
  10. Update weights          (POST /v1/weights/update)
  11. Wake KV cache           (POST /v1/engines/wake, tags=["kv_cache"])

[Engines SERVING — ready for next rollout]
```

## PPO vs GRPO

This example uses PPO, which differs from the Slime/GRPO example in several ways:

| Aspect | PPO (this example) | GRPO (Slime example) |
|---|---|---|
| Advantage estimation | GAE (critic network) | Group-relative (no critic) |
| Critic network | Yes, trained alongside actor | None |
| Reference policy | Frozen ref model for KL penalty | Optional |
| Responses per prompt | Typically 1 | Multiple (group size, e.g., 4) |
| Training phases | 11 (actor + critic + ref + GAE) | 7 (simpler pipeline) |

The llm-d API calls are identical — only the training-side computation differs.

## veRL to llm-d mapping

| veRL operation | llm-d API call |
|---|---|
| `async_rollout_manager.generate_sequences()` | `POST /v1/generate` (concurrent, routed through EPP) |
| `checkpoint_manager.sleep_replicas()` | `POST /v1/engines/sleep {"level": 2}` |
| `rollout.resume(tags=["weights"])` | `POST /v1/engines/wake {"tags": ["weights"]}` |
| `rollout.resume(tags=["kv_cache"])` | `POST /v1/engines/wake {"tags": ["kv_cache"]}` |
| `CheckpointEngineManager.__init__(backend="nccl")` | `POST /v1/weights/init` |
| `ActorRolloutRefWorker.update_weights()` (CUDA IPC + ZMQ) | `POST /v1/weights/update` (single call) |

### Weight sync comparison

veRL's colocated weight sync (naive backend) does 5 operations per engine:
1. `actor.engine.get_per_tensor_param()` — yield tensors one by one
2. Allocate CUDA IPC buffer and get handle via `reduce_tensor()`
3. Send IPC handle + bucket metadata via ZMQ REQ/REP
4. Server imports buffer via `cudaIpcOpenMemHandle` and copies weights
5. `actor.engine.to("cpu")` — offload trainer model

veRL's disaggregated weight sync (NCCL backend) uses Ray collective API:
1. `abort_all_requests()` on all replicas
2. `build_process_group()` for NCCL topology
3. `NCCLCheckpointEngine.send_weights()` — collective broadcast
4. `finalize()` communication
5. `resume_all_requests()`

llm-d collapses either pattern into a single `POST /v1/weights/update` — the coordinator handles the full pause-transfer-resume lifecycle internally.

## Running the example

### Dry run (no infrastructure needed)

```bash
python3 examples/verl/ppo_training_loop.py --dry-run
```

### Against a mock engine

```bash
# Terminal 1: start a mock vLLM server
# (any server that handles GET /health and POST /v1/completions)

# Terminal 2: start the rollout controller
go run ./cmd/rollout-controller \
  --engines http://localhost:8000 \
  --simulate-lifecycle

# Terminal 3: run the training loop
python3 examples/verl/ppo_training_loop.py \
  --controller-url http://localhost:8090
```

### Against real vLLM engines

```bash
# Terminal 1: vLLM with dev mode
VLLM_SERVER_DEV_MODE=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --load-format dummy --enforce-eager --port 8000

# Terminal 2: rollout controller
go run ./cmd/rollout-controller \
  --engines http://localhost:8000

# Terminal 3: training loop
python3 examples/verl/ppo_training_loop.py \
  --controller-url http://localhost:8090
```

## CLI options

```
--controller-url URL     Rollout controller URL (default: http://localhost:8090)
--num-steps N            Number of training steps (default: 10)
--batch-size N           Prompts per step (default: 4)
--responses-per-prompt N Responses per prompt (default: 1, PPO typically uses 1)
--sleep-level {0,1,2}    GPU memory management level (default: 2)
--weight-sync-backend    nccl | nixl | checkpoint (default: nccl)
--trainer-address ADDR   NCCL rendezvous address (default: localhost)
--trainer-port PORT      NCCL rendezvous port (default: 29500)
--dry-run                Log HTTP calls without sending them
-v, --verbose            Debug logging
```

## Architecture

```
Training loop (this script)        llm-d Rollout Controller (Go)         vLLM engine pool
───────────────────────────        ──────────────────────────────         ────────────────

POST /v1/generate ─────────────►   pick ready engine (PickReadyEngine)
                                   forward to /v1/completions  ────────►  generate tokens
                               ◄── translate response back              ◄──

POST /v1/engines/sleep ────────►   coordinator.SleepAll()
                                   fan out to each engine  ─────────────►  POST /sleep
                               ◄── {"status": "sleeping"}

  [reward model, ref policy,       [engines sleeping, GPU freed]           [GPU freed]
   critic, GAE, actor train,
   critic train]

POST /v1/engines/wake ─────────►   coordinator.WakeUpAll(["weights"])
                                   fan out to each engine  ─────────────►  POST /wake_up

POST /v1/weights/update ───────►   coordinator.UpdateWeights()
                                   1. pause all engines    ─────────────►  POST /pause
                                   2. trigger NCCL receive ─────────────►  POST /update_weights
                                      [trainer broadcasts via NCCL data plane]
                                   3. reset prefix cache   ─────────────►  POST /reset_prefix_cache
                                   4. resume all engines   ─────────────►  POST /resume
                               ◄── {"status": "updated"}

POST /v1/engines/wake ─────────►   coordinator.WakeUpAll(["kv_cache"])
                                   fan out to each engine  ─────────────►  POST /wake_up
```

The control plane (HTTP) and data plane (NCCL/NIXL) are intentionally separate. The controller orchestrates the lifecycle but does not proxy weight tensors — those flow directly from trainer GPUs to engine GPUs.

## veRL source references

Key veRL files this example maps from:

- `verl/trainer/ppo/ray_trainer.py` — `RayPPOTrainer.fit()` main training loop
- `verl/workers/engine_workers.py` — `ActorRolloutRefWorker.update_weights()` colocated weight sync
- `verl/checkpoint_engine/base.py` — `CheckpointEngineManager` weight sync orchestration
- `verl/checkpoint_engine/nccl_checkpoint_engine.py` — NCCL collective communication
- `verl/experimental/agent_loop/agent_loop.py` — `AgentLoopManager` generation orchestration
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — `ServerAdapter` vLLM lifecycle management
- `verl/workers/rollout/replica.py` — `RolloutReplica` server initialization
