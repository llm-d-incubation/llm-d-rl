# GRPO training loop with llm-d rollout controller

This example demonstrates how [Slime's](https://github.com/THUDM/slime) GRPO training loop would work using llm-d's rollout controller instead of Ray actors and direct SGLang API calls.

## What it shows

A colocated RL training loop where inference engines and the trainer share the same GPUs. Each training step follows a 7-phase lifecycle:

```
[Engines SERVING — full GPU memory for inference]
  1. Generate rollouts (POST /v1/generate, concurrent)
  2. Sleep engines    (POST /v1/engines/sleep, level=2)

[Engines SLEEPING — GPU memory freed for training]
  3. Compute rewards  (simulated)
  4. Training step    (simulated)

[Weight update]
  5. Wake weights     (POST /v1/engines/wake, tags=["weights"])
  6. Update weights   (POST /v1/weights/update)
  7. Wake KV cache    (POST /v1/engines/wake, tags=["kv_cache"])

[Engines SERVING — ready for next rollout]
```

## Slime to llm-d mapping

| Slime operation | llm-d API call |
|---|---|
| `POST http://{sglang_router}/generate` | `POST /v1/generate` (routed through EPP) |
| `engine.release_memory_occupation.remote()` | `POST /v1/engines/sleep {"level": 2}` |
| `engine.resume_memory_occupation(tags=[WEIGHTS])` | `POST /v1/engines/wake {"tags": ["weights"]}` |
| `engine.resume_memory_occupation(tags=[KV_CACHE])` | `POST /v1/engines/wake {"tags": ["kv_cache"]}` |
| `engine.pause_generation.remote()` | `POST /v1/engines/pause {"mode": "keep"}` |
| `engine.continue_generation.remote()` | `POST /v1/engines/resume` |
| NCCL group setup inside `SGLangEngine.init()` | `POST /v1/weights/init` |
| `pause` + `flush` + NCCL broadcast + `continue` (5 ops) | `POST /v1/weights/update` (single call) |

The key difference: Slime orchestrates 5 separate Ray remote calls across all engines for each weight update. llm-d collapses this into a single HTTP call — the coordinator handles the full pause-transfer-resume lifecycle internally.

## Running the example

### Dry run (no infrastructure needed)

```bash
python3 examples/grpo_training_loop.py --dry-run
```

Logs all HTTP calls without sending them. Useful for reading the flow.

### Against llm-d-inference-sim

Requires Go and [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim).

```bash
# Terminal 1: start the inference simulator
llm-d-inference-sim --port 8000

# Terminal 2: start the rollout controller
go run ./cmd/rollout-controller \
  --engines http://localhost:8000 \
  --simulate-lifecycle

# Terminal 3: run the training loop
python3 examples/grpo_training_loop.py \
  --controller-url http://localhost:8090
```

The `--simulate-lifecycle` flag makes lifecycle operations (sleep, wake, pause, resume, weight sync) no-op while generation requests forward to the real inference simulator. This lets the full 7-phase loop run without GPUs.

### Against real vLLM engines

Requires vLLM with dev-mode endpoints enabled.

```bash
# Terminal 1: start vLLM with dev mode
VLLM_SERVER_DEV_MODE=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --load-format dummy \
  --enforce-eager \
  --port 8000

# Terminal 2: start the rollout controller (no --simulate-lifecycle)
go run ./cmd/rollout-controller \
  --engines http://localhost:8000

# Terminal 3: run the training loop
python3 examples/grpo_training_loop.py \
  --controller-url http://localhost:8090
```

## CLI options

```
--controller-url URL     Rollout controller URL (default: http://localhost:8090)
--num-steps N            Number of training steps (default: 10)
--batch-size N           Prompts per step (default: 4)
--responses-per-prompt N GRPO group size (default: 4)
--sleep-level {0,1,2}    GPU memory management level (default: 2)
--weight-sync-backend    nccl | nixl | checkpoint (default: nccl)
--trainer-address ADDR   NCCL rendezvous address (default: localhost)
--trainer-port PORT      NCCL rendezvous port (default: 29500)
--dry-run                Log HTTP calls without sending them
-v, --verbose            Debug logging
```

## Architecture

```
Training Loop (Python)          llm-d Rollout Controller (Go)         vLLM Engine Pool
─────────────────────          ──────────────────────────────         ────────────────

POST /v1/generate ──────────►  pick ready engine (PickReadyEngine)
                               forward to engine /v1/completions  ──► generate tokens
                           ◄── translate response back            ◄── return tokens

POST /v1/engines/sleep ─────►  coordinator.SleepAll()
                               fan out to each engine  ───────────►  POST /sleep
                           ◄── {"status": "sleeping"}

     [training step]           [engines sleeping, GPU freed]          [GPU memory freed]

POST /v1/engines/wake ──────►  coordinator.WakeUpAll(["weights"])
                               fan out to each engine  ───────────►  POST /wake_up

POST /v1/weights/update ────►  coordinator.UpdateWeights()
                               1. pause all engines    ───────────►  POST /pause
                               2. trigger NCCL receive ───────────►  POST /update_weights
                                  [trainer broadcasts weights via NCCL data plane]
                               3. reset prefix cache   ───────────►  POST /reset_prefix_cache
                               4. resume all engines   ───────────►  POST /resume
                           ◄── {"status": "updated"}

POST /v1/engines/wake ──────►  coordinator.WakeUpAll(["kv_cache"])
                               fan out to each engine  ───────────►  POST /wake_up
```

The control plane (HTTP) and data plane (NCCL/NIXL) are intentionally separate. The controller orchestrates the lifecycle but does not proxy weight tensors — those flow directly from trainer GPUs to engine GPUs.
