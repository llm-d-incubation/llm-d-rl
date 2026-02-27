# Rollout controller implementation

This document describes the implementation of the llm-d rollout controller — the control plane component that sits between RL training frameworks and vLLM inference engine pools.

## Overview

The rollout controller is a Go HTTP server that exposes a framework-agnostic API for managing inference engines during RL training. It handles three concerns:

1. Generation routing — forwarding token generation requests to healthy engines
2. Weight synchronization — orchestrating the pause-transfer-resume lifecycle for weight updates
3. Engine lifecycle — managing sleep/wake transitions for GPU memory sharing in colocated deployments

The controller manages the control plane only. Weight tensor data flows directly from trainer GPUs to engine GPUs via NCCL or NIXL — the controller never proxies tensor data.

## Project structure

```
cmd/
  rollout-controller/     Entry point, CLI flags, process lifecycle
  weight-sync-proxy/      (placeholder for future weight transfer proxy)

api/
  v1alpha1/
    types.go              All request/response types and enums

pkg/
  rollout/
    server.go             HTTP server, route registration, request handlers
  lifecycle/
    pool.go               Engine pool management, health monitoring
  weightsync/
    coordinator.go        Weight sync orchestration across engine pool
    vllm_client.go        HTTP client for vLLM dev-mode endpoints
    simulated_client.go   No-op client for GPU-free demos
  routing/                (placeholder for EPP integration)

examples/
  slime/
    grpo_training_loop.py   Slime/GRPO training loop (7 phases, group-relative advantages)
    README.md               Slime-to-llm-d mapping and run instructions
  verl/
    ppo_training_loop.py    veRL/PPO training loop (11 phases, critic + GAE)
    README.md               veRL-to-llm-d mapping and run instructions
  README.md                 Examples index
```

## API types

All API types live in `api/v1alpha1/types.go`. This package defines the contract between training frameworks and the controller.

### Enums

- `PauseMode` — how to handle in-flight requests during weight updates. `abort` discards them, `wait` drains them, `keep` freezes them in place (most efficient, requires vLLM v1+).

- `SleepLevel` — GPU memory management during sleep. Level 0 pauses scheduling only. Level 1 offloads weights to CPU. Level 2 discards all GPU memory (preferred for colocated RL).

- `WeightSyncBackend` — transport mechanism for weight transfer. NCCL (default), NIXL/RDMA, or checkpoint (shared filesystem).

- `PoolPhase` — operational state of the engine pool: Serving, Sleeping, Syncing, or Rolling.

### Request types

`GenerateRequest` carries tokenized prompts, sampling parameters, and an optional session ID for KV cache affinity. `WeightTransferInit` configures the NCCL/NIXL data plane (master address, port, world size). `WeightUpdateRequest` triggers a weight update with a target version and pause mode.

### Response types

`GenerateResponse` returns generated token IDs, log probabilities, the weight version that produced the output, and the engine ID. `PoolStatus` reports the pool phase, engine count, and per-engine health.

## HTTP server

The server (`pkg/rollout/server.go`) registers 10 routes across four groups:

```
Generation
  POST /v1/generate              Forward to engine /v1/completions
  POST /v1/generate/abort        Abort in-flight requests (not yet implemented)

Weight management
  POST /v1/weights/init          Initialize weight transfer data plane
  POST /v1/weights/update        Trigger weight update across pool
  GET  /v1/weights/version       Get current weight version

Engine lifecycle
  POST /v1/engines/pause         Pause all engines
  POST /v1/engines/resume        Resume all engines
  POST /v1/engines/sleep         Sleep all engines
  POST /v1/engines/wake          Wake all engines

Status
  GET  /v1/pool/status           Pool phase, engine count, weight version
  GET  /v1/health                Controller health check
```

### Generation handler

`handleGenerate` is the main request path. It:

1. Decodes a `GenerateRequest` from the request body.
2. Calls `pool.PickReadyEngine()` to select a healthy engine.
3. Translates the request to OpenAI `/v1/completions` format (vLLM's native API).
4. Forwards the HTTP request to the engine.
5. Translates the OpenAI response back to a `GenerateResponse`.
6. Tags the response with the current weight version and engine ID.

The translation in `forwardToEngine` maps fields as follows:

| GenerateRequest field | OpenAI completions field |
|---|---|
| `prompt_token_ids` | `prompt` (vLLM accepts token ID arrays) |
| `sampling_params.temperature` | `temperature` |
| `sampling_params.top_p` | `top_p` |
| `sampling_params.max_tokens` | `max_tokens` |
| `sampling_params.n` | `n` |
| `sampling_params.stop` | `stop` |
| `return_logprobs` | `logprobs: 1` |

### Weight update handler

`handleUpdateWeights` sets the pool phase to Syncing, delegates to the coordinator's `UpdateWeights` method, updates the tracked weight version, and sets the phase back to Serving. This endpoint is the single-call equivalent of what Slime does in five separate Ray remote calls.

### Lifecycle handlers

`handleSleep` sets the pool phase to Sleeping and fans out to all engines. `handleWakeUp` fans out to all engines and sets the phase back to Serving. `handlePause` and `handleResume` delegate to the coordinator without changing pool phase (they're used internally during weight updates).

## Engine pool management

The pool manager (`pkg/lifecycle/pool.go`) tracks registered engines and their health status.

### Engine registration

Engines are registered at startup via the `--engines` CLI flag. Each engine gets an `EngineInfo` with an ID (e.g., `engine-0`), an address (the URL), and a client (either `VLLMClient` or `SimulatedEngineClient`). The engine starts as not-ready and becomes ready after passing its first health check.

### Health checks

`RunHealthChecks` runs a concurrent health check on every engine. It snapshots the engine list under a read lock, releases the lock, then spawns a goroutine per engine. Each goroutine calls `client.Health(ctx)` (an HTTP GET to `/health`) with a 5-second timeout. After the check, each goroutine acquires the write lock briefly to update state.

On success, the engine is marked ready and its failure counter resets. On failure, the counter increments. After 3 consecutive failures, the engine is marked unready and the `OnEngineUnhealthy` callback fires (intended for triggering Kubernetes pod replacement).

`StartHealthLoop` runs health checks on a configurable interval (default 30 seconds). An initial `RunHealthChecks` call happens synchronously at startup so engines are ready before the first request arrives.

### Engine selection

`PickReadyEngine` iterates the engine map under a read lock and returns the first ready engine. This is a placeholder for more sophisticated routing — the roadmap calls for integrating with the EPP for KV-cache-aware dispatch.

## Weight sync coordinator

The coordinator (`pkg/weightsync/coordinator.go`) orchestrates weight synchronization across the engine pool.

### EngineClient interface

The coordinator operates through the `EngineClient` interface, which has 9 methods:

```go
Pause(ctx, mode)                   // pause generation
Resume(ctx)                        // resume generation
InitWeightTransfer(ctx, init)      // set up NCCL/NIXL data plane
UpdateWeights(ctx, req)            // trigger weight reception
GetWeightVersion(ctx)              // query current weight version
Sleep(ctx, level)                  // enter sleep mode
WakeUp(ctx, tags)                  // exit sleep mode
Health(ctx)                        // health check
ResetPrefixCache(ctx)              // clear KV cache
```

Two implementations exist:

- `VLLMClient` — makes real HTTP calls to vLLM's dev-mode endpoints (`/pause`, `/resume`, `/sleep`, `/wake_up`, `/init_weight_transfer_engine`, `/update_weights`, `/reset_prefix_cache`). Requires `VLLM_SERVER_DEV_MODE=1`.

- `SimulatedEngineClient` — forwards `Health` to a real server but no-ops everything else. Used with `--simulate-lifecycle` for GPU-free demos.

### Weight update lifecycle

`UpdateWeights` executes the full weight update in 5 steps:

1. Pause all engines (using the requested pause mode, defaulting to `keep`)
2. Trigger weight reception on all engines (the trainer broadcasts tensors via NCCL/NIXL concurrently)
3. Reset KV caches if requested (stale caches after policy weight changes)
4. Resume all engines
5. Update the coordinator's weight version

If step 2 fails, the coordinator attempts to resume engines before returning the error.

The `InitTransfer` call must happen before the first `UpdateWeights`. It initializes the NCCL/NIXL communication group on each engine. If an engine is unregistered, the group is invalidated and needs re-initialization.

### Concurrent fan-out

`forEachEngine` runs an operation on all engines concurrently using goroutines. Errors are collected and aggregated. This is used by `PauseAll`, `ResumeAll`, `SleepAll`, `WakeUpAll`, and internally by `UpdateWeights`.

## Entry point

`cmd/rollout-controller/main.go` wires everything together:

1. Parses CLI flags (`--port`, `--engines`, `--health-check-interval`, `--simulate-lifecycle`, `--version`)
2. Creates a `Coordinator` and a `PoolManager`
3. For each comma-separated engine URL in `--engines`:
   - Creates a `VLLMClient` (or `SimulatedEngineClient` if `--simulate-lifecycle`)
   - Registers it with both the pool manager and coordinator
4. Starts the health check loop in a background goroutine
5. Runs an initial health check to mark engines ready
6. Starts the HTTP server
7. Handles graceful shutdown on SIGINT/SIGTERM (10-second drain timeout)

## Design decisions

### Control plane vs data plane separation

The controller never proxies weight tensors. It orchestrates the lifecycle (pause, init, update, resume) while NCCL/NIXL handles the actual tensor transfer between trainer and engine GPUs. This keeps the controller lightweight and avoids it becoming a throughput bottleneck.

### Framework-agnostic HTTP API

The API uses standard HTTP/JSON so any language can consume it. The Python example (`examples/slime/grpo_training_loop.py`) uses only `urllib` from the standard library — zero external dependencies. This makes it easy for framework authors to write adapters without importing Go-specific libraries.

### Simulated lifecycle for testing

The `SimulatedEngineClient` enables end-to-end testing of the full 7-phase training loop without GPUs, vLLM dev-mode, or NCCL. Health checks still hit a real server (to validate the engine is reachable), but all lifecycle operations are logged no-ops. This lets you develop and debug the training loop integration before deploying to a GPU cluster.

### Single-call weight update

Slime's weight update requires 5 separate Ray remote calls across all engines: pause, flush, NCCL broadcast, continue, plus health verification. The rollout controller collapses this into a single `POST /v1/weights/update` — the coordinator handles the full lifecycle internally. This reduces the surface area for partial failures and makes the training loop code simpler.

## Running the demo

### Dry run (no infrastructure)

```bash
python3 examples/slime/grpo_training_loop.py --dry-run
```

### Against a mock engine

```bash
# Terminal 1: mock vLLM server
python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        n = body.get('n', 1)
        choices = [{'text': f' Answer {i}', 'index': i, 'finish_reason': 'stop',
                     'logprobs': {'token_logprobs': [-0.5], 'tokens': ['Answer']}}
                    for i in range(n)]
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'choices': choices}).encode())
    def log_message(self, *a): pass

HTTPServer(('', 8000), H).serve_forever()
"

# Terminal 2: rollout controller
go run ./cmd/rollout-controller \
  --engines http://localhost:8000 \
  --simulate-lifecycle

# Terminal 3: training loop
python3 examples/slime/grpo_training_loop.py \
  --controller-url http://localhost:8090 \
  --num-steps 3
```

### Against real vLLM engines

```bash
# Terminal 1: vLLM with dev mode
VLLM_SERVER_DEV_MODE=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --load-format dummy --enforce-eager --port 8000

# Terminal 2: rollout controller (no --simulate-lifecycle)
go run ./cmd/rollout-controller \
  --engines http://localhost:8000

# Terminal 3: training loop
python3 examples/slime/grpo_training_loop.py \
  --controller-url http://localhost:8090
```

## Current limitations

- Engine selection is first-available, not load-aware (EPP integration planned for Phase 1)
- No gRPC API yet (HTTP only)
- `GetWeightVersion` on the vLLM client is a placeholder (vLLM doesn't expose this endpoint yet)
- `POST /v1/generate/abort` is not implemented
- No retry or circuit breaker logic on engine communication
- Weight transfer initialization is sequential across engines (could be parallelized)
