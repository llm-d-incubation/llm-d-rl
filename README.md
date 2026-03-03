# llm-d-RL

Framework-agnostic RL rollout infrastructure for the [llm-d](https://github.com/llm-d) inference serving stack.

llm-d-rl provides the control plane that RL training frameworks need to orchestrate inference engines during post-training. It handles weight synchronization, engine lifecycle management, and generation routing so that framework authors don't have to reimplement these primitives.

## Problem

Every RL post-training framework (veRL, OpenRLHF, SkyRL, NeMo-RL, Slime) implements its own inference orchestration layer: weight sync via NCCL, engine health monitoring, request routing, sleep/wake for GPU sharing. These implementations are tightly coupled to specific inference backends and use framework-specific RPCs (usually Ray). llm-d-rl provides these as reusable, backend-agnostic HTTP APIs that any training loop can consume.

The controller is a standalone Go binary with **no Kubernetes dependencies**. It talks HTTP to vLLM engines and runs on any infrastructure — Slurm, bare metal, Docker, or Kubernetes. Integration with llm-d's [inference scheduler (EPP)](https://github.com/llm-d/llm-d-inference-scheduler) for KV-cache-aware routing is planned and will require Kubernetes, but the core controller and weight sync path do not.

## Architecture

```
Training loop (any framework)       Rollout controller (this repo)       vLLM engine pool
─────────────────────────────       ──────────────────────────────       ────────────────

POST /v1/generate ──────────────►   pick ready engine
                                    forward to /v1/completions  ──────►  generate tokens
                                ◄── translate response            ◄──

POST /v1/engines/sleep ─────────►   fan out to all engines  ──────────►  free GPU memory
                                ◄── {"status": "sleeping"}

     [training step on GPU]         [engines sleeping]                   [GPU freed]

POST /v1/engines/wake ──────────►   fan out to all engines  ──────────►  restore memory
POST /v1/weights/update ────────►   1. pause all engines    ──────────►  pause generation
                                    2. trigger NCCL receive ──────────►  receive weights
                                       [trainer broadcasts via NCCL data plane]
                                    3. reset prefix cache   ──────────►  clear KV cache
                                    4. resume all engines   ──────────►  resume generation
                                ◄── {"status": "updated"}
```

The control plane (HTTP) and data plane (NCCL/NIXL) are intentionally separate. The controller orchestrates the lifecycle but never proxies weight tensors.

## Components

### Rollout controller

The main binary. Exposes an HTTP API for generation, weight management, engine lifecycle, and pool status.

```
POST /v1/generate              Generate token sequences
POST /v1/weights/init          Initialize weight transfer data plane
POST /v1/weights/update        Trigger weight update across pool
GET  /v1/weights/version       Get current weight version
POST /v1/engines/pause         Pause all engines
POST /v1/engines/resume        Resume all engines
POST /v1/engines/sleep         Sleep engines (free GPU memory)
POST /v1/engines/wake          Wake engines (restore GPU memory)
GET  /v1/pool/status           Pool phase, engine count, weight version
GET  /v1/health                Controller health check
```

### API types

`api/v1alpha1/types.go` defines the contract between training frameworks and the controller. Key concepts:

- `PauseMode` — how to handle in-flight requests during weight updates (abort, wait, or keep)
- `SleepLevel` — GPU memory management during sleep (level 0-2)
- `WeightSyncBackend` — transport for weight transfer (NCCL, NIXL, or checkpoint)
- `PoolPhase` — operational state (Serving, Sleeping, Syncing, Rolling)

### Engine clients

Two implementations of the `EngineClient` interface:

- `VLLMClient` — real HTTP calls to vLLM dev-mode endpoints (requires `VLLM_SERVER_DEV_MODE=1`)
- `SimulatedEngineClient` — forwards health checks to a real server, no-ops everything else (for GPU-free testing)

## Quick start

### Build

```bash
make build
```

### Dry run (no infrastructure needed)

```bash
# Slime/GRPO example
python3 examples/slime/grpo_training_loop.py --dry-run

# veRL/PPO example
python3 examples/verl/ppo_training_loop.py --dry-run
```

Logs all HTTP calls without sending them.

### Against a mock engine

```bash
# Terminal 1: start a mock vLLM server (any OpenAI-compatible server works)
# See examples/README.md for details

# Terminal 2: start the rollout controller
go run ./cmd/rollout-controller \
  --engines http://localhost:8000 \
  --simulate-lifecycle

# Terminal 3: run either training loop
python3 examples/slime/grpo_training_loop.py --controller-url http://localhost:8090
python3 examples/verl/ppo_training_loop.py --controller-url http://localhost:8090
```

### Against real vLLM engines

```bash
# Terminal 1: vLLM with dev mode
VLLM_SERVER_DEV_MODE=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --load-format dummy --enforce-eager --port 8000

# Terminal 2: rollout controller
go run ./cmd/rollout-controller \
  --engines http://localhost:8000

# Terminal 3: run either training loop
python3 examples/slime/grpo_training_loop.py --controller-url http://localhost:8090
python3 examples/verl/ppo_training_loop.py --controller-url http://localhost:8090
```

### On Kubernetes

```bash
# 1. Build and push the controller image
export REGISTRY=ghcr.io/youruser
make docker-build && make docker-push

# 2. Create namespace and HuggingFace token secret
kubectl apply -f deploy/cks/namespace.yaml
kubectl create secret generic hf-token \
  --namespace=llm-d-rl \
  --from-literal=token=hf_YOUR_TOKEN

# 3. Deploy vLLM engines (wait ~2-5 min for model load)
kubectl apply -f deploy/cks/vllm-engine.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=vllm-engine --timeout=600s

# 4. Deploy rollout controller
kubectl apply -f deploy/cks/rollout-controller.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=rollout-controller --timeout=120s

# 5. Run the NCCL weight trainer
kubectl apply -f deploy/cks/trainer-job.yaml
kubectl -n llm-d-rl logs -f job/nccl-trainer
```

See `deploy/cks/` for the full manifests.

### On Slurm (untested)

The controller has no Kubernetes dependencies — it's a standalone binary that talks HTTP to vLLM engines. See `deploy/slurm/` for example sbatch scripts and manual launch instructions.

### CLI flags

```
--port                    HTTP server port (default: 8090)
--engines                 Comma-separated engine URLs (e.g., http://localhost:8000,http://localhost:8001)
--health-check-interval   Interval between engine health checks (default: 30s)
--simulate-lifecycle      No-op lifecycle operations for GPU-free demos
--version                 Print version and exit
```

## Development

```bash
make test           # run tests with race detection
make lint           # run golangci-lint
make fmt            # format code
make generate       # generate protobuf code
make docker-build   # build container image
```

## Documentation

- [Rollout controller implementation](docs/rollout-controller.md) — how the controller is built, package by package
- [Examples](examples/README.md) — Slime/GRPO and veRL/PPO training loop examples
  - [Slime/GRPO](examples/slime/) — GRPO with group-relative advantages (7 phases)
  - [veRL/PPO](examples/verl/) — PPO with critic network and GAE (11 phases)
- [Slurm deployment](deploy/slurm/) — sbatch scripts for Slurm clusters
- [Kubernetes deployment](deploy/cks/) — manifests for Kubernetes clusters (tested on CoreWeave CKS)
- [North star design](docs/proposals/rl-rollout-northstar.md) — technical specification and framework analysis
- [Roadmap](docs/proposals/roadmap.md) — multi-phase implementation plan

This is an experimentation project. The rollout controller runs end-to-end with simulated lifecycle operations and has deployment manifests for testing with real vLLM and NCCL weight sync. 

## License

Apache License 2.0
