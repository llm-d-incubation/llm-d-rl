# llmd_verl — veRL ↔ llm-d Integration

A drop-in plugin that lets [veRL](https://github.com/volcengine/verl) use
[llm-d](https://github.com/llm-d/llm-d) as its rollout backend instead of
veRL's own Ray-managed vLLM processes.

Training (FSDP + PPO/GRPO) still runs as normal Ray actors on GPUs.
Inference runs in **dedicated llm-d vLLM pods**,
load-balanced by a Go controller. Weight synchronization between the two
happens over a shared NCCL group, without going through veRL's rollout
workers.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  KubeRay cluster                                                       │
│                                                                        │
│   Head pod (driver)         Worker pods (training GPUs)                │
│   ┌──────────────┐          ┌───────────────────────────┐              │
│   │ fit() loop   │          │ FSDP rank 0  …  FSDP rank N│             │
│   │              │          │ rank 0 = NCCL master       │             │
│   │ LlmdVerl     │──Ray────▶│                            │             │
│   │ CheckpointEng│          │ LlmdNcclCheckpointEngine   │             │
│   │ Manager      │          └──────────┬─────────────────┘             │
│   │              │                     │                                │
│   │ LlmdAgent    │                     │ NCCL broadcast                 │
│   │ LoopManager  │                     │ (only rank 0 sends)            │
│   └──────┬───────┘                     │                                │
│          │                             │                                │
│          │  ┌─────────────────────┐    │                                │
│          ├─▶│ AgentLoopWorker 0   │    │                                │
│          ├─▶│ AgentLoopWorker 1   │    │                                │
│          ├─▶│ AgentLoopWorker …   │    │                                │
│          └─▶│ AgentLoopWorker N   │    │                                │
│             └─────────┬───────────┘    │                                │
│                       │ HTTP           │                                │
└───────────────────────┼────────────────┼────────────────────────────────┘
                        │                ▼
                        │      ┌────────────────────┐
                        │      │ vLLM pod 0         │
                        ▼      │ vLLM pod 1         │
                 ┌──────────────┐ vLLM pod …         │   llm-d Go controller
                 │ Go controller│ vLLM pod N         │   orchestrates
                 │              │────────────────────┘   (sleep/wake/pause/
                 └──────────────┘                         weight-transfer)
```

**Control plane** (HTTP):
- `LlmdAgentLoopManager` → spawns N `LlmdAgentLoopWorker` Ray actors →
  each worker sends HTTP requests to Go controller → vLLM pods: generation.
- `LlmdVerlCheckpointEngineManager` → Go controller → vLLM pods: lifecycle
  (sleep, wake, init weight-transfer, orchestrate weight update).

**Data plane** (NCCL):
- Trainer FSDP rank 0 ↔ all vLLM pods: one process group, weight broadcast.
- Other FSDP ranks participate only in the FSDP AllGather (to materialize
  full params) and skip the broadcast.

---

## Files

| File | Purpose |
|---|---|
| `config.py`                    | `LlmdRolloutConfig` dataclass — controller URL, timeouts, `use_packed` toggle. |
| `client.py`                    | `AsyncRolloutControllerClient` — aiohttp-based async HTTP client for the Go controller API. |
| `agent_loop.py`                | `LlmdSingleTurnAgentLoop` — subclass of veRL's `SingleTurnAgentLoop`, calls llm-d HTTP API directly. |
| `agent_loop_manager.py`        | `LlmdAgentLoopManager` + `LlmdAgentLoopWorker` — multi-process rollout orchestration. |
| `checkpoint_engine_manager.py` | `LlmdVerlCheckpointEngineManager` — veRL `CheckpointEngineManager` replacement. |
| `checkpoint_engine.py`         | `LlmdNcclCheckpointEngine` — NCCL broadcast engine that runs on trainer GPU workers. Registered in veRL's `CheckpointEngineRegistry` as `"llmd"`. |

---

## Classes

### `LlmdRolloutConfig` (`config.py`)

Plain dataclass read from veRL's YAML (`rollout.custom.llmd.*` or
`checkpoint_engine.engine_kwargs.llmd.*`). Carries the llm-d-specific
knobs:

| Field | Default | Purpose |
|---|---|---|
| `controller_url`          | `http://localhost:8090` | Go controller endpoint. |
| `master_port`             | `29500`                 | Port trainer rank 0 listens on for NCCL rendezvous. |
| `nccl_timeout_s`          | `300`                   | NCCL `StatelessProcessGroup` store timeout. |
| `http_timeout_s`          | `120.0`                 | HTTP client timeout. |
| `use_packed`              | `True`                  | Pack multiple tensors per NCCL broadcast (vLLM 0.17+). |

The packed-tensor bucket size is **not** configured here — it comes from
veRL's `rollout.checkpoint_engine.update_weights_bucket_megabytes` so the
trainer and vLLM agree on one value.

### `AsyncRolloutControllerClient` (`client.py`)

Async HTTP client (aiohttp) for the Go controller's REST API. Endpoints used:

| Method | Endpoint | Called from |
|---|---|---|
| `generate()`             | `POST /v1/generate`        | Agent loop (per prompt) |
| `init_weight_transfer()` | `POST /v1/weights/init`    | Checkpoint manager (once, at start) |
| `update_weights()`       | `POST /v1/weights/update`  | Checkpoint manager (every step) |
| `sleep()` / `wake_up()`  | `POST /v1/engines/{sleep,wake}` | Lifecycle |
| `pause()` / `resume()`   | `POST /v1/engines/{pause,resume}` | Lifecycle |
| `get_pool_status()`      | `GET  /v1/pool/status`     | Discover vLLM pod count |

### `LlmdSingleTurnAgentLoop` (`agent_loop.py`)

Subclass of veRL's `SingleTurnAgentLoop`. Registered as `"llmd_single_turn"`
via the `@register` decorator.

Copies veRL's `SingleTurnAgentLoop.run()` exactly, with only these differences:
1. The structure of the request body passed to `self.server_manager.generate()` matches
   the Go controller's `/v1/generate` API (not vLLM's internal RPC format).
2. Includes `prompt` (rendered text) in the body — needed by the Go controller for routing.
3. Calls our `AsyncRolloutControllerClient` (injected as `self.server_manager` by
   `LlmdAgentLoopWorker`) instead of veRL's `AsyncLLMServerManager`.

### `LlmdAgentLoopWorker` (`agent_loop_manager.py`)

Subclass of veRL's `AgentLoopWorker`. Each instance is a separate **Ray actor**
(separate process, separate event loop).

Sets `self.server_manager = AsyncRolloutControllerClient(...)` before calling
`super().__init__()`. veRL's `AgentLoopWorker.__init__` has a guard
(`if not hasattr(self, "server_manager")`) that skips its own server creation
when `server_manager` is already set.

After `super().__init__()`, the worker inherits from veRL:
- `self.tokenizer` / `self.processor` — loaded from model path.
- `generate_sequences()` — splits chunk into per-sample async tasks.
- `_run_agent_loop()` → `hydra.utils.instantiate(...)` — creates
  `LlmdSingleTurnAgentLoop` with `server_manager=self.server_manager`.
- `_postprocess()` — padding, reward computation, DataProto assembly.

### `LlmdAgentLoopManager` (`agent_loop_manager.py`)

Replaces veRL's `AgentLoopManager`. veRL constructs it via `create()` and
calls `generate_sequences(prompts_dataproto)` each training step.

Responsibilities:
1. Spawn N `LlmdAgentLoopWorker` Ray actors, distributed across available nodes.
2. On each `generate_sequences` call: chunk the batch across workers.
3. Dispatch in parallel via `asyncio.gather(*[worker.generate_sequences.remote(chunk) ...])`.
4. Concatenate results with `DataProto.concat(outputs)`.
5. Compute and attach performance metrics (min/max/mean timings).

### `LlmdVerlCheckpointEngineManager` (`checkpoint_engine_manager.py`)

Replaces veRL's `CheckpointEngineManager`. Coordinates the two-sided
weight sync each training step.

Key methods:
- `sleep_replicas()` / `wake_up_replicas()`  — HTTP to controller.
- `update_weights(global_steps)`             — called every PPO step (see below).
- `_init_nccl_group()`                       — lazy, first-call setup of the NCCL rendezvous.

**Weight sync flow (per step):**

1. Trainer rank 0's `LlmdNcclCheckpointEngine` starts a non-blocking Ray
   RPC: FSDP AllGather + NCCL broadcast of packed tensor buckets.
2. **Concurrently**, the manager sends `POST /v1/weights/update` to the Go
   controller, which fans out to every vLLM pod so they can receive.
3. `ray.get()` on the trainer refs waits for the broadcast to finish.

The manager pulls the bucket size from veRL's
`update_weights_bucket_megabytes` at init and forwards it to vLLM via
`packed_buffer_size_bytes` on every update — ensuring both sides of the
NCCL broadcast use the same buffer layout.

### `LlmdNcclCheckpointEngine` (`checkpoint_engine.py`)

Runs on every trainer FSDP worker. Registered as backend `"llmd"` in
veRL's `CheckpointEngineRegistry`, so setting
`rollout.checkpoint_engine.backend: llmd` is enough for veRL to pick it up.

Only **rank 0** joins the NCCL group shared with vLLM pods. Other ranks
participate in the FSDP AllGather to materialize full params but skip the
broadcast itself.

Two broadcast modes (toggled by `cfg.use_packed`):

- **Packed**: pack tensors into contiguous buffers up to
  `bucket_size` bytes, broadcast per bucket with `torch.cat`. Matches
  vLLM's `packed_broadcast_producer` algorithm exactly ("add then check"
  bucket boundary, variable bucket size).
- **Non-packed**: one `pynccl.broadcast` per tensor.

Both modes overlap CPU packing with GPU broadcasts via `BroadcastOperation`
(runs NCCL in a background executor thread, matching veRL's native
`nccl_checkpoint_engine` pattern).

---

## Minimal veRL config

To switch any existing veRL PPO/GRPO run onto llm-d, change **only these
fields**:

```yaml
actor_rollout_ref:
  rollout:
    # ── plugin hooks ────────────────────────────────────────────
    checkpoint_manager_class: llmd_verl.checkpoint_engine_manager.LlmdVerlCheckpointEngineManager
    agent:
      agent_loop_manager_class: llmd_verl.agent_loop_manager.LlmdAgentLoopManager
      default_agent_loop: llmd_single_turn
      num_workers: 4
    custom:
      llmd:
        controller_url: ${oc.env:LLMD_CONTROLLER_URL,http://localhost:8090}

    # ── NCCL weight broadcast ───────────────────────────────────
    checkpoint_engine:
      _target_: verl.workers.config.CheckpointEngineConfig
      backend: llmd
      custom_backend_module: llmd_verl.checkpoint_engine   # triggers @register("llmd") on worker
      update_weights_bucket_megabytes: 1024
      engine_kwargs:
        llmd:
          controller_url: ${oc.env:LLMD_CONTROLLER_URL,http://localhost:8090}

    # ── inference GPUs live in llm-d pods, not Ray ──────────────
    gpu_memory_utilization: 0.0
    load_format: dummy
```

Everything else (FSDP config, optimizer, PPO/GRPO hyperparams, reward
model, dataset) stays exactly the same. In the given one-step-off policy
example, all of this is already configured in the provided config maps.

---

## Concurrency model

```
LlmdAgentLoopManager.generate_sequences(batch)
    │
    ├── prompts.chunk(num_workers)
    │
    ├── asyncio.gather(
    │       worker_0.generate_sequences.remote(chunk_0),
    │       worker_1.generate_sequences.remote(chunk_1),
    │       ...
    │       worker_N.generate_sequences.remote(chunk_N),
    │   )
    │
    │   Each worker is a separate Ray actor (separate process, separate event loop):
    │   ┌─────────────────────────────────────────────────┐
    │   │  LlmdAgentLoopWorker (Ray actor)                │
    │   │                                                 │
    │   │  generate_sequences(chunk)  [inherited from veRL]│
    │   │    │                                            │
    │   │    ├── asyncio.create_task(_run_agent_loop(sample_0))
    │   │    ├── asyncio.create_task(_run_agent_loop(sample_1))
    │   │    │   ...                                      │
    │   │    └── asyncio.create_task(_run_agent_loop(sample_N))
    │   │         │                                       │
    │   │         └── LlmdSingleTurnAgentLoop.run()       │
    │   │               └── await client.generate(...)    │
    │   │                     └── aiohttp POST to controller│
    │   └─────────────────────────────────────────────────┘
    │
    ├── DataProto.concat(outputs)
    └── return output
```

Each worker has its own event loop and aiohttp session — no cross-loop
issues. HTTP connections are distributed across N processes, providing
true parallelism for the network I/O.

---

## Logging

All modules use the `VERL_LOGGING_LEVEL` env var:

```python
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
```

Set it in your RayCluster yaml to control verbosity globally:

```yaml
env:
  - name: VERL_LOGGING_LEVEL
    value: "INFO"   # DEBUG / INFO / WARN / ERROR
```

- `INFO` — NCCL init, "broadcast N tensors in X.XXXs", weight sync done, worker spawn count.
- `DEBUG` — per-bucket sizes, per-50-tensor progress.

---

## Timing & metrics emitted by `generate_sequences`

`LlmdAgentLoopManager.generate_sequences` attaches a `timing` dict to
the returned `DataProto.meta_info["timing"]`, which veRL's trainer
merges into its `timing_raw` every step. The entries are:

| Metric | What it captures |
|---|---|
| `agent_loop/generate_sequences/min\|max\|mean` | Per-sample HTTP generation time (from `simple_timer` inside `run()`). |
| `agent_loop/tool_calls/min\|max\|mean` | Per-sample tool-call time (0 for single-turn). |
| `agent_loop/compute_score/min\|max\|mean` | Per-sample reward computation time. |
| `agent_loop/num_preempted/min\|max\|mean` | Number of preemptions per sample (0 for llm-d). |
| `agent_loop/slowest/*` | Breakdown of the slowest sample in the batch. |

---

## Deployment

See [`../../deploy/verl/README.md`](../../deploy/verl/README.md) for
the full end-to-end deployment guide (RayCluster + llm-d + training).
