# llmd_verl — veRL ↔ llm-d Integration

A drop-in plugin that lets [veRL](https://github.com/volcengine/verl) use
[llm-d](https://github.com/llm-d/llm-d) as its rollout backend instead of
veRL's own Ray-managed vLLM processes.

Training (FSDP + PPO/GRPO) still runs as normal Ray actors on GPUs.
Inference (generation + KV cache) runs in **dedicated llm-d vLLM pods**,
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
│          │ HTTP                        │                                │
└──────────┼─────────────────────────────┼────────────────────────────────┘
           │                             ▼
           │                   ┌────────────────────┐
           │                   │ vLLM pod 0         │
           ▼                   │ vLLM pod 1         │
    ┌──────────────┐           │ vLLM pod …         │   llm-d Go controller
    │ Go controller│──HTTP────▶│ vLLM pod N         │   orchestrates
    │              │           └────────────────────┘   (sleep/wake/pause/
    └──────────────┘                                     weight-transfer)
```

**Control plane** (HTTP):
- `LlmdAgentLoopManager` → Go controller → vLLM pods: generation requests.
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
| `client.py`                    | `RolloutControllerClient` — urllib-based HTTP client for the Go controller API. |
| `agent_loop_manager.py`        | `LlmdAgentLoopManager` — veRL `AgentLoopManager` replacement for rollout. |
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

### `RolloutControllerClient` (`client.py`)

Thin HTTP client around the Go controller's REST API. Endpoints used:

| Method | Endpoint | Called from |
|---|---|---|
| `generate()`             | `POST /v1/generate`        | Agent loop manager (per prompt) |
| `init_weight_transfer()` | `POST /v1/weights/init`    | Checkpoint manager (once, at start) |
| `update_weights()`       | `POST /v1/weights/update`  | Checkpoint manager (every step) |
| `sleep()` / `wake_up()`  | `POST /v1/engines/{sleep,wake}` | Lifecycle |
| `pause()` / `resume()`   | `POST /v1/engines/{pause,resume}` | Lifecycle |
| `get_pool_status()`      | `GET  /v1/pool/status`     | Discover vLLM pod count |

The client is sync (urllib-based). The managers call it via
`asyncio.to_thread(...)` to avoid blocking the event loop.

### `LlmdAgentLoopManager` (`agent_loop_manager.py`)

Replaces veRL's `AgentLoopManager`. veRL constructs it and calls
`generate_sequences(prompts_dataproto)` each training step.

Responsibilities:
1. Unpack the input `DataProto` (`input_ids`, `attention_mask`, sampling flags).
2. Fan out concurrent `POST /v1/generate` requests to the Go controller
   (via a `ThreadPoolExecutor`, one thread per prompt, bounded by
   `_MAX_CONCURRENT`).
3. Collect `output_token_ids` + `logprobs` from responses.
4. Pad all samples to `max_response_length`, build
   `prompts / responses / input_ids / attention_mask / position_ids /
   response_mask` tensors in the shape veRL's fit loop expects.
5. **Launch reward computation as each generation completes** — see below.
6. Return a `DataProto` that the rest of the pipeline consumes unchanged.

#### Streaming reward (reward launched during generation)

As soon as a per-sample generation response arrives (inside the
`ThreadPoolExecutor` `as_completed` loop), the manager immediately fires
a Ray RPC to a `reward_loop_worker` for that sample — **without waiting
for the rest of the batch**. The `ObjectRef` is stashed in
`reward_refs[idx]`, and after all generations are in, the manager does
a single `ray.get(reward_refs)` to collect scores.

This design is **mirrored from upstream veRL**: in veRL's own
`verl/experimental/agent_loop/agent_loop.py`, each sample's
`_run_agent_loop` task calls `await self._compute_score(...)` at the
end of its own generation, so reward compute naturally overlaps with
the rest of the batch still generating. We do the same thing, just
driven by a threadpool `as_completed` loop instead of asyncio tasks
(because the llm-d HTTP client is sync).

Practical consequences:
- Reward compute is almost entirely **hidden under generation** — the
  wall-clock cost of reward is paid during generate, not after.
- The "reward" phase after generation is just a `ray.get` barrier that
  waits for any stragglers. See the
  [Timing & metrics](#timing--metrics-emitted-by-generate_sequences)
  section below for how this affects the reported metrics.

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

- **Packed** (production): pack tensors into contiguous buffers up to
  `bucket_size` bytes, broadcast per bucket with `torch.cat`. Matches
  vLLM's `packed_broadcast_producer` algorithm exactly ("add then check"
  bucket boundary, variable bucket size).
- **Non-packed** (debugging): one `pynccl.broadcast` per tensor.

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
model, dataset) stays exactly the same.

---

## Logging

All four modules use the `VERL_LOGGING_LEVEL` env var (same pattern as
veRL core):

```python
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
```

Set it in your RayCluster yaml to control verbosity globally:

```yaml
env:
  - name: VERL_LOGGING_LEVEL
    value: "INFO"   # DEBUG / INFO / WARN / ERROR
```

- `INFO` — NCCL init, "broadcast N tensors in X.XXXs", weight sync done.
- `DEBUG` — per-bucket sizes, per-50-tensor progress, generation sample previews.

---

## Timing & metrics emitted by `generate_sequences`

`LlmdAgentLoopManager.generate_sequences` attaches a `timing` dict to
the returned `DataProto.meta_info["timing"]`, which veRL's trainer
merges into its `timing_raw` / wandb panel every step. The entries are:

| Metric | Wall-clock window | What it captures | What it does NOT capture |
|---|---|---|---|
| `generate_sequences` | `t_start → t_end` (function entry → batch tensors assembled) | Prompt tokenization + all `POST /v1/generate` HTTPs + response parsing + padded-tensor build. Reward RPCs are **launched** inside this window but their Ray futures are not awaited yet. | The `ray.get` barrier on reward results. |
| `llmd/http_generate` | `t_gen_start → t_gen_end` (threadpool submit → all generate responses collected) | Pure "how long did the fan-out of `POST /v1/generate` take" — dominated by vLLM decode time + controller overhead. | Tokenization, tensor build, reward. |
| `llmd/reward_straggler_wait` | `t_end → ray.get completes` | Time spent **after** generation finished, waiting for the slowest reward Ray task to return. Named "straggler wait" on purpose. | **The true reward compute cost.** That cost was paid in parallel with generation and is effectively hidden inside `generate_sequences`. |

### Why `llmd/reward_straggler_wait` is not "reward time"

Because reward RPCs are launched the moment each generation response
arrives (see the [Streaming reward](#streaming-reward-reward-launched-during-generation)
section), by the time generation as a whole is done, most — often all —
reward futures are already resolved. `ray.get` then returns quickly and
`llmd/reward_straggler_wait ≈ 0`. It does **not** mean reward is free;
it means reward finished overlapping with generation.

Two cases:

- Reward is faster than generation (typical for rule-based GSM8K,
  millisecond-level scoring): every reward future is ready when gen
  ends → `llmd/reward_straggler_wait ≈ 0`.
- Reward is slower than generation (heavy GenRM, network pressure,
  etc.): the barrier measures only the **excess** time, not the full
  cost. So this metric always **lower-bounds** reward compute cost —
  never upper-bounds it.

If you want an honest reward-cost metric, instrument the reward worker
itself and have it report per-sample latency, then summarize
(max / p95 / sum) on the caller side. The current timers are kept
honestly named so the wandb panel isn't misleading.

### Relationship to veRL's own timers

veRL's trainer wraps `generate_sequences` in its own `generate_async`
`marked_timer`. That outer timer therefore captures
`generate_sequences + llmd/reward_straggler_wait` (both happen inside
the manager call). veRL's separate `reward` timer wraps
`_fit_compute_reward`, which under the streaming agent-reward-loop
path is effectively free (just a `batch["rm_scores"]` lookup). Net
result: when reading veRL wandb panels for this run, treat
`generate_async` as "gen + reward" and `reward` as "≈ 0 by
construction."

---

## Deployment

See [`../../deploy/verl/README.md`](../../deploy/verl/README.md) for
the full end-to-end deployment guide (RayCluster + llm-d + training).
