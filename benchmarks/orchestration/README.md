# Weight Sync Validation

## Why a Control Plane Matters

In RL training with disaggregated inference, the trainer must sync updated weights to all vLLM engines after every gradient step. This involves a precise multi-phase lifecycle (sleep engines to free GPU memory, broadcast weights via NCCL, wake engines, restore KV cache) and every phase has ordering constraints that are easy to get wrong at multi-engine scale.

Without a control plane, every RL framework has to implement this lifecycle itself: manage NCCL group creation with correct rank assignments, ensure concurrent fan-out for collective operations, handle sleep/wake sequencing, and track weight versions. Getting any of these wrong causes silent hangs or deadlocks (see [Bugs Found](#bugs-found)).

llm-d-rl's rollout controller handles all of this behind an HTTP API. The trainer calls `POST /v1/weights/update` and the controller does the rest, correctly, concurrently, and without consuming any GPUs for coordination.

## Validation Results

We validated the controller at 1, 2, and 4 engine scales on CKS with NVIDIA H200 GPUs, running Llama-3.1-8B-Instruct (16.1 GB at bf16).

**4 engines, InfiniBand, same node:**

| Metric | Result |
|--------|--------|
| Full weight sync cycle | **0.99s** |
| NCCL broadcast (16.1 GB) | 0.34s (47.8 GB/s) |
| Orchestration overhead | 0.53s |
| Weight version correctness | 20/20 steps, no gaps |

**Orchestration overhead** (everything except generate, train, and NCCL broadcast) is ~527ms per step. This is dominated by vLLM's internal GPU memory management during sleep/wake — not HTTP round-trips or controller logic. We confirmed this by running an equivalent Ray-based harness that calls the same vLLM endpoints: orchestration overhead was identical (~520-600ms) at every scale.

<p align="center"><img src="charts/orchestration_overhead.png" width="75%"></p>

## Bugs Found

Running at multi-engine scale exposed three deadlock patterns in vLLM's weight transfer API. These affect any system orchestrating NCCL weight sync — the controller now handles all of them correctly.

1. **NCCL init requires concurrent fan-out.** `init_weight_transfer_engine` must be called on all engines simultaneously. NCCL's `StatelessProcessGroup.create()` blocks until all ranks connect — sequential calls hang on the first engine.

2. **Rank offsets must be unique.** Assigning all engines `rank_offset=1` causes rank collisions at 2+ engines. The controller assigns incrementing offsets (trainer=0, engine-0=1, engine-1=2, ...).

3. **update_weights requires concurrent fan-out.** `POST /update_weights` blocks during NCCL receive. Sequential calls at 2+ engines deadlock because the collective never forms.

## Where Time Goes

<p align="center"><img src="charts/step_breakdown.png" width="75%"></p>

With InfiniBand, NCCL broadcast takes 0.34s — fast enough that orchestration overhead (sleep/wake lifecycle) becomes the dominant cost at 53% of step time. The next optimization target is vLLM's GPU memory management during these transitions.

## Setup

- **Cluster**: CoreWeave CKS, NVIDIA H200 (143 GB VRAM), 8 GPUs/node
- **Model**: Llama-3.1-8B-Instruct (8.03B params, 16.1 GB bf16)
- **Transport**: NCCL over InfiniBand with GPU Direct RDMA
- **vLLM**: v0.16.0, dev mode, enforce eager, max model len 2048
- **Protocol**: 5 warmup + 20 measured steps, wall-clock timing per phase

## Remaining Work

- [ ] Optimize vLLM sleep/wake lifecycle (53% of step time with IB)
- [ ] Benchmark with NIXL backend
- [ ] End-to-end veRL integration

<details>
<summary>Appendix: TCP baseline and Ray comparison</summary>

We also ran the full sweep over TCP sockets as a baseline, and compared against a Ray-based harness that calls the same vLLM endpoints with the same NCCL code path.

| Engines | llm-d-rl step | Ray step | Orch. overhead (llm-d-rl / Ray) |
|---------|---------------|----------|---------------------------------|
| 1 | 3.289s | 3.314s | 524ms / 516ms |
| 2 | 7.905s | 5.587s | 558ms / 544ms |
| 4 | 8.106s | 9.066s | 589ms / 604ms |

Over TCP, NCCL dominates step time (80-91%), making orchestration overhead hard to distinguish. Step time differences at 2-4 engines reflect network conditions during the runs, not control plane differences — orchestration overhead is within 15ms at every scale.

The Ray harness failed to complete NCCL IB initialization due to a Ray actor interaction with NCCL's IB bootstrap. TCP results confirmed identical orchestration overhead between systems.

<p align="center"><img src="charts/step_time_scaling.png" width="75%"></p>
<p align="center"><img src="charts/nccl_dominance.png" width="75%"></p>
<p align="center"><img src="charts/nccl_throughput.png" width="75%"></p>
<p align="center"><img src="charts/step_distribution.png" width="75%"></p>

</details>

<details>
<summary>Appendix: Files</summary>

| File | Purpose |
|------|---------|
| `llmd_bench.py` | Instrumented llm-d-rl trainer with per-phase timing |
| `ray_bench.py` | Ray orchestration harness (sanity check) |
| `run_sweep.sh` | Automated sweep across 1/2/4 engine scales |
| `analyze.py` | Parse results and generate comparison tables |
| `generate_charts.py` | Generate matplotlib visualizations |
| `results/*.json` | Raw per-step timing data |
| `charts/*.png` | Generated visualizations |

</details>
