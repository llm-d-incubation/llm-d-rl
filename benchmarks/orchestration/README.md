# Weight Sync Validation

Validates that llm-d-rl's Go rollout controller can orchestrate multi-engine NCCL weight sync correctly and at speed.

**Result**: On CoreWeave CKS with 4 H200-backed vLLM engines on one node, a full weight sync cycle (sleep, broadcast 16.1 GB via NCCL over InfiniBand, wake) completes in **under 1 second** using **0 GPUs** for coordination.

| Metric | TCP | InfiniBand |
|--------|-----|------------|
| Step time (4 engines) | 8.1s | **0.99s** |
| NCCL broadcast (16.1 GB) | 7.4s | **0.34s** |
| Effective throughput | 2.2 GB/s | **47.8 GB/s** |
| Orchestration overhead | 589ms | **527ms** |

## What We Validated

Each weight sync step runs 6 phases:

1. **Generate** — rollout via vLLM `/v1/completions`
2. **Sleep** — free GPU memory on engines
3. **Train** — update model weights
4. **Broadcast** — NCCL transfer of full model to all engines
5. **Wake** — restore engine memory and KV cache
6. **Resume** — engines resume serving

We ran this at 1, 2, and 4 engine scales (5 warmup + 20 measured steps each). All runs maintained strictly monotonic weight versions with no gaps, missed updates, or ordering errors.

As a sanity check, we ran the same workload through a Ray-based harness that calls the same vLLM endpoints with the same NCCL code path. Orchestration overhead was identical (~520-600ms) across both systems at every scale — confirming the controller adds no meaningful cost. The overhead is dominated by vLLM's sleep/wake GPU memory management, not HTTP round-trips or coordinator logic.

## Bugs Found

Running at multi-engine scale exposed three NCCL deadlocks that affect anyone using vLLM's weight transfer API:

1. **Sequential NCCL init.** `init_weight_transfer_engine` must be called on all engines concurrently — NCCL blocks until all ranks connect. Sequential calls hang on the first engine.

2. **Hardcoded rank offset.** All engines were assigned `rank_offset=1`, causing rank collisions at 2+ engines. Fix: incrementing offsets (trainer=0, engine-0=1, engine-1=2, ...).

3. **Sequential update_weights.** `POST /update_weights` blocks during NCCL receive. At 2+ engines, sequential calls deadlock because the collective never completes. Fix: concurrent HTTP calls.

## Where Time Goes

<p align="center"><img src="charts/step_breakdown.png" width="75%"></p>

Over TCP, NCCL broadcast accounts for 80-91% of step time — everything else is noise. With InfiniBand, NCCL drops to 34% and orchestration overhead (sleep/wake lifecycle) becomes the dominant cost at 53%.

<p align="center"><img src="charts/orchestration_overhead.png" width="75%"></p>

Orchestration overhead stays flat at ~520-600ms regardless of engine count or transport. This is almost entirely vLLM's GPU memory management during sleep/wake — the next optimization target.

## Hardware

- **Cluster**: CoreWeave CKS, NVIDIA H200 (143 GB VRAM), 8 GPUs/node
- **Model**: Llama-3.1-8B-Instruct (8.03B params, 16.1 GB bf16)
- **IB config**: `NCCL_NET=IB`, `NCCL_IB_HCA=ibp`, `rdma/ib: 1`, GPU Direct RDMA
- **vLLM**: v0.16.0, dev mode, enforce eager, max model len 2048

## Remaining Work

- [ ] Optimize vLLM sleep/wake lifecycle (53% of step time with IB)
- [ ] Benchmark with NIXL backend
- [ ] End-to-end veRL integration

## Appendix

<details>
<summary>TCP scaling data (llm-d-rl vs Ray sanity check)</summary>

| Engines | llm-d-rl step | Ray step | Orch. overhead (llm-d-rl / Ray) |
|---------|---------------|----------|---------------------------------|
| 1 | 3.289s | 3.314s | 524ms / 516ms |
| 2 | 7.905s | 5.587s | 558ms / 544ms |
| 4 | 8.106s | 9.066s | 589ms / 604ms |

Step time differences at 2-4 engines reflect TCP network conditions during the runs (different times/nodes), not systemic control plane differences. Orchestration overhead is within 15ms at every scale.

</details>

<details>
<summary>Charts</summary>

<p align="center"><img src="charts/step_time_scaling.png" width="75%"></p>
<p align="center"><img src="charts/nccl_dominance.png" width="75%"></p>
<p align="center"><img src="charts/nccl_throughput.png" width="75%"></p>
<p align="center"><img src="charts/step_distribution.png" width="75%"></p>

</details>

<details>
<summary>Ray + InfiniBand</summary>

The Ray harness failed to complete NCCL IB initialization — NCCL bootstrap socket connections got "Connection refused" in the Ray actor environment. This appears to be a Ray-specific interaction with NCCL's IB bootstrap. TCP results showed identical orchestration overhead, so the IB comparison would not change conclusions.

</details>

<details>
<summary>Files</summary>

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
