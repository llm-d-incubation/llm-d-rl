# Orchestration Overhead Benchmark: llm-d-rl vs Ray

## What This Measures

This benchmark isolates the **orchestration overhead** of RL training step coordination. Both systems perform identical GPU operations (NCCL weight broadcast, vLLM sleep/wake, generation) on the same hardware. The only difference is the control plane:

- **llm-d-rl**: Go HTTP server coordinates vLLM engines via HTTP
- **Ray harness**: Ray actor coordinates the same vLLM engines via HTTP (simulating veRL's `CheckpointEngineManager` pattern)

### Per-Step Lifecycle (6 phases)

1. **Generate** — produce a rollout via vLLM
2. **Sleep** — free GPU memory on engines (level 2)
3. **Train** — perturb model weights (simulating a gradient step)
4. **Wake weights** — restore weight memory on engines
5. **Weight sync** — NCCL broadcast from trainer to engines + HTTP coordination
6. **Wake KV cache** — restore KV cache memory

**Orchestration overhead** = step_total - generate - train - nccl_broadcast

This removes the GPU-bound operations to isolate what the control plane adds.

## Prerequisites

- CoreWeave CKS cluster with H200 GPUs
- `kubectl` configured for the cluster
- Namespace `llm-d-rl` with `hf-token` and `ghcr-creds` secrets
- Rollout controller deployed (`deploy/cks/rollout-controller.yaml`)

## Quick Start

```bash
# 1. Deploy 8B vLLM engines (4 replicas)
kubectl apply -f deploy/cks/vllm-engine-8b.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=vllm-engine --timeout=600s

# 2. Run full sweep (1/2/4 engines x 2 systems)
cd benchmarks/orchestration
./run_sweep.sh ./results

# 3. Analyze
python analyze.py --results-dir ./results --detailed
```

## Manual Runs

### llm-d-rl benchmark (single scale)

```bash
# Ensure controller is deployed and pointing at the right engines
kubectl apply -f deploy/cks/bench-trainer-job.yaml
kubectl -n llm-d-rl logs -f job/llmd-bench
```

### Ray benchmark (single scale)

```bash
# Update ENGINE_URLS env var in ray-trainer-job.yaml to match engine count
kubectl apply -f deploy/cks/ray-trainer-job.yaml
kubectl -n llm-d-rl logs -f job/ray-bench
```

## Output

Results are JSON files with per-step timing data:

```json
{
  "system": "llm-d-rl",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "num_engines": 4,
  "steps": [
    {
      "generate": 0.234,
      "sleep": 0.012,
      "train": 0.089,
      "wake_weights": 0.015,
      "nccl_broadcast": 0.156,
      "update_weights_http": 0.203,
      "wake_kvcache": 0.011,
      "step_total": 0.982
    }
  ]
}
```

The `analyze.py` script computes median, p95, p99 per phase and generates comparison tables.

## Files

| File | Purpose |
|------|---------|
| `llmd_bench.py` | Instrumented llm-d-rl trainer with per-phase timing |
| `ray_bench.py` | Ray orchestration harness with per-phase timing |
| `run_sweep.sh` | Automated sweep across 1/2/4 engine scales |
| `analyze.py` | Parse results and generate comparison tables |
