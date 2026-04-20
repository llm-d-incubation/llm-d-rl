# Inference Router Mode

By default the llm-d-rl rollout controller dispatches generation requests directly
to the first ready vLLM engine in its pool. For production workloads you can instead
point `--router-url` at an **inference router** — a dedicated scheduling layer that
picks the optimal engine for each request based on live metrics.

## Why use a router

Direct dispatch picks the first ready engine regardless of load. A router adds:

- **KV-cache-aware routing** — steer requests to the engine that already holds the
  relevant KV cache entries, reducing recomputation
- **Load-aware routing** — avoid engines with deep queues or high KV cache pressure
- **Session affinity** — multi-turn rollouts land on the same engine so the KV cache
  built during earlier turns is reused

## Separation of concerns

The router handles **generation scheduling only**. Weight synchronization is always
performed directly by the llm-d-rl rollout controller — it talks pod-to-pod using
the engine pool it discovers independently via label selector. The router is never
in the weight sync path.

```
Training Loop
  └─ POST /v1/generate
       └─ Rollout Controller  (--router-url=http://<router>)
            ├─ Generation: POST /v1/completions ──► Router ──► best vLLM pod
            └─ Weight sync: pause / update_weights / resume
                  ──► each vLLM pod directly (by pod IP, bypasses router)
```

## Router options

### llm-d EPP (Envoy Gateway + Inference Scheduler)

Prefix-cache-aware routing using the Kubernetes Gateway API Inference Extension.
The EPP scores pods by KV-cache prefix hit rate and decode load. Requires
cluster-level CRDs (Gateway API + Istio) and the `llm-d.ai/*` pod labels.

**When to use:** You already have the llm-d inference stack installed, or you need
prefix-cache-aware routing for multi-turn workloads where prompt prefixes repeat.

→ **[Deployment guide: README-llmd.md](README-llmd.md)**

### py-inference-scheduler

Load-aware routing using queue depth and KV cache utilization, implemented as a
lightweight Python proxy. No Kubernetes infrastructure dependencies beyond a
standard cluster.

**When to use:** You want router-mode benefits without the llm-d stack, or your
cluster does not have Gateway API CRDs and Istio installed.

→ **[Deployment guide: README-py-is.md](README-py-is.md)**

---

## Comparison

| | llm-d EPP | py-inference-scheduler |
|---|---|---|
| **Scheduling signal** | KV-cache prefix hit rate + decode load | Queue depth + KV cache utilization |
| **Prompt format** | Text (`--tokens-in=false`) | Text (`--tokens-in=false`) |
| **Infrastructure** | Gateway API CRDs + Istio + InferencePool CRD | None |
| **Rollout controller manifest** | `rollout-controller-router-llmd.yaml` | `rollout-controller-router-py-is.yaml` |
| **vLLM manifest** | `llmd-vllm-engine.yaml` | Either (`llmd-vllm-engine.yaml` or `vllm-engine.yaml`) |
