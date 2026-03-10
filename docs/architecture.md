# llm-d-rl Architecture

## Component Architecture

```mermaid
flowchart TB
    TRAIN["Train Job"]
    subgraph A["Rollout Controller  (llm-d-rl)"]
        SRV["HTTP Server<br/>pkg/rollout"] ~~~ PM["Pool Manager<br/>pkg/lifecycle"]
        WS["Weight Sync Coordinator<br/>pkg/weightsync"] ~~~ PW["Pod Watcher<br/>pkg/discovery"]
    end
    subgraph B["Inference Scheduler Pod"]
        SCHED["Inference Scheduler<br/>(Envoy Gateway + EPP)<br/>routes to best pod"]
    end
    subgraph C["vLLM Engine Pool"]
        direction LR
        E0["vLLM Pod 0"] ~~~ E1["vLLM Pod 1"]
    end
    TRAIN -- "generate rollouts<br/>weight sync" --> A
    %% Generate path: RC → Scheduler → vLLM
    A -- "POST /v1/completions<br/>+ X-Session-ID" --> B
    B -- "HTTP to chosen pod" --> C

    %% Weight sync path: RC → vLLM (direct, bypasses scheduler)
    A -. "weight sync<br/>pause / update_weights / resume<br/>sleep / wake (direct)" .-> C

    style A fill:#e8e8e8,stroke:#999,color:#000
    style B fill:#e8e8e8,stroke:#999,color:#000
    style E0 fill:#e8e8e8,stroke:#999,color:#000
    style E1 fill:#e8e8e8,stroke:#999,color:#000
```

## RL Training Step — Interaction Sequence

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'actorTextColor': '#000000', 'signalTextColor': '#000000', 'noteBkgColor': '#e8e8e8', 'noteTextColor': '#000000', 'signalColor': '#000000', 'actorBkg': '#e8e8e8', 'actorLineColor': '#000000'}}}%%
sequenceDiagram
    participant TL as Training Loop
    participant RC as Rollout Controller
    participant GW as Envoy Gateway + EPP
    participant vLLM as vLLM Pod(s)
    participant TR as Trainer GPU

    Note over RC,vLLM: Startup — Pod Watcher fires as pods become Ready
    RC->>vLLM: health check (per pod, every 30s)

    rect rgb(200, 220, 255)
        Note over TL,vLLM: ── Generation phase ──
        TL->>RC: POST /v1/generate {prompt_token_ids, session_id}
        RC->>GW: POST /v1/completions + X-Session-ID
        GW->>GW: EPP picks best pod (KV-cache / load aware)
        GW->>vLLM: POST /v1/completions (to chosen pod)
        vLLM-->>GW: {choices, logprobs}
        GW-->>RC: response
        RC-->>TL: {output_token_ids, logprobs, weight_version, engine_id}
    end

    Note over TL: compute rewards, advantages, gradient update

    rect rgb(255, 213, 179)
        Note over TL,vLLM: ── Sleep (GPU hand-off to trainer) ──
        TL->>RC: POST /v1/engines/sleep {level: 2}
        RC->>vLLM: POST /sleep {level:2}  [fan-out, all pods]
        vLLM-->>RC: ok  (GPU memory freed)
        RC-->>TL: {status: sleeping}
    end

    Note over TR: training step uses full GPU memory

    rect rgb(200, 235, 200)
        Note over TL,vLLM: ── Wake + Weight Sync ──
        TL->>RC: POST /v1/engines/wake {tags:[weights]}
        RC->>vLLM: POST /wake_up {tags:[weights]}  [fan-out]
        vLLM-->>RC: ok

        TL->>RC: POST /v1/weights/init {backend:nccl, master_addr, ...}
        RC->>vLLM: POST /init_weight_transfer_engine  [fan-out, parallel]
        Note over TR,vLLM: NCCL rendezvous — trainer rank 0, engines rank 1…N
        vLLM-->>RC: ok (transfer group ready)
        RC-->>TL: {status: ready}

        TL->>RC: POST /v1/weights/update {target_version, param_names, ...}
        RC->>vLLM: POST /pause {mode:keep}  [fan-out]
        RC->>vLLM: POST /update_weights  [fan-out — triggers receive]
        TR-->>vLLM: NCCL broadcast weight tensors (GPU→GPU, direct)
        RC->>vLLM: POST /reset_prefix_cache  [fan-out]
        RC->>vLLM: POST /resume  [fan-out]
        vLLM-->>RC: ok
        RC-->>TL: {status: updated, weight_version: N+1}
    end

    Note over TL,vLLM: repeat from generation phase
```

## Key Design Points

| Path | Route | Why |
|---|---|---|
| Inference | Training Loop → RC → Inference Scheduler → vLLM pod | KV-cache-aware, session-affinity routing |
| Weight sync | RC → each vLLM pod directly | Per-engine control operations (pause/resume/sync) |
| Weight data | Trainer GPU → vLLM GPUs (NCCL/NIXL) | RC orchestrates the lifecycle but never proxies tensors |
| Pod discovery | Kubernetes API → Pod Watcher → Pool Manager + Coordinator | Label selector; no static IPs |
