# Deploying llm-d-rl with the llm-d Inference Stack

This guide deploys llm-d-rl with the full llm-d inference scheduling stack (inference router + gateway) for prefix-cache-aware, load-aware routing of RL rollout generation requests.

All resources deploy into a single `llm-d-rl` namespace.

## Architecture

```
Trainer Job
  └─ POST /v1/generate
       └─ Rollout Controller
            ├─ Generation: POST /v1/completions + X-Session-ID ──► Gateway ──► Inference Router ──► best vLLM pod
            └─ Weight sync: pause/update_weights/resume ──► each vLLM pod directly (by pod IP)
```

The inference router (llm-d inference scheduler) selects the optimal vLLM pod for each request using prefix-cache hit rate and load metrics. Weight sync operations always go directly to each pod.

### Pool design

The rollout controller uses a single `EnginePool` for both concerns:

- **Generation routing** — when `--router-url` is set, all `/v1/generate` requests are forwarded to the gateway. The controller never picks a pod for generation.
- **Weight sync** — the controller's Kubernetes pod watcher independently discovers engine pods by label selector (`--engine-selector`) and talks directly to each one for pause/update_weights/resume.

The two paths are decoupled: the gateway handles routing; the controller handles weight sync.

## Prerequisites

### Cluster-level (once per cluster)

Install Gateway API CRDs, Inference Extension CRDs, and the Istio gateway controller:

```bash
./deploy/llm-d/prereqs.sh
```

This installs:
- Gateway API CRDs v1.4.0
- Gateway API Inference Extension CRDs v1.3.0 (adds InferencePool)
- Istio v1.28.1 with `ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true`

Verify:
```bash
kubectl api-resources --api-group=inference.networking.k8s.io
# Should show: InferencePool
```

## Deploy the Stack

### 1. Create namespace and HuggingFace token secret

```bash
kubectl apply -f deploy/cks/namespace.yaml
kubectl -n llm-d-rl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_YOUR_TOKEN
```

### 2. Deploy Inference Router

```bash
kubectl apply -f deploy/cks/epp.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l app=llm-d-epp --timeout=300s
```

This creates:
- ClusterRole + ClusterRoleBinding for the EPP service account (required to watch pods and InferencePools)
- Inference router Deployment + Service (prefix-cache-aware routing on port 9002)

The first deploy may take a while as the image is pulled.

### 3. Deploy Gateway + InferencePool

```bash
kubectl apply -f deploy/cks/gateway.yaml
```

This creates:
- InferencePool CR (watches pods with `llm-d.ai/inference-serving=true`)
- Gateway (Istio Envoy proxy on port 80)
- HTTPRoute (routes traffic through the InferencePool)

### 4. Deploy vLLM engines

```bash
kubectl apply -f deploy/cks/llmd-vllm-engine.yaml
```

Wait for readiness:
```bash
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l llm-d.ai/inference-serving=true --timeout=600s
```

### 5. Deploy rollout controller

```bash
kubectl apply -f deploy/cks/llmd-rollout-controller.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l app=rollout-controller --timeout=120s
```

Key flags used by `llmd-rollout-controller.yaml`:

| Flag | Value | Purpose |
|------|-------|---------|
| `--engine-selector` | `llm-d.ai/inference-serving=true` | Kubernetes pod watch for weight sync |
| `--router-url` | `http://llm-d-inference-gateway-istio.llm-d-rl.svc.cluster.local:80` | Gateway URL for generation routing |
| `--tokens-in` | `false` (default) | Send text prompts; required for prefix-cache-aware routing via the gateway |

### 6. Run a trainer job

For the llm-d inference router path (text prompts for prefix-cache routing):
```bash
kubectl apply -f deploy/cks/trainer-job-text.yaml
kubectl -n llm-d-rl logs -f job/nccl-trainer-text
```

For direct engine dispatch (token IDs, no router):
```bash
kubectl apply -f deploy/cks/trainer-job.yaml
kubectl -n llm-d-rl logs -f job/nccl-trainer
```

## Verify

```bash
# InferencePool exists
kubectl get inferencepools -n llm-d-rl

# vLLM pods discovered by inference router
kubectl get pods -n llm-d-rl -l llm-d.ai/inference-serving=true

# Inference router running
kubectl get pods -n llm-d-rl -l app=llm-d-epp

# Gateway programmed
kubectl get gateway -n llm-d-rl

# Rollout controller sees engines (for weight sync)
kubectl exec -n llm-d-rl deploy/rollout-controller -- \
  wget -qO- http://localhost:8090/v1/pool/status
```

## Customization

### Change the model

Edit `deploy/cks/llmd-vllm-engine.yaml`:
- `--model` arg
- Adjust `--max-model-len` and resource requests as needed

### Scale vLLM replicas

```bash
kubectl -n llm-d-rl scale statefulset/vllm-engine --replicas=4
```

The rollout controller's pod watcher and the inference router both discover new pods automatically — no restart or config change needed.

### Use a different gateway controller

Edit `deploy/cks/gateway.yaml`, change the Gateway's `gatewayClassName`:
- `istio` (default)
- `kgateway`
- `gke-l7-regional-external-managed` (GKE)

And update `--router-url` in `deploy/cks/llmd-rollout-controller.yaml` to match the gateway service name (e.g., for kgateway the service name pattern differs from Istio's `{name}-istio`).

### Skip the inference router (direct dispatch)

Use the original manifests instead:
```bash
kubectl apply -f deploy/cks/vllm-engine.yaml
kubectl apply -f deploy/cks/rollout-controller.yaml
```

These use `--engine-selector=llm-d-role=rollout-engine`, no `--router-url`, and `--tokens-in=true` (token ID arrays sent directly to vLLM).

## Architecture Overview

### Request Flow

Inference requests flow through three layers:

1. **Trainer Job** sends generation requests to the **Rollout Controller**
2. The Rollout Controller forwards them to the **Gateway** (Istio Envoy proxy)
3. The Gateway calls the **inference router** via gRPC to pick the best vLLM pod, then routes the request there

Weight sync operations (pause/update_weights/resume) bypass the Gateway entirely — the Rollout Controller talks directly to each vLLM pod by IP.

### vLLM Discovery (Inference Router)

The inference router discovers vLLM pods dynamically through the Kubernetes **InferencePool** CRD:

1. An `InferencePool` resource is created with a label selector (`llm-d.ai/inference-serving=true`)
2. The router is pointed at this pool via `--pool-name` and `--pool-namespace`
3. The router watches the Kubernetes API for pods matching the selector in that namespace
4. As pods become Ready or are removed, the router's internal endpoint list updates automatically

This means scaling vLLM (e.g., `kubectl scale statefulset/vllm-engine --replicas=4`) is picked up immediately — no restart or config change needed.

### vLLM Discovery (Rollout Controller)

The Rollout Controller has its own independent discovery mechanism for weight sync operations:

1. A **PodWatcher** (`pkg/discovery`) uses a Kubernetes informer with a label selector
2. On Add/Update events, it checks if the pod is Ready and has an IP — if so, it calls `pool.AddEngine(id, address)`
3. On Delete or NotReady transitions, it calls `pool.RemoveEngine(id)`
4. The **EnginePool** (`pkg/pool`) receives these calls and maintains the live engine pool, running periodic health checks (every 15s by default)

### Scheduling Plugins

The inference router uses a plugin chain to pick the optimal pod for each request:

- **prefix-cache-scorer** — scores pods by KV-cache prefix hit rate (weighted 2x)
- **decode-filter** — filters out pods with high decode load
- **max-score-picker** — picks the highest-scoring pod

This is configured in the `epp-config` ConfigMap (`deploy/cks/epp.yaml`).

### Prompt Format

| Mode | `--tokens-in` | Prompt field sent | Used with |
|------|--------------|-------------------|-----------|
| Router (text) | `false` (default) | `"prompt": "<text>"` | `--router-url` + `trainer-job-text.yaml` |
| Direct (tokens) | `true` | `"prompt_token_ids": [...]` | no `--router-url` + `trainer-job.yaml` |

The gateway's inference router requires a text string in `"prompt"` for prefix-cache tokenization. Direct-to-vLLM accepts token ID arrays natively.

## Development

### Prerequisites

- Go 1.24+
- Docker (for container builds)
- Access to a Kubernetes cluster with GPU nodes (for end-to-end testing)

### Building

The rollout controller has two build variants:

```bash
# Lightweight build (no Kubernetes dependencies — uses static engine list via --engines flag)
make build

# Kubernetes-enabled build (includes pod discovery via --engine-selector)
make build-k8s
```

The `k8s` variant uses a Go build tag to include the `pkg/discovery` package, which pulls in the `k8s.io/client-go` dependency.

### Building the Container Image

The Dockerfile always builds the `k8s` variant (with discovery support):

```bash
# Build with auto-detected version tag
REGISTRY=quay.io/youruser/myrepo make docker-build

# Tag as latest and push
docker tag quay.io/youruser/myrepo/llm-d-rl:$(git describe --tags --always --dirty) \
           quay.io/youruser/myrepo/llm-d-rl:latest
docker push quay.io/youruser/myrepo/llm-d-rl:latest
```

Update the image in `deploy/cks/llmd-rollout-controller.yaml` to match.

### Running Locally (without Kubernetes)

For development without a cluster, use the static engine list:

```bash
make build
./bin/rollout-controller \
  --engines=http://localhost:8000 \
  --tokens-in=true \
  --port=8090
```

### Running Locally (with Kubernetes)

Point at your cluster's vLLM pods using a label selector:

```bash
make build-k8s
./bin/rollout-controller-k8s \
  --engine-selector="llm-d.ai/inference-serving=true" \
  --namespace=llm-d-rl \
  --kubeconfig=$HOME/.kube/config \
  --router-url=http://llm-d-inference-gateway-istio.llm-d-rl.svc.cluster.local:80 \
  --port=8090
```

### Running Tests

```bash
make test    # unit tests with race detector
make lint    # golangci-lint
make fmt     # gofmt
```

## File Reference

| File | Purpose |
|---|---|
| `deploy/llm-d/prereqs.sh` | CRDs + Istio (cluster-level) |
| `deploy/cks/namespace.yaml` | Namespace |
| `deploy/cks/epp.yaml` | Inference router Deployment + Service + RBAC |
| `deploy/cks/gateway.yaml` | InferencePool + Gateway + HTTPRoute |
| `deploy/cks/llmd-vllm-engine.yaml` | vLLM with llm-d.ai labels + dev mode |
| `deploy/cks/llmd-rollout-controller.yaml` | Rollout controller with router routing |
| `deploy/cks/trainer-job-text.yaml` | NCCL weight trainer (text prompts, router path) |
| `deploy/cks/trainer-job.yaml` | NCCL weight trainer (token IDs, direct dispatch) |
