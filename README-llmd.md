# Deploying llm-d-rl with the llm-d Inference Stack

For background on router mode and a comparison with the py-inference-scheduler option,
see [README-inference-router.md](README-inference-router.md).

All resources deploy into a single `llm-d-rl` namespace.

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
kubectl apply -f deploy/cks/rollout-controller-router-llmd.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l app=rollout-controller --timeout=120s
```

Key flags used by `rollout-controller-router-llmd.yaml`:

| Flag | Value | Purpose |
|------|-------|---------|
| `--engine-selector` | `llm-d.ai/inference-serving=true` | Kubernetes pod watch for weight sync |
| `--router-url` | `http://llm-d-inference-gateway-istio.llm-d-rl.svc.cluster.local:80` | Gateway URL for generation routing |
| `--tokens-in` | `false` (default) | Send text prompts; required for prefix-cache-aware routing via the gateway |

### 6. Run a trainer job

For the llm-d inference router path (text prompts for prefix-cache routing):
```bash
kubectl apply -f deploy/cks/trainer-job-textinput.yaml
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

And update `--router-url` in `deploy/cks/rollout-controller-router-llmd.yaml` to match the gateway service name (e.g., for kgateway the service name pattern differs from Istio's `{name}-istio`).

### Skip the inference router (direct dispatch)

Use the original manifests instead:
```bash
kubectl apply -f deploy/cks/vllm-engine.yaml
kubectl apply -f deploy/cks/rollout-controller.yaml
```

These use `--engine-selector=llm-d-role=rollout-engine`, no `--router-url`, and `--tokens-in=true` (token ID arrays sent directly to vLLM).

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

Update the image in `deploy/cks/rollout-controller-router-llmd.yaml` to match.

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
| `deploy/cks/rollout-controller-router-llmd.yaml` | Rollout controller with router routing |
| `deploy/cks/trainer-job-textinput.yaml` | NCCL weight trainer (text prompts, router path) |
| `deploy/cks/trainer-job.yaml` | NCCL weight trainer (token IDs, direct dispatch) |
