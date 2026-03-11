# Deploying llm-d-rl with the llm-d Inference Stack

This guide deploys llm-d-rl with the full llm-d inference scheduling stack (EPP + Gateway) for KV-cache-aware, load-aware routing of RL rollout generation requests.

All resources deploy into a single `llm-d-rl` namespace.

## Architecture

```
Trainer Job
  └─ POST /v1/generate
       └─ Rollout Controller
            ├─ Generation: POST /v1/completions + X-Session-ID ──► Gateway ──► EPP ──► best vLLM pod
            └─ Weight sync: pause/update_weights/resume ──► each vLLM pod directly (by pod IP)
```

The EPP (Endpoint Picker Plugin) selects the optimal vLLM pod for each request using KV-cache hit rate and load metrics. Weight sync operations always go directly to each pod.

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

### 2. Deploy EPP

```bash
kubectl apply -f deploy/cks/epp.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l app=llm-d-epp --timeout=300s
```

This creates the EPP Deployment + Service (KV-cache-aware routing on port 9002). The first deploy may take a while as the image is pulled.

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

### 6. Run a trainer job

```bash
kubectl apply -f deploy/cks/trainer-job.yaml
kubectl -n llm-d-rl logs -f job/nccl-trainer
```

## Verify

```bash
# InferencePool exists
kubectl get inferencepools -n llm-d-rl

# vLLM pods discovered by EPP
kubectl get pods -n llm-d-rl -l llm-d.ai/inference-serving=true

# EPP running
kubectl get pods -n llm-d-rl -l app=llm-d-epp

# Gateway programmed
kubectl get gateway -n llm-d-rl

# Rollout controller sees engines
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

### Use a different gateway controller

Edit `deploy/cks/gateway.yaml`, change the Gateway's `gatewayClassName`:
- `istio` (default)
- `kgateway`
- `gke-l7-regional-external-managed` (GKE)

And update the `--epp-url` in `deploy/cks/llmd-rollout-controller.yaml` to match the gateway service name (e.g., for kgateway the service name pattern differs from Istio's `{name}-istio`).

### Skip the EPP (direct dispatch)

Use the original manifests instead:
```bash
kubectl apply -f deploy/cks/vllm-engine.yaml
kubectl apply -f deploy/cks/rollout-controller.yaml
```

These use `--engine-selector=llm-d-role=rollout-engine` and no `--epp-url` (round-robin direct dispatch).

## File Reference

| File | Purpose |
|---|---|
| `deploy/llm-d/prereqs.sh` | CRDs + Istio (cluster-level) |
| `deploy/cks/namespace.yaml` | Namespace |
| `deploy/cks/epp.yaml` | EPP Deployment + Service |
| `deploy/cks/gateway.yaml` | InferencePool + Gateway + HTTPRoute |
| `deploy/cks/llmd-vllm-engine.yaml` | vLLM with llm-d.ai labels + dev mode |
| `deploy/cks/llmd-rollout-controller.yaml` | Rollout controller with EPP routing |
| `deploy/cks/trainer-job.yaml` | Example NCCL weight trainer |
