# Deploying llm-d-rl with py-inference-scheduler

This guide deploys llm-d-rl with [py-inference-scheduler](https://github.com/llm-d-incubation/py-inference-scheduler) as the inference routing backend. It provides load-aware and KV-cache-aware routing without requiring the full llm-d Kubernetes stack (no Gateway API CRDs, no Istio, no InferencePool CRD).

All resources deploy into the `llm-d-rl` namespace.

## Architecture

```
Trainer Job
  └─ POST /v1/generate
       └─ Rollout Controller  (--router-url=http://py-is-server:8080)
            ├─ Generation: POST /v1/completions ──► py-is Server ──► best vLLM pod
            └─ Weight sync: pause/update_weights/resume ──► each vLLM pod (direct)
```

The py-is server is a separate pod that:
1. Polls `GET /v1/pool/status` from the rollout controller every 5s to keep its engine list in sync
2. Polls `GET /metrics` from each vLLM pod every 1s for KV cache and queue depth
3. Uses the py-inference-scheduler scoring plugins to pick the best pod per request
4. Proxies the `/v1/completions` request to the chosen pod

Weight sync operations bypass the py-is server entirely — the rollout controller talks directly to each vLLM pod.

## Comparison with the llm-d EPP stack

| | py-is (this guide) | llm-d EPP ([README-llmd.md](README-llmd.md)) |
|---|---|---|
| **Routing** | py-inference-scheduler (Python) | EPP + Envoy Gateway (Go) |
| **Prerequisites** | None beyond a K8s cluster | Gateway API CRDs + Istio |
| **InferencePool CRD** | Not required | Required |
| **Scheduling plugins** | waiting_queue + running_queue + kv_cache | prefix-cache + decode-filter |

## Prerequisites

- A Kubernetes cluster with GPU nodes
- A HuggingFace token secret (for model download)
- The py-is server container image built and pushed (see [Building the image](#building-the-image))

## Deploy the Stack

### 1. Create namespace and HuggingFace token secret

```bash
kubectl apply -f deploy/cks/namespace.yaml
kubectl -n llm-d-rl create secret generic hf-token \
  --from-literal=token=hf_YOUR_TOKEN
```

### 2. Deploy vLLM engines

```bash
kubectl apply -f deploy/cks/vllm-engine.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l app=vllm-engine --timeout=600s
```

### 3. Deploy the py-is server

```bash
kubectl apply -f deploy/cks/py-is-server.yaml
kubectl -n llm-d-rl wait --for=condition=ready pod \
  -l app=py-is-server --timeout=120s
```

This creates:
- `ConfigMap` `py-is-scheduler-config` with the scoring profile
- `Deployment` + `Service` for the py-is proxy on port 8080

### 4. Deploy the rollout controller pointed at py-is

Use `llmd-rollout-controller.yaml` as a base but swap the `--router-url` to point at the py-is server:

```bash
kubectl apply -f deploy/cks/llmd-rollout-controller.yaml
kubectl -n llm-d-rl patch deployment rollout-controller \
  --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args/4","value":"--router-url=http://py-is-server.llm-d-rl.svc.cluster.local:8080"}]'
```

Or edit `llmd-rollout-controller.yaml` locally and change:
```yaml
- "--router-url=http://llm-d-inference-gateway-istio.llm-d-rl.svc.cluster.local:80"
```
to:
```yaml
- "--router-url=http://py-is-server.llm-d-rl.svc.cluster.local:8080"
```

### 5. Run a trainer job

```bash
kubectl apply -f deploy/cks/trainer-job-textinput.yaml
kubectl -n llm-d-rl logs -f job/nccl-trainer-text
```

## Verify

```bash
# py-is server is up and sees engines
kubectl exec -n llm-d-rl deploy/py-is-server -- \
  wget -qO- http://localhost:8080/v1/health
# Expected: {"status":"ok","ready_endpoints":N}

# Rollout controller is up
kubectl exec -n llm-d-rl deploy/rollout-controller -- \
  wget -qO- http://localhost:8090/v1/pool/status

# py-is server logs show routing decisions
kubectl -n llm-d-rl logs deploy/py-is-server
# Expected: "routing to engine-0 (http://...)"
```

## Customization

### Change the scoring weights

Edit the `scheduler.yaml` in the `py-is-scheduler-config` ConfigMap:

```yaml
profiles:
  default:
    scorers:
      - type: waiting_queue
        weight: 1.0   # increase to prioritize low queue depth more
      - type: running_queue
        weight: 1.0
      - type: kv_cache
        weight: 2.0   # increase to prioritize low KV cache pressure more
    picker:
      type: max_score
```

Apply with:
```bash
kubectl apply -f deploy/cks/py-is-server.yaml
kubectl -n llm-d-rl rollout restart deploy/py-is-server
```

### Scale vLLM replicas

```bash
kubectl -n llm-d-rl scale statefulset/vllm-engine --replicas=4
```

The rollout controller's pod watcher picks up new pods automatically. The py-is server polls `/v1/pool/status` every 5s and will see them within one poll cycle.

## Building the image

The py-is server image is built from `integration/py-is/Dockerfile` in this repository.
The Dockerfile copies `integration/py-is/` into the container and installs
`py-inference-scheduler` from GitHub along with `fastapi`, `uvicorn`, and `httpx`.

```bash
export REGISTRY=quay.io/youruser/myrepo
docker build -f integration/py-is/Dockerfile \
  -t ${REGISTRY}/llm-d-rl-py-is:latest .
docker push ${REGISTRY}/llm-d-rl-py-is:latest
```

Update the `image:` field in `deploy/cks/py-is-server.yaml` to match.

## Local dev (no Kubernetes)

```bash
# Terminal 1 — vLLM engines (or mocks)
# Terminal 2 — rollout controller
./bin/rollout-controller \
  --engines=http://localhost:8000,http://localhost:8001 \
  --router-url=http://localhost:8080 \
  --simulate-lifecycle

# Terminal 3 — py-is server
cd integration/py-is
pip install fastapi "uvicorn[standard]" httpx \
  "git+https://github.com/llm-d-incubation/py-inference-scheduler.git"
ROLLOUT_CONTROLLER_URL=http://localhost:8090 \
ROUTER_CONFIG_PATH=scheduler.yaml \
uvicorn server:app --port 8080

# Terminal 4 — training loop
python3 examples/slime/grpo_training_loop.py --controller-url http://localhost:8090
```
