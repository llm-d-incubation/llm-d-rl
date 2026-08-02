# Deploying vime on KubeRay

Step-by-step guide for deploying the vime + llm-d cluster and running training. For architecture and how the routing stack works, see [../../README.md](../../README.md).

## Prerequisites

- Kubernetes cluster with GPU nodes (4 GPUs on a single node for the Qwen3-4B example)
- KubeRay operator installed (see [setting-kuberay.md](setting-kuberay.md))
- `envsubst` and `kubectl` on your PATH

## Step 1 — Configure

Export your namespace (required - not stored in a file):
```bash
export NAMESPACE=<your-namespace>
```

Images are defined in `deploy.env` — edit tags there rather than in the manifest:

| Variable | Image |
|---|---|
| `IMG_VIME` | `inferactinc/public:vime-latest` |
| `IMG_CRANE` | `gcr.io/go-containerregistry/crane@sha256:1b1fb24d2b1bb27a9daf81a588157e68463876904e8e537a812edba6284fb252` |
| `IMG_EPP` | `ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:f762cfe4fe4d53be7a91a134f7cd82faddcb0347` |
| `IMG_ENVOY` | `docker.io/envoyproxy/envoy:distroless-v1.33.2` |

If needed, adjust the manifest:
- **GPU count** — `resources.limits.nvidia.com/gpu` defaults to 4; edit to match your node
- **Node placement** — `nodeAffinity` has two knobs:
  - `NotIn` — exclude known-faulty nodes; replace the placeholder hostnames with your own
  - `In` — pin to specific nodes; uncomment the block and add the target hostnames

## Step 2 — Deploy

```bash
bash deploy.sh apply
```

This builds the `llmd-epp-configs-vime` ConfigMap (from `epp-config.yaml`, `envoy.yaml`, `router_shim.py`, `run-qwen3-4B.sh`) and applies the rendered cluster manifest.

Useful sub-commands:
```bash
bash deploy.sh render     # print rendered manifest (no kubectl)
bash deploy.sh configmap  # rebuild ConfigMap only
bash deploy.sh delete     # tear down the cluster
```

## Step 3 — Wait for setup

```bash
kubectl get pods -n $NAMESPACE -w

HEAD=$(kubectl get pod -n "$NAMESPACE" -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Check setup log:
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/setup_log.txt

# Setup is done when this exists:
kubectl exec -n $NAMESPACE $HEAD -- test -f /tmp/vime_ready.txt && echo "ready"
```

PostStart completes when vime is installed and llm-d routing is running. Model download and weight conversion happen in the next step.

## Step 4 — Health check (optional)

Verify all three services are up before submitting a training job:

```bash
# EPP — gRPC port open
kubectl exec -n $NAMESPACE $HEAD -- bash -c \
  'echo > /dev/tcp/127.0.0.1/9002 && echo "EPP OK" || echo "EPP DOWN"'

# Envoy — HTTP port open
kubectl exec -n $NAMESPACE $HEAD -- bash -c \
  'echo > /dev/tcp/127.0.0.1/8081 && echo "Envoy OK" || echo "Envoy DOWN"'

# Shim — HTTP port open
kubectl exec -n $NAMESPACE $HEAD -- bash -c \
  'echo > /dev/tcp/127.0.0.1/3001 && echo "Shim OK" || echo "Shim DOWN"'
```

## Step 5 — Run training

Exec into the head pod and run the training script:

```bash
kubectl exec -it -n "$NAMESPACE" "$HEAD" -- bash

# Inside the head pod:

# Validate with vime's built-in router first (optional):
bash /etc/llmd-configs/run-qwen3-4B.sh --native

# Run with llm-d routing:
bash /etc/llmd-configs/run-qwen3-4B.sh --llmd
```

The script downloads Qwen3-4B and the dapo-math-17k dataset, converts weights to Megatron format (once, skipped on re-runs), then submits the Ray job.

To stop a running job:
```bash
ray job list --address=http://127.0.0.1:8265          # find the job ID
ray job stop <job-id> --address=http://127.0.0.1:8265 # graceful stop
```

## Logs

| File | Component |
|---|---|
| `/tmp/setup_log.txt` | postStart setup (vime install, service startup) |
| `/tmp/epp.log` | EPP |
| `/tmp/envoy.log` | Envoy |
| `/tmp/shim.log` | Registration shim |
| `/tmp/ray/session_latest/logs/worker-*.out` | vLLM engine output |

```bash
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/epp.log
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/envoy.log
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/shim.log
```

## EPP config

`../epp-config.yaml` is the active config. To update it on a running cluster:

```bash
bash deploy.sh configmap
```

Then restart EPP to pick up the new config — the ConfigMap is mounted read-only, so the file on disk updates automatically, but EPP only reads it at startup:

```bash
kubectl exec -n $NAMESPACE $HEAD -- bash -c 'kill $(pgrep epp)'
# EPP will not auto-restart; re-run the postStart start command manually:
kubectl exec -n $NAMESPACE $HEAD -- bash -c '
  nohup /opt/llm-d-bins/epp \
    --config-file /etc/llmd-configs/epp-config.yaml \
    --pool-name file-discovery \
    --pool-namespace default \
    --grpc-port 9002 \
    --grpc-health-port 9003 \
    --metrics-port 9090 \
    --secure-serving=false \
    --tracing=false \
    >> /tmp/epp.log 2>&1 &
'
```

## Increasing verbosity

All components default to quiet logging. Set these env vars in the `env:` block of `ray-cluster.yaml.tmpl`:

| Env var | Component | Default | `info` | `debug` | `trace` |
|---------|-----------|---------|--------|---------|---------|
| `VIME_EPP_VERBOSITY` | EPP subprocess (`-v`) | `1` | `1`-`3` | `4` | `5` |
| `VIME_ENVOY_LOG_LEVEL` | Envoy proxy (`--log-level`) | `info` | `info` | `debug` | `trace` |
