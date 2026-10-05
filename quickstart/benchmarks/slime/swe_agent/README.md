# slime + llm-d: coding_agent_rl on KubeRay

This benchmark runs slime's `coding_agent_rl` on a KubeRay cluster with llm-d
routing. It uses [Claude Code](https://claude.ai/code) as the coding agent and
[SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) as the
workload.

Slime's `coding_agent_rl` talks to sandboxes through a `Sandbox` protocol;
its built-in implementation is E2B. This exampale binds a DinD client instead, using
slime's `--custom-generate-function-path` hook (the same plug-in point as
slime's own examples). `dind/dind_sandbox.py` is an HTTP client for the FastAPI
in each sandbox-runner pod (`server.py` in
`manifests/sandbox-runner.yaml.tmpl`). `dind/dind_generate.py` patches that
client into slime's coding_agent_rl example (`examples/coding_agent_rl/generate.py`
for the agent sandbox, `swe.py` for the eval sandbox), which we leave
unchanged. Both ship in the ConfigMap at `/etc/llmd-configs`.
No slime source changes are needed.

---

## Architecture

```
Ray head pod
  ├── Megatron training
  └── SGLang replicas
        └── requests → Envoy :8081 → EPP → replica

sandbox-runner pods
  ├── sandbox-api  FastAPI server — /sandboxes, /images/prefetch
  └── dind         Docker-in-Docker daemon
        └── SWE-bench containers (one per active rollout)
```

### Routing modes

| Flag | Router |
|------|--------|
| `--mode llm-d` | Envoy + EPP |
| `--mode native` | slime's built-in sglang-router |

---

## Files

| File | Purpose |
|------|---------|
| `run_coding_agent_kuberay.sh` | Main launcher — downloads model, submits Ray job |
| `benchmark.env` | Hyperparameter overrides (batch size, steps, context length) |
| `dind/dind_sandbox.py` | slime `Sandbox` client for the sandbox-runner FastAPI |
| `dind/dind_generate.py` | slime `--custom-generate-function-path` entry: binds `DinDSandbox` |
| `setup_sandbox.sh` | One-shot setup: sandbox runner deployment, dataset preparation, and image prefetch |
| `teardown_sandbox.sh` | Deletes only the sandbox-runner Deployments and Services |
| `scripts/prepare_sandbox.sh` | Prepares the dataset, runner assignments, and image cache. Used by `setup_sandbox.sh` |
| `scripts/prepare_swe_rebench.py` | Downloads SWE-rebench-V2 from HuggingFace → training JSONL on PVC. Used by `setup_sandbox.sh`|
| `manifests/sandbox-runner.yaml.tmpl` | Sandbox runner Services, Deployments, and sandbox HTTP API |
---

## Quickstart

Running this benchmark has two parts: deploying the kuberay cluster,
similarly to the other frameworks, and an additional step of deploying
the sandbox-runner pod pool for the agent tool execution.

For the cluster deployment, refer to
[`../../../kuberay/README.md`](../../../kuberay/README.md)
(`--framework slime-swe-agent`). Then:

### 1. Prepare the sandboxes

Before running, set the workload size and training parameters in `benchmark.env`.

The `setup_sandbox.sh` script deploys the sandbox-runner pods, downloads the dataset,
and prefetches the Docker images onto the sandbox-runner pods. The prepared dataset
and runner configuration go to `slime-cache`; the pulled image tarballs go to
`sandbox-images`. This is safe to rerun — already-cached images are skipped.

Each workload image is mapped to one sandbox runner. Images are assigned
round-robin in dataset order so every training step is spread evenly across the
runner pool. All samples in a GRPO group share the same image and are routed to
the same runner. Each runner prefetches only its assigned images.
`DinDSandbox` uses the image value to select the runner when creating a
sandbox, and later requests for that sandbox continue to use the same runner.

```bash
NAMESPACE=<your-namespace> bash setup_sandbox.sh
```

What it does:
1. Calculates the sandbox pool size and the number of images per pod:
   - `steps = NUM_ROLLOUT`
   - `images_per_runner = floor(SANDBOX_MAX_PER_POD / N_SAMPLES)`
   - `runner_count = ceil(ROLLOUT_BATCH_SIZE / images_per_runner)`
   - `total_images = steps × ROLLOUT_BATCH_SIZE`

   Steps affect the number of unique images to prefetch, but not the runner
   count because the steps run sequentially.
2. Prepares `total_images` rows so each step consumes new prompts and images →
   `/tmp/slime/data/swe_train.jsonl`.
3. Assigns each successive batch evenly across runners and prefetches all
   unique images in parallel.
4. Writes `/tmp/slime/data/sandbox_assignment.json` and
   `sandbox_runners.txt` to the PVC.

To delete only the sandbox-runner Deployments and Services while leaving the
Ray cluster, EPP, Envoy, and PVC running:

```bash
NAMESPACE=<your-namespace> bash teardown_sandbox.sh
```

### 2. Launch training

```bash
HEAD=$(kubectl get pod -n $NAMESPACE -l ray.io/node-type=head \
       -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n $NAMESPACE $HEAD -- \
  bash /etc/llmd-configs/run_coding_agent_kuberay.sh --mode llm-d
```

Use `--mode native` to route through slime's built-in sglang-router instead.

Logs land on the PVC at `/tmp/slime/data/<mode>-run-YYYYmmdd_HHMMSS/driver.log`,
alongside `epp.log` and `envoy.log` in `--mode llm-d`. The newest run is
`ls -dt /tmp/slime/data/*-run-* | head -1`.