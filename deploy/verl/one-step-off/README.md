# veRL + llm-d: one-step off-policy

End-to-end deployment for running veRL RL training with **llm-d** as the rollout backend, or **vanilla veRL** as a baseline, for one-step off-policy experiments.

It is built from these pieces:

1. A **KubeRay cluster** — trainer **head** pod (driver) and **worker** pod(s) (FSDP training GPUs).
2. An **llm-d deployment** (Go controller + vLLM) for rollout generation — **only** when you use the llm-d RayCluster and `llm-d-config.yaml`.

---

## Prerequisites

- `kubectl` configured for your cluster.
- A namespace with GPU capacity.
- `helm` (to install KubeRay).
- For llm-d runs: a running llm-d stack reachable from the RayCluster (controller URL, e.g. `http://<controller-svc>.<namespace>.svc.cluster.local:8090`).

---

## GPU layout

The one-step-off setup targets **2 GPUs for training** and **4 for rollout** (**6** accelerator GPUs in total; the Ray head is CPU-only).

### Vanilla veRL path (`ray-cluster-verl.yaml`)

| Where | GPUs | Role |
|---|---:|---|
| Ray **head** | **0** | Driver, `fit()`, Ray GCS (CPU only). |
| Ray **worker** | **6** (see `ray-cluster-verl.yaml`) | **2** for FSDP (`actor_rollout_ref.actor.n_gpus_per_node` in `verl-config.yaml`), **4** for vLLM rollout (`rollout.n_gpus_per_node`). |

Everything runs **inside** the Ray cluster. The worker `resources.limits` must cover both pools (here **6×** `nvidia.com/gpu` with `replicas: 1`). Hydra uses **4** rollout GPUs with `tensor_model_parallel_size: 1`, so one vLLM engine per GPU.

### llm-d path (`ray-cluster-llmd.yaml` + vLLM manifests)

| Where | GPUs | Role |
|---|---:|---|
| Ray **head** | **0** | Same as baseline (CPU-only driver). |
| Ray **worker** | **2** | FSDP training only. |
| llm-d vLLM StatefulSet (`vllm-engine*.yaml` in this folder) | **4** (e.g. **4** pods × **1** GPU) | Rollout only; Go controller load-balances HTTP to these engines. |

---

## Deploying

Run commands from this directory: `deploy/verl/one-step-off/`.

### 1. Install KubeRay (once per namespace)

```bash
export NAMESPACE=<your-namespace>
bash prereqs.sh
```

Installs the KubeRay operator and CRDs into `$NAMESPACE`.

### 2. Deploy llm-d (llm-d runs only)

Deploy the llm-d controller, EPP, and related pieces as described in `README-llmd.md`, except for the vLLM engines. For the vLLM engines defined **in this folder**, apply the following:

```bash
kubectl apply -f vllm-engine.yaml
```

Wait until vLLM pods are `Ready`:

```bash
kubectl get pods -l app=vllm -w
```

The `vllm-engine` manifest here is aligned with the veRL trainer image (vLLM 0.17.x) and similar engine flags to vanilla veRL’s vLLM rollout.

### 3. Deploy the RayCluster

Choose one:

```bash
# Vanilla veRL (rollout on Ray workers) — verl-config.yaml
kubectl apply -f ray-cluster-verl.yaml

# llm-d (rollout on llm-d pods) — llm-d-config.yaml
kubectl apply -f ray-cluster-llmd.yaml
```

Each file defines a CPU-only head, one GPU worker group, and shared env vars such as `LLMD_CONTROLLER_URL`, `VERL_LOGGING_LEVEL`.

Wait until the cluster is ready:

```bash
kubectl get raycluster verl-cluster -w
# STATUS = ready
```

### 4. Apply the ConfigMap

Hydra configs are shipped in `verl-configs-cm.yaml` (inline `verl-config.yaml` and `llm-d-config.yaml`). The RayCluster manifests mount them at `/etc/verl-configs` on head and workers.

```bash
kubectl apply -f verl-configs-cm.yaml
```

Check the mount (set `HEAD_POD` from your cluster):

```bash
kubectl exec "${HEAD_POD}" -- ls /etc/verl-configs
# expect: llm-d-config.yaml  verl-config.yaml
```

Notes:

- The ConfigMap `namespace:` must match the RayCluster namespace.
- After editing and re-applying the ConfigMap, mounted files refresh within ~60s, or delete pods for an immediate reload:
  ```bash
  kubectl delete pods -l ray.io/cluster=verl-cluster
  ```
- To rebuild the ConfigMap from the loose files instead of editing the inlined YAML:
  ```bash
  kubectl create configmap verl-configs \
      --from-file=verl-config.yaml \
      --from-file=llm-d-config.yaml \
      --dry-run=client -o yaml | kubectl apply -f -
  ```

### 5. Run training

Shell into the head pod:

```bash
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -ti "${HEAD_POD}" -- bash
cd /opt/verl
```

**llm-d backend** (rollout via llm-d vLLM pods):

```bash
python -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path /etc/verl-configs \
    --config-name llm-d-config \
    data.train_files=/tmp/verl/data/gsm8k/train.parquet \
    data.val_files=/tmp/verl/data/gsm8k/test.parquet
```

**Vanilla veRL** (rollout inside Ray workers):

```bash
python -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path /etc/verl-configs \
    --config-name verl-config \
    data.train_files=/tmp/verl/data/gsm8k/train.parquet \
    data.val_files=/tmp/verl/data/gsm8k/test.parquet
```

Logs go to stdout (`kubectl logs "${HEAD_POD}"`) and Weights & Biases (e.g. project `verl-benchmark`).

---

### Log level

Set `VERL_LOGGING_LEVEL` in the RayCluster YAML, re-apply, and recreate pods so they pick up env vars:

```yaml
- name: VERL_LOGGING_LEVEL
  value: "DEBUG"    # or INFO / WARN / ERROR
```
---

## See also

- [`../../python/llmd_verl/README.md`](../../python/llmd_verl/README.md) — `llmd_verl` integration (classes and architecture).
