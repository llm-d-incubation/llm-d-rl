# Running the quickstart on KubeRay

One cluster definition serves every RL framework in this repo. Nothing here is
required to *use* the integration - see
[`integrations/README.md`](../../integrations/README.md) for that. This is the
worked example we run on our own cluster.

## Layout

| File | What it is |
|---|---|
| `deploy.sh` | the only entry point: apply, provision, check, render, delete |
| `frameworks.env` | per-framework cluster shape: topology, sizes, names |
| `deploy.env` | values that belong to *this* cluster |
| `ray-cluster.head-workers.yaml.tmpl` | CPU head + GPU worker group (verl) |
| `ray-cluster.single-pod.yaml.tmpl` | one pod, GPUs on the head (vime, slime) |
| `provision/<framework>.sh` | how each framework is installed on a node |
| `pvc.yaml` | durable cache for checkpoints and datasets |
| `setting-kuberay.md` | installing the KubeRay operator itself |

Two templates, keyed by **topology** rather than by framework, so adding a
framework costs a `provision/<name>.sh` and a block in `frameworks.env` - never a
manifest. Everything else is a variable.

## Prerequisites

`kubectl`, `envsubst` (GNU gettext), `python3` with `pyyaml`, a namespace, and the
KubeRay operator ([`setting-kuberay.md`](setting-kuberay.md)).

```bash
export NAMESPACE=<your-namespace>
```

## 1. Create the durable cache (once per namespace)

```bash
./deploy.sh pvc --framework verl
```

`deploy.sh delete` never removes it, so models and datasets survive a recreate.

## 2. Start the cluster

```bash
./deploy.sh apply --framework verl                  # vLLM (default engine)
./deploy.sh apply --framework verl --engine sglang
./deploy.sh apply --framework vime
./deploy.sh apply --framework slime
./deploy.sh render --framework verl                 # inspect first, no kubectl
```

Every framework renders the **same** cluster name, so they are mutually exclusive
within a namespace - applying one replaces the other. That is deliberate: the
scripts in `../benchmarks/scripts` all resolve pods by label and take `items[0]`,
which cannot tell two clusters apart.

## 3. Provision the nodes

Nothing is installed at pod start. Wait for the pods to be `Ready`, then:

```bash
./deploy.sh provision --framework verl
./deploy.sh check     --framework verl
```

`provision` clones the framework at the ref in
`integrations/verl/environments.env`, installs it and the integration packages on
every node, verifies the engine version against what the framework declares it
needs, and writes `/tmp/llmd-provisioned.json` on each pod. `check` reads those
markers back and fails if any pod is unprovisioned or if the pods disagree.

**Pod `Ready` does not mean "provisioned"** - it means Ray is up. `check` is what
tells you the rest, and `run_on_head.sh` runs it before every launch.

Why it works this way: changing a framework ref no longer needs a pod recreate,
which on a shared cluster risks losing the GPU allocation; a failure is an exit
code on your terminal instead of a restart loop; and provisioning your own
working tree is a first-class mode rather than copying files into site-packages:

```bash
./deploy.sh provision --framework verl --from-local        # this checkout
./deploy.sh provision --framework verl --from-local /path  # another one
```

Re-running is cheap and idempotent (a fetch, not a fresh clone). Do it after any
pod restart - a restarted pod comes back with Ray but no framework, and `check`
is what catches that.

## 4. Run something

verl:

```bash
cd ../benchmarks
scripts/run_on_head.sh --mode epp --task gsm8k --steps 1
```

vime:

```bash
cd ../benchmarks
FRAMEWORK=vime scripts/run_on_head.sh --mode llm-d --steps 1
```

slime:

```bash
cd ../benchmarks
FRAMEWORK=slime scripts/run_on_head.sh --mode llm-d --steps 1
```

See [`../benchmarks/verl/README.md`](../benchmarks/verl/README.md) for the
workloads and modes, and `llm-d-rl-verl-overrides --list` for what each mode sets.

## Changing versions

| To change | Edit |
|---|---|
| EPP / Envoy / sidecar build | `integrations/common/src/llm_d_rl_common/configs/versions.env` |
| an EPP scorer config | `configs/epp/{base,profiles,modifiers}` then `deploy.sh configmap` |
| framework commit or image | `integrations/<framework>/environments.env`, then `deploy.sh provision` |
| cluster shape (GPUs, dshm, topology) | `frameworks.env`, then re-`apply` |
| EPP binary only, fast loop | `../benchmarks/scripts/push-epp.sh` |

The EPP image and the EPP configs are pinned together in `versions.env` on
purpose: a config can require a plugin or a stability flag that only some builds
have, and keeping the two apart is what let two frameworks silently score against
different EPP versions.

## Logs

| Path on the pod | What |
|---|---|
| `/tmp/train.log` | the training run (what `run_on_head.sh` tails) |
| `/tmp/llmd-provisioned.json` | provisioning provenance, read by `deploy.sh check` |
| `/tmp/router.log` | EPP + Envoy, on frameworks that start them from provisioning |
| `/tmp/shim.log` | the registration shim (vime, slime) |
| `/tmp/verl/reqlog/*.jsonl` | per-request timing records |
| `/tmp/vllm_metrics.csv` | scraped engine `/metrics` |

Raise verbosity with `LLMD_EPP_VERBOSITY` / `LLMD_ENVOY_LOG_LEVEL` in the
manifest's env block.

## Per-framework notes

**verl** - CPU head running the driver, one GPU worker group. The framework clone
lives on the PVC, so every node installs the same commit rather than cloning
independently. PD and P2P modes need the sidecar, which the init container
extracts.

**vime** - single pod, GPUs on the head (vime cannot run a CPU-only head).
Provisioning also starts EPP, Envoy and the registration shim, because vime has
no in-process hook to start them from.

**slime** - single pod like vime. It needs an EPP image carrying
`sglanghttp-parser`; `integrations/slime/environments.env` ships a placeholder and
`deploy.sh` refuses to apply until you replace it.
