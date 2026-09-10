# swe_rebench - SWE-reBench agentic RL training (uni-agent harness)

[`nebius/SWE-rebench`](https://huggingface.co/datasets/nebius/SWE-rebench) is a large, continuously
mined collection of real GitHub issues and their merged fixes, each packaged with the repo's own
`FAIL_TO_PASS`/`PASS_TO_PASS` test suites and a ready-to-run Docker image
(`swerebench/sweb.eval.x86_64.<instance_id>`) that reproduces the repo state right before the fix. An
episode gives the model an issue description and the repo checked out at that pre-fix commit, with the
repo's own future git history (tags, later commits) stripped out so the fix can't leak in; reward is
whether the model's own patch makes the held-out tests pass, not a learned or string-matched score.
This workload trains on the `filtered` split and validates on SWE-bench Verified.

Resolving one of these issues is **agentic and multi-turn**, not a single completion: the
[uni-agent](https://github.com/verl-project/uni-agent) ReAct loop drives a per-episode Docker sandbox
through a bounded sequence of tool calls (`stateful_shell`, `str_replace_editor`, `submit` -- up to
`max_steps=60` turns) to explore the repo, reproduce the bug, edit files, and run tests before
submitting a patch. Turns are seconds to minutes apart (shell commands, test runs), and the sandbox
stays alive and stateful for the whole episode -- so, unlike a single-turn text task, keeping every
turn of one episode on the same replica (cross-turn affinity) matters for routing here.

Training is GRPO (`algorithm.adv_estimator=grpo`) via `verl.trainer.main_ppo`, launched through
`ray job submit` with `trainer.v1.trainer_mode=colocate_async` and uni-agent's own rollout adapter
sitting in `agent_loop_manager_class` (`uni_agent.framework.entry.AgentFrameworkRolloutAdapter` for
vanilla routing, `...llmd_bridge.LlmdRouterAgentFramework` for llm-d/EPP routing) -- so this workload
ships its own self-contained launcher (`train_swe_rebench.sh`) rather than a `task.env`.

**Prerequisite**: a provisioned Ray cluster with a working Docker (out of scope for this README).

## 1. Install uni-agent on every cluster node, then add the llm-d bridge

`uni_agent/tasks/swe_rebench/` and everything else this workload needs are already on
[verl-project/uni-agent](https://github.com/verl-project/uni-agent)'s `main` branch. The one
exception is `uni_agent/framework/llmd_bridge.py` (the `LlmdRouterAgentFramework` class the llmd
arm needs) -- not upstreamed, so it ships in this folder (`uni_agent_overlay/`) and is copied
into the clone as a small overlay.

On **every** cluster node running a ray-head or ray-worker process for this workload:

```bash
git clone -b main https://github.com/verl-project/uni-agent /tmp/uni_agent_src
cd /tmp/uni_agent_src && git checkout b49d017cadbbdf8675819eadd524322487b57f2a
pip install -e .
pip install swebench==4.1.0
```

`b49d017cadbbdf8675819eadd524322487b57f2a` is the commit this workload was last validated against.

Then, on every node: copy this whole folder (however you move files onto your cluster's nodes --
`scp`, a shared/mounted volume, your provisioning tool's file-push) to
`/tmp/uni_agent_src/examples/quickstart/training/swe_rebench_workload/`, and copy the llm-d bridge
into the cloned package (needed for the llmd arm; harmless to have for vanilla too):

```bash
cp /tmp/uni_agent_src/examples/quickstart/training/swe_rebench_workload/uni_agent_overlay/uni_agent/framework/llmd_bridge.py \
   /tmp/uni_agent_src/uni_agent/framework/llmd_bridge.py
```

`train_swe_rebench.sh` and `prepare_env.sh` below assume they are run from inside that copied
folder, on the node.

## 2. Prepare data and images

On the **ray-head** node:

```bash
cd /tmp/uni_agent_src/examples/quickstart/training/swe_rebench_workload
bash prepare_env.sh
```

This preprocesses `nebius/SWE-rebench` (`split="filtered"`) into `swe_rebench_filtered.parquet` and
SWE-bench Verified into `swe_bench_verified.parquet` under `/tmp/verl/data/uni_agent/`, then pre-pulls
and patches (adds `tmux`, needed by `stateful_shell`; swaps in a reachable apt mirror) every Docker
image those rows reference, on **this node's** local Docker daemon.

**Repeat on every ray-worker node too** -- each node runs an independent Docker cache, and a Ray task
can land on any of them. The dataset preprocessing step is harmless to repeat (it just re-reads the
same rows), so the simplest path is: copy the two parquet files from the ray-head node's
`/tmp/verl/data/uni_agent/` to the same path on each worker, then run `bash prepare_env.sh` again
there -- it skips straight to pre-pulling images it finds an already-written parquet pair.

`MAX_INSTANCES` (default 64, set as an env var before `prepare_env.sh`) keeps this bounded (~64
multi-GB images, tens of minutes); set `MAX_INSTANCES=0` for the full 6542-row split, at which point
pre-pulling every image stops being practical -- pre-fix a curated subset instead and accept
occasional cold pulls for the rest.

## 3. Run

Both arms use the same script; only `ROUTING_ARM` differs, so every other override (model, batch
size, response length, GRPO group size, task config) is identical by construction.

### Without the llm-d router (baseline)

```bash
ROUTING_ARM=vanilla bash train_swe_rebench.sh trainer.total_training_steps=5
```

### With the llm-d router (EPP)

```bash
ROUTING_ARM=llmd RUNTIME_ENV=runtime_env_reqlog.yaml \
bash train_swe_rebench.sh trainer.total_training_steps=5
```

`ray job submit --no-wait` returns immediately; monitor with `ray job list` / `ray job logs <id>` on
the ray-head node. The two arms differ by exactly one class swap plus three EPP-only overrides --
the vanilla command is a strict subset of the llmd one by construction, never a separately
maintained branch.

### Routing policy

The llmd arm defaults to `EPP_CONFIG_FILE=/etc/llmd-configs/epp-config-persistent.yaml` (llm-d-rl's
existing prefix-cache + load-aware profile). A measured upgrade for this workload's long multi-turn
episodes is `epp-config-session-affinity.yaml` (shipped in this folder): `session-affinity-scorer`
(keyed on the `x-request-id` header, which equals uni-agent's per-episode `session_id` and is stable
across all of an episode's turns) plus `active-request-scorer` as a tiebreaker. In one measured run,
switching to it raised the fraction of episodes staying pinned to one replica across their whole
episode from **5.6% to 67.7%**:

```bash
EPP_CONFIG_FILE=/tmp/swe_rebench_configs/epp-config-session-affinity.yaml \
ROUTING_ARM=llmd RUNTIME_ENV=runtime_env_reqlog.yaml \
bash train_swe_rebench.sh trainer.total_training_steps=5
```

(Copy `epp-config-session-affinity.yaml` to a path outside the cluster's managed `/etc/llmd-configs`
ConfigMap first, e.g. `/tmp/swe_rebench_configs/`, so provisioning doesn't overwrite it.) That number
depends on `epp_report_completion=true`, which `train_swe_rebench.sh` already sets by default for the
llmd arm -- see the comment above `EPP_REPORT_COMPLETION` in the script for why it's required for any
identity-keyed scorer.

## 4. Verify it actually trained

- The Ray job log shows `verl.trainer.ppo` training-loop lines, `global_step` incrementing past 0/1,
  and `actor/pg_loss` + `actor/grad_norm` present and non-NaN (the strongest signal `update_actor` ran).
- The resolved Hydra config dump shows `algorithm.adv_estimator: grpo`, not a silently-defaulted `gae`.
- A checkpoint lands at the first `save_freq=5` boundary, under `trainer.default_local_dir`.
- **llmd arm only:** each `GatewayActor` process writes its own `reqlog-<pid>.jsonl` under
  `VERL_REQLOG_DIR` (`/tmp/reqlog`) on *its own node's* local filesystem. Copy `/tmp/reqlog` off of
  **every** ray-head and ray-worker node and concatenate before reading routing evidence out of it,
  or you will silently undercount:

  ```bash
  # on each node, however you move files off your cluster's nodes:
  cat /tmp/reqlog/*.jsonl > /tmp/reqlog_<node-name>.jsonl
  # then, once collected locally from every node:
  cat reqlog_*.jsonl > merged_reqlog.jsonl
  ```

## Files

| File | Purpose |
|---|---|
| `train_swe_rebench.sh` | The launcher. `ROUTING_ARM=vanilla\|llmd` toggles routing; every other override is shared. |
| `prepare_env.sh` | One-time setup: preprocess datasets, pre-pull+fix this node's Docker images (run once per node, §2). |
| `task_config_react.yaml` | uni-agent task config: `swe_rebench` (train) + `swe_bench` (val), `provider: docker`. |
| `runtime_env.yaml` | Baseline `ray job submit --runtime-env` file (working_dir + compat env vars). |
| `runtime_env_reqlog.yaml` | Same, plus `VERL_REQLOG_DIR` for per-request routing evidence (llmd arm). |
| `epp-config-session-affinity.yaml` | The measured session-affinity + active-request EPP config (§3, "Routing policy"). |
| `uni_agent_overlay/` | `llmd_bridge.py`, not upstreamed -- copy onto the clone (§1). |
