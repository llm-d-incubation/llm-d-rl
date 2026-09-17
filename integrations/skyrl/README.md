# SkyRL + llm-d

[SkyRL](https://github.com/NovaSky-AI/SkyRL) is a vLLM/torch-based RL training framework. This
integration fronts SkyRL's inference engines with **llm-d routing** (EPP + Envoy).

**Zero code changes to SkyRL.** SkyRL already exposes an external-router seam:
`generator.inference_engine.external_proxy_url` (data plane - all `generate()` calls) and
`generator.inference_engine.external_server_urls` (control plane - pause/resume/sleep/wake/
weight-sync, fanned out directly). SkyRL's inference client (`RemoteInferenceClient`) is HTTP-only,
so this integration is the **llm-d serving** mechanism (Envoy in the data path, calling EPP over gRPC ext_proc to pick a replica).

SkyRL's own config validation forbids `trainer.placement.colocate_all=true` together with
either external-URL field, so routing through EPP requires disaggregated placement: training
and inference use separate GPU allocations, unlike the documented GSM8K quick start's default
(fully colocated).

Use `generator.inference_engine.weight_sync_backend=delta`, not SkyRL's own `nccl` default -
`nccl` fails deterministically in this mode.

## Components

The [llm-d router stack](../common/README.md#llm-d-router-stack) (EPP, Envoy) and the
[registration shim](../common/README.md#registration-shim), started the following way:

```
llm-d-rl-router --epp-config /etc/llmd-configs/epp-config.yaml --envoy-config /etc/llmd-configs/envoy.yaml
llm-d-registration-shim --engine-type vllm
```

SkyRL's own `skyrl.train.entrypoints.serve` entrypoint launches the standalone vLLM engines used
in EPP mode (the same engines a colocated `run_gsm8k.sh`-style run would otherwise launch
in-process); each engine's URL (from `serve.py`'s printed `server_urls`) is registered with the
shim so EPP's endpoints YAML picks it up.

## Get started

- **[quickstart/kuberay/](../../quickstart/kuberay/README.md)** - end-to-end KubeRay example:
  `./deploy.sh apply --framework skyrl`, `provision`, `check`, then
  `FRAMEWORK=skyrl ../benchmarks/scripts/run_on_head.sh --mode epp --task gsm8k --steps 3`.
