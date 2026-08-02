# vime + llm-d

[vime](https://github.com/vllm-project/vime) is a vLLM-based RL training framework. This integration replaces vime's built-in `vllm_router` with **llm-d routing**.

**Zero code changes to vime.** When `--vllm-router-ip`/`--vllm-router-port` is set, vime's `_start_router` returns immediately without starting the built-in router process ([`rollout.py:1035`](https://github.com/vllm-project/vime/blob/main/vime/ray/rollout.py#L1035)). All rollout traffic goes to that address instead. We point it at Envoy, which forwards inference requests through EPP and registration requests to the shim.

## Components

| Component | Port | Role |
|---|---|---|
| EPP | 9002 (gRPC) | ext_proc filter; scores and picks the target vLLM engine per request |
| Envoy | 8081 (HTTP) | Single entry point; routes `/workers*` to shim, everything else through EPP ext_proc |
| Registration shim | 3001 (HTTP, localhost) | Tracks engine registry; writes `/tmp/epp-endpoints.yaml` for EPP |


## Registration shim API

The shim (`vime-router-shim`) is the only custom component in this stack. Envoy proxies all `/workers*` traffic to it; vime never talks to it directly.

| Method | Path | Called by | Description |
|--------|------|-----------|-------------|
| `POST` | `/workers` | vLLM engine on startup | Register an engine. Idempotent — same URL returns the existing ID. |
| `GET` | `/workers` | vime `abort()` | List all registered engines. |
| `DELETE` | `/workers/{ref}` | vLLM engine on shutdown | Deregister an engine. `{ref}` is the percent-encoded engine URL. |

`POST /workers` body: `{"url": "http://host:port"}` → `{"url": "http://host:port"}`. (`worker_type` is optional — used by vime for prefill/decode disaggregation; defaults to `"regular"`.)

On every change the shim atomically rewrites `/tmp/epp-endpoints.yaml` (configurable via `--endpoints-file`). EPP's `file-discovery` plugin watches that file and picks up the new endpoint list immediately.

## How to use llm-d routing

The only change from a standard vime run is two extra flags on `train.py`:

```
--vllm-router-ip  <envoy-host>  # host where Envoy is running
--vllm-router-port 8081         # Envoy listener
```

## Get started

- **[deploy/README.md](deploy/README.md)** — general deployment guide: install, binaries, config, run. Works on any Ray cluster.
- **[deploy/kuberay/README.md](deploy/kuberay/README.md)** — complete end-to-end KubeRay example: cluster manifest, configs, and scripts for deploy and train.
