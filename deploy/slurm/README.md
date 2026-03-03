# Slurm Deployment

The rollout controller is a standalone Go binary with no Kubernetes dependencies. It communicates with vLLM engines over HTTP and can run anywhere — Slurm, bare metal, Docker, or your laptop.

## Quick Start

```bash
# 1. Start vLLM engines (one per GPU)
srun --gres=gpu:1 --job-name=engine0 \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 --enforce-eager --max-model-len 2048 \
    --weight-transfer-config '{"backend":"nccl"}' &

srun --gres=gpu:1 --job-name=engine1 \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8001 --enforce-eager --max-model-len 2048 \
    --weight-transfer-config '{"backend":"nccl"}' &

# 2. Start rollout controller (no GPU needed)
./rollout-controller \
  --engines=http://engine0-host:8000,http://engine1-host:8001 \
  --port=8090 &

# 3. Run training
srun --gres=gpu:1 python ppo_with_llmd.py \
  --controller-url http://controller-host:8090 \
  --model-name meta-llama/Llama-3.1-8B-Instruct
```

## Sbatch Example

See `run.sbatch` for a self-contained Slurm batch script that launches engines, controller, and trainer as steps within a single allocation.

## InfiniBand

For NCCL over InfiniBand, set these environment variables on the engines and trainer:

```bash
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5   # adapter name varies by cluster
export VLLM_SERVER_DEV_MODE=1
```
