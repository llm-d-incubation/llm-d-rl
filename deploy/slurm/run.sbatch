#!/bin/bash
#SBATCH --job-name=llmd-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:5              # 4 engines + 1 trainer
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=llmd-rl-%j.log

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_ENGINES=4
CONTROLLER_PORT=8090
ENGINE_BASE_PORT=8000
NUM_STEPS=10

# ─── InfiniBand (uncomment if available) ─────────────────────────────────────
# export NCCL_IB_DISABLE=0
# export NCCL_NET=IB
# export NCCL_IB_HCA=mlx5

# ─── vLLM dev mode ──────────────────────────────────────────────────────────
export VLLM_SERVER_DEV_MODE=1

# ─── Start engines ──────────────────────────────────────────────────────────
ENGINE_URLS=""
for i in $(seq 0 $((NUM_ENGINES - 1))); do
    PORT=$((ENGINE_BASE_PORT + i))
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --enforce-eager \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.85 \
        --weight-transfer-config '{"backend":"nccl"}' &
    if [ -n "$ENGINE_URLS" ]; then
        ENGINE_URLS="${ENGINE_URLS},"
    fi
    ENGINE_URLS="${ENGINE_URLS}http://localhost:${PORT}"
done

echo "Waiting for engines to start..."
for i in $(seq 0 $((NUM_ENGINES - 1))); do
    PORT=$((ENGINE_BASE_PORT + i))
    until curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; do
        sleep 5
    done
    echo "  Engine $i ready (port $PORT)"
done

# ─── Start controller ───────────────────────────────────────────────────────
./rollout-controller \
    --engines="$ENGINE_URLS" \
    --port="$CONTROLLER_PORT" &
CONTROLLER_PID=$!

sleep 2
echo "Controller ready (port $CONTROLLER_PORT)"

# ─── Run training ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$NUM_ENGINES python examples/verl/ppo_with_llmd.py \
    --controller-url "http://localhost:${CONTROLLER_PORT}" \
    --model-name "$MODEL" \
    --num-steps "$NUM_STEPS"

# ─── Cleanup ─────────────────────────────────────────────────────────────────
kill $CONTROLLER_PID
wait
echo "Done"
