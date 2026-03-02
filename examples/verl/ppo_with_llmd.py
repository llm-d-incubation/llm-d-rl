"""PPO training loop using llm-d-rl via the llmd_verl package.

This example demonstrates how veRL's PPO training loop uses llm-d-rl
as its rollout backend. It replaces veRL's Ray-based engine management
with llm-d-rl's Go controller + NCCL data plane.

The training loop follows veRL's standard PPO pattern:
    for step in training_loop:
        responses = generate_rollouts(prompts)      # via llm-d-rl controller
        manager.sleep_replicas()                     # free GPU memory
        rewards = compute_rewards(responses)         # reward model
        advantages = compute_gae(rewards, values)    # GAE
        train_policy(model, batch)                   # gradient step
        manager.update_weights(model)                # NCCL broadcast + lifecycle
        manager.resume_replicas()                    # wake engines

Prerequisites:
    - llm-d-rl rollout controller deployed and running
    - vLLM engines deployed with weight-transfer-config
    - Model accessible via HuggingFace

Usage:
    # On CKS cluster with GPU
    python ppo_with_llmd.py \
        --controller-url http://rollout-controller.llm-d-rl.svc:8090 \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --num-steps 10

    # Dry run (against simulated controller)
    python ppo_with_llmd.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

log = logging.getLogger("ppo-llmd")


def compute_rewards(responses: list[dict]) -> list[float]:
    """Placeholder reward function. Replace with actual reward model."""
    rewards = []
    for resp in responses:
        tokens = resp.get("output_token_ids", [])
        # Simple reward: longer responses score higher (placeholder)
        rewards.append(len(tokens) / 100.0)
    return rewards


def train_step(model: torch.nn.Module, step: int) -> float:
    """Simulated gradient step. Replace with actual PPO update."""
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.001
    return 0.01 * (1.0 / (1.0 + step * 0.1))  # Simulated loss


def main():
    parser = argparse.ArgumentParser(description="PPO with llm-d-rl rollout")
    parser.add_argument("--controller-url",
                        default="http://rollout-controller.llm-d-rl.svc.cluster.local:8090")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without controller (simulated)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device if not args.dry_run else "cpu")

    # --- Load model ---
    log.info("Loading model: %s", args.model_name)
    load_start = time.perf_counter()

    if args.dry_run:
        # Small model for dry run
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_config(config)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
        ).to(device)

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    load_time = time.perf_counter() - load_start
    log.info("Model loaded: %d params (%.1f GB) in %.1fs",
             num_params, num_params * 2 / 1e9, load_time)

    # --- Initialize llm-d-rl manager ---
    from llmd_verl.config import LlmdRolloutConfig
    from llmd_verl.manager import LlmdCheckpointEngineManager

    config = LlmdRolloutConfig(controller_url=args.controller_url)

    if args.dry_run:
        log.info("DRY RUN: skipping controller connection and NCCL setup")
        manager = None
    else:
        manager = LlmdCheckpointEngineManager(config)
        status = manager.init(timeout=600)
        log.info("llm-d-rl ready: %d engines, phase=%s",
                 status.get("total_engines", 0), status.get("phase", "unknown"))

    # --- Training loop ---
    log.info("")
    log.info("Starting PPO training: %d steps, batch_size=%d", args.num_steps, args.batch_size)
    log.info("=" * 60)

    all_metrics = []

    for step in range(1, args.num_steps + 1):
        step_start = time.perf_counter()
        metrics = {"step": step}

        # Phase 1: Generate rollouts
        log.info("[Step %d] Generating rollouts (batch_size=%d)...", step, args.batch_size)
        gen_start = time.perf_counter()

        if args.dry_run:
            responses = [{"output_token_ids": list(range(args.max_tokens))}
                         for _ in range(args.batch_size)]
        else:
            responses = []
            for i in range(args.batch_size):
                prompt_ids = list(range(100 + i * 20, 120 + i * 20))
                resp = manager.client.generate(prompt_ids, max_tokens=args.max_tokens)
                responses.append(resp)

        metrics["generate_s"] = time.perf_counter() - gen_start
        log.info("  Generate: %.3fs (%d responses)", metrics["generate_s"], len(responses))

        # Phase 2: Sleep engines (free GPU memory for training)
        sleep_start = time.perf_counter()
        if manager:
            manager.sleep_replicas()
        metrics["sleep_s"] = time.perf_counter() - sleep_start

        # Phase 3: Compute rewards
        rewards = compute_rewards(responses)
        avg_reward = sum(rewards) / len(rewards)
        metrics["avg_reward"] = avg_reward

        # Phase 4: Train (simulated PPO update)
        train_start = time.perf_counter()
        loss = train_step(model, step)
        metrics["train_s"] = time.perf_counter() - train_start
        metrics["loss"] = loss

        # Phase 5: Sync weights to engines
        sync_start = time.perf_counter()
        if manager:
            version = manager.update_weights(model, device)
            metrics["weight_version"] = version
        else:
            metrics["weight_version"] = step
        metrics["sync_s"] = time.perf_counter() - sync_start

        # Phase 6: Resume engines
        resume_start = time.perf_counter()
        if manager:
            manager.resume_replicas()
        metrics["resume_s"] = time.perf_counter() - resume_start

        metrics["step_total_s"] = time.perf_counter() - step_start

        log.info("  Sleep: %.3fs | Train: %.3fs (loss=%.4f) | Sync: %.3fs | Resume: %.3fs",
                 metrics["sleep_s"], metrics["train_s"], loss,
                 metrics["sync_s"], metrics["resume_s"])
        log.info("  Step total: %.3fs | Reward: %.3f | Version: %d",
                 metrics["step_total_s"], avg_reward, metrics["weight_version"])
        log.info("")

        all_metrics.append(metrics)

    # --- Summary ---
    log.info("=" * 60)
    log.info("Training complete: %d steps", args.num_steps)

    if all_metrics:
        avg_step = sum(m["step_total_s"] for m in all_metrics) / len(all_metrics)
        avg_sync = sum(m["sync_s"] for m in all_metrics) / len(all_metrics)
        log.info("  Avg step time: %.3fs", avg_step)
        log.info("  Avg sync time: %.3fs", avg_sync)
        log.info("  Final weight version: %d", all_metrics[-1]["weight_version"])

    # Print structured results
    print("===BENCH_RESULTS_START===")
    print(json.dumps({"system": "llmd-verl", "metrics": all_metrics}, indent=2))
    print("===BENCH_RESULTS_END===")

    if manager:
        manager.finalize()


if __name__ == "__main__":
    main()
