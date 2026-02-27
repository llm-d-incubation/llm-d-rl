"""PPO training loop using llm-d rollout controller.

This script demonstrates how veRL's PPO training loop (RayPPOTrainer)
would work using llm-d's rollout controller instead of Ray actors,
CUDA IPC weight transfer, and direct vLLM/SGLang server management.

veRL's loop (verl/trainer/ppo/ray_trainer.py RayPPOTrainer.fit()):
    for epoch in range(total_epochs):
        for batch in train_dataloader:
            gen_output = async_rollout_manager.generate_sequences(batch)
            checkpoint_manager.sleep_replicas()
            reward = reward_loop_manager.compute_reward(gen_output)
            old_log_prob = actor.compute_log_prob(batch)
            advantages = compute_gae(rewards, values)
            actor_output = actor_rollout_wg.update_actor(batch)
            critic_output = critic_wg.train_mini_batch(batch)
            checkpoint_manager.update_weights()

llm-d equivalent (this script):
    for step in range(num_steps):
        responses = client.generate_batch(requests)
        client.sleep(level=2)
        rewards = compute_rewards(responses)
        advantages = compute_gae(rewards, values)
        train_actor(batch)
        train_critic(batch)
        client.wake_up(tags=["weights"])
        client.update_weights(target_version)
        client.wake_up(tags=["kv_cache"])

Usage:
    # Dry run (no controller needed)
    python ppo_training_loop.py --dry-run

    # Against a running rollout controller
    #   Terminal 1: mock vLLM or llm-d-inference-sim on port 8000
    #   Terminal 2: go run ./cmd/rollout-controller --engines http://localhost:8000 --simulate-lifecycle
    #   Terminal 3: python ppo_training_loop.py --controller-url http://localhost:8090
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

log = logging.getLogger("ppo")


# ---------------------------------------------------------------------------
# API types (mirrors api/v1alpha1/types.go)
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: int = 512
    n: int = 1
    stop: list[str] = field(default_factory=list)


@dataclass
class GenerateRequest:
    prompt_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    session_id: str = ""
    return_logprobs: bool = False
    weight_version: int = 0


@dataclass
class GenerateResponse:
    output_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    weight_version: int = 0
    engine_id: str = ""
    finish_reason: str = ""


@dataclass
class WeightTransferInit:
    backend: str = "nccl"
    master_address: str = ""
    master_port: int = 0
    trainer_world_size: int = 1
    packed: bool = False
    is_checkpoint_format: bool = False


@dataclass
class WeightUpdateRequest:
    target_version: int = 0
    pause_mode: str = "keep"
    reset_kv_cache: bool = True


@dataclass
class PoolStatus:
    phase: str = ""
    total_engines: int = 0
    ready_engines: int = 0
    weight_version: int = 0
    last_weight_sync: str = ""
    engines: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rollout controller client
# ---------------------------------------------------------------------------

class RolloutControllerError(Exception):
    def __init__(self, method: str, path: str, status_code: int, body: str):
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body
        super().__init__(f"{method} {path} returned {status_code}: {body}")


class RolloutControllerClient:
    """HTTP client for the llm-d rollout controller.

    Wraps all endpoints from pkg/rollout/server.go Handler().
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        dry_run: bool = False,
        max_workers: int = 16,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dry_run = dry_run
        self.max_workers = max_workers
        self._dry_run_weight_version = 0

    # --- Generation ---

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """POST /v1/generate — generate token sequences.

        veRL equivalent:
            async_rollout_manager.generate_sequences(gen_batch_output)
            -> AgentLoopManager distributes to AgentLoopWorker actors
            -> each worker calls vLLM/SGLang server via Ray actor handle
            (verl/experimental/agent_loop/agent_loop.py)

        llm-d difference: HTTP requests routed through EPP for KV-cache-aware
        dispatch. veRL uses Ray actor handles with load-balanced chunking
        across AgentLoopWorkers.
        """
        data = self._post("/v1/generate", _serialize(req))
        return GenerateResponse(**data) if data else GenerateResponse()

    def generate_batch(
        self, requests: list[GenerateRequest],
    ) -> list[GenerateResponse]:
        """Send multiple generation requests concurrently.

        veRL equivalent:
            AgentLoopManager.generate_sequences() chunks prompts across
            workers: ray.get([worker.generate_sequences.remote(chunk)
                              for worker, chunk in zip(workers, chunks)])
            (verl/experimental/agent_loop/agent_loop.py)

        llm-d difference: concurrent HTTP requests via ThreadPoolExecutor
        instead of Ray remote calls with chunk-based load balancing.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.generate, req): i
                for i, req in enumerate(requests)
            }
            indexed: list[tuple[int, GenerateResponse]] = []
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    indexed.append((idx, future.result()))
                except Exception as exc:
                    log.warning("  Request %d failed: %s", idx, exc)
                    indexed.append((idx, GenerateResponse(finish_reason="error")))

        indexed.sort(key=lambda x: x[0])
        return [resp for _, resp in indexed]

    # --- Weight management ---

    def init_weight_transfer(self, init: WeightTransferInit) -> None:
        """POST /v1/weights/init — initialize weight transfer data plane.

        veRL equivalent:
            CheckpointEngineManager.__init__() sets up the backend
            -> NCCLCheckpointEngine.init_process_group() for NCCL
            -> builds Ray collective communication topology
            (verl/checkpoint_engine/nccl_checkpoint_engine.py)

        llm-d difference: explicit HTTP call to initialize the data plane
        on each engine. veRL sets up NCCL groups during worker initialization
        via Ray collective API.
        """
        self._post("/v1/weights/init", _serialize(init))

    def update_weights(self, req: WeightUpdateRequest) -> dict[str, Any]:
        """POST /v1/weights/update — orchestrate weight sync across pool.

        veRL equivalent (colocated/hybrid mode):
            ActorRolloutRefWorker.update_weights()
            1. per_tensor_param = actor.engine.get_per_tensor_param()
            2. rollout.update_weights(per_tensor_param)  # CUDA IPC + ZMQ
            3. actor.engine.to("cpu")  # offload after sync
            (verl/workers/engine_workers.py lines 622-694)

        veRL equivalent (disaggregated mode):
            CheckpointEngineManager.update_weights()
            1. abort_all_requests() on all replicas
            2. build_process_group() for NCCL topology
            3. ray.get(trainer.update_weights() + rollout.update_weights())
            4. finalize() communication
            5. resume_all_requests()
            (verl/checkpoint_engine/base.py lines 298-424)

        llm-d difference: single HTTP call. The coordinator handles the
        full pause-transfer-resume lifecycle. Weight tensors flow directly
        via NCCL/NIXL data plane — the controller only orchestrates.
        """
        return self._post("/v1/weights/update", _serialize(req))

    def get_weight_version(self) -> int:
        """GET /v1/weights/version"""
        data = self._get("/v1/weights/version")
        return data.get("weight_version", 0) if data else 0

    # --- Engine lifecycle ---

    def sleep(self, level: int = 2) -> None:
        """POST /v1/engines/sleep — free GPU memory for training.

        veRL equivalent:
            CheckpointEngineManager.sleep_replicas()
            -> asyncio.gather(*[r.sleep() for r in self.replicas])
            -> ServerAdapter.release()
            -> vLLM: engine.sleep(level=VLLM_SLEEP_LEVEL)
            (verl/checkpoint_engine/base.py, verl/workers/rollout/vllm_rollout/vllm_rollout.py)

        llm-d difference: HTTP call to controller which fans out to all
        engines. veRL uses Ray remote calls to each RolloutReplica actor.
        """
        self._post("/v1/engines/sleep", {"level": level})

    def wake_up(self, tags: list[str]) -> None:
        """POST /v1/engines/wake — restore specified GPU resources.

        veRL equivalent:
            ActorRolloutRefWorker.update_weights() calls:
            -> rollout.resume(tags=["weights"])  # restore weight memory
            -> [do weight transfer]
            -> rollout.resume(tags=["kv_cache"]) # restore KV cache
            (verl/workers/engine_workers.py lines 640, 672)

        The two-phase wake pattern is the same in both veRL and llm-d:
        wake weights first, transfer new weights, then wake KV cache.
        """
        self._post("/v1/engines/wake", {"tags": tags})

    def pause(self, mode: str = "keep") -> None:
        """POST /v1/engines/pause — pause generation on all engines.

        veRL equivalent:
            CheckpointEngineManager.update_weights() in disaggregated mode:
            -> abort_all_requests() on all replicas
            (verl/checkpoint_engine/base.py)
        """
        self._post("/v1/engines/pause", {"mode": mode})

    def resume(self) -> None:
        """POST /v1/engines/resume — resume generation on all engines.

        veRL equivalent:
            CheckpointEngineManager.update_weights() in disaggregated mode:
            -> resume_all_requests() on all replicas
            (verl/checkpoint_engine/base.py)
        """
        self._post("/v1/engines/resume", {})

    # --- Pool status ---

    def get_pool_status(self) -> PoolStatus:
        """GET /v1/pool/status"""
        data = self._get("/v1/pool/status")
        if not data:
            return PoolStatus()
        return PoolStatus(
            phase=data.get("phase", ""),
            total_engines=data.get("total_engines", 0),
            ready_engines=data.get("ready_engines", 0),
            weight_version=data.get("weight_version", 0),
            last_weight_sync=data.get("last_weight_sync", ""),
            engines=data.get("engines", []),
        )

    def health_check(self) -> bool:
        """GET /v1/health"""
        try:
            data = self._get("/v1/health")
            return data.get("status") == "ok" if data else False
        except Exception:
            return False

    # --- HTTP helpers ---

    def _post(self, path: str, body: dict | None = None) -> dict:
        url = self.base_url + path
        if self.dry_run:
            log.info("  [dry-run] POST %s %s", path, json.dumps(body or {}))
            return self._dry_run_response(path, body)

        data = json.dumps(body or {}).encode()
        req = Request(url, data=data, method="POST",
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
                return json.loads(content) if content else {}
        except HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RolloutControllerError("POST", path, exc.code, body_text) from exc

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        if self.dry_run:
            log.info("  [dry-run] GET %s", path)
            return self._dry_run_response(path)

        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
                return json.loads(content) if content else {}
        except HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RolloutControllerError("GET", path, exc.code, body_text) from exc

    def _dry_run_response(self, path: str, body: dict | None = None) -> dict:
        if path == "/v1/health":
            return {"status": "ok"}
        if path == "/v1/pool/status":
            return {
                "phase": "Serving",
                "total_engines": 4,
                "ready_engines": 4,
                "weight_version": self._dry_run_weight_version,
            }
        if path == "/v1/weights/version":
            return {"weight_version": self._dry_run_weight_version}
        if path == "/v1/weights/update":
            if body and "target_version" in body:
                self._dry_run_weight_version = body["target_version"]
            return {"status": "updated", "weight_version": self._dry_run_weight_version}
        if path == "/v1/generate":
            return {
                "output_token_ids": [42, 43, 44, 45],
                "logprobs": [-0.5, -0.3, -0.8, -0.1],
                "weight_version": self._dry_run_weight_version,
                "engine_id": "engine-0",
                "finish_reason": "stop",
            }
        return {"status": "ok"}


def _serialize(obj: Any) -> dict:
    d = asdict(obj)
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Simulated training components
# ---------------------------------------------------------------------------

def fake_tokenize(text: str) -> list[int]:
    """Simulate tokenization (placeholder for HuggingFace tokenizer)."""
    return [ord(c) for c in text]


def fake_detokenize(token_ids: list[int]) -> str:
    return "".join(chr(t) for t in token_ids if 32 <= t < 127)


def simulate_reward_model(
    prompts: list[str],
    responses: list[GenerateResponse],
) -> list[float]:
    """Simulate reward model scoring.

    veRL equivalent:
        reward_loop_manager.compute_reward(gen_output)
        -> RewardLoopWorker processes batches
        -> reward model forward pass on responses
        (verl/trainer/ppo/ray_trainer.py _compute_reward_colocate)

    In production, this would call a reward model (e.g., a classifier or
    LLM-as-judge) to score each response. For math tasks, it might check
    whether the answer is correct.
    """
    import random
    return [random.uniform(0.0, 1.0) for _ in responses]


def simulate_compute_values(
    responses: list[GenerateResponse],
) -> list[float]:
    """Simulate critic value estimation.

    veRL equivalent:
        critic_wg.compute_values(batch)
        -> TrainingWorker forward pass through critic network
        (verl/trainer/ppo/ray_trainer.py)

    PPO requires value estimates from a critic network to compute
    advantages via GAE. GRPO does not use a critic (group-relative
    advantages instead).
    """
    return [0.5 + 0.1 * i for i in range(len(responses))]


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """Compute generalized advantage estimation (GAE).

    veRL equivalent:
        compute_gae_advantage() called on the driver (not distributed)
        -> uses rewards, values, and next_values to compute advantages
        (verl/trainer/ppo/ray_trainer.py)

    GAE is the standard advantage estimator for PPO. It balances bias
    and variance using the lambda parameter.
    """
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def simulate_actor_training_step(
    step: int,
    advantages: list[float],
) -> dict:
    """Simulate actor (policy) training step.

    veRL equivalent:
        actor_rollout_wg.update_actor(training_batch)
        -> ActorRolloutRefWorker.update_actor()
        -> FSDP/Megatron distributed training step
        -> computes clipped PPO loss with KL penalty
        (verl/workers/engine_workers.py, verl/trainer/ppo/actor/base.py)

    Out of scope for llm-d — training stays in the RL framework.
    """
    time.sleep(1.5)
    policy_loss = 1.0 / (1 + step * 0.1)
    kl_div = max(0.01, 0.05 - step * 0.005)
    return {
        "policy_loss": round(policy_loss, 4),
        "kl_divergence": round(kl_div, 4),
        "mean_advantage": round(sum(advantages) / max(len(advantages), 1), 4),
    }


def simulate_critic_training_step(
    step: int,
    values: list[float],
    rewards: list[float],
) -> dict:
    """Simulate critic (value) training step.

    veRL equivalent:
        critic_wg.train_mini_batch(training_batch)
        -> TrainingWorker trains critic network
        -> MSE loss between predicted values and returns
        (verl/trainer/ppo/ray_trainer.py)

    PPO trains a separate critic network to estimate state values.
    GRPO does not have a critic — this is a key architectural difference.
    """
    time.sleep(0.5)
    value_loss = sum((r - v) ** 2 for r, v in zip(rewards, values)) / max(len(values), 1)
    return {
        "value_loss": round(value_loss, 4),
    }


def simulate_ref_log_probs(
    responses: list[GenerateResponse],
) -> list[float]:
    """Simulate reference policy log probabilities.

    veRL equivalent:
        ref_wg.compute_ref_log_prob(batch)
        -> TrainingWorker forward pass through frozen reference model
        (verl/trainer/ppo/ray_trainer.py)

    PPO uses a frozen reference policy to compute KL divergence penalty.
    This prevents the policy from diverging too far from the initial model.
    """
    return [-0.5 - 0.1 * i for i in range(len(responses))]


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------

PROMPTS = [
    "What is 2 + 3?",
    "Solve: x^2 = 16",
    "What is the derivative of x^3?",
    "Simplify: (a+b)^2",
    "What is 7 * 8?",
    "Find the integral of 2x dx",
    "What is sqrt(144)?",
    "Solve: 3x + 5 = 20",
]


def run_ppo_training_loop(
    controller_url: str,
    num_steps: int = 10,
    batch_size: int = 4,
    responses_per_prompt: int = 1,
    sleep_level: int = 2,
    weight_sync_backend: str = "nccl",
    trainer_address: str = "localhost",
    trainer_port: int = 29500,
    dry_run: bool = False,
) -> None:
    """Run a PPO training loop using llm-d's rollout controller.

    This demonstrates the full colocated training lifecycle matching
    veRL's RayPPOTrainer.fit() method. The key difference from the GRPO
    example is that PPO uses a separate critic network and GAE for
    advantage estimation, while GRPO uses group-relative advantages.

    veRL lifecycle per step (from verl/trainer/ppo/ray_trainer.py):

        [Engines SERVING]
        1. generate_sequences()          -> POST /v1/generate
        2. sleep_replicas()              -> POST /v1/engines/sleep

        [Engines SLEEPING — GPU freed for training]
        3. compute_reward()              (simulated)
        4. compute_ref_log_prob()        (simulated)
        5. compute_values()              (simulated)
        6. compute_gae()                 (simulated)
        7. update_actor()                (simulated)
        8. train_critic()                (simulated)

        [Weight update]
        9.  resume(tags=["weights"])      -> POST /v1/engines/wake
        10. update_weights()             -> POST /v1/weights/update
        11. resume(tags=["kv_cache"])    -> POST /v1/engines/wake

        [Engines SERVING — ready for next rollout]
    """
    client = RolloutControllerClient(
        controller_url, dry_run=dry_run,
    )

    # === INITIALIZATION ===
    #
    # veRL equivalent: RayPPOTrainer.__init__() and init_workers()
    # -> creates ResourcePoolManager, RayWorkerGroups for actor/critic/ref
    # -> initializes RolloutReplicas (hybrid, colocated, or standalone)
    # -> launches vLLM/SGLang servers via RolloutReplica.launch_servers()

    log.info("Connecting to rollout controller at %s", controller_url)

    if not client.health_check():
        log.error("Rollout controller is not healthy")
        sys.exit(1)
    log.info("Health check passed")

    status = client.get_pool_status()
    log.info(
        "Pool: phase=%s, engines=%d/%d ready, weight_version=%d",
        status.phase, status.ready_engines, status.total_engines,
        status.weight_version,
    )

    if status.ready_engines == 0 and not dry_run:
        log.error("No ready engines in the pool")
        sys.exit(1)

    # Initialize weight transfer data plane.
    #
    # veRL equivalent: CheckpointEngineManager.__init__() configures the
    # backend (naive, nccl, nixl). For NCCL, it creates Ray collective
    # communication groups between trainer and rollout workers.
    # (verl/checkpoint_engine/base.py, verl/checkpoint_engine/nccl_checkpoint_engine.py)
    log.info("Initializing weight transfer (backend=%s)", weight_sync_backend)
    client.init_weight_transfer(WeightTransferInit(
        backend=weight_sync_backend,
        master_address=trainer_address,
        master_port=trainer_port,
        trainer_world_size=1,
        packed=True,
        is_checkpoint_format=True,
    ))

    weight_version = status.weight_version

    # === TRAINING LOOP ===
    #
    # veRL equivalent: RayPPOTrainer.fit() main loop
    # (verl/trainer/ppo/ray_trainer.py lines 1221-1608)

    for step in range(1, num_steps + 1):
        log.info("")
        log.info("=" * 60)
        log.info("Step %d/%d  (weight_version=%d)", step, num_steps, weight_version)
        log.info("=" * 60)
        step_start = time.time()

        # ------------------------------------------------------
        # PHASE 1: GENERATE ROLLOUTS
        # Pool phase: Serving
        #
        # veRL: gen_batch_output = async_rollout_manager.generate_sequences(batch)
        #   -> AgentLoopManager chunks across AgentLoopWorkers
        #   -> each worker dispatches to vLLM/SGLang server
        #   -> results collected via ray.get()
        #   (verl/experimental/agent_loop/agent_loop.py)
        #
        # llm-d: POST /v1/generate for each prompt. The controller
        # picks a ready engine and forwards to /v1/completions.
        # Future: EPP integration for KV-cache-aware routing.
        # ------------------------------------------------------

        log.info("[1/11] Generating rollouts (%d prompts x %d responses)...",
                 batch_size, responses_per_prompt)

        step_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]

        gen_requests = [
            GenerateRequest(
                prompt_token_ids=fake_tokenize(prompt),
                sampling_params=SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=256,
                    n=responses_per_prompt,
                ),
                return_logprobs=True,
                weight_version=weight_version,
            )
            for prompt in step_prompts
        ]

        responses = client.generate_batch(gen_requests)

        gen_time = time.time() - step_start
        log.info("  Generated %d responses in %.2fs", len(responses), gen_time)

        # ------------------------------------------------------
        # PHASE 2: SLEEP ENGINES (free GPU for training)
        # Pool phase: Serving -> Sleeping
        #
        # veRL: checkpoint_manager.sleep_replicas()
        #   -> asyncio.gather(*[r.sleep() for r in replicas])
        #   -> ServerAdapter.release()
        #   -> vLLM engine.sleep(level=VLLM_SLEEP_LEVEL)
        #   (verl/checkpoint_engine/base.py)
        #
        # llm-d: POST /v1/engines/sleep. The coordinator fans out
        # to all engines via the EngineClient interface.
        # ------------------------------------------------------

        log.info("[2/11] Sleeping engines (level=%d)...", sleep_level)
        client.sleep(level=sleep_level)
        log.info("  GPU memory freed for training")

        # ------------------------------------------------------
        # PHASE 3: COMPUTE REWARDS
        #
        # veRL: reward_tensor = reward_loop_manager.compute_reward(batch)
        #   -> RewardLoopWorker processes batches asynchronously
        #   -> supports colocated RM or standalone RM server
        #   (verl/trainer/ppo/ray_trainer.py _compute_reward_colocate)
        #
        # Out of scope for llm-d.
        # ------------------------------------------------------

        log.info("[3/11] Computing rewards (simulated)...")
        rewards = simulate_reward_model(step_prompts, responses)
        log.info("  Mean reward: %.4f", sum(rewards) / len(rewards))

        # ------------------------------------------------------
        # PHASE 4: COMPUTE REFERENCE LOG PROBS
        #
        # veRL: ref_log_prob = ref_wg.compute_ref_log_prob(batch)
        #   -> TrainingWorker forward pass through frozen ref model
        #   -> used for KL divergence penalty in PPO loss
        #   (verl/trainer/ppo/ray_trainer.py)
        #
        # Out of scope for llm-d. Note: GRPO does not use a ref model.
        # ------------------------------------------------------

        log.info("[4/11] Computing reference log probs (simulated)...")
        ref_log_probs = simulate_ref_log_probs(responses)

        # ------------------------------------------------------
        # PHASE 5: COMPUTE VALUES (critic forward pass)
        #
        # veRL: values = critic_wg.compute_values(batch)
        #   -> TrainingWorker runs critic network forward pass
        #   (verl/trainer/ppo/ray_trainer.py)
        #
        # Out of scope for llm-d. GRPO does not use a critic.
        # ------------------------------------------------------

        log.info("[5/11] Computing values (simulated)...")
        values = simulate_compute_values(responses)

        # ------------------------------------------------------
        # PHASE 6: COMPUTE ADVANTAGES (GAE)
        #
        # veRL: advantages = compute_gae_advantage(rewards, values)
        #   -> runs on driver process (not distributed)
        #   -> GAE(gamma=0.99, lambda=0.95) is the standard choice
        #   (verl/trainer/ppo/ray_trainer.py)
        #
        # Out of scope for llm-d. GRPO uses group-relative advantages
        # (normalize within each prompt's response group) instead of GAE.
        # ------------------------------------------------------

        log.info("[6/11] Computing advantages (GAE)...")
        advantages = compute_gae(rewards, values)
        log.info("  Mean advantage: %.4f", sum(advantages) / len(advantages))

        # ------------------------------------------------------
        # PHASE 7: ACTOR (POLICY) TRAINING STEP
        #
        # veRL: actor_output = actor_rollout_wg.update_actor(batch)
        #   -> ActorRolloutRefWorker.update_actor()
        #   -> FSDP/Megatron distributed training step
        #   -> clipped PPO loss with KL penalty:
        #      L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) - beta * KL
        #   (verl/workers/engine_workers.py, verl/trainer/ppo/actor/base.py)
        #
        # Out of scope for llm-d.
        # ------------------------------------------------------

        log.info("[7/11] Actor training step (simulated)...")
        actor_metrics = simulate_actor_training_step(step, advantages)
        log.info("  policy_loss=%.4f  kl=%.4f  mean_adv=%.4f",
                 actor_metrics["policy_loss"],
                 actor_metrics["kl_divergence"],
                 actor_metrics["mean_advantage"])

        # ------------------------------------------------------
        # PHASE 8: CRITIC TRAINING STEP
        #
        # veRL: critic_output = critic_wg.train_mini_batch(batch)
        #   -> TrainingWorker trains critic on MSE(V(s), returns)
        #   (verl/trainer/ppo/ray_trainer.py)
        #
        # Out of scope for llm-d. GRPO does not train a critic.
        # ------------------------------------------------------

        log.info("[8/11] Critic training step (simulated)...")
        critic_metrics = simulate_critic_training_step(step, values, rewards)
        log.info("  value_loss=%.4f", critic_metrics["value_loss"])

        # ------------------------------------------------------
        # PHASE 9: WAKE ENGINES — weights only
        # Pool phase: Sleeping -> (partial wake)
        #
        # veRL: rollout.resume(tags=["weights"])
        #   -> ServerAdapter.resume(tags=["weights"])
        #   -> vLLM: engine.wake_up(tags=["weights"])
        #   (verl/workers/engine_workers.py line 640)
        #
        # llm-d: POST /v1/engines/wake with tags=["weights"].
        # The two-phase wake pattern is identical in veRL and llm-d.
        # ------------------------------------------------------

        log.info("[9/11] Waking engines (weights only)...")
        client.wake_up(tags=["weights"])
        log.info("  Weight memory restored")

        # ------------------------------------------------------
        # PHASE 10: UPDATE WEIGHTS
        # Pool phase: Syncing
        #
        # veRL colocated (naive backend):
        #   ActorRolloutRefWorker.update_weights()
        #   1. per_tensor_param = actor.engine.get_per_tensor_param()
        #   2. rollout.update_weights(per_tensor_param)
        #      -> allocate CUDA IPC buffer
        #      -> ZMQ REQ/REP coordination
        #      -> bucket streaming (64MB chunks)
        #      -> server imports via cudaIpcOpenMemHandle
        #   3. actor.engine.to("cpu")
        #   (verl/workers/engine_workers.py lines 622-694)
        #   (verl/workers/rollout/vllm_rollout/vllm_rollout.py update_weights)
        #
        # veRL disaggregated (nccl backend):
        #   CheckpointEngineManager.update_weights()
        #   -> NCCLCheckpointEngine.send_weights()
        #   -> Ray collective broadcast within process group
        #   (verl/checkpoint_engine/nccl_checkpoint_engine.py)
        #
        # llm-d: POST /v1/weights/update. The coordinator handles the
        # full pause-transfer-resume lifecycle internally.
        # ------------------------------------------------------

        weight_version += 1
        log.info("[10/11] Updating weights (version %d)...", weight_version)

        result = client.update_weights(WeightUpdateRequest(
            target_version=weight_version,
            pause_mode="keep",
            reset_kv_cache=True,
        ))
        log.info("  Weight update: %s", result)

        # ------------------------------------------------------
        # PHASE 11: WAKE ENGINES — KV cache
        # Pool phase: -> Serving
        #
        # veRL: rollout.resume(tags=["kv_cache"])
        #   -> ServerAdapter.resume(tags=["kv_cache"])
        #   -> vLLM: engine.wake_up(tags=["kv_cache"])
        #   (verl/workers/engine_workers.py line 672)
        #
        # llm-d: POST /v1/engines/wake with tags=["kv_cache"].
        # After this, the pool returns to Serving phase.
        # ------------------------------------------------------

        log.info("[11/11] Waking engines (KV cache)...")
        client.wake_up(tags=["kv_cache"])

        status = client.get_pool_status()
        log.info(
            "  Pool: phase=%s, engines=%d/%d ready, weight_version=%d",
            status.phase, status.ready_engines, status.total_engines,
            status.weight_version,
        )

        step_time = time.time() - step_start
        log.info("Step %d complete (%.2fs)", step, step_time)

    log.info("")
    log.info("Training complete. Final weight version: %d", weight_version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO training loop using llm-d rollout controller (veRL mapping)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python ppo_training_loop.py --dry-run
  python ppo_training_loop.py --controller-url http://localhost:8090
  python ppo_training_loop.py --dry-run --num-steps 20 --batch-size 8
""",
    )
    parser.add_argument(
        "--controller-url", default="http://localhost:8090",
        help="URL of the llm-d rollout controller (default: http://localhost:8090)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=10,
        help="number of training steps (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="number of prompts per step (default: 4)",
    )
    parser.add_argument(
        "--responses-per-prompt", type=int, default=1,
        help="responses per prompt (PPO typically uses 1, default: 1)",
    )
    parser.add_argument(
        "--sleep-level", type=int, default=2, choices=[0, 1, 2],
        help="sleep level for GPU memory management (default: 2)",
    )
    parser.add_argument(
        "--weight-sync-backend", default="nccl",
        choices=["nccl", "nixl", "checkpoint"],
        help="weight transfer backend (default: nccl)",
    )
    parser.add_argument(
        "--trainer-address", default="localhost",
        help="trainer address for NCCL rendezvous (default: localhost)",
    )
    parser.add_argument(
        "--trainer-port", type=int, default=29500,
        help="trainer port for NCCL rendezvous (default: 29500)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="log HTTP calls without sending them",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_ppo_training_loop(
        controller_url=args.controller_url,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        responses_per_prompt=args.responses_per_prompt,
        sleep_level=args.sleep_level,
        weight_sync_backend=args.weight_sync_backend,
        trainer_address=args.trainer_address,
        trainer_port=args.trainer_port,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
