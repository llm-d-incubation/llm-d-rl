"""GRPO training loop using llm-d rollout controller.

This script demonstrates how a GRPO (Group Relative Policy Optimization)
training loop interacts with llm-d's rollout controller for inference
engine management. It shows the full colocated deployment lifecycle where
inference engines and the trainer share the same GPUs.

The example maps directly from Slime's training loop (Ray actors + SGLang)
to llm-d's HTTP API. Each API call includes a comment showing the
equivalent Slime operation for comparison.

Slime's loop (train.py):
    for rollout_id in range(num_rollouts):
        rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))
        ray.get(rollout_manager.offload.remote())
        ray.get(actor_model.async_train(rollout_id, rollout_data))
        actor_model.update_weights()
        ray.get(rollout_manager.onload.remote())

llm-d equivalent (this script):
    for step in range(num_steps):
        responses = client.generate_batch(requests)
        client.sleep(level=2)
        train(responses)                         # simulated
        client.wake_up(tags=["weights"])
        client.update_weights(target_version)
        client.wake_up(tags=["kv_cache"])

Usage:
    # Dry run (no controller needed, logs all HTTP calls)
    python grpo_training_loop.py --dry-run

    # Against a running rollout controller + llm-d-inference-sim
    #   Terminal 1: llm-d-inference-sim --port 8000
    #   Terminal 2: go run ./cmd/rollout-controller --engines http://localhost:8000 --simulate-lifecycle
    #   Terminal 3: python grpo_training_loop.py --controller-url http://localhost:8090

    # With custom training parameters
    python grpo_training_loop.py --dry-run --num-steps 20 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

log = logging.getLogger("grpo")


# ---------------------------------------------------------------------------
# API types (mirrors api/v1alpha1/types.go)
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    """Generation sampling parameters. Mirrors api/v1alpha1.SamplingParams."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: int = 512
    n: int = 1
    stop: list[str] = field(default_factory=list)


@dataclass
class GenerateRequest:
    """Request to generate token sequences. Mirrors api/v1alpha1.GenerateRequest."""

    prompt_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    session_id: str = ""
    return_logprobs: bool = False
    weight_version: int = 0


@dataclass
class GenerateResponse:
    """Response from a generation request. Mirrors api/v1alpha1.GenerateResponse."""

    output_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    weight_version: int = 0
    engine_id: str = ""
    finish_reason: str = ""


@dataclass
class WeightTransferInit:
    """Configures the weight transfer data plane. Mirrors api/v1alpha1.WeightTransferInit."""

    backend: str = "nccl"
    master_address: str = ""
    master_port: int = 0
    trainer_world_size: int = 1
    packed: bool = False
    is_checkpoint_format: bool = False


@dataclass
class WeightUpdateRequest:
    """Triggers a weight update across the engine pool. Mirrors api/v1alpha1.WeightUpdateRequest."""

    target_version: int = 0
    pause_mode: str = "keep"
    reset_kv_cache: bool = True


@dataclass
class PoolStatus:
    """Current state of the engine pool. Mirrors api/v1alpha1.PoolStatus."""

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
    """Error from the rollout controller API."""

    def __init__(self, method: str, path: str, status_code: int, body: str):
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body
        super().__init__(f"{method} {path} returned {status_code}: {body}")


class RolloutControllerClient:
    """HTTP client for the llm-d rollout controller.

    Wraps all endpoints from pkg/rollout/server.go Handler().
    This is a prototype for the future llm-d-rollout-client Python SDK
    (see roadmap Phase 2: framework adapters).
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
        """POST /v1/generate — route through EPP for KV-cache-aware dispatch.

        Slime equivalent:
            POST http://{sglang_router_ip}:{sglang_router_port}/generate
            (slime/rollout/sglang_rollout.py generate())

        llm-d difference: requests go through the inference scheduler (EPP)
        which provides KV-cache-aware routing, prefix-cache affinity, and
        predicted latency scoring. Slime sends directly to SGLang's router
        which only does least-inflight dispatch.
        """
        data = self._post("/v1/generate", _serialize(req))
        return GenerateResponse(**data) if data else GenerateResponse()

    def generate_batch(
        self, requests: list[GenerateRequest],
    ) -> list[GenerateResponse]:
        """Send multiple generation requests concurrently.

        Uses ThreadPoolExecutor since the Go server does not yet have a
        batch endpoint. Each request is dispatched through EPP independently.

        Future: POST /v1/generate/batch (see northstar doc BatchGenerateRequest)
        """
        responses: list[GenerateResponse] = []

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
        responses = [resp for _, resp in indexed]
        return responses

    # --- Weight management ---

    def init_weight_transfer(self, init: WeightTransferInit) -> None:
        """POST /v1/weights/init — initialize NCCL/NIXL transfer group.

        Slime equivalent:
            NCCL group setup inside SGLangEngine.init() and
            init_weights_update_group() in update_weight_from_distributed.py

        llm-d difference: explicit API call to set up the data plane.
        The coordinator (pkg/weightsync/coordinator.go InitTransfer) calls
        InitWeightTransfer on each engine, which maps to vLLM's
        /init_weight_transfer_engine HTTP endpoint.
        """
        self._post("/v1/weights/init", _serialize(init))

    def update_weights(self, req: WeightUpdateRequest) -> dict[str, Any]:
        """POST /v1/weights/update — orchestrate pause -> transfer -> resume.

        Slime equivalent (5 separate operations):
            1. pause_generation() on all engines
            2. flush_cache() on all engines
            3. NCCL broadcast weights from trainer rank 0
            4. continue_generation() on all engines
            5. Increment weight_version
            (slime/backends/megatron_utils/update_weight/*.py)

        llm-d difference: single API call. The coordinator
        (pkg/weightsync/coordinator.go UpdateWeights) handles the full
        lifecycle internally, including error recovery (auto-resume on
        failure).
        """
        return self._post("/v1/weights/update", _serialize(req))

    def get_weight_version(self) -> int:
        """GET /v1/weights/version — get current pool weight version."""
        data = self._get("/v1/weights/version")
        return data.get("weight_version", 0) if data else 0

    # --- Engine lifecycle ---

    def sleep(self, level: int = 2) -> None:
        """POST /v1/engines/sleep — put all engines to sleep.

        Slime equivalent:
            ray.get([engine.release_memory_occupation.remote()
                     for engine in engines])
            (slime/ray/rollout.py RolloutManager.offload())

        Sleep levels (from api/v1alpha1/types.go):
            0: pause scheduling only (no GPU memory changes)
            1: offload weights to CPU, discard KV cache
            2: discard all GPU memory (weights + KV cache)
               Preferred for colocated RL (vLLM >= 0.8.5)
        """
        self._post("/v1/engines/sleep", {"level": level})

    def wake_up(self, tags: list[str]) -> None:
        """POST /v1/engines/wake — restore specified resources.

        Slime equivalent:
            ray.get([engine.resume_memory_occupation.remote(tags=[...])
                     for engine in engines])
            (slime/ray/rollout.py RolloutManager.onload_weights/onload_kv)

        Tags: "weights", "kv_cache"
        Can be called in stages: wake(["weights"]) then wake(["kv_cache"])
        to separate weight restoration from KV cache allocation.
        """
        self._post("/v1/engines/wake", {"tags": tags})

    # --- Pool status ---

    def get_pool_status(self) -> PoolStatus:
        """GET /v1/pool/status — get current pool status."""
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
        """GET /v1/health — check if the rollout controller is healthy."""
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
        """Return a plausible mock response for dry-run mode."""
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
    """Convert a dataclass to a JSON-serializable dict, dropping None values."""
    d = asdict(obj)
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Simulated training components
# ---------------------------------------------------------------------------

def fake_tokenize(text: str) -> list[int]:
    """Simulate tokenization by mapping characters to integer IDs.

    In production, use the model's actual tokenizer (e.g., from HuggingFace
    transformers). This is just a placeholder so the example has no ML
    framework dependencies.
    """
    return [ord(c) for c in text]


def fake_detokenize(token_ids: list[int]) -> str:
    """Inverse of fake_tokenize."""
    return "".join(chr(t) for t in token_ids if 32 <= t < 127)


def simulate_grpo_training_step(
    prompts: list[str],
    responses: list[GenerateResponse],
    step: int,
) -> dict:
    """Simulate a GRPO training step.

    In a real system, this would:
    1. Compute rewards using a reward function (e.g., math correctness)
    2. Group responses by prompt (GRPO groups N responses per prompt)
    3. Compute advantages within each group (group-relative normalization)
    4. Compute policy gradient loss with clipped ratio and KL penalty
    5. Backward pass and optimizer step

    Slime: ray.get(actor_model.async_train(rollout_id, rollout_data))
    veRL:  self._update_actor(batch)

    This is OUT OF SCOPE for llm-d -- the rollout controller manages
    inference; training stays in the RL framework.
    """
    time.sleep(2)
    fake_loss = 1.0 / (1 + step * 0.1)
    return {
        "loss": round(fake_loss, 4),
        "num_prompts": len(prompts),
        "num_responses": len(responses),
    }


# ---------------------------------------------------------------------------
# GRPO training loop
# ---------------------------------------------------------------------------

# Example math prompts (in production, these come from a dataset).
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


def run_grpo_training_loop(
    controller_url: str,
    num_steps: int = 10,
    batch_size: int = 4,
    responses_per_prompt: int = 4,
    sleep_level: int = 2,
    weight_sync_backend: str = "nccl",
    trainer_address: str = "localhost",
    trainer_port: int = 29500,
    dry_run: bool = False,
) -> None:
    """Run a GRPO training loop using llm-d's rollout controller.

    This demonstrates the full colocated training lifecycle where inference
    engines and the trainer share the same GPUs.

    Lifecycle per step:

        [Engines SERVING — full GPU memory for inference]
        1. Generate rollouts (multiple responses per prompt)
        2. Sleep engines (free GPU memory for training)

        [Engines SLEEPING — GPU memory available for training]
        3. Compute rewards (simulated)
        4. GRPO training step (simulated)

        [Weight update phase]
        5. Wake engines (restore weight memory only)
        6. Update weights (pause -> NCCL transfer -> resume)
        7. Wake engines (restore KV cache)

        [Engines SERVING — ready for next rollout]
    """
    client = RolloutControllerClient(
        controller_url, dry_run=dry_run,
    )

    # === INITIALIZATION ===

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

    # Initialize weight transfer data plane (once, before the loop).
    #
    # Slime: this happens inside SGLangEngine.init() where NCCL groups
    #   are created between training ranks and engine workers.
    # llm-d: explicit API call. The coordinator calls each engine's
    #   /init_weight_transfer_engine endpoint (vLLM dev-mode API).
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
        # Slime: rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))
        #   which calls POST http://{sglang_router}/generate for each prompt
        #
        # llm-d: POST /v1/generate for each prompt (concurrent via
        #   ThreadPoolExecutor). The controller routes through EPP for
        #   KV-cache-aware dispatch instead of SGLang's router.
        # ------------------------------------------------------

        log.info("[1/7] Generating rollouts (%d prompts x %d responses)...",
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

        for resp in responses:
            if resp.weight_version != weight_version and resp.weight_version != 0:
                log.warning(
                    "  Weight version mismatch: expected %d, got %d from %s",
                    weight_version, resp.weight_version, resp.engine_id,
                )

        # ------------------------------------------------------
        # PHASE 2: SLEEP ENGINES (free GPU for training)
        # Pool phase: Serving -> Sleeping
        #
        # Slime: ray.get(rollout_manager.offload.remote())
        #   which calls engine.release_memory_occupation.remote()
        #   on each SGLang engine actor.
        #
        # llm-d: POST /v1/engines/sleep with level=2 (discard all GPU
        #   memory). The coordinator (coordinator.go SleepAll) calls
        #   each engine's /sleep endpoint (vLLM dev-mode API).
        # ------------------------------------------------------

        log.info("[2/7] Sleeping engines (level=%d)...", sleep_level)
        client.sleep(level=sleep_level)
        log.info("  GPU memory freed for training")

        # ------------------------------------------------------
        # PHASE 3: COMPUTE REWARDS (simulated)
        # Out of scope for llm-d — this is RL framework code.
        #
        # Slime: reward computation happens inside training step
        # veRL: extract_reward(batch)
        # OpenRLHF: reward_model.forward_value(sequences)
        # ------------------------------------------------------

        log.info("[3/7] Computing rewards (simulated)...")
        time.sleep(0.5)

        # ------------------------------------------------------
        # PHASE 4: GRPO TRAINING STEP (simulated)
        # Out of scope for llm-d — this is RL framework code.
        # The trainer has full GPU memory because engines are sleeping.
        #
        # Slime: ray.get(actor_model.async_train(rollout_id, rollout_data))
        # veRL: self._update_actor(batch)
        # OpenRLHF: actor_model.train()
        # ------------------------------------------------------

        log.info("[4/7] GRPO training step (simulated)...")
        metrics = simulate_grpo_training_step(step_prompts, responses, step)
        log.info("  loss=%.4f  prompts=%d  responses=%d",
                 metrics["loss"], metrics["num_prompts"], metrics["num_responses"])

        # ------------------------------------------------------
        # PHASE 5: WAKE ENGINES — weights only
        # Pool phase: Sleeping -> (partial wake)
        #
        # Slime: engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        #   (slime/ray/rollout.py onload_weights())
        #
        # llm-d: POST /v1/engines/wake with tags=["weights"].
        #   The coordinator (coordinator.go WakeUpAll) calls each engine's
        #   /wake_up endpoint. We wake weights first so the engine can
        #   receive the new weights via NCCL before allocating KV cache.
        # ------------------------------------------------------

        log.info("[5/7] Waking engines (weights only)...")
        client.wake_up(tags=["weights"])
        log.info("  Weight memory restored")

        # ------------------------------------------------------
        # PHASE 6: UPDATE WEIGHTS
        # Pool phase: Syncing
        #
        # Slime (5 separate operations across all engines):
        #   1. recover_rollout_engines() — restart dead engines
        #   2. pause_generation() on all engines
        #   3. flush_cache() on all engines
        #   4. NCCL broadcast from trainer rank 0 -> all engine workers
        #   5. continue_generation() on all engines
        #   (slime/backends/megatron_utils/update_weight/*.py)
        #
        # llm-d (single API call):
        #   POST /v1/weights/update
        #   The coordinator (coordinator.go UpdateWeights) handles:
        #     1. Pause all engines (PauseMode.KEEP)
        #     2. Signal engines to start receiving (NCCL data plane)
        #     3. Reset prefix caches
        #     4. Resume all engines
        #     5. Update weight version
        #   On failure, automatically resumes engines before returning error.
        # ------------------------------------------------------

        weight_version += 1
        log.info("[6/7] Updating weights (version %d)...", weight_version)

        result = client.update_weights(WeightUpdateRequest(
            target_version=weight_version,
            pause_mode="keep",
            reset_kv_cache=True,
        ))
        log.info("  Weight update: %s", result)

        current_version = client.get_weight_version()
        if current_version != weight_version and not dry_run:
            log.warning(
                "  Version mismatch after update: expected %d, got %d",
                weight_version, current_version,
            )

        # ------------------------------------------------------
        # PHASE 7: WAKE ENGINES — KV cache
        # Pool phase: -> Serving
        #
        # Slime: engine.resume_memory_occupation(
        #            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
        #   (slime/ray/rollout.py onload_kv())
        #
        # llm-d: POST /v1/engines/wake with tags=["kv_cache"].
        #   This allocates KV cache memory and rebuilds CUDA graphs.
        #   After this call, the pool phase returns to Serving.
        # ------------------------------------------------------

        log.info("[7/7] Waking engines (KV cache)...")
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
        description="GRPO training loop using llm-d rollout controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python grpo_training_loop.py --dry-run
  python grpo_training_loop.py --controller-url http://localhost:8090
  python grpo_training_loop.py --dry-run --num-steps 20 --batch-size 8
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
        "--responses-per-prompt", type=int, default=4,
        help="number of responses per prompt / GRPO group size (default: 4)",
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

    run_grpo_training_loop(
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
