"""Real NCCL weight trainer for llm-d end-to-end demo.

This trainer loads a model, perturbs its weights (simulating a training step),
and broadcasts the updated weights to vLLM via NCCL — orchestrated by the
llm-d rollout controller.

The trainer is rank 0 in the NCCL group. vLLM engines join as rank 1+.
Weight tensors flow directly via NCCL (data plane). The rollout controller
only orchestrates the lifecycle via HTTP (control plane).

Usage:
    python nccl_weight_trainer.py \
        --controller-url http://rollout-controller:8090 \
        --model-name meta-llama/Llama-3.2-1B \
        --num-steps 5
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import time
import threading
from datetime import timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import torch
from transformers import AutoModelForCausalLM

log = logging.getLogger("nccl-trainer")


# ---------------------------------------------------------------------------
# NCCL setup using vLLM's stateless process group
# ---------------------------------------------------------------------------

def get_pod_ip() -> str:
    """Get this pod's IP address for NCCL rendezvous."""
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


def setup_nccl_group(master_address: str, master_port: int, world_size: int, device: torch.device):
    """Create NCCL communicator as rank 0 (trainer is master).

    Uses vLLM's StatelessProcessGroup which is independent of
    torch.distributed — the trainer's own distributed setup (if any)
    is unaffected.

    Returns (pynccl_communicator, process_group).
    """
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    log.info("Creating StatelessProcessGroup: rank=0, world_size=%d, master=%s:%d",
             world_size, master_address, master_port)

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
    )

    pynccl = PyNcclCommunicator(pg, device=device)
    log.info("NCCL communicator created")

    return pynccl, pg


def broadcast_weights(pynccl, model: torch.nn.Module, device: torch.device) -> None:
    """Broadcast all model parameters via NCCL from rank 0.

    vLLM engines (rank 1+) receive via their /update_weights handler
    which also calls NCCL receive on each parameter.
    """
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for name, param in model.named_parameters():
            tensor = param.data.contiguous()
            pynccl.broadcast(tensor, src=0, stream=stream)
    stream.synchronize()


# ---------------------------------------------------------------------------
# Rollout controller HTTP client
# ---------------------------------------------------------------------------

class ControllerClient:
    """Minimal HTTP client for the rollout controller."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict | None = None) -> dict:
        url = self.base_url + path
        data = json.dumps(body or {}).encode()
        req = Request(url, data=data, method="POST",
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
                return json.loads(content) if content else {}
        except HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"POST {path} returned {exc.code}: {body_text}") from exc

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
                return json.loads(content) if content else {}
        except HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"GET {path} returned {exc.code}: {body_text}") from exc

    def health_check(self) -> bool:
        try:
            data = self._get("/v1/health")
            return data.get("status") == "ok"
        except Exception:
            return False

    def wait_for_ready(self, timeout: float = 600) -> dict:
        """Wait for controller to be healthy with ready engines."""
        start = time.time()
        while time.time() - start < timeout:
            if self.health_check():
                status = self._get("/v1/pool/status")
                if status.get("ready_engines", 0) > 0:
                    return status
            log.info("  Waiting for controller + engines...")
            time.sleep(10)
        raise TimeoutError(f"Controller not ready after {timeout}s")

    def init_weight_transfer(self, master_address: str, master_port: int, world_size: int) -> dict:
        return self._post("/v1/weights/init", {
            "backend": "nccl",
            "master_address": master_address,
            "master_port": master_port,
            "trainer_world_size": world_size,
            "packed": False,
            "is_checkpoint_format": True,
        })

    def update_weights(self, target_version: int, model: torch.nn.Module) -> dict:
        """POST /v1/weights/update — blocks until NCCL transfer completes."""
        names = []
        dtype_names = []
        shapes = []
        for name, param in model.named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).replace("torch.", ""))
            shapes.append(list(param.shape))
        return self._post("/v1/weights/update", {
            "target_version": target_version,
            "pause_mode": "keep",
            "reset_kv_cache": True,
            "param_names": names,
            "param_dtypes": dtype_names,
            "param_shapes": shapes,
        })

    def generate(self, prompt_token_ids: list[int], max_tokens: int = 32) -> dict:
        return self._post("/v1/generate", {
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": {"temperature": 0.7, "max_tokens": max_tokens},
        })

    def sleep(self, level: int = 2) -> dict:
        return self._post("/v1/engines/sleep", {"level": level})

    def wake_up(self, tags: list[str]) -> dict:
        return self._post("/v1/engines/wake", {"tags": tags})

    def get_weight_version(self) -> int:
        data = self._get("/v1/weights/version")
        return data.get("weight_version", 0)

    def get_pool_status(self) -> dict:
        return self._get("/v1/pool/status")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NCCL weight trainer for llm-d rollout controller",
    )
    parser.add_argument(
        "--controller-url", default="http://rollout-controller.llm-d-rl.svc.cluster.local:8090",
        help="rollout controller URL",
    )
    parser.add_argument(
        "--model-name", default="meta-llama/Llama-3.2-1B",
        help="model to load (must match vLLM engine)",
    )
    parser.add_argument(
        "--master-port", type=int, default=29500,
        help="NCCL rendezvous port on this pod",
    )
    parser.add_argument(
        "--num-steps", type=int, default=5,
        help="number of training steps",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="torch device",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    client = ControllerClient(args.controller_url)

    # === STEP 1: Load model ===
    log.info("Loading model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %d parameters (%.1f GB)",
             num_params, num_params * 2 / 1e9)  # bf16 = 2 bytes

    # === STEP 2: Wait for controller + vLLM ===
    log.info("Waiting for controller at %s", args.controller_url)
    status = client.wait_for_ready(timeout=600)
    log.info("Controller ready: phase=%s, engines=%d/%d",
             status.get("phase"), status.get("ready_engines"), status.get("total_engines"))

    # === STEP 3: Set up NCCL group + init engines concurrently ===
    master_address = get_pod_ip()
    master_port = args.master_port
    total_engines = status.get("total_engines", 1)
    world_size = 1 + total_engines  # 1 trainer + N engines

    log.info("NCCL rendezvous: master=%s:%d, world_size=%d",
             master_address, master_port, world_size)

    # StatelessProcessGroup.create() starts a TCP store and blocks until
    # all ranks connect. We must tell the engines to join concurrently,
    # otherwise create() waits forever.
    nccl_result = [None, None]  # (pynccl, pg)
    nccl_error = [None]

    def create_nccl_group():
        try:
            pynccl, pg = setup_nccl_group(master_address, master_port, world_size, device)
            nccl_result[0] = pynccl
            nccl_result[1] = pg
        except Exception as e:
            nccl_error[0] = e
            log.error("NCCL group setup failed: %s", e)

    nccl_thread = threading.Thread(target=create_nccl_group, daemon=True)
    nccl_thread.start()

    # Give the TCP store a moment to start listening
    time.sleep(1)

    # Tell the controller to make vLLM engines join our NCCL group
    log.info("Initializing weight transfer on engines...")
    client.init_weight_transfer(master_address, master_port, world_size)
    log.info("Weight transfer init sent to controller")

    # Wait for NCCL group to complete (all ranks connected)
    nccl_thread.join(timeout=300)
    if nccl_error[0]:
        raise RuntimeError(f"NCCL group setup failed: {nccl_error[0]}")

    pynccl, pg = nccl_result
    log.info("NCCL group established — engines joined")

    # === STEP 5: Training loop ===
    weight_version = 0

    for step in range(1, args.num_steps + 1):
        log.info("")
        log.info("=" * 60)
        log.info("Step %d/%d  (weight_version=%d)", step, args.num_steps, weight_version)
        log.info("=" * 60)
        step_start = time.time()

        # --- 5a. Generate a rollout ---
        log.info("[1/6] Generating rollout...")
        prompt_ids = list(range(100, 120))  # dummy token IDs
        resp = client.generate(prompt_ids, max_tokens=16)
        log.info("  Generated: finish_reason=%s, engine=%s, tokens=%d",
                 resp.get("finish_reason"), resp.get("engine_id"),
                 len(resp.get("output_token_ids", [])))

        # --- 5b. Sleep engines ---
        log.info("[2/6] Sleeping engines (level=2)...")
        client.sleep(level=2)
        log.info("  GPU memory freed on engines")

        # --- 5c. Perturb weights (simulate training step) ---
        log.info("[3/6] Perturbing weights (simulating gradient step)...")
        with torch.no_grad():
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.001

        # --- 5d. Wake engines (weights only) ---
        log.info("[4/6] Waking engines (weights only)...")
        client.wake_up(tags=["weights"])
        log.info("  Engine weight memory restored")

        # --- 5e. Concurrent weight sync ---
        #
        # The controller's POST /v1/weights/update calls vLLM's
        # POST /update_weights which starts an NCCL receive (blocks).
        # We must be broadcasting from rank 0 at the same time.
        #
        # Solution: run NCCL broadcast in a background thread.
        # Both sides are NCCL collectives that block until the other
        # is ready, so the ordering is safe.

        weight_version += 1
        log.info("[5/6] Updating weights (version %d) — concurrent NCCL broadcast...",
                 weight_version)

        broadcast_error = [None]

        def do_broadcast():
            try:
                time.sleep(0.5)  # let HTTP call propagate to vLLM first
                broadcast_weights(pynccl, model, device)
                log.info("  NCCL broadcast complete")
            except Exception as e:
                broadcast_error[0] = e
                log.error("  NCCL broadcast failed: %s", e)

        broadcast_thread = threading.Thread(target=do_broadcast, daemon=True)
        broadcast_thread.start()

        # This blocks until the NCCL collective completes on the engine side
        result = client.update_weights(weight_version, model)
        log.info("  Controller returned: %s", result)

        broadcast_thread.join(timeout=120)
        if broadcast_error[0]:
            raise RuntimeError(f"NCCL broadcast failed: {broadcast_error[0]}")

        # --- 5f. Wake engines (KV cache) ---
        log.info("[6/6] Waking engines (KV cache)...")
        client.wake_up(tags=["kv_cache"])

        status = client.get_pool_status()
        step_time = time.time() - step_start
        log.info("  Pool: phase=%s, weight_version=%d", status.get("phase"), status.get("weight_version"))
        log.info("Step %d complete (%.1fs)", step, step_time)

    # === Done ===
    log.info("")
    log.info("Training complete. Final weight version: %d", weight_version)

    final_version = client.get_weight_version()
    log.info("Controller weight version: %d", final_version)

    if final_version == weight_version:
        log.info("SUCCESS: weight versions match")
    else:
        log.error("MISMATCH: expected %d, got %d", weight_version, final_version)


if __name__ == "__main__":
    main()
