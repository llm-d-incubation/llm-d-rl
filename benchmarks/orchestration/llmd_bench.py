"""Instrumented llm-d-rl trainer for orchestration overhead benchmarking.

Runs the same 6-phase lifecycle as the demo trainer, but records per-phase
wall-clock times using time.perf_counter() for precise measurement.

Results are written as structured JSON for comparison with the Ray harness.

Usage:
    python llmd_bench.py \
        --controller-url http://rollout-controller:8090 \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --warmup-steps 5 \
        --measured-steps 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
import threading
from contextlib import contextmanager
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import torch
from transformers import AutoModelForCausalLM

log = logging.getLogger("llmd-bench")


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str, record: dict):
    """Context manager that records elapsed time in seconds."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    record[label] = elapsed


# ---------------------------------------------------------------------------
# NCCL setup
# ---------------------------------------------------------------------------

def get_pod_ip() -> str:
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


def setup_nccl_group(master_address: str, master_port: int, world_size: int, device: torch.device):
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
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for name, param in model.named_parameters():
            tensor = param.data.contiguous()
            pynccl.broadcast(tensor, src=0, stream=stream)
    stream.synchronize()


# ---------------------------------------------------------------------------
# Controller HTTP client
# ---------------------------------------------------------------------------

class ControllerClient:
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
# Main benchmark loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="llm-d-rl orchestration benchmark")
    parser.add_argument("--controller-url",
                        default="http://rollout-controller.llm-d-rl.svc.cluster.local:8090")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="/results/llmd_results.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    client = ControllerClient(args.controller_url)
    total_steps = args.warmup_steps + args.measured_steps

    # === Load model ===
    log.info("Loading model: %s", args.model_name)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    model_load_time = time.perf_counter() - load_start
    num_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %d parameters (%.1f GB) in %.1fs",
             num_params, num_params * 2 / 1e9, model_load_time)

    # === Wait for controller ===
    log.info("Waiting for controller at %s", args.controller_url)
    status = client.wait_for_ready(timeout=600)
    num_engines = status.get("total_engines", 1)
    log.info("Controller ready: %d engines", num_engines)

    # === Set up NCCL group ===
    master_address = get_pod_ip()
    master_port = args.master_port
    world_size = 1 + num_engines

    log.info("NCCL rendezvous: master=%s:%d, world_size=%d",
             master_address, master_port, world_size)

    nccl_result = [None, None]
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
    time.sleep(1)

    nccl_init_start = time.perf_counter()
    client.init_weight_transfer(master_address, master_port, world_size)
    nccl_thread.join(timeout=300)
    nccl_init_time = time.perf_counter() - nccl_init_start

    if nccl_error[0]:
        raise RuntimeError(f"NCCL group setup failed: {nccl_error[0]}")

    pynccl, pg = nccl_result
    log.info("NCCL group established in %.3fs", nccl_init_time)

    # === Benchmark loop ===
    weight_version = 0
    all_timings = []

    for step in range(1, total_steps + 1):
        is_warmup = step <= args.warmup_steps
        phase_label = "WARMUP" if is_warmup else "MEASURED"
        log.info("")
        log.info("=" * 60)
        log.info("%s Step %d/%d  (weight_version=%d)", phase_label, step, total_steps, weight_version)
        log.info("=" * 60)

        t = {}
        step_start = time.perf_counter()

        # 1. Generate
        with timed("generate", t):
            prompt_ids = list(range(100, 120))
            resp = client.generate(prompt_ids, max_tokens=16)
        log.info("  [1/6] Generate: %.3fs (tokens=%d)",
                 t["generate"], len(resp.get("output_token_ids", [])))

        # 2. Sleep
        with timed("sleep", t):
            client.sleep(level=2)
        log.info("  [2/6] Sleep: %.3fs", t["sleep"])

        # 3. Train (perturb weights)
        with timed("train", t):
            with torch.no_grad():
                for param in model.parameters():
                    param.data += torch.randn_like(param.data) * 0.001
        log.info("  [3/6] Train: %.3fs", t["train"])

        # 4. Wake (weights)
        with timed("wake_weights", t):
            client.wake_up(tags=["weights"])
        log.info("  [4/6] Wake weights: %.3fs", t["wake_weights"])

        # 5. NCCL broadcast + update_weights
        weight_version += 1
        broadcast_error = [None]
        broadcast_elapsed = [0.0]

        def do_broadcast():
            try:
                time.sleep(0.5)
                bstart = time.perf_counter()
                broadcast_weights(pynccl, model, device)
                broadcast_elapsed[0] = time.perf_counter() - bstart
            except Exception as e:
                broadcast_error[0] = e

        broadcast_thread = threading.Thread(target=do_broadcast, daemon=True)
        broadcast_thread.start()

        with timed("update_weights_http", t):
            result = client.update_weights(weight_version, model)

        broadcast_thread.join(timeout=120)
        if broadcast_error[0]:
            raise RuntimeError(f"NCCL broadcast failed: {broadcast_error[0]}")

        t["nccl_broadcast"] = broadcast_elapsed[0]
        log.info("  [5/6] Weight sync: http=%.3fs, nccl=%.3fs",
                 t["update_weights_http"], t["nccl_broadcast"])

        # 6. Wake (KV cache)
        with timed("wake_kvcache", t):
            client.wake_up(tags=["kv_cache"])
        log.info("  [6/6] Wake KV cache: %.3fs", t["wake_kvcache"])

        t["step_total"] = time.perf_counter() - step_start
        t["step_num"] = step
        t["is_warmup"] = is_warmup
        t["weight_version"] = weight_version
        t["num_engines"] = num_engines

        log.info("  Step total: %.3fs", t["step_total"])

        if not is_warmup:
            all_timings.append(t)

    # === Write results ===
    results = {
        "system": "llm-d-rl",
        "model": args.model_name,
        "num_engines": num_engines,
        "num_params": num_params,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "model_load_time_s": model_load_time,
        "nccl_init_time_s": nccl_init_time,
        "steps": all_timings,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results written to %s", args.output)

    # Print results to stdout with delimiters for reliable extraction from logs
    print("===BENCH_RESULTS_START===")
    print(json.dumps(results, indent=2))
    print("===BENCH_RESULTS_END===")

    # === Verify ===
    final_version = client.get_weight_version()
    log.info("Final weight version: controller=%d, expected=%d",
             final_version, weight_version)
    if final_version == weight_version:
        log.info("SUCCESS: weight versions match")
    else:
        log.error("MISMATCH: expected %d, got %d", weight_version, final_version)


if __name__ == "__main__":
    main()
