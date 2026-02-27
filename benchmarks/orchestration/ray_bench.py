"""Ray orchestration harness for orchestration overhead benchmarking.

Simulates veRL's CheckpointEngineManager pattern: a Ray actor on a GPU
orchestrates the same vLLM engines through the same 6-phase lifecycle,
calling vLLM dev-mode endpoints directly (bypassing the Go controller).

This isolates the orchestration overhead: Ray actor management vs Go HTTP.

Usage:
    python ray_bench.py \
        --engine-urls http://vllm-engine-0.vllm-engine:8000 \
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

log = logging.getLogger("ray-bench")


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str, record: dict):
    start = time.perf_counter()
    yield
    record[label] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# NCCL setup (same as llm-d-rl trainer — identical NCCL path)
# ---------------------------------------------------------------------------

def get_pod_ip() -> str:
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


def setup_nccl_group(master_address: str, master_port: int, world_size: int, device: torch.device):
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl, pg


def broadcast_weights(pynccl, model: torch.nn.Module, device: torch.device) -> None:
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for name, param in model.named_parameters():
            tensor = param.data.contiguous()
            pynccl.broadcast(tensor, src=0, stream=stream)
    stream.synchronize()


# ---------------------------------------------------------------------------
# Direct vLLM HTTP client (bypasses Go controller)
# ---------------------------------------------------------------------------

class VLLMDirectClient:
    """Calls vLLM dev-mode endpoints directly, same as the Go controller does."""

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
            self._get("/health")
            return True
        except Exception:
            return False

    def discover_model_name(self) -> str:
        try:
            data = self._get("/v1/models")
            models = data.get("data", [])
            if models:
                return models[0].get("id", "default")
        except Exception:
            pass
        return "default"

    def generate(self, model_name: str, prompt_token_ids: list[int], max_tokens: int = 32) -> dict:
        return self._post("/v1/completions", {
            "model": model_name,
            "prompt": prompt_token_ids,
            "temperature": 0.7,
            "max_tokens": max_tokens,
        })

    def sleep(self, level: int = 2) -> dict:
        return self._post("/sleep", {"level": level, "mode": "keep"})

    def wake_up(self, tags: list[str]) -> dict:
        return self._post("/wake_up", {"tags": tags})

    def init_weight_transfer(self, master_address: str, master_port: int, world_size: int) -> dict:
        return self._post("/init_weight_transfer_engine", {
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": world_size,
            },
        })

    def update_weights(self, model: torch.nn.Module) -> dict:
        names = []
        dtype_names = []
        shapes = []
        for name, param in model.named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).replace("torch.", ""))
            shapes.append(list(param.shape))
        return self._post("/update_weights", {
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": False,
            },
        })


# ---------------------------------------------------------------------------
# Ray trainer actor
# ---------------------------------------------------------------------------

def make_ray_trainer_class():
    """Factory to create the Ray remote class (deferred import of ray)."""
    import ray

    @ray.remote(num_gpus=1)
    class RayTrainerActor:
        """Simulates veRL's CheckpointEngineManager orchestration pattern.

        Loads the model on a GPU, then orchestrates the same vLLM engines
        through the same lifecycle — but via Ray actor instead of Go HTTP.
        """

        def __init__(self, engine_urls: list[str], model_name: str,
                     master_port: int, device: str = "cuda:0"):
            self.device = torch.device(device)
            self.engine_urls = engine_urls
            self.model_name = model_name
            self.master_port = master_port
            self.clients = [VLLMDirectClient(url) for url in engine_urls]
            self.model = None
            self.pynccl = None
            self.pg = None
            self.vllm_model_name = None

        def setup(self) -> dict:
            """Load model and set up NCCL group."""
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )

            # Load model
            load_start = time.perf_counter()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.model.eval()
            model_load_time = time.perf_counter() - load_start

            num_params = sum(p.numel() for p in self.model.parameters())

            # Discover model name from vLLM
            self.vllm_model_name = self.clients[0].discover_model_name()

            # Set up NCCL group
            master_address = get_pod_ip()
            world_size = 1 + len(self.clients)

            nccl_result = [None, None]
            nccl_error = [None]

            def create_nccl():
                try:
                    p, g = setup_nccl_group(master_address, self.master_port, world_size, self.device)
                    nccl_result[0] = p
                    nccl_result[1] = g
                except Exception as e:
                    nccl_error[0] = e

            nccl_thread = threading.Thread(target=create_nccl, daemon=True)
            nccl_thread.start()
            time.sleep(1)

            nccl_init_start = time.perf_counter()
            for client in self.clients:
                client.init_weight_transfer(master_address, self.master_port, world_size)
            nccl_thread.join(timeout=300)
            nccl_init_time = time.perf_counter() - nccl_init_start

            if nccl_error[0]:
                raise RuntimeError(f"NCCL group setup failed: {nccl_error[0]}")

            self.pynccl, self.pg = nccl_result

            return {
                "model_load_time_s": model_load_time,
                "nccl_init_time_s": nccl_init_time,
                "num_params": num_params,
                "num_engines": len(self.clients),
            }

        def run_step(self, step_num: int, weight_version: int) -> dict:
            """Run one benchmark step, return per-phase timings."""
            t = {}
            step_start = time.perf_counter()

            # 1. Generate (pick first engine, like llm-d-rl does)
            with timed("generate", t):
                prompt_ids = list(range(100, 120))
                self.clients[0].generate(
                    self.vllm_model_name, prompt_ids, max_tokens=16)

            # 2. Sleep all engines
            with timed("sleep", t):
                for client in self.clients:
                    client.sleep(level=2)

            # 3. Train (perturb weights)
            with timed("train", t):
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.data += torch.randn_like(param.data) * 0.001

            # 4. Wake (weights)
            with timed("wake_weights", t):
                for client in self.clients:
                    client.wake_up(tags=["weights"])

            # 5. NCCL broadcast + update_weights
            new_version = weight_version + 1
            broadcast_error = [None]
            broadcast_elapsed = [0.0]

            def do_broadcast():
                try:
                    time.sleep(0.5)
                    bstart = time.perf_counter()
                    broadcast_weights(self.pynccl, self.model, self.device)
                    broadcast_elapsed[0] = time.perf_counter() - bstart
                except Exception as e:
                    broadcast_error[0] = e

            broadcast_thread = threading.Thread(target=do_broadcast, daemon=True)
            broadcast_thread.start()

            with timed("update_weights_http", t):
                for client in self.clients:
                    client.update_weights(self.model)

            broadcast_thread.join(timeout=120)
            if broadcast_error[0]:
                raise RuntimeError(f"NCCL broadcast failed: {broadcast_error[0]}")

            t["nccl_broadcast"] = broadcast_elapsed[0]

            # 6. Wake (KV cache)
            with timed("wake_kvcache", t):
                for client in self.clients:
                    client.wake_up(tags=["kv_cache"])

            t["step_total"] = time.perf_counter() - step_start
            t["step_num"] = step_num
            t["weight_version"] = new_version

            return t

    return RayTrainerActor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ray orchestration benchmark")
    parser.add_argument("--engine-urls", required=True,
                        help="Comma-separated vLLM engine URLs")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--output", default="/results/ray_results.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    import ray
    ray.init()

    engine_urls = [u.strip() for u in args.engine_urls.split(",")]
    total_steps = args.warmup_steps + args.measured_steps

    log.info("Engine URLs: %s", engine_urls)
    log.info("Model: %s", args.model_name)
    log.info("Steps: %d warmup + %d measured", args.warmup_steps, args.measured_steps)

    # Wait for engines to be healthy
    log.info("Waiting for engines to be healthy...")
    temp_client = VLLMDirectClient(engine_urls[0])
    start = time.time()
    while time.time() - start < 600:
        if temp_client.health_check():
            break
        time.sleep(10)
    else:
        raise TimeoutError("Engines not healthy after 600s")
    log.info("Engines healthy")

    # Create Ray trainer actor
    RayTrainerActor = make_ray_trainer_class()
    actor = RayTrainerActor.remote(engine_urls, args.model_name, args.master_port)

    # Setup (load model, NCCL)
    log.info("Setting up Ray trainer actor...")
    setup_info = ray.get(actor.setup.remote())
    log.info("Setup complete: %s", setup_info)

    # Benchmark loop
    all_timings = []
    weight_version = 0

    for step in range(1, total_steps + 1):
        is_warmup = step <= args.warmup_steps
        phase_label = "WARMUP" if is_warmup else "MEASURED"
        log.info("%s Step %d/%d", phase_label, step, total_steps)

        t = ray.get(actor.run_step.remote(step, weight_version))
        weight_version = t["weight_version"]

        t["is_warmup"] = is_warmup
        t["num_engines"] = len(engine_urls)
        log.info("  Step %d: total=%.3fs, generate=%.3fs, sleep=%.3fs, "
                 "train=%.3fs, nccl=%.3fs, http=%.3fs",
                 step, t["step_total"], t["generate"], t["sleep"],
                 t["train"], t["nccl_broadcast"], t["update_weights_http"])

        if not is_warmup:
            all_timings.append(t)

    # Write results
    results = {
        "system": "ray",
        "model": args.model_name,
        "num_engines": len(engine_urls),
        "num_params": setup_info["num_params"],
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "model_load_time_s": setup_info["model_load_time_s"],
        "nccl_init_time_s": setup_info["nccl_init_time_s"],
        "steps": all_timings,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results written to %s", args.output)

    ray.shutdown()


if __name__ == "__main__":
    main()
