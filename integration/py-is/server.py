"""
py-inference-scheduler HTTP proxy server for llm-d-rl.

Acts as the --router-url target for the llm-d-rl rollout controller.
Receives /v1/completions requests, uses py-inference-scheduler to pick
the best vLLM pod, and proxies the request there.

Engine discovery:
  - Primary:  polls GET {ROLLOUT_CONTROLLER_URL}/v1/pool/status every 5s.
              Reuses the rollout controller's already-health-checked pool
              as the single source of truth.
  - Fallback: VLLM_ENDPOINTS env var (comma-separated URLs) for local dev
              without a running rollout controller.

Required env vars:
  ROUTER_CONFIG_PATH          Path to the scheduler YAML config.
  ROLLOUT_CONTROLLER_URL      Rollout controller base URL (e.g. http://rollout-controller:8090).
                              Not needed when VLLM_ENDPOINTS is set.

Optional env vars:
  VLLM_ENDPOINTS              Comma-separated vLLM URLs for static local dev.
  POOL_POLL_INTERVAL_S        Seconds between pool status polls (default: 5).
"""

import asyncio
import logging
import os
import uuid
from typing import List

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from scheduling import Endpoint, LLMRequest, Scheduler

from metrics import metrics_polling_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="py-is routing proxy", version="0.1.0")

# Shared mutable state — both background tasks operate on these objects.
_endpoints: List[Endpoint] = []
_scheduler: Scheduler = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _scheduler

    _scheduler = Scheduler()  # reads ROUTER_CONFIG_PATH env var
    logger.info("scheduler loaded from %s", os.environ.get("ROUTER_CONFIG_PATH"))

    static = os.environ.get("VLLM_ENDPOINTS", "")
    if static:
        for url in (u.strip() for u in static.split(",") if u.strip()):
            _endpoints.append(Endpoint(name=url, attributes={"address": url, "routing_stats": {}}))
        logger.info("static endpoints: %s", [e.name for e in _endpoints])
    else:
        asyncio.create_task(_pool_polling_loop())

    asyncio.create_task(metrics_polling_loop(_endpoints))


# ---------------------------------------------------------------------------
# Pool polling — keeps _endpoints in sync with the rollout controller's pool
# ---------------------------------------------------------------------------

async def _pool_polling_loop() -> None:
    rc_url = os.environ.get("ROLLOUT_CONTROLLER_URL", "").rstrip("/")
    interval = float(os.environ.get("POOL_POLL_INTERVAL_S", "5"))

    if not rc_url:
        logger.warning("ROLLOUT_CONTROLLER_URL not set and VLLM_ENDPOINTS not set — no engines")
        return

    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{rc_url}/v1/pool/status", timeout=5.0)
                resp.raise_for_status()
                data = resp.json()
                _sync_endpoints(data.get("engines", []))
            except Exception as exc:
                logger.warning("pool status poll failed: %s", exc)
            await asyncio.sleep(interval)


def _sync_endpoints(engines: list) -> None:
    """Update _endpoints in-place from the pool status engines list."""
    ready_addresses = {e["address"] for e in engines if e.get("ready")}

    # Remove engines that are gone or not ready.
    _endpoints[:] = [ep for ep in _endpoints if ep.attributes["address"] in ready_addresses]

    # Add new engines.
    existing = {ep.attributes["address"] for ep in _endpoints}
    for e in engines:
        if e.get("ready") and e["address"] not in existing:
            _endpoints.append(Endpoint(
                name=e["id"],
                attributes={"address": e["address"], "routing_stats": {}},
            ))
            logger.info("registered engine %s at %s", e["id"], e["address"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    if not _endpoints:
        return JSONResponse({"error": "no ready engines"}, status_code=503)

    raw_body = await request.body()
    req = LLMRequest(request_id=str(uuid.uuid4()), body=raw_body)

    selected = _scheduler.run(req, list(_endpoints))
    chosen = selected[0].endpoint if selected else _endpoints[0]

    if not selected:
        logger.warning("scheduler returned no selection, falling back to first endpoint")

    chosen_url = chosen.attributes["address"]
    logger.info("routing to %s (%s)", chosen.name, chosen_url)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{chosen_url}/v1/completions",
            content=raw_body,
            headers={"Content-Type": "application/json"},
            timeout=300.0,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/v1/models")
async def models() -> Response:
    if not _endpoints:
        return JSONResponse({"object": "list", "data": []})
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{_endpoints[0].attributes['address']}/v1/models",
                timeout=5.0,
            )
            return Response(content=resp.content, media_type="application/json")
        except Exception:
            return JSONResponse({"object": "list", "data": []})


@app.get("/v1/health")
async def health() -> dict:
    return {"status": "ok", "ready_endpoints": len(_endpoints)}
