"""
Prometheus metrics poller for vLLM engines.

Polls each vLLM pod's /metrics endpoint in a background asyncio loop and
writes the results into endpoint.attributes["routing_stats"]. The scorers
in py-inference-scheduler read exactly these keys:
  - num_waiting_reqs  (WaitingQueueScorer)
  - num_running_reqs  (RunningQueueScorer)
  - kv                (KVCacheScorer)
"""

import asyncio
import logging
import re
from typing import List, Any

import httpx

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0

# vLLM Prometheus metric patterns.
# Labels vary (model_name, version, etc.) so we match any label set.
_WAITING = re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)", re.MULTILINE)
_RUNNING = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)", re.MULTILINE)
_KV      = re.compile(r"^vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)", re.MULTILINE)


def _parse(text: str) -> dict:
    def first(pattern: re.Pattern) -> float:
        m = pattern.search(text)
        return float(m.group(1)) if m else 0.0

    return {
        "num_waiting_reqs": first(_WAITING),
        "num_running_reqs": first(_RUNNING),
        "kv":               first(_KV),
    }


async def _poll_one(client: httpx.AsyncClient, endpoint: Any) -> None:
    address = endpoint.attributes.get("address", "")
    try:
        resp = await client.get(f"{address}/metrics", timeout=2.0)
        resp.raise_for_status()
        endpoint.attributes["routing_stats"] = _parse(resp.text)
    except Exception as exc:
        logger.warning("metrics poll failed for %s: %s", address, exc)


async def metrics_polling_loop(endpoints: List[Any]) -> None:
    """Continuously poll all endpoints. The list is shared with server.py and
    updated in-place by the pool polling loop, so new/removed endpoints are
    picked up automatically on the next iteration."""
    async with httpx.AsyncClient() as client:
        while True:
            if endpoints:
                await asyncio.gather(
                    *[_poll_one(client, ep) for ep in list(endpoints)],
                    return_exceptions=True,
                )
            await asyncio.sleep(POLL_INTERVAL_S)
