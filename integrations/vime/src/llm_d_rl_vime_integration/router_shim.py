#!/usr/bin/env python3
"""
Registration shim for the vime + llm-d routing stack.

Maintains the in-memory engine registry and atomically rewrites
/tmp/epp-endpoints.yaml on every change so EPP's file-discovery plugin stays in sync.

Endpoints (proxied from Envoy :8081/workers*, never called by vime directly):
  POST   /workers          register engine — body: {url, worker_type?} → {url}
  GET    /workers          list engines
  DELETE /workers/{ref}    deregister engine — ref is the percent-encoded engine URL
"""

import argparse
import logging
import urllib.parse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from llm_d_rl_common.endpoints import write_rollout_endpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

_workers: dict[str, dict[str, Any]] = {}   # url → {url, worker_type}
_endpoints_file: Path = Path("/tmp/epp-endpoints.yaml")


def _write_endpoints() -> None:
    write_rollout_endpoints(str(_endpoints_file), list(_workers.keys()))
    logger.info("endpoints: %d worker(s) → %s", len(_workers), _endpoints_file)


@app.post("/workers")
async def add_worker(request: Request):
    body = await request.json()
    url = body["url"]
    if url in _workers:
        logger.info("already registered %s", url)
        return {"url": url}
    _workers[url] = {"url": url, "worker_type": body.get("worker_type", "regular")}
    _write_endpoints()
    logger.info("registered %s", url)
    return {"url": url}


@app.get("/workers")
def list_workers():
    return {"workers": list(_workers.values())}


@app.delete("/workers/{worker_ref:path}")
def remove_worker(worker_ref: str):
    url = urllib.parse.unquote(worker_ref)
    removed = _workers.pop(url, None)
    if removed:
        _write_endpoints()
        logger.info("deregistered %s", url)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail={"status": "not_found"})

def main():
    parser = argparse.ArgumentParser(description="Vime EPP registration shim")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: localhost — Envoy proxies to us)")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--endpoints-file", default="/tmp/epp-endpoints.yaml")
    args = parser.parse_args()

    global _endpoints_file
    _endpoints_file = Path(args.endpoints_file)
    _endpoints_file.parent.mkdir(parents=True, exist_ok=True)

    # Write empty file so EPP starts cleanly without a missing-file error
    _write_endpoints()
    logger.info("shim listening on %s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
