"""slime Sandbox client over the in-cluster sandbox-runner HTTP API.

DinDSandbox is a drop-in for slime's E2BSandbox.
Both generate.py and swe.py stay unmodified; dind_generatebinds this class into 
both modules at import time.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_SVCS = [
    u.strip()
    for u in os.environ.get("DIND_SANDBOX_SERVICE_URLS", "").split(",")
    if u.strip()
]
if not _SVCS:
    raise RuntimeError(
        "DIND_SANDBOX_SERVICE_URLS is empty — run setup_sandbox.sh "
        "so the launcher can pass the sandbox-runner Service URLs"
    )
_ASSIGNMENT_FILE = os.environ.get(
    "SANDBOX_ASSIGNMENT_FILE", "/tmp/slime/data/sandbox_assignment.json"
)
_assignment: dict[str, int] | None = None

_DEFAULT_LIFETIME_SEC = 3600


def _load_assignment() -> dict[str, int]:
    global _assignment
    if _assignment is None:
        with open(_ASSIGNMENT_FILE) as f:
            _assignment = json.load(f)
    return _assignment


def _pick_svc(image: str) -> str:
    assignment = _load_assignment()
    if image not in assignment:
        raise RuntimeError(
            f"image '{image}' not in assignment table "
            f"({_ASSIGNMENT_FILE}) — re-run setup_sandbox.sh"
        )
    runner = assignment[image]
    if not 0 <= runner < len(_SVCS):
        raise RuntimeError(
            f"image '{image}' maps to runner {runner}, but only "
            f"{len(_SVCS)} runner URL(s) are configured"
        )
    return _SVCS[runner]


class DinDSandbox:
    """Async context manager around one DinD container on a sandbox-runner pod."""

    def __init__(self, image: str, **_ignored) -> None:
        self.image = image
        self.timeout = _DEFAULT_LIFETIME_SEC
        self.sandbox_id = ""
        self._cid = ""
        self._svc = ""
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> DinDSandbox:
        self._svc = _pick_svc(self.image)
        self._client = httpx.AsyncClient(base_url=self._svc, timeout=630.0)
        try:
            r = await self._client.post(
                "/sandboxes",
                json={"image": self.image, "timeout": self.timeout},
                timeout=600.0,
            )
            r.raise_for_status()
            self._cid = r.json()["id"]
            self.sandbox_id = self._cid[:12]
            return self
        except BaseException:
            await self._client.aclose()
            self._client = None
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._client is not None and self._cid:
                await self._client.delete(f"/sandboxes/{self._cid}", timeout=30.0)
        except Exception as e:
            logger.warning("[dind] kill %s failed: %s", self.sandbox_id, e)
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> tuple[int, str, str]:
        body: dict = {"cmd": cmd, "envs": env or {}, "timeout": timeout, "user": user}
        r = await self._client.post(
            f"/sandboxes/{self._cid}/exec",
            json=body,
            timeout=timeout + 30,
        )
        r.raise_for_status()
        d = r.json()
        exit_code, stdout, stderr = d["exit_code"], d["stdout"] or "", d["stderr"] or ""
        if check and exit_code != 0:
            raise RuntimeError(
                f"dind exec failed (exit={exit_code}): {cmd[:120]}\n{stderr[:400]}"
            )
        return exit_code, stdout, stderr

    async def write_file(
        self, sandbox_path: str, content: str | bytes | Path, *, user: str = "root"
    ) -> None:
        if isinstance(content, Path):
            data = content.read_bytes()
        elif isinstance(content, str):
            data = content.encode()
        else:
            data = bytes(content)
        r = await self._client.post(
            f"/sandboxes/{self._cid}/files/write",
            params={"path": sandbox_path},
            content=data,
            headers={"content-type": "application/octet-stream"},
            timeout=630.0,
        )
        r.raise_for_status()

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            r = await self._client.get(
                f"/sandboxes/{self._cid}/files/read",
                params={"path": sandbox_path},
                timeout=60.0,
            )
            if r.status_code != 200:
                return ""
            return r.json()["content"]
        except Exception:
            return ""
