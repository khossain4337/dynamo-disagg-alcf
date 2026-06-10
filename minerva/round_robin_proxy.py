#!/usr/bin/env python3
"""
round_robin_proxy.py

Minimal async round-robin HTTP proxy for AGG-3 baseline benchmarking.
Distributes requests across N backend vLLM servers on localhost.

Usage:
    python3 round_robin_proxy.py --port 8000 --backends 8001 8002 8003

Features:
    - Round-robin request distribution
    - Full streaming response support (SSE / chunked transfer)
    - /health endpoint returns 200 only when ALL backends are healthy
    - All other paths forwarded verbatim (headers, body, method)

NOT intended for production — benchmark use only.
"""

import argparse
import asyncio
import itertools
import logging
import sys

import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [proxy] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("proxy")


# ── Round-robin counter ───────────────────────────────────────
class RoundRobin:
    def __init__(self, backends: list[str]):
        self._cycle = itertools.cycle(backends)
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            return next(self._cycle)


# ── Health check ──────────────────────────────────────────────
async def health_handler(request: web.Request) -> web.Response:
    """
    Returns 200 only when every backend /health returns 200.
    Returns 503 otherwise — matches vLLM health check contract.
    """
    backends: list[str] = request.app["backends"]
    session: aiohttp.ClientSession = request.app["session"]

    results = await asyncio.gather(
        *[_check_backend(session, b) for b in backends],
        return_exceptions=True,
    )

    if all(r is True for r in results):
        return web.Response(status=200, text="OK")

    unhealthy = [b for b, r in zip(backends, results) if r is not True]
    log.warning("Unhealthy backends: %s", unhealthy)
    return web.Response(status=503, text=f"Unhealthy backends: {unhealthy}")


async def _check_backend(session: aiohttp.ClientSession, base_url: str) -> bool:
    try:
        async with session.get(
            f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=2)
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Generic proxy handler ─────────────────────────────────────
async def proxy_handler(request: web.Request) -> web.StreamResponse:
    """
    Forward request to next backend in round-robin order.
    Streams response back to client — handles SSE and chunked transfer.
    """
    rr: RoundRobin = request.app["round_robin"]
    session: aiohttp.ClientSession = request.app["session"]

    backend = await rr.next()
    target_url = f"{backend}{request.path_qs}"

    # Forward all headers except Host (aiohttp sets it correctly)
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    body = await request.read()

    try:
        async with session.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            data=body if body else None,
            timeout=aiohttp.ClientTimeout(total=600),  # 10 min — long generations
            allow_redirects=False,
        ) as backend_resp:

            # Build response with same status + headers
            response = web.StreamResponse(
                status=backend_resp.status,
                headers={
                    k: v
                    for k, v in backend_resp.headers.items()
                    if k.lower()
                    not in ("transfer-encoding", "content-encoding", "content-length")
                },
            )
            await response.prepare(request)

            # Stream chunks back
            async for chunk in backend_resp.content.iter_any():
                await response.write(chunk)

            await response.write_eof()
            return response

    except aiohttp.ClientError as e:
        log.error("Backend %s error: %s", backend, e)
        return web.Response(status=502, text=f"Backend error: {e}")


# ── App factory ───────────────────────────────────────────────
async def create_app(listen_port: int, backend_ports: list[int]) -> web.Application:
    backends = [f"http://127.0.0.1:{p}" for p in backend_ports]
    log.info("Proxy listening on port %d", listen_port)
    log.info("Backends: %s", backends)

    app = web.Application()
    app["backends"] = backends
    app["round_robin"] = RoundRobin(backends)

    # Shared aiohttp session — reuse connections across requests
    connector = aiohttp.TCPConnector(limit=0)  # unlimited connections
    app["session"] = aiohttp.ClientSession(connector=connector)

    # Routes
    app.router.add_get("/health", health_handler)
    app.router.add_route("*", "/{path_info:.*}", proxy_handler)

    async def cleanup(app: web.Application) -> None:
        await app["session"].close()

    app.on_cleanup.append(cleanup)
    return app


# ── Entry point ───────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin HTTP proxy")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--backends",
        type=int,
        nargs="+",
        required=True,
        help="Backend ports to round-robin across (e.g. 8001 8002 8003)",
    )
    args = parser.parse_args()

    if not args.backends:
        print("ERROR: at least one backend port required", file=sys.stderr)
        sys.exit(1)

    async def run() -> None:
        app = await create_app(args.port, args.backends)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.port)
        await site.start()
        log.info("Proxy ready — ctrl+c to stop")
        # Run forever until killed
        await asyncio.Event().wait()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Proxy stopped")


if __name__ == "__main__":
    main()
