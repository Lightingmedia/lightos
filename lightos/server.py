"""HTTP API for LightOS - the backend a frontend app calls.

Standard library only, so the production image needs no web framework.

Endpoints
    GET  /healthz        liveness probe
    GET  /v1/catalog     query capacity and prices           (public)
    POST /v1/estimate    price a blueprint, provision nothing (public)
    POST /v1/jobs        launch a blueprint                   (token required)

`POST /v1/jobs` runs caller-supplied shell on provisioned hardware, so it is
an arbitrary-code-execution endpoint by design. It stays disabled until
LIGHTOS_API_TOKEN is set, and the read-only endpoints are the only ones a
public deployment should expose.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import yaml

from . import __version__
from .catalog import (CatalogError, CostCatalog, describe_request,
                      estimate_cost, total_vram_gb, vram_for_model)
from .orchestrator import Orchestrator
from .task import Task, TaskError

TOKEN_ENV = "LIGHTOS_API_TOKEN"
MAX_BODY_BYTES = 256 * 1024

# When rejecting an oversized body we first read and discard it, so the
# client can finish writing and then read our 413. Replying mid-upload makes
# the client see a connection reset instead of the status we sent. Bounded,
# because a caller can declare a Content-Length far larger than it will send.
MAX_DRAIN_BYTES = 4 * MAX_BODY_BYTES
_DRAIN_CHUNK = 64 * 1024

# Browser origins allowed to call this API, comma-separated, e.g.
# "https://agentic.lightos.sh,http://localhost:5173". A browser blocks
# cross-origin reads unless the server opts in, so a frontend served from a
# different host needs its origin listed here.
#
# The default is empty: no cross-origin access. "*" allows any origin and is
# only appropriate while every exposed endpoint is read-only and public.
ORIGINS_ENV = "LIGHTOS_ALLOWED_ORIGINS"
CORS_METHODS = "GET, POST, OPTIONS"
CORS_HEADERS = "Authorization, Content-Type"
CORS_MAX_AGE = "600"


class LightOSHandler(BaseHTTPRequestHandler):
    server_version = f"LightOS/{__version__}"

    # Injected by serve().
    catalog: CostCatalog = None
    orchestrator: Orchestrator = None
    api_token: Optional[str] = None
    allowed_origins: tuple = ()

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    def _allowed_origin(self) -> Optional[str]:
        """The value to echo in Access-Control-Allow-Origin, if permitted."""
        origin = self.headers.get("Origin")
        if not origin or not self.allowed_origins:
            return None
        if "*" in self.allowed_origins:
            # Echo the caller rather than "*" so the policy still works if
            # credentialed requests are enabled later.
            return origin
        return origin if origin in self.allowed_origins else None

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        """CORS preflight. Browsers send this before a cross-origin POST."""
        origin = self._allowed_origin()
        if origin is None:
            # No CORS headers: the browser then blocks the real request.
            self.send_response(HTTPStatus.FORBIDDEN)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", CORS_METHODS)
        self.send_header("Access-Control-Allow-Headers", CORS_HEADERS)
        self.send_header("Access-Control-Max-Age", CORS_MAX_AGE)
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok",
                                       "version": __version__})
        elif route == "/":
            self._json(HTTPStatus.OK, self._index())
        elif route == "/v1/catalog":
            self._handle_catalog()
        else:
            self._error(HTTPStatus.NOT_FOUND, f"No route for {route}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route == "/v1/estimate":
            self._handle_estimate()
        elif route == "/v1/jobs":
            self._handle_launch()
        else:
            self._error(HTTPStatus.NOT_FOUND, f"No route for {route}")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _index(self) -> Dict[str, Any]:
        return {
            "service": "lightos",
            "version": __version__,
            "clouds": self.catalog.clouds(),
            "accelerators": self.catalog.accelerators(),
            "endpoints": {
                "GET /healthz": "liveness probe",
                "GET /v1/catalog": "query capacity and prices",
                "POST /v1/estimate": "price a blueprint without provisioning",
                "POST /v1/jobs": "launch a blueprint (requires bearer token)",
            },
            "job_submission_enabled": bool(self.api_token),
            "allowed_origins": list(self.allowed_origins),
        }

    def _handle_catalog(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        accelerator = _first(params, "accelerator")
        precision = _first(params, "precision") or "bf16"
        try:
            gpus = int(_first(params, "gpus") or 1)
            hours = float(_first(params, "hours") or 1.0)
            min_vram = _first(params, "min_vram_gb")
            min_vram = int(min_vram) if min_vram else None
            model_size = _first(params, "model_size")
            model_size = float(model_size) if model_size else None
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST,
                        "'gpus', 'hours', 'min_vram_gb' and 'model_size' "
                        "must be numbers.")
            return

        # model_size is a convenience for frontends: it converts a parameter
        # count into the VRAM floor that actually drives selection.
        vram_from_model = None
        if model_size:
            try:
                vram_from_model = vram_for_model(model_size, precision)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            min_vram = max(min_vram or 0, vram_from_model)

        matches = self.catalog.find_matches(
            accelerator, gpus,
            clouds=params.get("cloud"),
            region=_first(params, "region"),
            min_vram_gb=min_vram)
        enriched = [dict(m, total_vram_gb=total_vram_gb(m),
                         projected_cost=estimate_cost(m, hours),
                         projected_hours=hours) for m in matches]
        self._json(HTTPStatus.OK, {
            "query": {"accelerator": accelerator, "gpus": gpus,
                      "cloud": params.get("cloud"),
                      "region": _first(params, "region"),
                      "min_vram_gb": min_vram,
                      "model_size": model_size,
                      "precision": precision if model_size else None,
                      "described": describe_request(accelerator, gpus,
                                                    min_vram)},
            "vram_required_gb": vram_from_model,
            "count": len(matches),
            "matches": enriched,
            # The same enriched shape as the list, so a frontend can render
            # either without special-casing the fields.
            "cheapest": enriched[0] if enriched else None,
            # Surfaced so a frontend can show a staleness banner rather than
            # presenting old prices as current.
            "stale_prices": len(self.catalog.stale_entries()),
        })

    def _handle_estimate(self) -> None:
        task = self._read_task()
        if task is None:
            return
        try:
            offer = self.catalog.find_cheapest_match(
                task.accelerator, task.gpu_count, task.clouds, task.region)
        except CatalogError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
            return
        self._json(HTTPStatus.OK, {
            "task": task.name,
            "hardware": (f"{task.gpu_count}x {task.accelerator}"
                         if task.accelerator else "CPU-only"),
            "selected": offer,
            "hourly_cost": offer["hourly_cost"],
            "provisioned": False,
        })

    def _handle_launch(self) -> None:
        if not self.api_token:
            self._error(HTTPStatus.FORBIDDEN,
                        f"Job submission is disabled. Set {TOKEN_ENV} to "
                        "enable it.")
            return
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED,
                        "Provide a valid 'Authorization: Bearer <token>'.")
            return
        task = self._read_task()
        if task is None:
            return

        # keep_blank_values so a bare `?dry_run` counts as true.
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        dry_run = _truthy(query, "dry_run")
        try:
            result = self.orchestrator.launch(task, dry_run=dry_run)
        except CatalogError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
            return

        status = HTTPStatus.OK if result.succeeded or dry_run \
            else HTTPStatus.BAD_GATEWAY
        self._json(status, {
            "job_id": result.job_id,
            "task": result.task_name,
            "succeeded": result.succeeded,
            "exit_code": result.exit_code,
            "instance_id": result.instance_id,
            "selected": result.offer,
            "duration_seconds": round(result.duration_seconds, 2),
            "estimated_cost": result.estimated_cost,
            "files_synced": result.files_synced,
            "terminated": result.terminated,
            "error": result.error,
            "dry_run": dry_run,
        })

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------
    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            return False
        # Constant-time so the endpoint does not leak the token byte by byte.
        return hmac.compare_digest(presented.strip(), self.api_token)

    def _read_body(self) -> Optional[bytes]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length.")
            return None
        if length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "Request body is required.")
            return None
        if length > MAX_BODY_BYTES:
            self._drain(min(length, MAX_DRAIN_BYTES))
            # Past the drain cap the request body is still arriving, so the
            # connection cannot be reused whatever we reply.
            self.close_connection = True
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"Body exceeds {MAX_BODY_BYTES} bytes.")
            return None
        return self.rfile.read(length)

    def _drain(self, remaining: int) -> None:
        """Read and discard up to `remaining` bytes of the request body."""
        while remaining > 0:
            chunk = self.rfile.read(min(_DRAIN_CHUNK, remaining))
            if not chunk:
                return
            remaining -= len(chunk)

    def _read_task(self) -> Optional[Task]:
        """Parse a blueprint from a JSON or YAML request body."""
        body = self._read_body()
        if body is None:
            return None
        try:
            # YAML is a JSON superset, so one parser covers both bodies.
            config = yaml.safe_load(body.decode("utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"Unparseable body: {exc}")
            return None
        try:
            return Task.from_dict(config)
        except TaskError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return None

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    def _json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self._allowed_origin()
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
        # Always vary: the response differs by Origin even when it is denied,
        # so a shared cache must not reuse one origin's response for another.
        self.send_header("Vary", "Origin")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message, "status": int(status)})

    def log_message(self, fmt: str, *args: Any) -> None:
        # One line per request on stdout, where the container collects logs.
        # `fmt` already quotes the request line, so do not re-quote it.
        print(f"[api] {self.address_string()} {fmt % args}", flush=True)


def _first(params: Dict[str, Any], key: str) -> Optional[str]:
    values = params.get(key)
    return values[0] if values else None


def _truthy(params: Dict[str, Any], key: str) -> bool:
    value = _first(params, key)
    return value is not None and value.lower() in ("1", "true", "yes", "")


def parse_origins(raw: Optional[str]) -> tuple:
    """Split the configured origin list, dropping blanks and trailing slashes.

    A browser's Origin header never has a trailing slash, so normalize here
    rather than silently failing to match a value a user typed with one.
    """
    if not raw:
        return ()
    return tuple(part.strip().rstrip("/") for part in raw.split(",")
                 if part.strip())


def build_server(host: str = "0.0.0.0", port: int = 8080,
                 catalog: Optional[CostCatalog] = None,
                 orchestrator: Optional[Orchestrator] = None,
                 api_token: Optional[str] = None,
                 allowed_origins: Optional[Sequence[str]] = None
                 ) -> ThreadingHTTPServer:
    catalog = catalog or CostCatalog()
    origins = (tuple(allowed_origins) if allowed_origins is not None
               else parse_origins(os.environ.get(ORIGINS_ENV)))
    handler = type("BoundLightOSHandler", (LightOSHandler,), {
        "catalog": catalog,
        "orchestrator": orchestrator or Orchestrator(catalog=catalog),
        "api_token": api_token if api_token is not None
                     else os.environ.get(TOKEN_ENV) or None,
        "allowed_origins": origins,
    })
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    httpd = build_server(host, port)
    handler = httpd.RequestHandlerClass
    token_state = ("enabled" if handler.api_token
                   else f"disabled (set {TOKEN_ENV} to enable)")
    origins = handler.allowed_origins
    cors_state = (", ".join(origins) if origins
                  else f"none (set {ORIGINS_ENV} for a browser frontend)")
    print(f"[lightos] serving on http://{host}:{port}", flush=True)
    print(f"[lightos] job submission: {token_state}", flush=True)
    print(f"[lightos] allowed browser origins: {cors_state}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[lightos] shutting down.", flush=True)
    finally:
        httpd.server_close()


def generate_token() -> str:
    """Convenience for operators bootstrapping LIGHTOS_API_TOKEN."""
    return secrets.token_urlsafe(32)


if __name__ == "__main__":
    serve(host=os.environ.get("LIGHTOS_HOST", "0.0.0.0"),
          port=int(os.environ.get("PORT", "8080")))
