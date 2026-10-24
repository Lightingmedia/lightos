"""HTTP API for LightOS - what runs behind agentic.lightos.sh.

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
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import yaml

from . import __version__
from .catalog import CatalogError, CostCatalog, estimate_cost
from .orchestrator import Orchestrator
from .task import Task, TaskError

TOKEN_ENV = "LIGHTOS_API_TOKEN"
MAX_BODY_BYTES = 256 * 1024


class LightOSHandler(BaseHTTPRequestHandler):
    server_version = f"LightOS/{__version__}"

    # Injected by serve().
    catalog: CostCatalog = None
    orchestrator: Orchestrator = None
    api_token: Optional[str] = None

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
        }

    def _handle_catalog(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        accelerator = _first(params, "accelerator")
        try:
            gpus = int(_first(params, "gpus") or 1)
            hours = float(_first(params, "hours") or 1.0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST,
                        "'gpus' and 'hours' must be numbers.")
            return

        matches = self.catalog.find_matches(
            accelerator, gpus,
            clouds=params.get("cloud"),
            region=_first(params, "region"))
        self._json(HTTPStatus.OK, {
            "query": {"accelerator": accelerator, "gpus": gpus,
                      "cloud": params.get("cloud"),
                      "region": _first(params, "region")},
            "count": len(matches),
            "matches": [
                dict(m, projected_cost=estimate_cost(m, hours),
                     projected_hours=hours) for m in matches
            ],
            "cheapest": matches[0] if matches else None,
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
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"Body exceeds {MAX_BODY_BYTES} bytes.")
            return None
        return self.rfile.read(length)

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


def build_server(host: str = "0.0.0.0", port: int = 8080,
                 catalog: Optional[CostCatalog] = None,
                 orchestrator: Optional[Orchestrator] = None,
                 api_token: Optional[str] = None) -> ThreadingHTTPServer:
    catalog = catalog or CostCatalog()
    handler = type("BoundLightOSHandler", (LightOSHandler,), {
        "catalog": catalog,
        "orchestrator": orchestrator or Orchestrator(catalog=catalog),
        "api_token": api_token if api_token is not None
                     else os.environ.get(TOKEN_ENV) or None,
    })
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    httpd = build_server(host, port)
    token_state = ("enabled" if httpd.RequestHandlerClass.api_token
                   else f"disabled (set {TOKEN_ENV} to enable)")
    print(f"[lightos] serving on http://{host}:{port} "
          f"(job submission {token_state})", flush=True)
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
