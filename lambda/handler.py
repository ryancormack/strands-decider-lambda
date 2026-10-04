"""Lambda Function URL handler serving Strands Decider over the Jev System One API.

POST /v1/systemone   Jev request in, Jev response out
GET  /health         served model and runtime details
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time
from typing import Any

from pydantic import ValidationError

from strands_decider.schema import SystemOneRequest

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_DIR = os.environ.get("DECIDER_MODEL_DIR", "/model/checkpoint")
MODEL_NAME = os.environ.get("DECIDER_MODEL_NAME", "strands-decider-2B-hobson-v19")
SYSTEMONE_PATH = "/v1/systemone"
HEALTH_PATH = "/health"
JSON_HEADERS = {"content-type": "application/json"}

_engine: Any = None


def _load_engine() -> Any:
    import torch
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderModel

    threads = int(os.environ.get("DECIDER_NUM_THREADS") or 0) or (os.cpu_count() or 1)
    torch.set_num_threads(threads)

    started = time.perf_counter()
    # SystemOneEngine upcasts the bf16 torso to fp32 on CPU, which must happen under
    # inference_mode because the model is built there.
    with torch.inference_mode():
        model = StrandsDeciderModel.load(MODEL_DIR)
        engine = SystemOneEngine(model, EngineConfig(device="cpu", model_name=MODEL_NAME))
    logger.info(
        json.dumps({"event": "model_loaded", "seconds": round(time.perf_counter() - started, 2), "threads": threads})
    )
    return engine


def get_engine() -> Any:
    global _engine
    if _engine is None:
        _engine = _load_engine()
    return _engine


def _response(status: int, body: Any, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {**JSON_HEADERS, **(headers or {})},
        "body": json.dumps(body),
    }


def _error(status: int, detail: Any) -> dict[str, Any]:
    return _response(status, {"detail": detail})


def _read_body(event: dict[str, Any]) -> str:
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        try:
            return base64.b64decode(body, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError("body is not valid base64-encoded UTF-8") from exc
    return str(body)


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


def handle_systemone(event: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(_read_body(event))
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(422, [{"loc": ["body"], "msg": f"invalid JSON: {exc}", "type": "json_invalid"}])
    if not isinstance(payload, dict):
        return _error(422, [{"loc": ["body"], "msg": "request body must be a JSON object", "type": "model_type"}])

    # The Jev API requires `model`; the upstream schema defaults it.
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        return _error(422, [{"loc": ["body", "model"], "msg": "Field required", "type": "missing"}])

    try:
        request = SystemOneRequest.model_validate(payload)
    except ValidationError as exc:
        return _error(422, _validation_detail(exc))

    started = time.perf_counter()
    try:
        result = get_engine().evaluate(request)
    except ValueError as exc:
        return _error(422, str(exc))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    return _response(200, result.model_dump(), {"x-decider-latency-ms": str(elapsed_ms)})


def handle_health() -> dict[str, Any]:
    engine = get_engine()
    config = engine.model.config
    return _response(
        200,
        {
            "status": "ok",
            "model": engine.cfg.model_name,
            "base_model": os.path.basename(config.base_model.rstrip("/")),
            "device": engine.cfg.device,
            "max_length": config.max_length,
            "prefix_cache": engine.cfg.use_prefix_cache,
        },
    )


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    http = (event.get("requestContext") or {}).get("http") or {}
    method = str(http.get("method") or "").upper()
    path = str(event.get("rawPath") or http.get("path") or "")
    if len(path) > 1:
        path = path.rstrip("/")

    if path == SYSTEMONE_PATH:
        if method != "POST":
            return _response(405, {"detail": "Method Not Allowed"}, {"allow": "POST"})
        return handle_systemone(event)
    if path == HEALTH_PATH:
        if method != "GET":
            return _response(405, {"detail": "Method Not Allowed"}, {"allow": "GET"})
        return handle_health()
    return _error(404, "Not Found")


def prefetch_model(root: str = "/model", workers: int = 32, chunk: int = 16 << 20) -> None:
    """Read every model file in parallel so the lazily-fetched image chunks arrive concurrently.

    Page faults from mmap'd safetensors fetch one chunk at a time, which took minutes.
    """
    from concurrent.futures import ThreadPoolExecutor

    spans = []
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in names:
            path = os.path.join(dirpath, name)
            size = os.path.getsize(path)
            spans.extend((path, off) for off in range(0, size, chunk))

    def read(span: tuple[str, int]) -> int:
        path, off = span
        try:
            with open(path, "rb", buffering=0) as f:
                f.seek(off)
                return len(f.read(chunk))
        except OSError:
            return 0

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        total = sum(pool.map(read, spans))
    logger.info(
        json.dumps({"event": "model_prefetched", "gib": round(total / 2**30, 2), "seconds": round(time.perf_counter() - started, 2)})
    )


# Runs once per sandbox at import. The load outlasts Lambda's 10 s init cap, so the first
# invoke absorbs the rest. Tests turn this off.
if os.environ.get("DECIDER_EAGER_LOAD", "1") == "1":
    prefetch_model()
    get_engine()
