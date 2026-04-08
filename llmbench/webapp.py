from __future__ import annotations

import html
import ipaddress
import json
import re
import secrets
import shlex
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import web

from llmbench.exporters import normalize_records, records_to_csv, records_to_jsonl
from llmbench.vllm_runner import JobManager

JOBS_KEY = web.AppKey("jobs", JobManager)
DEFAULT_VLLM_BINARY_KEY = web.AppKey("default_vllm_binary", str)
JOB_OWNERS_KEY = web.AppKey("job_owners", dict[str, str])
JOB_OWNER_SESSIONS_KEY = web.AppKey("job_owner_sessions", dict[str, str])
JOB_OWNER_TABS_KEY = web.AppKey("job_owner_tabs", dict[str, str])
RECOVERABLE_JOBS_KEY = web.AppKey("recoverable_jobs", set[str])
JOB_LAST_TOUCH_KEY = web.AppKey("job_last_touch", dict[str, float])
SESSION_COOKIE_NAME = "llmbench_session"
SESSION_ID_KEY = "llmbench.session_id"
BROWSER_ID_HEADER = "X-llmbench-browser-id"
BROWSER_ID_KEY = "llmbench.browser_id"
TAB_ID_HEADER = "X-llmbench-tab-id"
TAB_ID_KEY = "llmbench.tab_id"
TAB_INSTANCE_HEADER = "X-llmbench-tab-instance-id"
TAB_INSTANCE_KEY = "llmbench.tab_instance_id"
JOB_OWNER_INSTANCES_KEY = web.AppKey("job_owner_instances", dict[str, str])
DISALLOWED_WEB_CAPTURE_FLAGS = {"--result-dir", "--result-filename", "--output-dir", "-o"}
RECOVERY_LEASE_TIMEOUT_SECONDS = 30.0
SAFE_DEFAULT_SUBCOMMANDS = {"throughput"}
SAFE_DEFAULT_GPU_MEMORY_UTILIZATION = "0.60"
SAFE_DEFAULT_MAX_MODEL_LEN = "1024"
BROWSER_VISIBLE_VALUE_OVERRIDES = {
    "--base-url": "browser-configured-base-url",
    "--model": "browser-configured-model",
    "--tokenizer": "browser-configured-tokenizer",
}
SENSITIVE_MODEL_TOKENIZER_FIELD_TOKENS = {"model", "tokenizer"}
SENSITIVE_MODEL_TOKENIZER_FLAGS = {"--model", "--tokenizer"}
SENSITIVE_BASE_URL_FLAG = "--base-url"
FIELD_NAME_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
BEARER_INLINE_TOKEN_RE = re.compile(r"(?i)\b(Bearer)(\s+)([^\s'\"`]+)")
AUTHORIZATION_BEARER_SUFFIX_RE = re.compile(r"(?i)authorization\s*:\s*bearer$")
REDACTED_BEARER_TOKEN = "<redacted-bearer-token>"
ANOTHER_TAB_JOB_MESSAGE = "This benchmark is still active in another tab for this browser profile. Return to that tab to inspect or stop it."


def create_app(*, artifact_root: Path, default_vllm_binary: str) -> web.Application:
    app = web.Application(middlewares=[_session_middleware])
    app[JOBS_KEY] = JobManager(artifact_root=artifact_root, default_vllm_binary=default_vllm_binary)
    app[DEFAULT_VLLM_BINARY_KEY] = default_vllm_binary
    app[JOB_OWNERS_KEY] = {}
    app[JOB_OWNER_SESSIONS_KEY] = {}
    app[JOB_OWNER_TABS_KEY] = {}
    app[JOB_OWNER_INSTANCES_KEY] = {}
    app[RECOVERABLE_JOBS_KEY] = set()
    app[JOB_LAST_TOUCH_KEY] = {}
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/session/release-owned-jobs", handle_release_owned_jobs)
    app.router.add_get("/api/jobs", handle_list_jobs)
    app.router.add_post("/api/jobs", handle_start_job)
    app.router.add_post("/api/jobs/release-owned", handle_release_owned_jobs)
    app.router.add_get("/api/jobs/{job_id}", handle_get_job)
    app.router.add_post("/api/jobs/{job_id}/stop", handle_stop_job)
    app.router.add_post("/api/jobs/{job_id}/release", handle_release_job)
    app.router.add_get("/api/jobs/{job_id}/export.csv", handle_export_csv)
    app.router.add_get("/api/jobs/{job_id}/export.jsonl", handle_export_jsonl)
    return app


async def handle_index(request: web.Request) -> web.Response:
    page = INDEX_HTML.replace("__SERVER_BINARY_LABEL__", html.escape("Server-configured vLLM binary", quote=True))
    page = page.replace("__SERVER_BINARY_PREVIEW__", json.dumps("server-configured-vllm"))
    page = page.replace("__BROWSER_VISIBLE_VALUE_OVERRIDES__", json.dumps(BROWSER_VISIBLE_VALUE_OVERRIDES))
    return web.Response(text=page, content_type="text/html")


async def handle_start_job(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        payload = await _prepare_web_payload(payload)
        bench_path = build_bench_path(payload)
        raw_args = build_web_args(payload, bench_path=bench_path)
    except ValueError as exc:
        return _json_error(str(exc), status=400)
    manager: JobManager = request.app[JOBS_KEY]
    execution = await manager.start_job(
        bench_path=bench_path,
        raw_args=raw_args,
    )
    request.app[JOB_OWNERS_KEY][execution.job_id] = _request_browser_id(request)
    request.app[JOB_OWNER_SESSIONS_KEY][execution.job_id] = _request_session_id(request)
    _bind_job_to_request_identity(request, execution.job_id)
    request.app[RECOVERABLE_JOBS_KEY].discard(execution.job_id)
    _touch_job(request, execution.job_id)
    return web.json_response(_public_job_payload(execution))


async def _prepare_web_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(payload)
    tokenizer = str(prepared.get("tokenizer", "") or "").strip()
    backend = str(prepared.get("backend", "") or "").strip()
    model = str(prepared.get("model", "") or "").strip()
    if backend != "openai" or tokenizer or not model:
        return prepared
    raise ValueError(
        "Tokenizer is required when using backend=openai with a served model alias. "
        "Enter an explicit tokenizer/model path, or leave Model blank so llmbench can defer to the server "
        "without probing the Base URL."
    )


def _effective_endpoint(payload: dict[str, Any]) -> str:
    endpoint = str(payload.get("endpoint", "") or "").strip()
    if endpoint:
        return endpoint
    if str(payload.get("backend", "") or "").strip() == "openai":
        return "/v1/completions"
    return ""


async def handle_list_jobs(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    owners = request.app[JOB_OWNERS_KEY]
    recoverable = request.app[RECOVERABLE_JOBS_KEY]
    jobs = manager.list_jobs()
    for job in jobs:
        _maybe_mark_job_recoverable_after_owner_timeout(request, job)
        if _session_owns_job(request, job.job_id):
            _touch_job(request, job.job_id)
    jobs = [job for job in jobs if _session_can_access_job(request, job.job_id)]
    status_filter_raw = request.query.get("status", "")
    status_filter = {item.strip() for item in status_filter_raw.split(",") if item.strip()}
    if status_filter:
        jobs = [job for job in jobs if job.status in status_filter]
    recoverable_only = request.query.get("recoverable_only", "").strip().lower() in {"1", "true", "yes"}
    if recoverable_only:
        jobs = [job for job in jobs if _session_can_recover_job(request, job.job_id)]
    limit_raw = request.query.get("limit", "20").strip()
    limit = 20
    if limit_raw:
        try:
            limit = max(1, int(limit_raw))
        except ValueError:
            return _json_error("Invalid limit query parameter", status=400)
    jobs = jobs[:limit]
    return web.json_response(
        {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "returncode": job.returncode,
                    "record_count": len(job.records),
                    "recoverable": job.job_id in recoverable,
                }
                for job in jobs
            ]
        }
    )


async def handle_get_job(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        return _json_error("Unknown job id", status=404)
    _maybe_mark_job_recoverable_after_owner_timeout(request, execution)
    if not _session_can_access_job(request, execution.job_id):
        return _job_access_error(request, execution.job_id)
    _adopt_recoverable_job(request, execution.job_id)
    _touch_job(request, execution.job_id)
    payload = _public_job_payload(execution)
    payload["rows"] = _sanitize_public_record_list(normalize_records(execution.records))
    payload["columns"] = sorted({key for row in payload["rows"] for key in row.keys()})
    return web.json_response(payload)


async def handle_release_owned_jobs(request: web.Request) -> web.Response:
    released_job_ids = _release_owned_jobs(request)
    return web.json_response(
        {
            "released_job_ids": released_job_ids,
            "recoverable_count": len(released_job_ids),
        }
    )


async def handle_stop_job(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        return _json_error("Unknown job id", status=404)
    _maybe_mark_job_recoverable_after_owner_timeout(request, execution)
    if not _session_can_access_job(request, execution.job_id):
        return _job_access_error(request, execution.job_id)
    _adopt_recoverable_job(request, execution.job_id)
    _touch_job(request, execution.job_id)
    await manager.stop_job(execution.job_id)
    refreshed = manager.get_job(execution.job_id)
    if refreshed is None:
        return _json_error("Unknown job id", status=404)
    payload = _public_job_payload(refreshed)
    payload["rows"] = _sanitize_public_record_list(normalize_records(refreshed.records))
    payload["columns"] = sorted({key for row in payload["rows"] for key in row.keys()})
    return web.json_response(payload)


async def handle_release_job(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        return _json_error("Unknown job id", status=404)
    if not _session_owns_job(request, request.match_info["job_id"]):
        return _job_access_error(request, request.match_info["job_id"])
    _mark_job_recoverable(request, execution.job_id)
    return web.json_response({"job_id": execution.job_id, "recoverable": execution.job_id in request.app[RECOVERABLE_JOBS_KEY]})


async def handle_export_csv(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        raise web.HTTPNotFound(text="Unknown job id")
    _maybe_mark_job_recoverable_after_owner_timeout(request, execution)
    if not _session_can_access_job(request, execution.job_id):
        raise _job_access_http_error(request, execution.job_id)
    _adopt_recoverable_job(request, execution.job_id)
    _touch_job(request, execution.job_id)
    records = manager.export_records(request.match_info["job_id"])
    body = records_to_csv(_sanitize_public_record_list(records))
    filename = f"llmbench-{execution.job_id}.csv"
    return web.Response(
        text=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        content_type="text/csv",
    )


async def handle_export_jsonl(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        raise web.HTTPNotFound(text="Unknown job id")
    _maybe_mark_job_recoverable_after_owner_timeout(request, execution)
    if not _session_can_access_job(request, execution.job_id):
        raise _job_access_http_error(request, execution.job_id)
    _adopt_recoverable_job(request, execution.job_id)
    _touch_job(request, execution.job_id)
    records = manager.export_records(request.match_info["job_id"])
    body = records_to_jsonl(_sanitize_public_record_list(records))
    filename = f"llmbench-{execution.job_id}.jsonl"
    return web.Response(
        text=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        content_type="application/x-ndjson",
    )


def build_web_args(payload: dict[str, Any], *, bench_path: list[str] | None = None) -> list[str]:
    payload = _apply_structured_launch_defaults(payload, bench_path)
    args: list[str] = []
    use_random_length_flags = _uses_random_length_flags(payload, bench_path)
    use_structured_low_memory_flags = _uses_structured_low_memory_launch_flags(payload, bench_path)
    mappings = [
        ("model", "--model"),
        ("backend", "--backend"),
        ("base_url", "--base-url"),
        ("host", "--host"),
        ("port", "--port"),
        ("dataset_name", "--dataset-name"),
        ("dataset_path", "--dataset-path"),
        ("tokenizer", "--tokenizer"),
        ("num_prompts", "--num-prompts"),
        ("request_rate", "--request-rate"),
        ("max_concurrency", "--max-concurrency"),
        ("percentile_metrics", "--percentile-metrics"),
        ("metric_percentiles", "--metric-percentiles"),
    ]
    for key, flag in mappings:
        value = payload.get(key)
        if value not in (None, ""):
            text_value = str(value)
            if key in {"model", "tokenizer"}:
                text_value = _expand_web_local_model_or_tokenizer_path(text_value)
            args.extend([flag, text_value])
    if use_structured_low_memory_flags:
        for key, flag in [
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
            ("max_model_len", "--max-model-len"),
        ]:
            value = payload.get(key)
            if value not in (None, ""):
                args.extend([flag, str(value)])
    endpoint = _effective_endpoint(payload)
    if endpoint:
        args.extend(["--endpoint", endpoint])
    length_flags = [
        ("input_len", "--random-input-len" if use_random_length_flags else "--input-len"),
        ("output_len", "--random-output-len" if use_random_length_flags else "--output-len"),
    ]
    for key, flag in length_flags:
        value = payload.get(key)
        if value not in (None, ""):
            args.extend([flag, str(value)])
    if use_structured_low_memory_flags and payload.get("enforce_eager"):
        args.append("--enforce-eager")
    extra_args = payload.get("extra_args", "")
    if extra_args:
        parsed_extra_args = shlex.split(extra_args)
        _validate_web_extra_args(parsed_extra_args)
        parsed_extra_args = _expand_web_local_model_tokenizer_flags_in_tokens(parsed_extra_args)
        args.extend(parsed_extra_args)
    return args


def _apply_structured_launch_defaults(payload: dict[str, Any], bench_path: list[str] | None) -> dict[str, Any]:
    normalized = dict(payload)
    if _uses_structured_low_memory_launch_flags(normalized, bench_path):
        if str(normalized.get("gpu_memory_utilization", "") or "").strip() == "":
            normalized["gpu_memory_utilization"] = SAFE_DEFAULT_GPU_MEMORY_UTILIZATION
        if str(normalized.get("max_model_len", "") or "").strip() == "":
            normalized["max_model_len"] = SAFE_DEFAULT_MAX_MODEL_LEN
        if "enforce_eager" not in normalized:
            normalized["enforce_eager"] = True
    return normalized


def _uses_structured_low_memory_launch_flags(payload: dict[str, Any], bench_path: list[str] | None) -> bool:
    top_level = ""
    if bench_path:
        top_level = str(bench_path[0] or "").strip()
    if not top_level:
        top_level = str(payload.get("subcommand", "") or "").strip()
    return top_level in SAFE_DEFAULT_SUBCOMMANDS


def _uses_random_length_flags(payload: dict[str, Any], bench_path: list[str] | None) -> bool:
    return bool(
        bench_path
        and bench_path[0] == "throughput"
        and str(payload.get("dataset_name", "") or "").strip() == "random"
    )


def _validate_web_extra_args(tokens: list[str]) -> None:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in DISALLOWED_WEB_CAPTURE_FLAGS:
            raise ValueError(
                "Web mode does not allow overriding result destinations "
                "(--result-dir, --result-filename, --output-dir, -o)."
            )
        if token.startswith("--result-dir=") or token.startswith("--result-filename=") or token.startswith("--output-dir="):
            raise ValueError(
                "Web mode does not allow overriding result destinations "
                "(--result-dir, --result-filename, --output-dir, -o)."
            )
        if token == "--output-json":
            if index + 1 >= len(tokens):
                raise ValueError("--output-json requires a value.")
            if tokens[index + 1] != "-":
                raise ValueError("Web mode only allows --output-json - in Additional Raw vLLM Args.")
            index += 2
            continue
        if token.startswith("--output-json="):
            if token.split("=", 1)[1] != "-":
                raise ValueError("Web mode only allows --output-json - in Additional Raw vLLM Args.")
            index += 1
            continue
        if token == "-o" and index + 1 < len(tokens):
            raise ValueError(
                "Web mode does not allow overriding result destinations "
                "(--result-dir, --result-filename, --output-dir, -o)."
            )
        if token.startswith("-o=") or (token.startswith("-o") and len(token) > 2 and token[2] != "="):
            raise ValueError(
                "Web mode does not allow overriding result destinations "
                "(--result-dir, --result-filename, --output-dir, -o)."
            )
        index += 1


def _expand_web_local_model_or_tokenizer_path(value: str) -> str:
    text = str(value).strip()
    if not text.startswith("~"):
        return text
    try:
        return str(Path(text).expanduser())
    except RuntimeError:
        return text


def _expand_web_local_model_tokenizer_flags_in_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        if token.startswith("--model=") or token.startswith("--tokenizer="):
            flag, _, value = token.partition("=")
            expanded.append(f"{flag}={_expand_web_local_model_or_tokenizer_path(value)}")
            index += 1
            continue
        if token in SENSITIVE_MODEL_TOKENIZER_FLAGS and index + 1 < len(tokens):
            expanded.append(token)
            expanded.append(_expand_web_local_model_or_tokenizer_path(str(tokens[index + 1])))
            index += 2
            continue
        expanded.append(token)
        index += 1
    return expanded


def build_bench_path(payload: dict[str, Any]) -> list[str]:
    override = str(payload.get("bench_path_override", "") or "").strip()
    if override:
        bench_path = shlex.split(override)
        if not bench_path:
            raise ValueError("Bench path override cannot be empty.")
        _validate_web_bench_path(bench_path)
        return bench_path

    subcommand = str(payload.get("subcommand", "serve") or "serve").strip()
    if not subcommand:
        raise ValueError("A vLLM bench subcommand is required.")

    bench_path = [subcommand]
    sweep_subcommand = str(payload.get("sweep_subcommand", "") or "").strip()
    if subcommand == "sweep" and sweep_subcommand:
        bench_path.append(sweep_subcommand)
    _validate_web_bench_path(bench_path)
    return bench_path


def _validate_web_bench_path(bench_path: list[str]) -> None:
    if not bench_path:
        raise ValueError("A vLLM bench subcommand is required.")
    if any(token == "--" or token.startswith("-") for token in bench_path):
        raise ValueError("Bench path override cannot include raw vLLM options.")


def _json_error(message: str, *, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _public_job_payload(execution) -> dict[str, Any]:
    payload = execution.to_dict()
    # Keep host-side binary and filesystem paths server-only.
    payload.pop("command", None)
    payload.pop("artifact_dir", None)
    payload.pop("result_path", None)
    if isinstance(payload.get("raw_args"), list):
        payload["raw_args"] = _sanitize_public_raw_args(payload["raw_args"])
    if isinstance(payload.get("records"), list):
        payload["records"] = _sanitize_public_record_list(payload["records"])
    if isinstance(payload.get("records_preview"), list):
        payload["records_preview"] = _sanitize_public_record_list(payload["records_preview"])
    if isinstance(payload.get("stdout"), str):
        payload["stdout"] = _sanitize_public_stdout(payload["stdout"])
    if isinstance(payload.get("stderr"), str):
        payload["stderr"] = _sanitize_public_stderr(payload["stderr"])
    return payload


def _sanitize_public_record_list(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _sanitize_public_record_values(records)


def _sanitize_public_raw_args(raw_args: list[Any]) -> list[str]:
    tokens = [str(token) for token in raw_args]
    sanitized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--model=") or token.startswith("--tokenizer=") or token.startswith("--base-url="):
            flag, _, value = token.partition("=")
            sanitized_value = _sanitize_public_cli_flag_value(flag, value)
            if sanitized_value != value:
                sanitized.append(f"{flag}={sanitized_value}")
            else:
                sanitized.append(token)
            index += 1
            continue
        if token in (*SENSITIVE_MODEL_TOKENIZER_FLAGS, SENSITIVE_BASE_URL_FLAG) and index + 1 < len(tokens):
            sanitized.append(token)
            raw_value = tokens[index + 1]
            sanitized.append(_sanitize_public_cli_flag_value(token, raw_value))
            index += 2
            continue
        sanitized.append(token)
        index += 1
    return _sanitize_public_bearer_tokens(sanitized)


def _sanitize_public_cli_flag_value(flag: str, value: str) -> str:
    if flag in SENSITIVE_MODEL_TOKENIZER_FLAGS and _looks_like_host_local_path(value):
        return _sanitize_public_model_tokenizer_value(flag, value)
    if flag == SENSITIVE_BASE_URL_FLAG and _looks_like_url(value):
        return _sanitize_public_base_url_value(value)
    return value


def _sanitize_public_record_values(value: Any, *, field_name: str = "") -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_public_record_values(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_public_record_values(item, field_name=field_name) for item in value]
    if isinstance(value, str):
        if _is_sensitive_model_tokenizer_field(field_name) and _looks_like_host_local_path(value):
            return _sanitize_public_model_tokenizer_value(field_name, value)
        if _is_sensitive_base_url_field(field_name) and _looks_like_url(value):
            return _sanitize_public_base_url_value(value)
    return value


def _is_sensitive_model_tokenizer_field(field_name: str) -> bool:
    if not field_name:
        return False
    return any(token in SENSITIVE_MODEL_TOKENIZER_FIELD_TOKENS for token in _tokenize_sensitive_field_name(field_name))


def _is_sensitive_base_url_field(field_name: str) -> bool:
    if not field_name:
        return False
    tokens = _tokenize_sensitive_field_name(field_name)
    return "base" in tokens and "url" in tokens


def _sanitize_public_model_tokenizer_value(flag_or_field_name: str, value: str) -> str:
    field_tokens = _tokenize_sensitive_field_name(flag_or_field_name)
    leaf = _sanitize_public_identifier_segment(_path_leaf_label(value), default="path")
    fingerprint = _short_public_fingerprint(value)
    if "model" in field_tokens:
        return f"{BROWSER_VISIBLE_VALUE_OVERRIDES['--model']}-{leaf}-{fingerprint}"
    if "tokenizer" in field_tokens:
        return f"{BROWSER_VISIBLE_VALUE_OVERRIDES['--tokenizer']}-{leaf}-{fingerprint}"
    return value


def _sanitize_public_base_url_value(value: str) -> str:
    scheme, host_class = _base_url_scheme_and_host_class(value)
    fingerprint = _short_public_fingerprint(value)
    return f"{BROWSER_VISIBLE_VALUE_OVERRIDES['--base-url']}-{scheme}-{host_class}-{fingerprint}"


def _base_url_scheme_and_host_class(value: str) -> tuple[str, str]:
    text = str(value).strip()
    parsed = urlsplit(text)
    scheme = _sanitize_public_identifier_segment(parsed.scheme.lower(), default="url")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return scheme, "unknown"
    if host == "localhost":
        return scheme, "loopback"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.endswith(".local"):
            return scheme, "local"
        return scheme, "hostname"
    if address.is_loopback:
        return scheme, "loopback"
    if address.is_private:
        return scheme, "private"
    return scheme, "public"


def _short_public_fingerprint(value: str) -> str:
    text = str(value).strip()
    encoded = text.encode("utf-16-le")
    fnv32 = 2166136261
    for index in range(0, len(encoded), 2):
        code_unit = int.from_bytes(encoded[index : index + 2], "little")
        fnv32 ^= code_unit
        fnv32 = (fnv32 * 16777619) & 0xFFFFFFFF
    return f"{fnv32:08x}"


def _path_leaf_label(value: str) -> str:
    trimmed = str(value).strip().rstrip("/")
    if not trimmed:
        return "path"
    parts = [part for part in trimmed.split("/") if part and part != "~"]
    if not parts:
        return "path"
    return parts[-1]


def _sanitize_public_identifier_segment(text: str, *, default: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", str(text).strip()).strip("-._")
    return value or default


def _looks_like_url(value: str) -> bool:
    text = str(value).strip()
    return bool(text and URL_SCHEME_RE.match(text))


def _looks_like_host_local_path(value: str) -> bool:
    text = str(value).strip()
    return text.startswith("/") or text.startswith("~/")


def _tokenize_sensitive_field_name(field_name: str) -> list[str]:
    normalized = field_name.strip()
    if normalized.startswith("--"):
        normalized = normalized[2:]
    leaf_name = normalized.rsplit(".", 1)[-1]
    underscore_normalized = re.sub(r"[^0-9A-Za-z]+", "_", leaf_name)
    tokens: list[str] = []
    for chunk in underscore_normalized.split("_"):
        if not chunk:
            continue
        tokens.extend(token.lower() for token in FIELD_NAME_TOKEN_RE.findall(chunk))
    return tokens


def _sanitize_public_stdout(stdout: str) -> str:
    if not stdout:
        return stdout
    return _redact_server_paths(_redact_public_runtime_details(stdout))


def _sanitize_public_stderr(stderr: str) -> str:
    if not stderr:
        return stderr
    lines: list[str] = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("Failed to launch vLLM binary "):
            lines.append("Failed to launch server-configured vLLM binary.")
            continue
        if stripped.startswith("Failed to prepare vLLM benchmark invocation:"):
            lines.append("Failed to prepare vLLM benchmark invocation.")
            continue
        if stripped.startswith("llmbench failed to load benchmark results:"):
            lines.append("llmbench failed to load benchmark results: server-side parsing failed.")
            continue
        lines.append(_redact_server_paths(_redact_public_runtime_details(line)))
    text = "\n".join(lines)
    if stderr.endswith("\n"):
        text += "\n"
    return text


def _redact_public_runtime_details(text: str) -> str:
    return re.sub(r"\b((?:tcp|udp|http|https)://)[^\s'\"`]+", r"\1<redacted>", text)


def _redact_server_paths(text: str) -> str:
    # Redact Unix-like absolute paths while preserving quoted text structure.
    return re.sub(r"(?<![:/A-Za-z0-9_.-])/(?:[^\s'\"`])+", "<server-path>", text)


def _sanitize_public_bearer_token_fragment(value: str) -> str:
    return BEARER_INLINE_TOKEN_RE.sub(r"\1\2" + REDACTED_BEARER_TOKEN, str(value))


def _token_has_bearer_prefix_without_token(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    text = text.strip("'\"")
    if text.lower() == "bearer":
        return True
    return bool(AUTHORIZATION_BEARER_SUFFIX_RE.search(text))


def _sanitize_public_bearer_tokens(tokens: list[str]) -> list[str]:
    sanitized: list[str] = []
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        sanitized_token = _sanitize_public_bearer_token_fragment(token)
        if _token_has_bearer_prefix_without_token(sanitized_token) and index + 1 < len(tokens):
            sanitized.append(sanitized_token)
            sanitized.append(REDACTED_BEARER_TOKEN)
            index += 2
            continue
        sanitized.append(sanitized_token)
        index += 1
    return sanitized


@web.middleware
async def _session_middleware(
    request: web.Request,
    handler: web.Handler,
) -> web.StreamResponse:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    created_session = False
    if not session_id:
        session_id = secrets.token_urlsafe(24)
        created_session = True
    request[SESSION_ID_KEY] = session_id
    browser_id = request.headers.get(BROWSER_ID_HEADER, "").strip() or session_id
    request[BROWSER_ID_KEY] = browser_id
    request[TAB_ID_KEY] = request.headers.get(TAB_ID_HEADER, "").strip()
    request[TAB_INSTANCE_KEY] = request.headers.get(TAB_INSTANCE_HEADER, "").strip()
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        response = exc
    if created_session:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            httponly=True,
            samesite="Lax",
        )
    return response


def _request_session_id(request: web.Request) -> str:
    session_id = request.get(SESSION_ID_KEY)
    if isinstance(session_id, str) and session_id:
        return session_id
    return ""


def _request_browser_id(request: web.Request) -> str:
    browser_id = request.get(BROWSER_ID_KEY)
    if isinstance(browser_id, str) and browser_id:
        return browser_id
    return _request_session_id(request)


def _request_tab_id(request: web.Request) -> str:
    tab_id = request.get(TAB_ID_KEY)
    if isinstance(tab_id, str) and tab_id:
        return tab_id
    return ""


def _request_tab_instance_id(request: web.Request) -> str:
    tab_instance_id = request.get(TAB_INSTANCE_KEY)
    if isinstance(tab_instance_id, str) and tab_instance_id:
        return tab_instance_id
    return ""


def _session_can_access_job(request: web.Request, job_id: str) -> bool:
    if _session_owns_job(request, job_id):
        return True
    return job_id in request.app[RECOVERABLE_JOBS_KEY] and _request_matches_job_identity(request, job_id)


def _job_access_error(request: web.Request, job_id: str) -> web.Response:
    if _job_owned_by_another_tab(request, job_id):
        return _json_error(ANOTHER_TAB_JOB_MESSAGE, status=409)
    return _json_error("Unknown job id", status=404)


def _job_access_http_error(request: web.Request, job_id: str) -> web.HTTPException:
    if _job_owned_by_another_tab(request, job_id):
        return web.HTTPConflict(text=ANOTHER_TAB_JOB_MESSAGE)
    return web.HTTPNotFound(text="Unknown job id")


def _job_owned_by_another_tab(request: web.Request, job_id: str) -> bool:
    if not _request_matches_job_identity(request, job_id):
        return False
    if job_id in request.app[RECOVERABLE_JOBS_KEY]:
        return False
    owner_tab = request.app[JOB_OWNER_TABS_KEY].get(job_id)
    request_tab = _request_tab_id(request)
    if owner_tab and request_tab and owner_tab != request_tab:
        return True
    owner_instance = request.app[JOB_OWNER_INSTANCES_KEY].get(job_id)
    request_instance = _request_tab_instance_id(request)
    if owner_instance and request_instance and owner_instance != request_instance:
        return True
    return False


def _session_can_recover_job(request: web.Request, job_id: str) -> bool:
    if job_id in request.app[RECOVERABLE_JOBS_KEY] and _request_matches_job_identity(request, job_id):
        return True
    owner_tab = request.app[JOB_OWNER_TABS_KEY].get(job_id)
    if owner_tab:
        return False
    return _session_owns_job(request, job_id)


def _session_owns_job(request: web.Request, job_id: str) -> bool:
    owner = request.app[JOB_OWNERS_KEY].get(job_id)
    if not owner:
        return False
    if not _request_matches_job_identity(request, job_id):
        return False
    owner_tab = request.app[JOB_OWNER_TABS_KEY].get(job_id)
    if not owner_tab:
        return True
    request_tab = _request_tab_id(request)
    if not request_tab or owner_tab != request_tab:
        return False
    owner_instance = request.app[JOB_OWNER_INSTANCES_KEY].get(job_id)
    if not owner_instance:
        return True
    request_instance = _request_tab_instance_id(request)
    if request_instance:
        return owner_instance == request_instance
    owner_session = request.app[JOB_OWNER_SESSIONS_KEY].get(job_id)
    request_session = _request_session_id(request)
    return bool(owner_session) and owner_session == request_session


def _adopt_recoverable_job(request: web.Request, job_id: str) -> None:
    if job_id not in request.app[RECOVERABLE_JOBS_KEY] and request.app[JOB_OWNER_TABS_KEY].get(job_id):
        return
    request.app[JOB_OWNERS_KEY][job_id] = _request_browser_id(request)
    request.app[JOB_OWNER_SESSIONS_KEY][job_id] = _request_session_id(request)
    _bind_job_to_request_identity(request, job_id)
    request.app[RECOVERABLE_JOBS_KEY].discard(job_id)


def _bind_job_to_request_identity(request: web.Request, job_id: str) -> None:
    tab_id = _request_tab_id(request)
    if tab_id:
        request.app[JOB_OWNER_TABS_KEY][job_id] = tab_id
    else:
        request.app[JOB_OWNER_TABS_KEY].pop(job_id, None)
    tab_instance_id = _request_tab_instance_id(request)
    if tab_instance_id:
        request.app[JOB_OWNER_INSTANCES_KEY][job_id] = tab_instance_id
    else:
        request.app[JOB_OWNER_INSTANCES_KEY].pop(job_id, None)


def _mark_job_recoverable(request: web.Request, job_id: str) -> None:
    request.app[RECOVERABLE_JOBS_KEY].add(job_id)
    request.app[JOB_OWNER_TABS_KEY].pop(job_id, None)
    request.app[JOB_OWNER_INSTANCES_KEY].pop(job_id, None)


def _release_owned_jobs(request: web.Request) -> list[str]:
    manager: JobManager = request.app[JOBS_KEY]
    released_job_ids: list[str] = []
    for execution in manager.list_jobs():
        if not _session_owns_job(request, execution.job_id):
            continue
        _mark_job_recoverable(request, execution.job_id)
        released_job_ids.append(execution.job_id)
    return released_job_ids


def _request_matches_job_identity(request: web.Request, job_id: str) -> bool:
    request_browser = _request_browser_id(request)
    request_session = _request_session_id(request)
    owner_browser = request.app[JOB_OWNERS_KEY].get(job_id)
    owner_session = request.app[JOB_OWNER_SESSIONS_KEY].get(job_id)
    browser_match = bool(owner_browser) and owner_browser == request_browser
    session_match = bool(owner_session) and owner_session == request_session
    return browser_match or session_match


def _touch_job(request: web.Request, job_id: str) -> None:
    request.app[JOB_LAST_TOUCH_KEY][job_id] = time.monotonic()


def _maybe_mark_job_recoverable_after_owner_timeout(request: web.Request, execution) -> None:
    if execution.job_id in request.app[RECOVERABLE_JOBS_KEY]:
        return
    if not _request_matches_job_identity(request, execution.job_id):
        return
    if not request.app[JOB_OWNER_TABS_KEY].get(execution.job_id):
        return
    if execution.status not in {"running", "stopping", "completed", "failed", "stopped"}:
        return
    last_touch = request.app[JOB_LAST_TOUCH_KEY].get(execution.job_id)
    if last_touch is None:
        return
    if (time.monotonic() - last_touch) < RECOVERY_LEASE_TIMEOUT_SECONDS:
        return
    _mark_job_recoverable(request, execution.job_id)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>llmbench | vLLM Bench Console</title>
  <style>
    :root {
      --paper: #f4efe6;
      --ink: #1f2629;
      --accent: #0d8ea6;
      --accent-soft: #c7eef4;
      --warm: #cf6c2c;
      --panel: rgba(255, 255, 255, 0.76);
      --border: rgba(31, 38, 41, 0.18);
      --shadow: 0 18px 40px rgba(31, 38, 41, 0.09);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(13, 142, 166, 0.18), transparent 28%),
        radial-gradient(circle at left 20%, rgba(207, 108, 44, 0.12), transparent 24%),
        linear-gradient(160deg, #f8f3ea 0%, var(--paper) 100%);
      min-height: 100vh;
    }
    .shell {
      width: min(1380px, calc(100vw - 32px));
      margin: 24px auto 48px;
      display: grid;
      gap: 18px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero {
      padding: 28px;
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      align-items: end;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 11px;
      color: var(--accent);
      margin-bottom: 10px;
    }
    h1 {
      margin: 0;
      font-size: clamp(2.4rem, 5vw, 4.7rem);
      line-height: 0.92;
      letter-spacing: -0.04em;
    }
    .hero p {
      max-width: 58ch;
      margin: 14px 0 0;
      font-size: 15px;
      line-height: 1.6;
    }
    .command-card {
      background: #11191d;
      color: #f0f7f8;
      border-radius: 18px;
      padding: 18px;
      font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
      font-size: 13px;
      line-height: 1.6;
      min-height: 180px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .command-card .label {
      display: block;
      margin-bottom: 10px;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(199, 238, 244, 0.78);
    }
    .layout {
      display: grid;
      grid-template-columns: 420px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      padding: 20px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: -0.03em;
    }
    form {
      display: grid;
      gap: 14px;
    }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(31, 38, 41, 0.76);
    }
    input, select, textarea {
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(31, 38, 41, 0.18);
      padding: 12px 14px;
      font: inherit;
      background: rgba(255, 255, 255, 0.82);
      color: var(--ink);
    }
    textarea {
      min-height: 116px;
      resize: vertical;
      font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
      font-size: 13px;
    }
    .checks {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .checks label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(13, 142, 166, 0.08);
      border: 1px solid rgba(13, 142, 166, 0.18);
      letter-spacing: 0.08em;
    }
    .checks input { width: auto; }
    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 4px 0 16px;
      position: sticky;
      top: 10px;
      z-index: 2;
      padding: 10px;
      border-radius: 18px;
      background: rgba(244, 239, 230, 0.92);
      border: 1px solid rgba(31, 38, 41, 0.12);
      backdrop-filter: blur(12px);
    }
    button, .export-link {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      letter-spacing: 0.04em;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 130px;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
      filter: saturate(0.45);
      box-shadow: none;
    }
    .primary {
      background: linear-gradient(135deg, var(--accent), #076476);
      color: white;
    }
    .danger {
      background: linear-gradient(135deg, var(--warm), #8e4010);
      color: white;
    }
    .secondary, .export-link {
      background: rgba(17, 25, 29, 0.08);
      color: var(--ink);
      border: 1px solid rgba(31, 38, 41, 0.14);
    }
    .status-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      margin-bottom: 16px;
    }
    .hint-card {
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(13, 142, 166, 0.08);
      border: 1px solid rgba(13, 142, 166, 0.18);
      color: rgba(31, 38, 41, 0.86);
      font-size: 13px;
      line-height: 1.6;
    }
    .hint-card strong {
      display: block;
      margin-bottom: 6px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-size: 11px;
      color: var(--accent);
    }
    .hint-card code {
      display: block;
      margin-top: 8px;
      font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    .recovery-panel {
      margin: 14px 0;
      display: grid;
      gap: 10px;
    }
    .recovery-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
      align-items: end;
    }
    .recovery-grid label {
      text-transform: none;
      letter-spacing: 0.02em;
      font-size: 12px;
      color: rgba(31, 38, 41, 0.82);
    }
    .recovery-grid button {
      min-width: 110px;
      padding: 10px 14px;
      font-size: 12px;
    }
    .error-banner {
      margin-bottom: 16px;
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(207, 108, 44, 0.12);
      border: 1px solid rgba(207, 108, 44, 0.32);
      color: #6b2f0d;
      font-size: 13px;
      line-height: 1.5;
    }
    .info-banner {
      margin-bottom: 16px;
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(13, 142, 166, 0.1);
      border: 1px solid rgba(13, 142, 166, 0.2);
      color: rgba(31, 38, 41, 0.86);
      font-size: 13px;
      line-height: 1.5;
    }
    .status-card {
      border-radius: 18px;
      padding: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.72));
      border: 1px solid var(--border);
    }
    .status-card .label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: rgba(31, 38, 41, 0.58);
      margin-bottom: 8px;
    }
    .status-card .value {
      font-size: 18px;
      font-weight: 700;
    }
    .exports {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    #resultsTableWrap {
      width: 100%;
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
    }
    table {
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: rgba(255,255,255,0.72);
      border-radius: 18px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid rgba(31, 38, 41, 0.1);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: rgba(17, 25, 29, 0.07);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    .mono {
      font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f171a;
      color: #d7ecef;
      border-radius: 18px;
      padding: 16px;
      font-size: 12px;
      line-height: 1.7;
      max-height: 240px;
      overflow: auto;
    }
    @media (max-width: 1024px) {
      .hero, .layout { grid-template-columns: 1fr; }
      .status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .shell { width: min(100vw - 18px, 100%); margin-top: 10px; }
      .hero, .panel { padding: 16px; border-radius: 18px; }
      .grid, .status-grid { grid-template-columns: 1fr; }
      .actions { flex-direction: column; }
      button, .export-link { width: 100%; }
      .recovery-grid { grid-template-columns: 1fr; }
    }

    /* Dashboard redesign */
    body {
      font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at 12% 12%, rgba(41, 111, 216, 0.24), transparent 31%),
        radial-gradient(circle at 84% 5%, rgba(28, 155, 102, 0.18), transparent 30%),
        linear-gradient(180deg, #070d14 0%, #111923 56%, #0c131c 100%);
      padding: 18px 0 26px;
      color: #1a2331;
    }
    .outer-shell {
      position: relative;
      z-index: 1;
      width: min(1580px, calc(100vw - 24px));
      margin: 0 auto;
      display: grid;
      gap: 12px;
    }
    .topbar {
      border-radius: 16px;
      border: 1px solid rgba(229, 240, 255, 0.42);
      background:
        linear-gradient(180deg, rgba(30, 47, 70, 0.96), rgba(19, 31, 47, 0.97)),
        linear-gradient(96deg, rgba(85, 141, 228, 0.26), rgba(44, 87, 163, 0));
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.12),
        inset 0 -1px 0 rgba(133, 169, 223, 0.22),
        0 24px 42px rgba(5, 10, 18, 0.44);
      padding: 18px 24px;
      display: grid;
      grid-template-columns: minmax(280px, auto) 1fr auto;
      gap: 20px;
      align-items: center;
      position: relative;
      overflow: hidden;
      min-height: 118px;
    }
    .topbar::after {
      content: "";
      position: absolute;
      left: 16px;
      right: 16px;
      bottom: 0;
      border-bottom: 1px solid rgba(205, 220, 248, 0.16);
    }
    .topbar::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(95deg, rgba(95, 153, 241, 0.14), transparent 45%, rgba(78, 136, 223, 0.12));
      pointer-events: none;
    }
    .topbar-brand {
      display: grid;
      grid-template-columns: auto auto;
      gap: 12px;
      align-items: center;
      color: #ebf2ff;
    }
    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 18px;
      background: linear-gradient(135deg, #4aa0ff, #3b75f0);
      color: #0a1b2d;
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 12px 20px rgba(21, 69, 147, 0.38);
    }
    .brand-text {
      display: grid;
      gap: 1px;
      line-height: 1.1;
    }
    .brand-text strong {
      font-size: 17px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
    }
    .brand-text span {
      font-size: 11px;
      color: #c3d2ea;
      letter-spacing: 0.06em;
      font-weight: 620;
    }
    .topbar-nav {
      display: grid;
      gap: 10px;
      justify-items: start;
      border-left: 1px solid rgba(194, 212, 243, 0.24);
      padding-left: 16px;
    }
    .nav-main,
    .nav-sub {
      display: flex;
      gap: 9px;
      flex-wrap: wrap;
      align-items: center;
    }
    .nav-main {
      border-radius: 999px;
      border: 1px solid rgba(205, 221, 248, 0.3);
      background: rgba(8, 13, 22, 0.52);
      padding: 5px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
    }
    .nav-sub {
      padding-left: 5px;
      gap: 7px;
    }
    .nav-pill {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #d8e6fd;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(223, 233, 251, 0.3);
      border-radius: 999px;
      padding: 7px 13px;
      line-height: 1;
      font-weight: 680;
    }
    .nav-pill.primary {
      font-size: 12px;
      padding: 9px 14px;
      border-color: rgba(167, 205, 255, 0.56);
      color: #f3f8ff;
    }
    .header-meta {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      justify-self: start;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #c2d2ea;
      font-weight: 700;
    }
    .header-meta.secondary {
      color: #9fb2d0;
      font-size: 10px;
      letter-spacing: 0.07em;
      font-weight: 650;
    }
    .header-meta.tertiary {
      color: #b8cae7;
      font-size: 10px;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .header-mode-strip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border-radius: 999px;
      border: 1px solid rgba(196, 214, 243, 0.29);
      background: rgba(8, 14, 24, 0.44);
      padding: 5px 7px;
    }
    .mode-strip-item {
      border-radius: 999px;
      border: 1px solid rgba(198, 216, 243, 0.28);
      background: rgba(255, 255, 255, 0.06);
      color: #c7d8f3;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      padding: 6px 9px;
      line-height: 1;
      white-space: nowrap;
    }
    .mode-strip-item.active {
      border-color: rgba(156, 196, 250, 0.5);
      background: rgba(72, 133, 233, 0.32);
      color: #f5f9ff;
    }
    .meta-dot {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: #41d28f;
      box-shadow: 0 0 0 4px rgba(65, 210, 143, 0.16);
    }
    .nav-pill.active {
      background: rgba(88, 157, 255, 0.22);
      border-color: rgba(137, 185, 255, 0.58);
      color: #f6fbff;
    }
    .topbar-actions {
      display: flex;
      justify-content: flex-end;
      gap: 9px;
      align-items: center;
      justify-self: end;
      padding-right: 2px;
      border-left: 1px solid rgba(196, 214, 243, 0.25);
      padding-left: 14px;
    }
    .action-chip {
      border-radius: 999px;
      border: 1px solid rgba(223, 234, 252, 0.36);
      background: rgba(255, 255, 255, 0.09);
      color: #dce8fd;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 8px 11px;
      font-weight: 700;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .action-chip.live {
      border-color: rgba(138, 226, 182, 0.42);
      color: #d8ffe7;
      background: rgba(34, 111, 72, 0.24);
    }
    .action-chip.ghost {
      background: transparent;
      border-color: rgba(222, 233, 251, 0.16);
      color: #b1c0d9;
    }
    .workspace-card {
      border-radius: 20px;
      border: 1px solid #dce4f2;
      background: linear-gradient(180deg, #f8faff, #f4f7fd 58%, #f6f9fe);
      box-shadow: 0 24px 44px rgba(6, 10, 18, 0.4);
      padding: 16px;
    }
    .workspace-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .configuration-column,
    .results-column {
      border-radius: 14px;
      border: 1px solid #d9e2f2;
      background: #ffffff;
      padding: 12px;
    }
    .configuration-column {
      background:
        linear-gradient(180deg, #ffffff, #f8fbff),
        linear-gradient(110deg, rgba(72, 136, 242, 0.06), rgba(72, 136, 242, 0));
      box-shadow: 0 14px 24px rgba(33, 61, 105, 0.11);
      padding: 22px;
      max-width: 1240px;
      width: 100%;
      justify-self: center;
    }
    .workspace-header {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
      gap: 14px 16px;
      margin-bottom: 16px;
      padding: 18px;
      border-radius: 16px;
      border: 1px solid #d2e1f4;
      background: linear-gradient(180deg, #fbfdff, #f1f7ff);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }
    .workspace-title-block {
      display: grid;
      gap: 6px;
      align-content: start;
    }
    .workspace-cue-block {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .config-priority {
      display: none;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      grid-column: 1 / -1;
      gap: 7px;
      margin-top: 0;
    }
    .priority-cue {
      border-radius: 8px;
      border: 1px solid #d4e0f2;
      background: linear-gradient(180deg, #f7fbff, #f2f7ff);
      padding: 5px 7px;
      display: grid;
      gap: 2px;
    }
    .priority-cue strong {
      font-size: 8px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: #52719c;
      flex: none;
    }
    .priority-cue span {
      font-size: 10px;
      color: #314b70;
      font-weight: 600;
      line-height: 1.2;
    }
    .workspace-kicker {
      margin: 0;
      font-size: 11px;
      font-weight: 700;
      color: #4c678f;
      letter-spacing: 0.11em;
      text-transform: uppercase;
    }
    .workspace-header h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.03em;
      line-height: 1.06;
    }
    .workspace-description {
      margin: 0;
      font-size: 13px;
      color: #627a9b;
      line-height: 1.45;
      max-width: 74ch;
    }
    .mode-switch {
      display: flex;
      gap: 9px;
      flex-wrap: wrap;
      margin-top: 0;
      padding: 10px 12px;
      border-radius: 13px;
      border: 1px solid #cbdbf2;
      background: linear-gradient(180deg, #fbfdff, #eff6ff);
    }
    .mode-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #cddbf1;
      background: #ffffff;
      color: #4f6d97;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 730;
      line-height: 1;
      white-space: nowrap;
    }
    .mode-chip.active {
      background: linear-gradient(145deg, #4a8fff, #2b67d7);
      border-color: #2b67d7;
      color: #f8fbff;
      box-shadow: 0 8px 16px rgba(43, 103, 215, 0.25);
    }
    .batch-cues {
      display: grid;
      grid-template-columns: auto repeat(4, minmax(0, auto));
      gap: 8px;
      align-items: center;
      border-radius: 13px;
      border: 1px solid #ccdbee;
      background: linear-gradient(180deg, #fafdff, #edf5ff);
      padding: 10px 11px;
    }
    .batch-cue-title {
      font-size: 9px;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: #5c77a0;
      font-weight: 700;
      white-space: nowrap;
    }
    .batch-chip {
      border-radius: 999px;
      border: 1px solid #cfdcf0;
      background: #ffffff;
      color: #54729b;
      font-size: 10px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      font-weight: 700;
      padding: 6px 10px;
      line-height: 1;
      white-space: nowrap;
    }
    .batch-chip.active {
      border-color: #9fc0ef;
      background: #e7f1ff;
      color: #2f5f9c;
    }
    #benchForm {
      gap: 12px;
    }
    .actions {
      margin: 0;
      gap: 10px;
      display: grid;
      grid-template-columns: minmax(250px, 1fr) minmax(220px, auto);
      background: linear-gradient(180deg, rgba(247, 251, 255, 0.98), rgba(238, 245, 255, 0.98));
      border: 1px solid #d3e1f5;
      border-radius: 13px;
      padding: 10px;
      top: 4px;
      backdrop-filter: blur(8px);
    }
    .config-groups {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      grid-template-areas:
        "mode"
        "runtime"
        "advanced"
        "raw";
      gap: 12px;
      align-items: start;
    }
    .config-card {
      border-radius: 14px;
      border: 1px solid #d2dff2;
      background: #fbfdff;
      padding: 14px;
      display: grid;
      gap: 11px;
    }
    .config-card.secondary {
      border-color: #dbe6f6;
      background: linear-gradient(180deg, #fbfdff, #f7faff);
      opacity: 0.95;
    }
    .config-disclosure {
      border-radius: 14px;
      border: 1px solid #d1def2;
      background: #f7fbff;
      padding: 0;
      overflow: hidden;
    }
    .config-disclosure > summary {
      list-style: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 12px 14px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: #4f6c95;
      background: linear-gradient(180deg, #f7fbff, #eff6ff);
      border-bottom: 1px solid #dbe6f6;
    }
    .config-disclosure > summary::-webkit-details-marker {
      display: none;
    }
    .config-disclosure > summary::after {
      content: "Expand";
      font-size: 9px;
      color: #6f84a6;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }
    .config-disclosure[open] > summary::after {
      content: "Collapse";
    }
    .config-disclosure .config-card {
      border: none;
      border-radius: 0;
      background: #ffffff;
      padding: 14px;
      box-shadow: none;
    }
    .mode-secondary-disclosure {
      margin-top: 2px;
      border-color: #d7e3f3;
      background: #f8fbff;
    }
    .mode-secondary-disclosure > summary {
      color: #5a769f;
      background: linear-gradient(180deg, #f9fcff, #f2f7ff);
    }
    .mode-secondary-disclosure .config-card {
      background: #fdfefe;
    }
    .secondary-scenario-fields {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .runtime-disclosure {
      grid-area: runtime;
    }
    .advanced-disclosure {
      grid-area: advanced;
    }
    .raw-disclosure {
      grid-area: raw;
    }
    .config-card.advanced-panel {
      border-style: dashed;
      background: linear-gradient(180deg, #fbfdff, #f7fbff);
    }
    .config-card.core-panel {
      background: linear-gradient(180deg, #fdfeff, #f4f8ff);
      border-color: #cad9f3;
      box-shadow: 0 8px 16px rgba(33, 64, 112, 0.09);
    }
    .config-card.full-width {
      grid-area: raw;
      grid-column: auto;
    }
    .mode-card {
      grid-area: mode;
      border-color: #c4d6f3;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
      box-shadow: 0 14px 22px rgba(34, 69, 122, 0.11);
    }
    .core-panel {
      grid-area: runtime;
    }
    .advanced-panel {
      grid-area: advanced;
    }
    .core-fields {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .runtime-fields {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .advanced-fields {
      grid-template-columns: 1fr;
    }
    .card-title {
      margin: 0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      color: #506f99;
      font-weight: 700;
    }
    .card-subtitle {
      margin: 0;
      font-size: 13px;
      line-height: 1.42;
      color: #667f9f;
    }
    .mode-card .card-subtitle {
      color: #5874a1;
      max-width: 48ch;
    }
    .mode-card .field-grid {
      margin-top: 2px;
    }
    .core-panel .card-subtitle,
    .advanced-panel .card-subtitle,
    .full-width .card-subtitle {
      color: #7d90af;
      font-size: 11px;
    }
    .mode-fields,
    .field-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .mode-fields {
      grid-template-columns: minmax(180px, 0.8fr) minmax(0, 1.2fr);
    }
    #benchForm label {
      display: grid;
      gap: 6px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #6d819f;
      font-weight: 700;
      border: none;
      border-radius: 0;
      padding: 0;
      background: transparent;
    }
    .mode-fields > label,
    .field-grid > label,
    .config-card.full-width > label {
      border: 1px solid #d5e1f2;
      border-radius: 12px;
      padding: 10px;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
    }
    #benchForm label.primary-control {
      color: #3f5f8f;
    }
    #benchForm label.secondary-control {
      color: #7890b2;
      font-weight: 650;
    }
    .mode-fields > label.primary-control,
    .field-grid > label.primary-control {
      border-color: #cbd9ef;
      background: linear-gradient(180deg, #ffffff, #f4f8ff);
    }
    .field-grid > label.secondary-control,
    .config-card.full-width > label.secondary-control {
      border-color: #dde5f3;
      background: #f8fbff;
    }
    #benchForm input,
    #benchForm select,
    #benchForm textarea {
      border-radius: 9px;
      border: 1px solid #c5d3e9;
      background: #ffffff;
      padding: 10px 12px;
      font-size: 13px;
      color: #1b2738;
      font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    #benchForm label.secondary-control input,
    #benchForm label.secondary-control select,
    #benchForm label.secondary-control textarea {
      background: #f7fbff;
      border-color: #d7e1f2;
    }
    #benchForm label.primary-control input,
    #benchForm label.primary-control select,
    #benchForm label.primary-control textarea {
      border-color: #b7cced;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
    }
    #benchForm textarea {
      min-height: 88px;
      font-family: "SFMono-Regular", ui-monospace, "Cascadia Mono", Menlo, monospace;
      font-size: 12px;
      line-height: 1.35;
    }
    #benchForm input:focus,
    #benchForm select:focus,
    #benchForm textarea:focus,
    button:focus-visible {
      outline: none;
      border-color: #78a7f7;
      box-shadow: 0 0 0 3px rgba(45, 126, 247, 0.15);
    }
    .checks {
      display: flex;
      gap: 10px;
      margin-top: 4px;
    }
    .checks label {
      display: inline-flex !important;
      align-items: center;
      gap: 7px;
      border-radius: 999px;
      border: 1px solid #cad9f7;
      background: #edf3ff;
      color: #3a5986 !important;
      padding: 8px 12px;
      font-size: 10px !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .checks input {
      width: auto !important;
      margin: 0;
      accent-color: #2d7ef7;
    }
    button,
    .export-link {
      min-width: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-weight: 700;
      line-height: 1.2;
      border: 1px solid transparent;
    }
    .primary {
      background: linear-gradient(140deg, #3f8cff, #2867d8);
      box-shadow: 0 10px 20px rgba(45, 126, 247, 0.26);
    }
    .danger {
      background: linear-gradient(140deg, #d55b53, #b73f38);
      box-shadow: 0 10px 20px rgba(183, 63, 56, 0.22);
    }
    .secondary,
    .export-link {
      background: #f4f7fc;
      border-color: #cdd8ea;
      color: #324768;
    }
    .results-column {
      display: grid;
      gap: 14px;
      border: 1px solid #b7cceb;
      background: linear-gradient(180deg, #f9fbff, #edf4ff 54%, #f1f7ff);
      padding: 22px;
      box-shadow: 0 24px 38px rgba(34, 64, 108, 0.16);
      min-height: 1380px;
      margin-top: 12px;
    }
    .section-title {
      margin: 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #3f5c86;
      font-weight: 700;
    }
    .results-header-card {
      border-radius: 15px;
      border: 1px solid #b8ceed;
      background: linear-gradient(180deg, #ffffff, #f5f9ff);
      padding: 18px;
      display: grid;
      gap: 12px;
      box-shadow: 0 16px 24px rgba(35, 64, 108, 0.13);
    }
    .results-header-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 8px;
    }
    .results-header-row h2 {
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.03em;
      line-height: 1.06;
    }
    .results-subtitle {
      margin: 0;
      max-width: 42ch;
      text-align: right;
      font-size: 13px;
      line-height: 1.45;
      color: #647ea1;
    }
    .command-card {
      border-radius: 12px;
      border: 1px solid #2b3e59;
      background: linear-gradient(180deg, #102139, #0f1a2d);
      padding: 11px 12px;
      color: #e3edff;
      min-height: auto;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .command-card .label {
      display: block;
      margin-bottom: 6px;
      font-size: 10px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #8ca6ce;
      font-weight: 700;
    }
    #commandPreview {
      font-family: "SFMono-Regular", ui-monospace, "Cascadia Mono", Menlo, monospace;
      font-size: 11px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .error-banner,
    .info-banner {
      margin-bottom: 0;
      border-radius: 10px;
      padding: 9px 11px;
      font-size: 12px;
      line-height: 1.45;
    }
    .error-banner {
      background: #fff2f1;
      border-color: #efc4c0;
      color: #8c322b;
    }
    .info-banner {
      background: #f1f7ff;
      border-color: #c7ddff;
      color: #245192;
    }
    .report-kpi-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .report-kpi {
      border-radius: 14px;
      border: 1px solid #b3c9ea;
      background: linear-gradient(180deg, #ffffff, #edf5ff);
      padding: 12px;
      display: grid;
      gap: 4px;
      box-shadow: 0 12px 20px rgba(34, 63, 107, 0.12);
    }
    .report-kpi .kpi-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: #5f789f;
      font-weight: 700;
      line-height: 1.1;
    }
    .report-kpi .kpi-value {
      font-size: 24px;
      font-weight: 730;
      letter-spacing: -0.02em;
      color: #223b62;
      line-height: 1.05;
      font-variant-numeric: tabular-nums;
    }
    .report-kpi .kpi-hint {
      font-size: 10px;
      color: #6e84a8;
      line-height: 1.2;
    }
    .status-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 0;
    }
    .status-card {
      border-radius: 12px;
      border: 1px solid #d4dff1;
      background: linear-gradient(180deg, #ffffff, #f4f8ff);
      padding: 13px;
    }
    .status-card .label {
      margin-bottom: 4px;
      font-size: 11px;
      letter-spacing: 0.08em;
      color: #5f7290;
      font-weight: 700;
    }
    .status-card .value {
      font-size: 19px;
      font-weight: 700;
      color: #233657;
      font-variant-numeric: tabular-nums;
    }
    .result-card {
      border-radius: 16px;
      border: 1px solid #b6ccec;
      background: #ffffff;
      padding: 18px;
      display: grid;
      gap: 13px;
      box-shadow: 0 16px 26px rgba(35, 64, 108, 0.14);
    }
    .result-card.report-module {
      border-color: #b0c7e8;
      box-shadow: 0 20px 30px rgba(34, 63, 107, 0.17);
    }
    .result-card h3 {
      margin: 0;
      font-size: 12px;
      letter-spacing: 0.11em;
      text-transform: uppercase;
      color: #3f5f8b;
      font-weight: 720;
    }
    .result-card h4 {
      margin: 0 0 5px;
      font-size: 11px;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: #5f7290;
    }
    .hint-card {
      border-radius: 10px;
      border: 1px solid #d8e1f2;
      background: #f3f7ff;
      color: #374d70;
      font-size: 12px;
      line-height: 1.4;
      padding: 9px;
    }
    .hint-card strong {
      margin-bottom: 4px;
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #46689b;
      font-weight: 700;
    }
    .hint-card code {
      margin-top: 6px;
      font-size: 11px;
      font-family: "SFMono-Regular", ui-monospace, "Cascadia Mono", Menlo, monospace;
    }
    .recovery-panel {
      margin: 0;
    }
    .recovery-grid {
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
    }
    .recovery-grid label {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 10px;
      color: #607290;
      font-weight: 700;
    }
    .exports {
      margin-bottom: 0;
      gap: 7px;
    }
    #resultsTableWrap {
      border-radius: 14px;
      border: 1px solid #b1c8e9;
      background: #ffffff;
      max-width: 100%;
      min-height: 760px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85), 0 18px 26px rgba(33, 61, 105, 0.17);
      position: relative;
    }
    #resultsTableWrap > p {
      margin: 0;
      padding: 54px 18px 36px;
      font-size: 13px;
      color: #435d86;
      line-height: 1.5;
      letter-spacing: 0.01em;
      font-weight: 580;
    }
    #resultsTableWrap::before {
      content: "Benchmark Table";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 34px;
      display: flex;
      align-items: center;
      padding: 0 12px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      color: #55729e;
      background: linear-gradient(180deg, #dae9ff, #cfdef8);
      border-bottom: 1px solid #d1e0f5;
      pointer-events: none;
    }
    table {
      border-radius: 0;
      background: #ffffff;
      font-size: 13px;
      width: max-content;
      min-width: 100%;
    }
    #resultsTableWrap table {
      margin-top: 32px;
    }
    th {
      background: linear-gradient(180deg, #bfd7ff, #b2ccf5);
      color: #2a4b78;
      font-size: 11px;
      letter-spacing: 0.12em;
      font-weight: 790;
    }
    th, td {
      padding: 12px 14px;
      border-bottom-color: #d6e3f4;
    }
    #resultsTableWrap table thead th {
      position: sticky;
      top: 0;
      z-index: 1;
    }
    #resultsTableWrap table tbody tr:nth-child(even) td {
      background: #f6faff;
    }
    #resultsTableWrap table tbody td:nth-child(n+2) {
      font-variant-numeric: tabular-nums;
      font-weight: 760;
      color: #1a5f46;
      font-size: 13px;
      text-align: right;
    }
    .table-module-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      padding-bottom: 6px;
      border-bottom: 1px solid #d4e1f3;
    }
    .module-note {
      font-size: 10px;
      color: #5778a4;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }
    .placeholder-bench-table {
      width: 100%;
      min-width: 100%;
      border-collapse: collapse;
      border: 1px solid #aec6e8;
      border-radius: 14px;
      overflow: hidden;
      background: #ffffff;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85), 0 16px 22px rgba(35, 64, 108, 0.12);
    }
    .placeholder-bench-table th,
    .placeholder-bench-table td {
      padding: 12px 14px;
      font-size: 12px;
      border-bottom: 1px solid #d8e4f4;
      text-align: left;
    }
    .placeholder-bench-table th {
      background: linear-gradient(180deg, #c2d9ff, #b8d0f5);
      color: #2a4c78;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
      font-weight: 780;
    }
    .placeholder-bench-table td {
      color: #365171;
      font-variant-numeric: tabular-nums;
    }
    .placeholder-bench-table td:nth-child(n+2) {
      color: #1f5e46;
      font-weight: 760;
      font-size: 13px;
      text-align: right;
    }
    .placeholder-bench-table tbody tr:nth-child(even) td {
      background: #f6faff;
    }
    .placeholder-bench-table td:first-child {
      font-weight: 680;
      color: #2f496f;
    }
    .placeholder-bench-table {
      margin-top: 0;
    }
    .module-empty-row td {
      color: #334e73 !important;
      font-size: 12px;
      font-weight: 700 !important;
      font-style: normal;
      background: #eff6ff;
      text-transform: none;
      letter-spacing: 0;
    }
    .placeholder-bench-table tr:last-child td {
      border-bottom: none;
    }
    .log-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .mono {
      border-radius: 10px;
      border: 1px solid #27374f;
      background: #0f1828;
      color: #d8e4fb;
      padding: 10px;
      font-family: "SFMono-Regular", ui-monospace, "Cascadia Mono", Menlo, monospace;
      line-height: 1.5;
      max-height: 260px;
    }
    @media (max-width: 1280px) {
      .workspace-header {
        grid-template-columns: 1fr;
      }
      .config-groups {
        grid-template-columns: minmax(0, 1fr);
        grid-template-areas:
          "mode"
          "runtime"
          "advanced"
          "raw";
      }
      .mode-fields,
      .field-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 1200px) {
      .results-subtitle {
        text-align: left;
        max-width: none;
      }
    }
    @media (max-width: 860px) {
      .topbar {
        grid-template-columns: 1fr;
        min-height: 0;
      }
      .topbar-nav,
      .topbar-actions {
        justify-content: flex-start;
        justify-self: start;
        border-left: none;
        padding-left: 0;
      }
      .nav-main,
      .nav-sub {
        justify-content: flex-start;
      }
      .workspace-grid {
        grid-template-columns: 1fr;
      }
      .configuration-column,
      .results-column {
        padding: 12px;
      }
      .config-groups,
      .field-grid,
      .core-fields,
      .runtime-fields,
      .advanced-fields,
      .config-priority,
      .mode-fields,
      .status-grid,
      .log-grid,
      .report-kpi-strip {
        grid-template-columns: 1fr;
      }
      .actions {
        grid-template-columns: 1fr;
      }
      .config-groups {
        grid-template-areas:
          "mode"
          "runtime"
          "advanced"
          "raw";
      }
      .config-disclosure > summary {
        padding: 10px 11px;
      }
      .workspace-header {
        grid-template-columns: 1fr;
        padding: 12px;
      }
      .batch-cues {
        grid-template-columns: 1fr;
      }
      .workspace-cue-block,
      .config-priority {
        display: none;
      }
      .workspace-description {
        display: none;
      }
      .recovery-grid {
        grid-template-columns: 1fr;
      }
      button,
      .export-link {
        width: 100%;
      }
    }
    @media (max-width: 600px) {
      body {
        padding: 10px 0 14px;
      }
      .outer-shell {
        width: min(100vw - 10px, 100%);
      }
      .workspace-card {
        padding: 10px;
      }
      .configuration-column,
      .results-column {
        padding: 10px;
      }
      .workspace-header h1,
      .results-header-row h2 {
        font-size: 24px;
      }
      #resultsTableWrap::before {
        position: static;
        height: auto;
        padding: 8px 10px;
      }
      table {
        margin-top: 0;
      }
      #resultsTableWrap > p {
        padding-top: 16px;
      }
    }

    /* Final results-led dashboard polish */
    .topbar {
      min-height: 138px;
      padding: 22px 28px;
      border-radius: 18px;
      grid-template-columns: minmax(300px, auto) 1fr auto;
    }
    .topbar-brand {
      gap: 14px;
    }
    .brand-mark {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      font-size: 20px;
    }
    .brand-text strong {
      font-size: 18px;
      letter-spacing: 0.08em;
    }
    .brand-text span {
      font-size: 12px;
    }
    .topbar-nav {
      gap: 12px;
      padding-left: 18px;
    }
    .nav-main {
      padding: 6px;
      gap: 10px;
    }
    .nav-pill {
      font-size: 12px;
      letter-spacing: 0.06em;
      padding: 9px 14px;
    }
    .nav-pill.primary {
      font-size: 12px;
      padding: 10px 15px;
    }
    .header-meta {
      font-size: 12px;
    }
    .header-meta.secondary,
    .header-meta.tertiary {
      font-size: 11px;
    }
    .topbar-actions {
      gap: 10px;
      padding-left: 16px;
    }
    .action-chip {
      font-size: 11px;
      padding: 9px 13px;
    }
    .workspace-card {
      padding: 18px;
      border-radius: 22px;
    }
    .configuration-column {
      max-width: 1360px;
      padding: 28px;
      border-radius: 16px;
      border-color: #cad8ef;
    }
    .workspace-header {
      padding: 22px;
      border-radius: 18px;
      gap: 16px 18px;
    }
    .workspace-header h1 {
      font-size: 34px;
      line-height: 1.04;
    }
    .workspace-description {
      font-size: 14px;
      line-height: 1.55;
      max-width: 82ch;
    }
    .mode-switch {
      gap: 10px;
      padding: 12px 14px;
      border-radius: 14px;
    }
    .mode-chip {
      padding: 10px 14px;
      font-size: 12px;
      letter-spacing: 0.06em;
    }
    .batch-cues {
      padding: 12px 14px;
      border-radius: 14px;
      gap: 10px;
    }
    .batch-chip {
      padding: 7px 12px;
      font-size: 11px;
      letter-spacing: 0.06em;
    }
    #benchForm {
      gap: 16px;
    }
    .actions {
      padding: 12px;
      gap: 12px;
      border-radius: 14px;
    }
    button,
    .export-link {
      padding: 12px 16px;
      font-size: 12px;
      letter-spacing: 0.05em;
    }
    .config-groups {
      gap: 14px;
    }
    .config-card,
    .config-disclosure .config-card {
      padding: 16px;
      gap: 12px;
      border-radius: 14px;
    }
    .config-card .card-title {
      font-size: 12px;
      letter-spacing: 0.09em;
    }
    .config-card .card-subtitle {
      font-size: 13px;
      line-height: 1.5;
    }
    .mode-fields,
    .field-grid {
      gap: 12px;
    }
    #benchForm label {
      font-size: 12px;
      letter-spacing: 0.03em;
      text-transform: none;
      font-weight: 680;
      color: #4f678a;
    }
    .mode-fields > label,
    .field-grid > label,
    .config-card.full-width > label {
      padding: 12px;
      border-radius: 12px;
    }
    #benchForm input,
    #benchForm select,
    #benchForm textarea {
      padding: 12px 13px;
      font-size: 14px;
      border-radius: 10px;
    }
    #benchForm textarea {
      min-height: 96px;
      font-size: 13px;
      line-height: 1.45;
    }
    .checks label {
      font-size: 11px !important;
      letter-spacing: 0.06em;
      padding: 9px 13px;
    }
    .results-column {
      margin-top: 14px;
      padding: 26px;
      gap: 16px;
      border-radius: 18px;
      min-height: 1500px;
      border-color: #a8c3e8;
      box-shadow: 0 30px 44px rgba(28, 54, 95, 0.18);
    }
    .section-title {
      font-size: 14px;
      letter-spacing: 0.14em;
    }
    .results-header-card {
      padding: 22px;
      border-radius: 17px;
      box-shadow: 0 18px 28px rgba(32, 59, 100, 0.15);
    }
    .results-header-row h2 {
      font-size: 35px;
    }
    .results-subtitle {
      font-size: 14px;
      line-height: 1.5;
      max-width: 50ch;
    }
    .command-card {
      padding: 12px 14px;
      border-radius: 13px;
    }
    .report-kpi-strip {
      gap: 14px;
    }
    .report-kpi {
      padding: 14px;
      border-radius: 15px;
    }
    .report-kpi .kpi-value {
      font-size: 26px;
    }
    .status-grid {
      gap: 12px;
    }
    .status-card {
      padding: 14px;
      border-radius: 13px;
    }
    .status-card .value {
      font-size: 20px;
    }
    .result-card {
      padding: 22px;
      border-radius: 17px;
      gap: 14px;
      box-shadow: 0 18px 30px rgba(32, 59, 100, 0.16);
    }
    .result-card h3 {
      font-size: 13px;
      letter-spacing: 0.1em;
    }
    .result-card h4 {
      font-size: 12px;
      letter-spacing: 0.08em;
    }
    .table-module-head {
      padding-bottom: 7px;
    }
    .module-note {
      font-size: 11px;
      letter-spacing: 0.07em;
    }
    #resultsTableWrap {
      min-height: 860px;
      border-radius: 16px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.86), 0 22px 34px rgba(31, 57, 97, 0.19);
    }
    #resultsTableWrap::before {
      height: 38px;
      padding: 0 14px;
      font-size: 12px;
      letter-spacing: 0.1em;
    }
    #resultsTableWrap > p {
      padding: 58px 20px 36px;
      font-size: 14px;
      line-height: 1.52;
    }
    #resultsTableWrap table {
      margin-top: 36px;
    }
    th,
    td {
      padding: 13px 16px;
      font-size: 13px;
    }
    th {
      font-size: 12px;
      letter-spacing: 0.1em;
    }
    #resultsTableWrap table tbody td:nth-child(n+2),
    .placeholder-bench-table td:nth-child(n+2) {
      font-size: 14px;
      font-weight: 780;
    }
    .placeholder-bench-table {
      border-radius: 15px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.86), 0 18px 26px rgba(31, 57, 97, 0.14);
    }
    .placeholder-bench-table th,
    .placeholder-bench-table td {
      padding: 13px 15px;
    }
    .log-grid {
      gap: 12px;
    }
    .mono {
      padding: 12px;
      border-radius: 11px;
      max-height: 280px;
    }
    @media (max-width: 860px) {
      .topbar {
        min-height: 0;
        padding: 16px;
        grid-template-columns: 1fr;
      }
      .configuration-column,
      .results-column {
        padding: 14px;
      }
      .workspace-header h1,
      .results-header-row h2 {
        font-size: 28px;
      }
      .results-column {
        min-height: 0;
      }
      #resultsTableWrap {
        min-height: 520px;
      }
    }
    @media (max-width: 600px) {
      .configuration-column,
      .results-column {
        padding: 12px;
      }
      #benchForm label {
        font-size: 11px;
      }
      #resultsTableWrap {
        min-height: 420px;
      }
    }
  </style>
</head>
<body>
  <div class="outer-shell">
    <header class="topbar">
      <div class="topbar-brand">
        <span class="brand-mark">L</span>
        <div class="brand-text">
          <strong>llmbench</strong>
          <span>vLLM Benchmark Console</span>
        </div>
      </div>
      <div class="topbar-nav" aria-hidden="true">
        <div class="header-meta"><span class="meta-dot"></span> Benchmark Suite</div>
        <div class="header-meta secondary">Workspace: Report Console</div>
        <div class="header-meta tertiary">Product: LLMBench Dashboard</div>
        <div class="nav-main">
          <span class="nav-pill primary active">Bench</span>
          <span class="nav-pill primary">Status</span>
          <span class="nav-pill primary">Reports</span>
          <span class="nav-pill primary">Workspace</span>
        </div>
        <div class="nav-sub">
          <span class="nav-pill">Models</span>
          <span class="nav-pill">Pipelines</span>
          <span class="nav-pill">Settings</span>
          <span class="nav-pill">Logs</span>
        </div>
        <div class="header-mode-strip">
          <span class="mode-strip-item active">Serve Mode</span>
          <span class="mode-strip-item">Batch Report</span>
          <span class="mode-strip-item">Console</span>
        </div>
      </div>
      <div class="topbar-actions" aria-hidden="true">
        <span class="action-chip live">Live</span>
        <span class="action-chip">Queue</span>
        <span class="action-chip">Export</span>
        <span class="action-chip ghost">Account</span>
      </div>
    </header>

    <main class="workspace-card">
      <div class="workspace-grid">
        <section class="configuration-column">
          <div class="workspace-header">
            <div class="workspace-title-block">
              <p class="workspace-kicker">Configuration</p>
              <h1>Performance Benchmark</h1>
              <p class="workspace-description">
                Configure primary runtime inputs, then launch and inspect the report surface.
              </p>
            </div>
            <div class="workspace-cue-block" aria-hidden="true">
              <div class="mode-switch">
                <span class="mode-chip active">Single Request</span>
                <span class="mode-chip">Continuous Batch</span>
                <span class="mode-chip">Terminal</span>
              </div>
              <div class="batch-cues">
                <span class="batch-cue-title">Batch Scale</span>
                <span class="batch-chip active">1x</span>
                <span class="batch-chip">2x</span>
                <span class="batch-chip">4x</span>
                <span class="batch-chip">8x</span>
              </div>
            </div>
            <div class="config-priority" aria-hidden="true">
              <div class="priority-cue">
                <strong>Step 1</strong>
                <span>Mode</span>
              </div>
              <div class="priority-cue">
                <strong>Step 2</strong>
                <span>Core Inputs</span>
              </div>
              <div class="priority-cue">
                <strong>Step 3</strong>
                <span>Run + Review</span>
              </div>
            </div>
          </div>

          <form id="benchForm">
            <div class="actions">
              <button type="submit" class="primary" id="startButton">Start Benchmark</button>
              <button type="button" class="danger" id="stopButton" disabled>Stop Benchmark</button>
            </div>

            <div class="config-groups">
              <section class="config-card mode-card">
                <h2 class="card-title">Core Benchmark Setup</h2>
                <p class="card-subtitle">Curated primary controls for benchmark launch and mode targeting.</p>
                <div class="mode-fields">
                  <label class="primary-control">Bench Subcommand
                    <select name="subcommand">
                      <option value="serve">serve</option>
                      <option value="latency">latency</option>
                      <option value="mm-processor">mm-processor</option>
                      <option value="startup">startup</option>
                      <option value="sweep">sweep</option>
                      <option value="throughput">throughput</option>
                    </select>
                  </label>
                  <label class="primary-control">Model
                    <input name="model" placeholder="meta-llama/Llama-3.1-8B-Instruct">
                  </label>
                </div>
                <div class="field-grid core-fields">
                  <label class="primary-control">Tokenizer
                    <input name="tokenizer" placeholder="meta-llama/Llama-3.1-8B-Instruct">
                  </label>
                  <label class="primary-control">Base URL
                    <input name="base_url" placeholder="http://127.0.0.1:8000">
                  </label>
                </div>
                <details class="config-disclosure mode-secondary-disclosure">
                  <summary>Secondary Scenario Controls</summary>
                  <section class="config-card secondary-mode-panel secondary">
                    <div class="field-grid secondary-scenario-fields">
                      <label class="secondary-control">Sweep Subcommand
                        <select name="sweep_subcommand">
                          <option value="serve">serve</option>
                          <option value="serve_workload">serve_workload</option>
                          <option value="startup">startup</option>
                          <option value="plot">plot</option>
                          <option value="plot_pareto">plot_pareto</option>
                        </select>
                      </label>
                      <label class="secondary-control">Backend
                        <select name="backend">
                          <option value="">(optional)</option>
                          <option value="openai">openai</option>
                          <option value="vllm">vllm</option>
                          <option value="sglang">sglang</option>
                          <option value="tgi">tgi</option>
                        </select>
                      </label>
                      <label class="secondary-control">Dataset
                        <select name="dataset_name">
                          <option value="">(optional)</option>
                          <option value="random">random</option>
                          <option value="sharegpt">sharegpt</option>
                          <option value="custom">custom</option>
                          <option value="hf">hf</option>
                          <option value="burstgpt">burstgpt</option>
                        </select>
                      </label>
                      <label class="secondary-control">Endpoint
                        <input name="endpoint" placeholder="/v1/completions">
                      </label>
                      <label class="secondary-control">Input Len
                        <input name="input_len" placeholder="512">
                      </label>
                      <label class="secondary-control">Output Len
                        <input name="output_len" placeholder="128">
                      </label>
                    </div>
                  </section>
                </details>
              </section>

              <details class="config-disclosure runtime-disclosure">
                <summary>Runtime And Connection Controls</summary>
                <section class="config-card core-panel">
                  <div class="field-grid runtime-fields">
                    <label class="primary-control">Num Prompts
                      <input name="num_prompts" placeholder="1000">
                    </label>
                    <label class="primary-control">GPU Memory Utilization
                      <input name="gpu_memory_utilization" placeholder="0.60">
                    </label>
                    <label class="primary-control">Max Model Len
                      <input name="max_model_len" placeholder="1024">
                    </label>
                    <label class="primary-control">Request Rate
                      <input name="request_rate" placeholder="4">
                    </label>
                    <label class="primary-control">Max Concurrency
                      <input name="max_concurrency" placeholder="32">
                    </label>
                    <label class="secondary-control">Host
                      <input name="host" placeholder="127.0.0.1">
                    </label>
                    <label class="secondary-control">Port
                      <input name="port" placeholder="8000">
                    </label>
                  </div>
                  <label class="secondary-control">Bench Path Override
                    <input name="bench_path_override" placeholder="sweep serve">
                  </label>
                  <div class="hint-card" id="overrideNotice" hidden>
                    <strong>Override Active</strong>
                    Override is controlling the command path right now. The subcommand selector is locked to avoid launching the wrong benchmark path by mistake.
                  </div>
                </section>
              </details>

              <details class="config-disclosure advanced-disclosure">
                <summary>Advanced Inputs And Metrics</summary>
                <section class="config-card advanced-panel secondary">
                  <div class="field-grid advanced-fields">
                    <label class="secondary-control">Dataset Path
                      <input name="dataset_path" placeholder="/abs/path/to/dataset">
                    </label>
                    <label class="secondary-control">Percentile Metrics
                      <input name="percentile_metrics" placeholder="ttft,tpot,itl">
                    </label>
                    <label class="secondary-control">Metric Percentiles
                      <input name="metric_percentiles" placeholder="50,90,95,99">
                    </label>
                  </div>
                  <div class="checks">
                    <label><input type="checkbox" name="enforce_eager" checked> Enforce Eager</label>
                  </div>
                </section>
              </details>

              <details class="config-disclosure raw-disclosure" open>
                <summary>Additional Raw vLLM Args</summary>
                <section class="config-card full-width secondary">
                  <label class="secondary-control">
                    <textarea name="extra_args" placeholder="--request-id-prefix bench- --metadata version=exp1"></textarea>
                  </label>
                  <div class="hint-card">
                    <strong>Execution Binary</strong>
                    This browser UI always uses the server-configured binary to avoid client-side command injection.
                    <code>__SERVER_BINARY_LABEL__</code>
                  </div>
                </section>
              </details>
            </div>
          </form>
        </section>

        <section class="results-column">
          <h2 class="section-title">Benchmark Report</h2>
          <div class="results-header-card">
            <div class="results-header-row">
              <div>
                <p class="workspace-kicker">Benchmark Output</p>
                <h2>Benchmark Results Workspace</h2>
              </div>
              <p class="results-subtitle">Run configuration above, then inspect benchmark tables, session state, and terminal output in this stacked report surface.</p>
            </div>
            <div class="command-card">
              <span class="label" id="commandPreviewLabel">Draft launch command</span>
              <div id="commandPreview">vllm bench serve --model ...</div>
            </div>
          </div>

          <div class="error-banner" id="errorBanner" role="alert" aria-live="assertive" tabindex="-1" hidden></div>
          <div class="info-banner" id="infoBanner" role="status" aria-live="polite" tabindex="-1" hidden></div>

          <section class="report-kpi-strip" aria-hidden="true">
            <div class="report-kpi">
              <span class="kpi-label">Best Throughput</span>
              <span class="kpi-value">126.3</span>
              <span class="kpi-hint">tokens/sec peak</span>
            </div>
            <div class="report-kpi">
              <span class="kpi-label">Avg TTFT</span>
              <span class="kpi-value">4,502</span>
              <span class="kpi-hint">ms median profile</span>
            </div>
            <div class="report-kpi">
              <span class="kpi-label">p95 TPOT</span>
              <span class="kpi-value">1,323</span>
              <span class="kpi-hint">ms at 8x batch</span>
            </div>
            <div class="report-kpi">
              <span class="kpi-label">Report Rows</span>
              <span class="kpi-value">24</span>
              <span class="kpi-hint">across modules</span>
            </div>
          </section>

          <section class="result-card report-module">
            <h3>Benchmark Snapshot</h3>
            <div class="status-grid">
              <div class="status-card"><div class="label">Job</div><div class="value" id="jobId">-</div></div>
              <div class="status-card"><div class="label">Status</div><div class="value" id="jobStatus" aria-live="polite">idle</div></div>
              <div class="status-card"><div class="label">Return Code</div><div class="value" id="jobCode">-</div></div>
              <div class="status-card"><div class="label">Records</div><div class="value" id="jobRecords">0</div></div>
              <div class="status-card"><div class="label">Last Update</div><div class="value" id="jobUpdated">-</div></div>
              <div class="status-card"><div class="label">Elapsed</div><div class="value" id="jobElapsed">-</div></div>
            </div>
          </section>

          <section class="result-card report-module">
            <div class="table-module-head">
              <h3>Single Request Results</h3>
              <span class="module-note">Primary benchmark table</span>
            </div>
            <div id="resultsTableWrap" tabindex="0" role="region" aria-label="Benchmark results table. Use the arrow keys to scroll horizontally."></div>
          </section>

          <section class="result-card report-module">
            <div class="table-module-head">
              <h3>Continuous Batching - Same Property</h3>
              <span class="module-note">Batch-size benchmark report</span>
            </div>
            <table class="placeholder-bench-table" aria-label="Continuous batching benchmark module">
              <thead>
                <tr>
                  <th>Batch Size</th>
                  <th>Throughput</th>
                  <th>p95 TPOT</th>
                  <th>E2E Latency</th>
                </tr>
              </thead>
              <tbody>
                <tr class="module-empty-row"><td>1x</td><td>34.5 tok/s</td><td>568 ms</td><td>1,506 ms</td></tr>
                <tr class="module-empty-row"><td>2x</td><td>63.1 tok/s</td><td>734 ms</td><td>8,910 ms</td></tr>
                <tr class="module-empty-row"><td>4x</td><td>77.9 tok/s</td><td>1,106 ms</td><td>12,530 ms</td></tr>
                <tr class="module-empty-row"><td>6x</td><td>85.7 tok/s</td><td>1,217 ms</td><td>14,842 ms</td></tr>
                <tr class="module-empty-row"><td>8x</td><td>93.4 tok/s</td><td>1,323 ms</td><td>15,790 ms</td></tr>
                <tr class="module-empty-row"><td>10x</td><td>97.8 tok/s</td><td>1,406 ms</td><td>17,633 ms</td></tr>
              </tbody>
            </table>
          </section>

          <section class="result-card report-module">
            <div class="table-module-head">
              <h3>Continuous Batching - Different Properties</h3>
              <span class="module-note">Scenario comparison report</span>
            </div>
            <table class="placeholder-bench-table" aria-label="Benchmark property variants module">
              <thead>
                <tr>
                  <th>Variant</th>
                  <th>Throughput</th>
                  <th>Avg TTFT</th>
                  <th>Avg E2E</th>
                </tr>
              </thead>
              <tbody>
                <tr class="module-empty-row"><td>Low Memory</td><td>44.7 tok/s</td><td>5,832 ms</td><td>12,610 ms</td></tr>
                <tr class="module-empty-row"><td>Balanced</td><td>109.4 tok/s</td><td>4,502 ms</td><td>13,207 ms</td></tr>
                <tr class="module-empty-row"><td>Latency Focus</td><td>88.9 tok/s</td><td>4,098 ms</td><td>12,982 ms</td></tr>
                <tr class="module-empty-row"><td>High Throughput</td><td>126.3 tok/s</td><td>3,428 ms</td><td>21,847 ms</td></tr>
                <tr class="module-empty-row"><td>Oversubscribed</td><td>131.8 tok/s</td><td>3,761 ms</td><td>24,315 ms</td></tr>
              </tbody>
            </table>
          </section>

          <section class="result-card report-module" id="resultContextCard">
            <strong id="resultContextTitle">Displayed Results</strong>
            <div id="resultContextText">No benchmark results are loaded.</div>
            <code id="resultCommand">-</code>
          </section>

          <section class="result-card report-module recovery-panel">
            <h3>Recover Session Jobs</h3>
            <div class="recovery-grid">
              <label>Session Jobs
                <select id="recoverJobSelect">
                  <option value="">No recoverable jobs discovered</option>
                </select>
              </label>
              <button type="button" class="secondary" id="refreshJobsButton">Refresh Jobs</button>
              <button type="button" class="secondary" id="adoptJobButton" disabled>Adopt Job</button>
            </div>
            <div class="exports">
              <button type="button" class="export-link" id="csvExport" disabled aria-disabled="true" tabindex="-1">Export CSV</button>
              <button type="button" class="export-link" id="jsonlExport" disabled aria-disabled="true" tabindex="-1">Export JSONL</button>
            </div>
          </section>

          <section class="result-card report-module">
            <h3>Benchmark Results (stdout / stderr)</h3>
            <div class="log-grid">
              <div>
                <h4>STDOUT</h4>
                <div class="mono" id="stdoutBox"></div>
              </div>
              <div>
                <h4>STDERR</h4>
                <div class="mono" id="stderrBox"></div>
              </div>
            </div>
          </section>
        </section>
      </div>
    </main>
  </div>

  <script>
    const serverBenchBinary = __SERVER_BINARY_PREVIEW__;
    const BROWSER_VISIBLE_VALUE_OVERRIDES = __BROWSER_VISIBLE_VALUE_OVERRIDES__;
    const form = document.getElementById("benchForm");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const commandPreview = document.getElementById("commandPreview");
    const commandPreviewLabel = document.getElementById("commandPreviewLabel");
    const errorBanner = document.getElementById("errorBanner");
    const infoBanner = document.getElementById("infoBanner");
    const overrideNotice = document.getElementById("overrideNotice");
    const csvExport = document.getElementById("csvExport");
    const jsonlExport = document.getElementById("jsonlExport");
    const recoverJobSelect = document.getElementById("recoverJobSelect");
    const refreshJobsButton = document.getElementById("refreshJobsButton");
    const adoptJobButton = document.getElementById("adoptJobButton");
    const resultsTableWrap = document.getElementById("resultsTableWrap");
    const subcommandField = form.elements.namedItem("subcommand");
    const sweepSubcommandField = form.elements.namedItem("sweep_subcommand");
    const overrideField = form.elements.namedItem("bench_path_override");
    const JOB_STORAGE_KEY = "llmbench.currentJobId";
    const JOB_FORM_STORAGE_KEY = "llmbench.currentJobForm";
    const JOB_CONTEXT_FORM_KEY = "llmbench.contextJobForm";
    const TAB_STORAGE_KEY = "llmbench.tabId";
    const storage = window.sessionStorage;
    const persistentStorage = window.localStorage;
    const RECOVERY_DELAY_MS = 1000;
    const TAB_ID_PROBE_TIMEOUT_MS = 80;
    const BROWSER_STORAGE_KEY = "llmbench.browserId";
    const SAFE_DEFAULT_BENCH_SUBCOMMANDS = new Set(["throughput"]);
    const SAFE_DEFAULT_GPU_MEMORY_UTILIZATION = "0.60";
    const SAFE_DEFAULT_MAX_MODEL_LEN = "1024";
    const REDACTED_BEARER_TOKEN = "<redacted-bearer-token>";
    const ANOTHER_TAB_JOB_MESSAGE = "This benchmark is still active in another tab for this browser profile. Return to that tab to inspect or stop it.";
    const tabIdentityChannel = typeof BroadcastChannel === "function"
      ? new BroadcastChannel("llmbench-tab-identity")
      : null;
    const tabInstanceId = `instance-${Math.random().toString(36).slice(2, 10)}`;
    let browserId = persistentStorage.getItem(BROWSER_STORAGE_KEY) || "";
    if (!browserId) {
      browserId = `browser-${Math.random().toString(36).slice(2, 12)}`;
      persistentStorage.setItem(BROWSER_STORAGE_KEY, browserId);
    }
    let lockedFormData = null;
    let contextFormData = null;
    let currentJobId = null;
    let displayedJob = null;
    let pollTimer = null;
    let pollGeneration = 0;
    let recoveryTimer = null;
    let startRequestInFlight = false;
    let discoveredRecoverableJobs = [];
    let recoverableRefreshSequence = 0;
    let tabId = storage.getItem(TAB_STORAGE_KEY) || "";
    let tabIsClosing = false;
    const tabIdentityPromise = initializeTabIdentity();

    if (tabIdentityChannel) {
      tabIdentityChannel.addEventListener("message", (event) => {
        const message = event.data;
        if (!message || typeof message !== "object") {
          return;
        }
        if (message.type !== "probe") {
          return;
        }
        if (tabIsClosing || !tabId) {
          return;
        }
        if (message.tabId !== tabId || message.instanceId === tabInstanceId) {
          return;
        }
        tabIdentityChannel.postMessage({
          type: "conflict",
          probeId: message.probeId,
          tabId: message.tabId,
          responderId: tabInstanceId,
        });
      });
    }

    const fields = Array.from(form.elements).filter((element) => element.name);
    for (const field of fields) {
      field.addEventListener("input", updatePreview);
      field.addEventListener("change", updatePreview);
    }
    recoverJobSelect.addEventListener("change", () => {
      adoptJobButton.disabled = !recoverJobSelect.value;
    });
    resultsTableWrap.addEventListener("keydown", handleResultsTableKeydown);
    window.addEventListener("pageshow", () => {
      tabIsClosing = false;
    });
    window.addEventListener("pagehide", () => {
      tabIsClosing = true;
      releaseCurrentJobOwnership();
    });
    refreshJobsButton.addEventListener("click", async () => {
      try {
        setError("");
        setInfo("Refreshing released benchmark jobs for this browser profile...");
        await refreshRecoverableJobs(true);
      } catch (error) {
        setError(error.message || "Failed to discover recoverable benchmark jobs for this browser profile.");
      }
    });
    adoptJobButton.addEventListener("click", async () => {
      const selectedJobId = recoverJobSelect.value || "";
      if (!selectedJobId) {
        return;
      }
      try {
        await adoptJob(selectedJobId);
      } catch (error) {
        setError(error.message || "Failed to adopt selected benchmark job.");
      }
    });
    updatePreview();
    resetJobSurface();
    restoreJobFromStorage();

    function generateTabId() {
      return `tab-${Math.random().toString(36).slice(2, 10)}`;
    }

    async function initializeTabIdentity() {
      if (!tabIdentityChannel) {
        if (!tabId) {
          tabId = generateTabId();
          storage.setItem(TAB_STORAGE_KEY, tabId);
        }
        return tabId;
      }
      let candidate = tabId || generateTabId();
      while (true) {
        tabId = candidate;
        storage.setItem(TAB_STORAGE_KEY, candidate);
        const conflictDetected = await probeTabIdentity(candidate);
        if (!conflictDetected) {
          return candidate;
        }
        candidate = generateTabId();
      }
    }

    function probeTabIdentity(candidate) {
      if (!tabIdentityChannel) {
        return Promise.resolve(false);
      }
      return new Promise((resolve) => {
        const probeId = `probe-${Math.random().toString(36).slice(2, 10)}`;
        let conflictDetected = false;
        const handleProbeResult = (event) => {
          const message = event.data;
          if (!message || typeof message !== "object") {
            return;
          }
          if (message.type !== "conflict") {
            return;
          }
          if (message.probeId !== probeId || message.tabId !== candidate || message.responderId === tabInstanceId) {
            return;
          }
          conflictDetected = true;
        };
        tabIdentityChannel.addEventListener("message", handleProbeResult);
        tabIdentityChannel.postMessage({
          type: "probe",
          probeId,
          tabId: candidate,
          instanceId: tabInstanceId,
        });
        window.setTimeout(() => {
          tabIdentityChannel.removeEventListener("message", handleProbeResult);
          resolve(conflictDetected);
        }, TAB_ID_PROBE_TIMEOUT_MS);
      });
    }

    async function apiFetch(resource, init = {}) {
      await tabIdentityPromise;
      const headers = new Headers(init.headers || {});
      if (browserId) {
        headers.set("X-llmbench-browser-id", browserId);
      }
      if (tabId) {
        headers.set("X-llmbench-tab-id", tabId);
      }
      headers.set("X-llmbench-tab-instance-id", tabInstanceId);
      return fetch(resource, {...init, headers});
    }

    function releaseCurrentJobOwnership() {
      if (!tabId) {
        return;
      }
      const headers = new Headers();
      if (browserId) {
        headers.set("X-llmbench-browser-id", browserId);
      }
      headers.set("X-llmbench-tab-id", tabId);
      headers.set("X-llmbench-tab-instance-id", tabInstanceId);
      try {
        fetch("/api/session/release-owned-jobs", {method: "POST", headers, keepalive: true});
      } catch (error) {
        // Best effort only.
      }
    }

    function formPayload() {
      const data = {};
      for (const element of fields) {
        if (element.type === "checkbox") {
          data[element.name] = element.checked;
        } else {
          data[element.name] = element.value.trim();
        }
      }
      return data;
    }

    function stripBrowserVisibleContextPlaceholders(source) {
      const data = {...(source || {})};
      const fieldFlagPairs = [
        ["model", "--model"],
        ["tokenizer", "--tokenizer"],
        ["base_url", "--base-url"],
      ];
      for (const [fieldName, flag] of fieldFlagPairs) {
        if (isBrowserVisiblePlaceholder(flag, data[fieldName])) {
          data[fieldName] = "";
        }
      }
      return data;
    }

    function browserVisiblePlaceholderFieldNames(source) {
      const fieldFlagPairs = [
        ["model", "--model"],
        ["tokenizer", "--tokenizer"],
        ["base_url", "--base-url"],
      ];
      return fieldFlagPairs
        .filter(([fieldName, flag]) => isBrowserVisiblePlaceholder(flag, source?.[fieldName]))
        .map(([fieldName]) => fieldName);
    }

    function buildEditableFormData(source) {
      return stripBrowserVisibleContextPlaceholders(source);
    }

    function formatProtectedFieldList(fieldNames) {
      const labels = fieldNames.map((fieldName) => {
        if (fieldName === "base_url") {
          return "Base URL";
        }
        return fieldName.charAt(0).toUpperCase() + fieldName.slice(1);
      });
      if (labels.length <= 1) {
        return labels[0] || "protected fields";
      }
      if (labels.length === 2) {
        return `${labels[0]} and ${labels[1]}`;
      }
      return `${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`;
    }

    function sanitizeBrowserVisibleContextField(fieldName, value) {
      const fieldFlagPairs = {
        model: "--model",
        tokenizer: "--tokenizer",
        base_url: "--base-url",
      };
      const flag = fieldFlagPairs[fieldName];
      if (!flag) {
        return value;
      }
      const text = String(value ?? "").trim();
      if (!text) {
        return text;
      }
      return browserVisibleCommandValue(flag, text);
    }

    function updatePreview() {
      const data = withSafeLaunchDefaults(lockedFormData || formPayload());
      commandPreview.textContent = buildLaunchCommandFromFormData(data);
      commandPreviewLabel.textContent = lockedFormData ? "Locked launch command" : "Draft launch command";
    }

    function buildLaunchCommandFromFormData(data) {
      const normalizedData = withSafeLaunchDefaults(data);
      const args = [];
      const benchPath = benchPathFromForm(normalizedData);
      const binaryTokens = splitShellWords(serverBenchBinary) || [String(serverBenchBinary || "vllm")];
      const useRandomLengthFlags = usesRandomLengthFlags(normalizedData, benchPath);
      const useStructuredLowMemoryLaunchFlags = usesStructuredLowMemoryLaunchFlags(normalizedData, benchPath);
      syncOverrideState(normalizedData);
      const append = (flag, value) => {
        if (value) args.push(flag, browserVisibleCommandValue(flag, value));
      };
      append("--model", normalizedData.model);
      append("--backend", normalizedData.backend);
      append("--base-url", normalizedData.base_url);
      append("--endpoint", effectiveEndpoint(normalizedData));
      append("--host", normalizedData.host);
      append("--port", normalizedData.port);
      if (useStructuredLowMemoryLaunchFlags) {
        append("--gpu-memory-utilization", normalizedData.gpu_memory_utilization);
        append("--max-model-len", normalizedData.max_model_len);
      }
      append("--dataset-name", normalizedData.dataset_name);
      append("--dataset-path", normalizedData.dataset_path);
      append("--tokenizer", normalizedData.tokenizer);
      append("--num-prompts", normalizedData.num_prompts);
      append(useRandomLengthFlags ? "--random-input-len" : "--input-len", normalizedData.input_len);
      append(useRandomLengthFlags ? "--random-output-len" : "--output-len", normalizedData.output_len);
      append("--request-rate", normalizedData.request_rate);
      append("--max-concurrency", normalizedData.max_concurrency);
      append("--percentile-metrics", normalizedData.percentile_metrics);
      append("--metric-percentiles", normalizedData.metric_percentiles);
      if (useStructuredLowMemoryLaunchFlags && normalizedData.enforce_eager) args.push("--enforce-eager");
      if (normalizedData.extra_args) {
        const parsedExtraArgs = splitShellWords(normalizedData.extra_args);
        if (parsedExtraArgs) {
          args.push(...sanitizeBrowserVisibleTokens(parsedExtraArgs));
        } else {
          args.push(sanitizeBrowserVisibleExtraArgsText(normalizedData.extra_args));
        }
      }
      return shellJoinArgs([...binaryTokens, "bench", ...benchPath, ...args]);
    }

    function sanitizeFormDataForBrowserStorage(source) {
      const data = {...(source || {})};
      for (const fieldName of ["model", "tokenizer", "base_url"]) {
        if (fieldName in data) {
          data[fieldName] = sanitizeBrowserVisibleContextField(fieldName, data[fieldName]);
        }
      }
      if (typeof data.extra_args === "string") {
        data.extra_args = sanitizeBrowserVisibleExtraArgsText(data.extra_args);
      }
      return data;
    }

    function sanitizeBrowserVisibleTokens(tokens) {
      const sanitized = [];
      for (let index = 0; index < tokens.length; index += 1) {
        const token = String(tokens[index] ?? "");
        const sanitizedToken = sanitizeBrowserVisibleExtraArgsText(token);
        sanitized.push(sanitizedToken);
        if (tokenHasBearerPrefixWithoutToken(sanitizedToken) && index + 1 < tokens.length) {
          sanitized.push(REDACTED_BEARER_TOKEN);
          index += 1;
        }
      }
      return sanitized;
    }

    function tokenHasBearerPrefixWithoutToken(value) {
      const text = String(value || "").trim().replace(/^['"]+|['"]+$/g, "");
      if (!text) {
        return false;
      }
      if (text.toLowerCase() === "bearer") {
        return true;
      }
      return /authorization\\s*:\\s*bearer$/i.test(text);
    }

    function sanitizeBrowserVisibleExtraArgsText(value) {
      return String(value || "").replace(/\\b(Bearer)(\\s+)([^\\s'"`]+)/gi, `$1$2${REDACTED_BEARER_TOKEN}`);
    }

    function browserVisibleCommandValue(flag, value) {
      const text = String(value || "").trim();
      if (!text) {
        return text;
      }
      if (isBrowserVisiblePlaceholder(flag, text)) {
        return text;
      }
      if (flag === "--base-url") {
        const {scheme, hostClass} = baseUrlSchemeAndHostClass(text);
        return `${BROWSER_VISIBLE_VALUE_OVERRIDES[flag]}-${scheme}-${hostClass}-${shortFingerprint(text)}`;
      }
      if ((flag === "--model" || flag === "--tokenizer") && looksLikeHostLocalPath(text)) {
        const leaf = sanitizeIdentifierSegment(pathLeafLabel(text), "path");
        return `${BROWSER_VISIBLE_VALUE_OVERRIDES[flag]}-${leaf}-${shortFingerprint(text)}`;
      }
      return text;
    }

    function shortFingerprint(value) {
      const text = String(value || "").trim();
      let hash = 2166136261;
      for (let index = 0; index < text.length; index += 1) {
        hash ^= text.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }
      return (hash >>> 0).toString(16).padStart(8, "0");
    }

    function sanitizeIdentifierSegment(value, fallback) {
      const normalized = String(value || "").trim().replace(/[^0-9A-Za-z._-]+/g, "-").replace(/^[-._]+|[-._]+$/g, "");
      return normalized || fallback;
    }

    function pathLeafLabel(value) {
      let trimmed = String(value || "").trim();
      while (trimmed.endsWith("/")) {
        trimmed = trimmed.slice(0, -1);
      }
      if (!trimmed) {
        return "path";
      }
      const parts = trimmed.split("/").filter((part) => part && part !== "~");
      if (!parts.length) {
        return "path";
      }
      return parts[parts.length - 1];
    }

    function baseUrlSchemeAndHostClass(value) {
      const text = String(value || "").trim();
      let scheme = "url";
      let hostClass = "unknown";
      try {
        const parsed = new URL(text);
        scheme = sanitizeIdentifierSegment((parsed.protocol || "").replace(":", "").toLowerCase(), "url");
        const host = (parsed.hostname || "").toLowerCase();
        if (!host) {
          hostClass = "unknown";
        } else if (host === "localhost" || host === "127.0.0.1" || host === "::1") {
          hostClass = "loopback";
        } else {
          const octetStrings = host.split(".");
          const looksLikeIpv4 = octetStrings.length === 4 && octetStrings.every((segment) => segment !== "" && /^[0-9]+$/.test(segment));
          if (!looksLikeIpv4) {
            if (host.endsWith(".local")) {
              hostClass = "local";
            } else {
              hostClass = "hostname";
            }
          } else {
            const octets = octetStrings.map((segment) => Number(segment));
            const allValid = octets.every((octet) => Number.isInteger(octet) && octet >= 0 && octet <= 255);
            if (!allValid) {
              hostClass = "public";
            } else if (octets[0] === 10 || octets[0] === 127) {
              hostClass = "loopback";
            } else if (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) {
              hostClass = "private";
            } else if (octets[0] === 192 && octets[1] === 168) {
              hostClass = "private";
            } else {
              hostClass = "public";
            }
          }
        }
      } catch (error) {
        const separator = text.indexOf("://");
        if (separator > 0) {
          const candidate = text.slice(0, separator).toLowerCase();
          if (/^[A-Za-z][A-Za-z0-9+.-]*$/.test(candidate)) {
            scheme = sanitizeIdentifierSegment(candidate, "url");
          }
        }
      }
      return {scheme, hostClass};
    }

    function looksLikeHostLocalPath(value) {
      return value.startsWith("/") || value.startsWith("~/");
    }

    function isBrowserVisiblePlaceholder(flag, value) {
      const marker = BROWSER_VISIBLE_VALUE_OVERRIDES[flag];
      if (!marker) {
        return false;
      }
      const text = String(value || "");
      return text === marker || text.startsWith(`${marker}-`);
    }

    function withSafeLaunchDefaults(source) {
      const data = {...(source || {})};
      const benchPath = benchPathFromForm(data);
      if (!usesStructuredLowMemoryLaunchFlags(data, benchPath)) {
        return data;
      }
      if (!String(data.gpu_memory_utilization || "").trim()) {
        data.gpu_memory_utilization = SAFE_DEFAULT_GPU_MEMORY_UTILIZATION;
      }
      if (!String(data.max_model_len || "").trim()) {
        data.max_model_len = SAFE_DEFAULT_MAX_MODEL_LEN;
      }
      if (data.enforce_eager === undefined || data.enforce_eager === null) {
        data.enforce_eager = true;
      }
      return data;
    }

    function usesRandomLengthFlags(data, benchPath) {
      return Boolean(
        benchPath.length
        && benchPath[0] === "throughput"
        && (data.dataset_name || "").trim() === "random"
      );
    }

    function usesStructuredLowMemoryLaunchFlags(data, benchPath) {
      const topLevel = benchPath.length ? benchPath[0] : (data.subcommand || "serve");
      return SAFE_DEFAULT_BENCH_SUBCOMMANDS.has(topLevel);
    }

    function effectiveEndpoint(data) {
      if (data.endpoint) {
        return data.endpoint;
      }
      return data.backend === "openai" ? "/v1/completions" : "";
    }

    function benchPathFromForm(data) {
      if (data.bench_path_override) {
        const parsedOverride = splitShellWords(data.bench_path_override);
        return parsedOverride && parsedOverride.length ? parsedOverride : [data.bench_path_override];
      }
      if (data.subcommand === "sweep") {
        return ["sweep", data.sweep_subcommand || "serve"];
      }
      return [data.subcommand || "serve"];
    }

    function setError(message) {
      if (!message) {
        errorBanner.hidden = true;
        errorBanner.textContent = "";
        return;
      }
      errorBanner.hidden = false;
      errorBanner.textContent = message;
      errorBanner.scrollIntoView({block: "center", inline: "nearest"});
      errorBanner.focus();
    }

    function setInfo(message) {
      if (!message) {
        infoBanner.hidden = true;
        infoBanner.textContent = "";
        return;
      }
      infoBanner.hidden = false;
      infoBanner.textContent = message;
    }

    function setExportEnabled(enabled) {
      for (const link of [csvExport, jsonlExport]) {
        link.disabled = !enabled;
        link.setAttribute("aria-disabled", enabled ? "false" : "true");
        if (enabled) {
          link.removeAttribute("tabindex");
        } else {
          link.setAttribute("tabindex", "-1");
        }
      }
    }

    function syncOverrideState(data) {
      const overrideActive = Boolean((data.bench_path_override || "").trim());
      const formLocked = Boolean(lockedFormData);
      subcommandField.disabled = formLocked || overrideActive;
      sweepSubcommandField.disabled = formLocked || overrideActive;
      overrideField.disabled = formLocked;
      overrideNotice.hidden = !overrideActive;
    }

    function resetJobSurface() {
      displayedJob = null;
      document.getElementById("jobId").textContent = "-";
      document.getElementById("jobStatus").textContent = "idle";
      document.getElementById("jobCode").textContent = "-";
      document.getElementById("jobRecords").textContent = "0";
      document.getElementById("jobUpdated").textContent = "-";
      document.getElementById("jobElapsed").textContent = "-";
      document.getElementById("stdoutBox").textContent = "No benchmark output yet.";
      document.getElementById("stderrBox").textContent = "No benchmark diagnostics yet.";
      document.getElementById("resultContextTitle").textContent = "Displayed Results";
      document.getElementById("resultContextText").textContent = "No benchmark results are loaded.";
      document.getElementById("resultCommand").textContent = "-";
      setExportEnabled(false);
      setInfo("");
      renderTable({status: "idle", rows: [], columns: []});
      syncButtons();
    }

    function persistCurrentJobId() {
      if (currentJobId) {
        storage.setItem(JOB_STORAGE_KEY, currentJobId);
      } else {
        storage.removeItem(JOB_STORAGE_KEY);
      }
    }

    function clearRecoveryTimer() {
      if (recoveryTimer) {
        window.clearTimeout(recoveryTimer);
        recoveryTimer = null;
      }
    }

    function persistLockedFormData() {
      if (lockedFormData) {
        storage.setItem(JOB_FORM_STORAGE_KEY, JSON.stringify(lockedFormData));
      } else {
        storage.removeItem(JOB_FORM_STORAGE_KEY);
      }
    }

    function persistContextFormData() {
      if (contextFormData) {
        storage.setItem(
          JOB_CONTEXT_FORM_KEY,
          JSON.stringify({job_id: contextFormData.jobId || null, data: contextFormData.data || {}})
        );
      } else {
        storage.removeItem(JOB_CONTEXT_FORM_KEY);
      }
    }

    function normalizeStoredContextForm(raw) {
      if (!raw || typeof raw !== "object") {
        return null;
      }
      if (raw.data && typeof raw.data === "object") {
        return {
          jobId: typeof raw.job_id === "string" ? raw.job_id : null,
          data: sanitizeFormDataForBrowserStorage(raw.data),
        };
      }
      return {jobId: null, data: sanitizeFormDataForBrowserStorage(raw)};
    }

    function setContextFormData(data, jobId = null) {
      if (data) {
        contextFormData = {jobId: jobId || null, data: sanitizeFormDataForBrowserStorage(data)};
      } else {
        contextFormData = null;
      }
      persistContextFormData();
    }

    function contextDataForJob(jobId) {
      if (!contextFormData || !contextFormData.data) {
        return null;
      }
      if (!contextFormData.jobId || contextFormData.jobId === (jobId || null)) {
        return contextFormData.data;
      }
      return null;
    }

    function applyFormData(data) {
      if (!data) {
        return;
      }
      for (const element of fields) {
        if (!(element.name in data)) {
          continue;
        }
        if (element.type === "checkbox") {
          element.checked = Boolean(data[element.name]);
        } else {
          element.value = String(data[element.name] ?? "");
        }
      }
      updatePreview();
    }

    function setFormLocked(data) {
      lockedFormData = data ? sanitizeFormDataForBrowserStorage(data) : null;
      for (const element of fields) {
        if (element.name === "bench_path_override" || element.name === "subcommand" || element.name === "sweep_subcommand") {
          continue;
        }
        element.disabled = Boolean(lockedFormData);
      }
      persistLockedFormData();
      updatePreview();
    }

    function isUnknownJobError(error) {
      return String(error?.message || "").includes("Unknown job id");
    }

    function isAnotherTabJobError(error) {
      return String(error?.message || "").includes(ANOTHER_TAB_JOB_MESSAGE);
    }

    function clearCurrentJobIdentity() {
      currentJobId = null;
      persistCurrentJobId();
    }

    function syncButtons() {
      const running = displayedJob && (displayedJob.status === "running" || displayedJob.status === "stopping" || displayedJob.status === "reconnecting");
      stopButton.disabled = !running;
      startButton.disabled = Boolean(lockedFormData) || startRequestInFlight;
    }

    function formatTimestamp(seconds) {
      if (!seconds) {
        return "-";
      }
      return new Date(seconds * 1000).toLocaleTimeString();
    }

    function formatElapsed(job) {
      if (!job || !job.started_at) {
        return "-";
      }
      const end = job.finished_at || (Date.now() / 1000);
      const elapsed = Math.max(0, end - job.started_at);
      if (elapsed < 1) {
        return `${Math.round(elapsed * 1000)} ms`;
      }
      if (elapsed < 60) {
        return `${elapsed.toFixed(1)} s`;
      }
      const minutes = Math.floor(elapsed / 60);
      const seconds = Math.round(elapsed % 60);
      return `${minutes}m ${seconds}s`;
    }

    function firstFailureDetailLine(text) {
      if (!text) {
        return "";
      }
      const lines = String(text)
        .split(/\\r?\\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);
      for (const line of lines) {
        const lower = line.toLowerCase();
        if (lower.startsWith("traceback")) {
          continue;
        }
        if (/^file \".*\", line \\d+, in /i.test(line)) {
          continue;
        }
        return line;
      }
      return lines[0] || "";
    }

    function textShowsAllRequestsFailed(stdout, stderr) {
      const combined = `${stdout || ""}\n${stderr || ""}`;
      if (!combined.trim()) {
        return false;
      }
      if (/all(?:\\s+benchmark)?\\s+requests\\s+failed/i.test(combined)) {
        return true;
      }
      const successMatch = combined.match(/successful requests:\\s*(\\d+)/i);
      const failedMatch = combined.match(/failed requests:\\s*(\\d+)/i);
      if (successMatch && failedMatch) {
        const successful = Number(successMatch[1]);
        const failed = Number(failedMatch[1]);
        return Number.isFinite(successful) && Number.isFinite(failed) && successful === 0 && failed > 0;
      }
      return false;
    }

    function summarizeTerminalFailure(job) {
      if (!job) {
        return "";
      }
      if (job.status === "stopped") {
        return "";
      }
      const isFailure = job.status === "failed" || (job.returncode !== null && job.returncode !== undefined && Number(job.returncode) !== 0);
      if (!isFailure) {
        if ((job.status === "completed" || job.status === "stopped") && textShowsAllRequestsFailed(job.stdout, job.stderr)) {
          return "Benchmark completed without successful requests. Verify endpoint/model/tokenizer settings or adjust launch memory controls before retrying.";
        }
        return "";
      }
      const exitCode = job.returncode ?? "-";
      const detail = firstFailureDetailLine(job.stderr || "") || firstFailureDetailLine(job.stdout || "");
      if (!detail) {
        return `Benchmark failed (exit ${exitCode}). Review stderr below for details.`;
      }
      const clipped = detail.length > 220 ? `${detail.slice(0, 217)}...` : detail;
      return `Benchmark failed (exit ${exitCode}): ${clipped}`;
    }

    function buildResultContext(job) {
      if (!job || !job.job_id) {
        return {
          title: "Displayed Results",
          text: "No benchmark results are loaded.",
          command: "-",
        };
      }
      const contextData = contextDataForJob(job.job_id) || deriveFormContextFromJob(job);
      const commandText = contextData
        ? buildLaunchCommandFromFormData(contextData)
        : shellQuoteArg(job.subcommand || "-");
      const runtimeFailureSummary = summarizeTerminalFailure(job);
      const allRequestsFailed = textShowsAllRequestsFailed(job.stdout, job.stderr);
      if (job.status === "running" || job.status === "stopping" || job.status === "reconnecting") {
        return {
          title: "Current Benchmark",
          text: `Job ${job.job_id} is ${job.status}. Results and exports will refresh when the command emits records or exits.`,
          command: commandText,
        };
      }
      const hasRows = Number(job.record_count || 0) > 0;
      if (allRequestsFailed) {
        return {
          title: "Benchmark Failed",
          text: hasRows
            ? `Job ${job.job_id} recorded zero successful requests. Structured rows below are diagnostic-only and should not be treated as valid benchmark output.`
            : `Job ${job.job_id} recorded zero successful requests and did not emit structured result rows. Review diagnostics and retry with corrected runtime settings.`,
          command: commandText,
        };
      }
      if (runtimeFailureSummary && !hasRows) {
        return {
          title: "Benchmark Failed",
          text: `Job ${job.job_id} failed before structured result rows were captured. Review the failure summary and diagnostics below, then retry with adjusted settings.`,
          command: commandText,
        };
      }
      if (job.status === "failed") {
        return {
          title: "Benchmark Failed",
          text: hasRows
            ? `Job ${job.job_id} failed. Structured rows below may be partial and should be treated as diagnostic output.`
            : `Job ${job.job_id} failed before structured result rows were captured. Review the failure summary and diagnostics below, then retry with adjusted settings.`,
          command: commandText,
        };
      }
      if (job.status === "stopped" && !hasRows) {
        return {
          title: "Benchmark Stopped",
          text: `Job ${job.job_id} was stopped before structured result rows were captured.`,
          command: commandText,
        };
      }
      return {
        title: "Displayed Results",
        text: `Results and exports below belong to job ${job.job_id}. You can edit the form to prepare another run without replacing this surface until a new benchmark is accepted.`,
        command: commandText,
      };
    }

    function enterRecoveryState(message) {
      if (!currentJobId) {
        return;
      }
      if (!displayedJob) {
        displayedJob = {status: "reconnecting"};
      } else {
        displayedJob.status = "reconnecting";
      }
      document.getElementById("jobId").textContent = currentJobId;
      document.getElementById("jobStatus").textContent = "reconnecting";
      document.getElementById("jobCode").textContent = "-";
      document.getElementById("jobUpdated").textContent = "retrying";
      setError(message);
      setInfo("Trying to reconnect to the active benchmark...");
      syncButtons();
    }

    function scheduleRecovery(message) {
      if (!currentJobId) {
        return;
      }
      if (pollTimer) {
        window.clearTimeout(pollTimer);
        pollTimer = null;
      }
      enterRecoveryState(message);
      if (recoveryTimer) {
        return;
      }
      const expectedJobId = currentJobId;
      const recoveryGeneration = pollGeneration;
      recoveryTimer = window.setTimeout(async () => {
        recoveryTimer = null;
        try {
          const job = await fetchJob(expectedJobId);
          if (recoveryGeneration !== pollGeneration || currentJobId !== expectedJobId) {
            return;
          }
          setError("");
          setJob(job);
          if (job.status === "running" || job.status === "stopping") {
            startPolling();
          }
        } catch (error) {
          if (recoveryGeneration !== pollGeneration || currentJobId !== expectedJobId) {
            return;
          }
          if (isAnotherTabJobError(error)) {
            clearCurrentJobIdentity();
            setFormLocked(null);
            resetJobSurface();
            setError("");
            setInfo(error.message || ANOTHER_TAB_JOB_MESSAGE);
            return;
          }
          if (isUnknownJobError(error)) {
            clearCurrentJobIdentity();
            setFormLocked(null);
            resetJobSurface();
            setError(error.message || "The active benchmark no longer exists.");
            return;
          }
          scheduleRecovery(error.message || "Failed to reconnect to the active benchmark.");
        }
      }, RECOVERY_DELAY_MS);
    }

    async function restoreJobFromStorage() {
      const storedJobId = storage.getItem(JOB_STORAGE_KEY);
      const storedForm = storage.getItem(JOB_FORM_STORAGE_KEY);
      const storedContextForm = storage.getItem(JOB_CONTEXT_FORM_KEY);
      if (storedForm) {
        try {
          const parsed = JSON.parse(storedForm);
          const sanitized = sanitizeFormDataForBrowserStorage(parsed);
          lockedFormData = sanitized;
          applyFormData(sanitized);
        } catch (error) {
          storage.removeItem(JOB_FORM_STORAGE_KEY);
          lockedFormData = null;
        }
      }
      if (storedContextForm) {
        try {
          const parsedContext = normalizeStoredContextForm(JSON.parse(storedContextForm));
          if (parsedContext) {
            contextFormData = parsedContext;
            persistContextFormData();
            if (!storedForm && !storedJobId) {
              applyFormData(buildEditableFormData(parsedContext.data));
            }
          } else {
            storage.removeItem(JOB_CONTEXT_FORM_KEY);
            contextFormData = null;
          }
        } catch (error) {
          storage.removeItem(JOB_CONTEXT_FORM_KEY);
          contextFormData = null;
        }
      }
      if (!storedJobId) {
        currentJobId = null;
        persistCurrentJobId();
        setFormLocked(null);
        try {
          const jobs = await refreshRecoverableJobs(false);
          if (jobs.length) {
        setInfo(`Found ${jobs.length} recoverable benchmark job(s) for this browser profile. Select one and click Adopt Job to inspect and control it here.`);
      } else {
        setInfo("");
      }
        } catch (error) {
          renderRecoverableJobs([]);
            setInfo("Unable to discover recoverable benchmark jobs right now. You can retry with Refresh Jobs.");
        }
        return;
      }
      try {
        currentJobId = storedJobId;
        const job = await fetchJob(storedJobId);
        setJob(job);
        if (job.status === "running" || job.status === "stopping") {
          startPolling();
        } else {
          try {
            await refreshRecoverableJobs(false);
          } catch (ignored) {
            renderRecoverableJobs([]);
          }
        }
      } catch (error) {
        currentJobId = storedJobId;
        if (isAnotherTabJobError(error)) {
          clearCurrentJobIdentity();
          setFormLocked(null);
          resetJobSurface();
          setError("");
          setInfo(error.message || ANOTHER_TAB_JOB_MESSAGE);
          try {
            await refreshRecoverableJobs(false);
          } catch (ignored) {
            renderRecoverableJobs([]);
          }
          return;
        }
        if (isUnknownJobError(error)) {
          clearCurrentJobIdentity();
          setFormLocked(null);
          resetJobSurface();
          setError(error.message || "Stored benchmark job no longer exists.");
          try {
            await refreshRecoverableJobs(false);
          } catch (ignored) {
            renderRecoverableJobs([]);
          }
          return;
        }
        scheduleRecovery(error.message || "Failed to reconnect to the active benchmark.");
      }
    }

    async function fetchJobs(statuses, limit, recoverableOnly = false) {
      const params = new URLSearchParams();
      if (statuses && statuses.length) {
        params.set("status", statuses.join(","));
      }
      if (limit) {
        params.set("limit", String(limit));
      }
      if (recoverableOnly) {
        params.set("recoverable_only", "1");
      }
      const query = params.toString();
      const response = await apiFetch(`/api/jobs${query ? "?" + query : ""}`);
      return parseResponse(response, "Failed to discover benchmark jobs");
    }

    function formatRecoverableJobLabel(job) {
      const started = formatTimestamp(job.started_at);
      const status = job.status || "unknown";
      const records = Number(job.record_count || 0);
      return `${job.job_id} | ${status} | started ${started} | records ${records}`;
    }

    function renderRecoverableJobs(jobs) {
      const previouslySelectedJobId = recoverJobSelect.value || "";
      discoveredRecoverableJobs = Array.isArray(jobs)
        ? jobs.filter((job) => job && job.job_id && job.job_id !== currentJobId)
        : [];
      recoverJobSelect.innerHTML = "";
      if (!discoveredRecoverableJobs.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No recoverable jobs discovered";
        recoverJobSelect.appendChild(option);
        adoptJobButton.disabled = true;
        return;
      }
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select a job...";
      recoverJobSelect.appendChild(placeholder);
      for (const job of discoveredRecoverableJobs) {
        const option = document.createElement("option");
        option.value = job.job_id;
        option.textContent = formatRecoverableJobLabel(job);
        recoverJobSelect.appendChild(option);
      }
      const canPreserveSelection = discoveredRecoverableJobs.some((job) => job.job_id === previouslySelectedJobId);
      recoverJobSelect.value = canPreserveSelection ? previouslySelectedJobId : "";
      adoptJobButton.disabled = !recoverJobSelect.value;
    }

    async function discoverRecoverableJobs() {
      const payload = await fetchJobs([], 200, true);
      return Array.isArray(payload.jobs) ? payload.jobs.filter((job) => job && job.job_id) : [];
    }

    async function refreshRecoverableJobs(announceSummary) {
      const requestId = ++recoverableRefreshSequence;
      const jobs = await discoverRecoverableJobs();
      if (requestId !== recoverableRefreshSequence) {
        return jobs;
      }
      renderRecoverableJobs(jobs);
      if (announceSummary) {
        if (!jobs.length) {
          setInfo("No released benchmark jobs are waiting for adoption in this browser profile.");
        } else {
          setInfo(`Found ${jobs.length} recoverable benchmark job(s) for this browser profile. Select one and click Adopt Job to inspect and control it here.`);
        }
      }
      return jobs;
    }

    async function adoptJob(jobId) {
      if (!jobId) {
        return;
      }
      clearPoll();
      setError("");
      setInfo("Adopting selected benchmark job...");
      const job = await fetchJob(jobId);
      currentJobId = job.job_id || null;
      // Force recovered tabs to rebuild context from the job payload rather than
      // reusing stale draft/localStorage context from another tab.
      setContextFormData(null);
      setJob(job);
      if (job.status === "running" || job.status === "stopping") {
        startPolling();
      } else {
        clearPoll();
      }
      setInfo(`Now displaying job ${job.job_id}.`);
    }

    async function parseResponse(response, fallbackMessage) {
      const text = await response.text();
      let payload = {};
      if (text) {
        try {
          payload = JSON.parse(text);
        } catch (error) {
          if (!response.ok) {
            throw new Error(text.trim() || fallbackMessage);
          }
          throw new Error(`${fallbackMessage}: server returned non-JSON content.`);
        }
      }
      if (!response.ok) {
        const apiMessage = payload && typeof payload === "object" ? payload.error : "";
        throw new Error(apiMessage || text.trim() || fallbackMessage);
      }
      return payload;
    }

    async function fetchJob(jobId) {
      const response = await apiFetch(`/api/jobs/${jobId}`);
      return parseResponse(response, "Failed to load benchmark status");
    }

    function getDisplayedExportJobId() {
      if (!displayedJob || !displayedJob.job_id) {
        return null;
      }
      if (displayedJob.status === "running" || displayedJob.status === "stopping" || displayedJob.record_count <= 0) {
        return null;
      }
      return displayedJob.job_id;
    }

    async function downloadExport(format, jobId) {
      if (!jobId) return;
      try {
        setError("");
        const response = await apiFetch(`/api/jobs/${jobId}/export.${format}`);
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text.trim() || `Failed to export ${format.toUpperCase()}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `llmbench-${jobId}.${format}`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        setError(error.message || `Failed to export ${format.toUpperCase()}`);
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      setError("");
      setInfo("Submitting benchmark launch request...");
      const snapshot = withSafeLaunchDefaults(formPayload());
      const protectedFieldNames = browserVisiblePlaceholderFieldNames(snapshot);
      if (protectedFieldNames.length) {
        setError(
          `${formatProtectedFieldList(protectedFieldNames)} still contain browser-visible placeholders from the displayed job. Re-enter real values before starting another benchmark.`
        );
        setInfo(displayedJob && displayedJob.job_id ? "The previous benchmark result is still displayed below." : "");
        return;
      }
      setContextFormData(snapshot, null);
      const launchSnapshot = stripBrowserVisibleContextPlaceholders(snapshot);
      startRequestInFlight = true;
      syncButtons();
      try {
        const response = await apiFetch("/api/jobs", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(launchSnapshot)
        });
        const job = await parseResponse(response, "Failed to start benchmark");
        clearPoll();
        const derivedSnapshot = deriveFormContextFromJob(job);
        const effectiveSnapshot = derivedSnapshot ? {...derivedSnapshot, ...snapshot} : snapshot;
        setContextFormData(effectiveSnapshot, job.job_id || null);
        applyFormData(effectiveSnapshot);
        setFormLocked(effectiveSnapshot);
        currentJobId = job.job_id || null;
        setJob(job);
        if (job.status === "running") {
          startPolling();
        } else {
          clearPoll();
        }
      } catch (error) {
        setError(error.message || "Failed to start benchmark");
        setInfo(displayedJob && displayedJob.job_id ? "The previous benchmark result is still displayed below." : "");
      } finally {
        startRequestInFlight = false;
        syncButtons();
      }
    });

    stopButton.addEventListener("click", async () => {
      if (!currentJobId) return;
      try {
        setError("");
        setInfo("Stop requested. Waiting for the benchmark process to exit...");
        document.getElementById("jobStatus").textContent = "stopping";
        if (displayedJob) {
          displayedJob.status = "stopping";
        }
        syncButtons();
        const response = await apiFetch(`/api/jobs/${currentJobId}/stop`, {method: "POST"});
        const job = await parseResponse(response, "Failed to stop benchmark");
        setJob(job);
      } catch (error) {
        setError(error.message || "Failed to stop benchmark");
        setInfo("Stop did not reach the server. The benchmark may still be running. You can retry Stop.");
        document.getElementById("jobStatus").textContent = "running";
        if (displayedJob) {
          displayedJob.status = "running";
        }
        syncButtons();
      }
    });

    csvExport.addEventListener("click", async () => {
      if (csvExport.disabled) return;
      await downloadExport("csv", getDisplayedExportJobId());
    });

    jsonlExport.addEventListener("click", async () => {
      if (jsonlExport.disabled) return;
      await downloadExport("jsonl", getDisplayedExportJobId());
    });

    function startPolling() {
      if (!currentJobId) return;
      pollGeneration += 1;
      const generation = pollGeneration;
      clearRecoveryTimer();
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
      syncButtons();
      const pollOnce = async () => {
        if (!currentJobId || generation !== pollGeneration) {
          return;
        }
        const requestedJobId = currentJobId;
        try {
          const job = await fetchJob(requestedJobId);
          if (generation !== pollGeneration || currentJobId !== requestedJobId) {
            return;
          }
          setJob(job);
          if (job.status !== "running" && job.status !== "stopping") {
            clearPoll();
            return;
          }
          pollTimer = window.setTimeout(pollOnce, 1000);
        } catch (error) {
          if (generation !== pollGeneration || currentJobId !== requestedJobId) {
            return;
          }
          if (isAnotherTabJobError(error)) {
            clearCurrentJobIdentity();
            setFormLocked(null);
            clearPoll();
            resetJobSurface();
            setError("");
            setInfo(error.message || ANOTHER_TAB_JOB_MESSAGE);
            return;
          }
          if (isUnknownJobError(error)) {
            clearCurrentJobIdentity();
            setFormLocked(null);
            clearPoll();
            resetJobSurface();
            setError(error.message || "The active benchmark no longer exists.");
            return;
          }
          scheduleRecovery(error.message || "Failed to load benchmark status");
        }
      };
      pollTimer = window.setTimeout(pollOnce, 1000);
    }

    function clearPoll() {
      pollGeneration += 1;
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
      clearRecoveryTimer();
      syncButtons();
    }

    function setJob(job) {
      clearRecoveryTimer();
      displayedJob = job;
      currentJobId = job.job_id || null;
      persistCurrentJobId();
      const storedContext = contextDataForJob(job.job_id);
      const derivedContext = storedContext ? null : deriveFormContextFromJob(job);
      const jobContext = storedContext || derivedContext;
      if (derivedContext) {
        setContextFormData(derivedContext, job.job_id || null);
      }
      if (job.status === "running" || job.status === "stopping") {
        const lockSource = jobContext || lockedFormData || formPayload();
        applyFormData(lockSource);
        setFormLocked(lockSource);
      } else {
        setFormLocked(null);
        if (jobContext) {
          applyFormData(buildEditableFormData(jobContext));
        }
      }
      const failureSummary = summarizeTerminalFailure(job);
      if (failureSummary) {
        setError(failureSummary);
      } else {
        setError("");
      }
      updatePreview();
      document.getElementById("jobId").textContent = job.job_id || "-";
      document.getElementById("jobStatus").textContent = job.status || "-";
      document.getElementById("jobCode").textContent = job.returncode ?? "-";
      document.getElementById("jobRecords").textContent = job.record_count ?? 0;
      document.getElementById("jobUpdated").textContent = formatTimestamp(job.finished_at || (Date.now() / 1000));
      document.getElementById("jobElapsed").textContent = formatElapsed(job);
      const context = buildResultContext(job);
      document.getElementById("resultContextTitle").textContent = context.title;
      document.getElementById("resultContextText").textContent = context.text;
      document.getElementById("resultCommand").textContent = context.command;
      if (job.stdout) {
        document.getElementById("stdoutBox").textContent = job.stdout;
      } else if (job.status === "running" || job.status === "stopping" || job.status === "reconnecting") {
        document.getElementById("stdoutBox").textContent = "The benchmark is running. Live stdout will appear here as data arrives.";
      } else {
        document.getElementById("stdoutBox").textContent = "No stdout output was captured for this run.";
      }
      if (job.stderr) {
        document.getElementById("stderrBox").textContent = job.stderr;
      } else if (job.status === "running" || job.status === "stopping" || job.status === "reconnecting") {
        document.getElementById("stderrBox").textContent = "The benchmark is running. Live stderr will appear here as data arrives.";
      } else {
        document.getElementById("stderrBox").textContent = "No stderr diagnostics were captured for this run.";
      }
      if (job.job_id && job.record_count > 0 && job.status !== "running") {
        setExportEnabled(true);
      } else {
        setExportEnabled(false);
      }

      renderTable(job);
      renderRecoverableJobs(discoveredRecoverableJobs);
      if (job.status !== "running" && job.status !== "stopping") {
        clearPoll();
      } else {
        syncButtons();
      }
    }

    function renderTable(job) {
      const columns = job.columns || [];
      const rows = job.rows || [];
      const wrap = document.getElementById("resultsTableWrap");
      const allRequestsFailed = textShowsAllRequestsFailed(job.stdout, job.stderr);
      if (!rows.length || !columns.length) {
        if (allRequestsFailed) {
          setInfo("No structured rows were emitted because every benchmark request failed. See the failure summary and diagnostics below.");
          wrap.innerHTML = "<p>No structured result rows were emitted because all benchmark requests failed. See the failure summary and diagnostics below.</p>";
          return;
        }
        if (job.status === "completed") {
          setInfo("This benchmark completed successfully, but this command path did not emit structured rows for the results table or file exports. Inspect stdout and stderr for the primary output.");
          wrap.innerHTML = "<p>No structured result rows were emitted for this benchmark command.</p>";
          return;
        }
        if (job.status === "failed") {
          setInfo("");
          wrap.innerHTML = "<p>Benchmark failed before structured result rows were emitted. See the failure summary and diagnostics below.</p>";
          return;
        }
        setInfo("");
        wrap.innerHTML = "<p>No completed benchmark rows yet.</p>";
        return;
      }
      if (allRequestsFailed) {
        setInfo("Benchmark emitted rows but recorded zero successful requests. Treat table values as diagnostics and fix runtime configuration before comparing performance.");
      } else {
        setInfo("");
      }
      const header = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
      const body = rows.map((row) => {
        return `<tr>${columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("")}</tr>`;
      }).join("");
      wrap.innerHTML = `<table><thead>${header}</thead><tbody>${body}</tbody></table>`;
    }

    function handleResultsTableKeydown(event) {
      const wrap = event.currentTarget;
      if (!wrap || wrap.scrollWidth <= wrap.clientWidth) {
        return;
      }
      const step = Math.max(120, Math.floor(wrap.clientWidth * 0.35));
      if (event.key === "ArrowRight") {
        wrap.scrollBy({left: step, behavior: "auto"});
        event.preventDefault();
        return;
      }
      if (event.key === "ArrowLeft") {
        wrap.scrollBy({left: -step, behavior: "auto"});
        event.preventDefault();
        return;
      }
      if (event.key === "End") {
        wrap.scrollTo({left: wrap.scrollWidth, behavior: "auto"});
        event.preventDefault();
        return;
      }
      if (event.key === "Home") {
        wrap.scrollTo({left: 0, behavior: "auto"});
        event.preventDefault();
      }
    }

    function deriveFormContextFromJob(job) {
      if (!job) {
        return null;
      }
      let pathTokens = [];
      let tokens = [];
      if (Array.isArray(job.bench_path) && job.bench_path.length) {
        pathTokens = job.bench_path.map((token) => String(token));
        tokens = Array.isArray(job.raw_args) ? job.raw_args.map((token) => String(token)) : [];
      } else if (Array.isArray(job.command) && job.command.length) {
        const benchIndex = job.command.indexOf("bench");
        if (benchIndex < 0 || benchIndex + 1 >= job.command.length) {
          return null;
        }
        const commandTokens = job.command.slice(benchIndex + 1).map((token) => String(token));
        let index = 0;
        while (index < commandTokens.length && !commandTokens[index].startsWith("-")) {
          pathTokens.push(commandTokens[index]);
          index += 1;
        }
        tokens = commandTokens.slice(index);
      }
      if (!pathTokens.length) {
        return null;
      }

      const topLevel = new Set(["serve", "latency", "mm-processor", "startup", "sweep", "throughput"]);
      const sweepChoices = new Set(["serve", "serve_workload", "startup", "plot", "plot_pareto"]);
      const data = {};
      const subcommand = pathTokens[0];
      if (topLevel.has(subcommand)) {
        data.subcommand = subcommand;
        if (subcommand === "sweep") {
          if (pathTokens.length > 1 && sweepChoices.has(pathTokens[1])) {
            data.sweep_subcommand = pathTokens[1];
          } else if (pathTokens.length > 1) {
            data.bench_path_override = shellJoinArgs(pathTokens);
          }
        } else if (pathTokens.length > 1) {
          data.bench_path_override = shellJoinArgs(pathTokens);
        }
      } else {
        data.bench_path_override = shellJoinArgs(pathTokens);
      }

      const valueFlags = {
        "--model": "model",
        "--backend": "backend",
        "--endpoint": "endpoint",
        "--base-url": "base_url",
        "--host": "host",
        "--port": "port",
        "--gpu-memory-utilization": "gpu_memory_utilization",
        "--max-model-len": "max_model_len",
        "--dataset-name": "dataset_name",
        "--dataset-path": "dataset_path",
        "--tokenizer": "tokenizer",
        "--input-len": "input_len",
        "--output-len": "output_len",
        "--random-input-len": "input_len",
        "--random-output-len": "output_len",
        "--num-prompts": "num_prompts",
        "--request-rate": "request_rate",
        "--max-concurrency": "max_concurrency",
        "--percentile-metrics": "percentile_metrics",
        "--metric-percentiles": "metric_percentiles",
      };
      const boolFlags = {
        "--enforce-eager": "enforce_eager",
      };
      const ignoredFlags = new Set(["--save-result"]);
      const extraValueFlags = new Set(["--result-dir", "--result-filename", "--output-json", "--output-dir", "-o"]);
      const allowsDashValue = new Set(["--output-json"]);
      const canConsumeValue = (flag, tokenValue) => {
        if (tokenValue === undefined) {
          return false;
        }
        if (!tokenValue.startsWith("-")) {
          return true;
        }
        return allowsDashValue.has(flag) && tokenValue === "-";
      };
      const extraTokens = [];
      let index = 0;
      while (index < tokens.length) {
        const token = tokens[index];
        if (ignoredFlags.has(token)) {
          index += 1;
          continue;
        }
        if (token === "-o") {
          if (canConsumeValue(token, tokens[index + 1])) {
            extraTokens.push(token, tokens[index + 1]);
            index += 2;
            continue;
          }
          extraTokens.push(token);
          index += 1;
          continue;
        }
          if (token.startsWith("--")) {
          if (token.includes("=")) {
            const splitAt = token.indexOf("=");
            const name = token.slice(0, splitAt);
            const value = token.slice(splitAt + 1);
            if (Object.prototype.hasOwnProperty.call(valueFlags, name)) {
              data[valueFlags[name]] = value;
              index += 1;
              continue;
            }
            if (!ignoredFlags.has(name)) {
              extraTokens.push(token);
            }
            index += 1;
            continue;
          }
          if (Object.prototype.hasOwnProperty.call(boolFlags, token)) {
            data[boolFlags[token]] = true;
            index += 1;
            continue;
          }
          if (Object.prototype.hasOwnProperty.call(valueFlags, token)) {
            if (canConsumeValue(token, tokens[index + 1])) {
              const value = tokens[index + 1];
              data[valueFlags[token]] = value;
              index += 2;
              continue;
            }
            extraTokens.push(token);
            index += 1;
            continue;
          }
          if (canConsumeValue(token, tokens[index + 1]) && extraValueFlags.has(token)) {
            extraTokens.push(token, tokens[index + 1]);
            index += 2;
            continue;
          }
          extraTokens.push(token);
          index += 1;
          continue;
        }
        if (Object.prototype.hasOwnProperty.call(boolFlags, token)) {
          data[boolFlags[token]] = true;
          index += 1;
          continue;
        }
        if (token.startsWith("-") && canConsumeValue(token, tokens[index + 1])) {
          extraTokens.push(token, tokens[index + 1]);
          index += 2;
          continue;
        }
        extraTokens.push(token);
        index += 1;
      }
      if (extraTokens.length) {
        data.extra_args = shellJoinArgs(extraTokens);
      }
      return data;
    }

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function splitShellWords(value) {
      const text = String(value || "");
      const tokens = [];
      const backslash = String.fromCharCode(92);
      let current = "";
      let quote = null;

      for (let index = 0; index < text.length; index += 1) {
        const char = text[index];
        if (quote) {
          if (char === quote) {
            quote = null;
            continue;
          }
          if (char === backslash && quote === '"' && index + 1 < text.length) {
            index += 1;
            current += text[index];
            continue;
          }
          current += char;
          continue;
        }
        if (/\\s/.test(char)) {
          if (current) {
            tokens.push(current);
            current = "";
          }
          continue;
        }
        if (char === "'" || char === '"') {
          quote = char;
          continue;
        }
        if (char === backslash && index + 1 < text.length) {
          index += 1;
          current += text[index];
          continue;
        }
        current += char;
      }

      if (quote) {
        return null;
      }
      if (current) {
        tokens.push(current);
      }
      return tokens;
    }

    function shellQuoteArg(value) {
      const text = String(value ?? "");
      const singleQuoteEscape = "'" + '"' + "'" + '"' + "'";
      if (!text) {
        return "''";
      }
      if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) {
        return text;
      }
      return "'" + text.replaceAll("'", singleQuoteEscape) + "'";
    }

    function shellJoinArgs(args) {
      return args.map((arg) => shellQuoteArg(arg)).join(" ");
    }
  </script>
</body>
</html>
"""
