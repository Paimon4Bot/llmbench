from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from codecs import getincrementaldecoder
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO


TOP_LEVEL_BENCH_COMMANDS = (
    "latency",
    "mm-processor",
    "serve",
    "startup",
    "sweep",
    "throughput",
)

SWEEP_BENCH_COMMANDS = (
    "serve",
    "serve_workload",
    "startup",
    "plot",
    "plot_pareto",
)

OUTPUT_JSON_BENCH_COMMANDS = (
    "latency",
    "mm-processor",
    "startup",
    "throughput",
)

SWEEP_OUTPUT_DIR_COMMANDS = (
    "serve",
    "serve_workload",
    "startup",
)

SHORT_OPTION_ALIASES = {
    "-o": "--output-dir",
}

CAPTURE_PATH_FLAGS = {"--result-dir", "--output-json", "--output-dir"}
VALUE_OPTIONS = {"--result-dir", "--result-filename", "--output-json", "--output-dir"}
SUCCESSFUL_REQUEST_KEYS = {
    "completed",
    "completedrequests",
    "successful",
    "successfulrequests",
    "numsuccessfulrequests",
    "successfulrequestcount",
    "succeeded",
}
FAILED_REQUEST_KEYS = {
    "failed",
    "failures",
    "failedrequests",
    "numfailedrequests",
    "failedrequestcount",
}
SUCCESSFUL_REQUESTS_RE = re.compile(r"Successful requests\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
FAILED_REQUESTS_RE = re.compile(r"Failed requests\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
ALL_REQUESTS_FAILED_RE = re.compile(r"all requests failed", re.IGNORECASE)
ALL_REQUESTS_FAILED_MESSAGE = (
    "llmbench detected that all benchmark requests failed; treating the run as failed. "
    "This usually means the benchmark target was unreachable or misconfigured."
)


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    mode: str
    target_path: Path | None
    command_args: list[str]
    baseline_signatures: dict[str, tuple[int, int]]


@dataclass(slots=True)
class BenchmarkExecution:
    job_id: str
    subcommand: str
    bench_path: list[str]
    raw_args: list[str]
    command: list[str]
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    started_at: float
    finished_at: float | None
    artifact_dir: str
    result_path: str | None
    records: list[dict[str, Any]] = field(default_factory=list)
    streamed_output: bool = False
    deferred_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("deferred_stderr", None)
        payload.pop("streamed_output", None)
        payload["records_preview"] = self.records[:5]
        payload["record_count"] = len(self.records)
        return payload


def build_vllm_command(
    *,
    vllm_binary: str,
    bench_path: list[str],
    raw_args: list[str],
    artifact_dir: Path,
    capture_results: bool = True,
) -> tuple[list[str], CaptureConfig]:
    if capture_results and not bench_path:
        raise ValueError("A vLLM bench command path is required when collecting results.")
    normalized_raw_args = _normalize_capture_path_args(raw_args)
    normalized_raw_args = _rewrite_output_json_dash_args(bench_path, normalized_raw_args, artifact_dir)
    options, _ = _extract_cli_options(normalized_raw_args)
    _validate_explicit_result_destination_values(options)
    capture = _build_capture_config(bench_path, normalized_raw_args, artifact_dir, capture_results)
    _prepare_capture_target(capture)
    command = _split_launch_command(vllm_binary) + ["bench", *bench_path] + list(normalized_raw_args)
    command.extend(capture.command_args)
    return command, capture


def build_launch_env(command: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    if not command:
        return env
    binary = Path(command[0]).expanduser()
    if binary.exists() and binary.parent.is_dir():
        current_path = env.get("PATH", "")
        path_parts = [part for part in current_path.split(os.pathsep) if part]
        binary_dir = str(binary.parent)
        if binary_dir not in path_parts:
            env["PATH"] = os.pathsep.join([binary_dir, *path_parts]) if path_parts else binary_dir
    return env


def run_benchmark_sync(
    *,
    vllm_binary: str,
    bench_path: list[str],
    raw_args: list[str],
    artifact_root: Path,
    capture_results: bool = True,
    stream_output: bool = False,
    live_stdout: TextIO | None = None,
    live_stderr: TextIO | None = None,
) -> BenchmarkExecution:
    job_id = f"cli-{uuid.uuid4().hex[:8]}"
    artifact_dir = artifact_root / job_id
    started_at = time.time()
    try:
        command, capture = build_vllm_command(
            vllm_binary=vllm_binary,
            bench_path=bench_path,
            raw_args=raw_args,
            artifact_dir=artifact_dir,
            capture_results=capture_results,
        )
    except (OSError, ValueError) as exc:
        return BenchmarkExecution(
            job_id=job_id,
            subcommand=" ".join(bench_path) or "bench",
            bench_path=list(bench_path),
            raw_args=list(raw_args),
            command=[],
            status="failed",
            returncode=2,
            stdout="",
            stderr=f"Failed to prepare vLLM benchmark invocation: {exc}",
            started_at=started_at,
            finished_at=time.time(),
            artifact_dir=str(artifact_dir),
            result_path=None,
            records=[],
        )
    try:
        launch_env = build_launch_env(command)
        if stream_output:
            returncode, stdout, stderr = _run_and_capture_streaming(
                command,
                env=launch_env,
                live_stdout=live_stdout,
                live_stderr=live_stderr,
            )
        else:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                start_new_session=True,
                env=launch_env,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
    except KeyboardInterrupt:
        return BenchmarkExecution(
            job_id=job_id,
            subcommand=" ".join(bench_path) or "bench",
            bench_path=list(bench_path),
            raw_args=list(raw_args),
            command=command,
            status="failed",
            returncode=130,
            stdout="",
            stderr="Interrupted by user.",
            started_at=started_at,
            finished_at=time.time(),
            artifact_dir=str(artifact_dir),
            result_path=None,
            records=[],
        )
    except OSError as exc:
        return BenchmarkExecution(
            job_id=job_id,
            subcommand=" ".join(bench_path) or "bench",
            bench_path=list(bench_path),
            raw_args=list(raw_args),
            command=command,
            status="failed",
            returncode=127,
            stdout="",
            stderr=f"Failed to launch vLLM binary '{vllm_binary}': {exc}",
            started_at=started_at,
            finished_at=time.time(),
            artifact_dir=str(artifact_dir),
            result_path=str(capture.target_path) if capture.target_path else None,
            records=[],
        )

    try:
        records = (
            load_result_records(
                capture,
                stdout,
                allow_stdout_fallback=(_capture_uses_stdout(capture) and returncode == 0),
            )
            if (capture_results and returncode == 0)
            else []
        )
        status, effective_returncode, stderr_text = _classify_execution_outcome(
            returncode=returncode,
            records=records,
            stdout=stdout,
            stderr=stderr,
        )
        result_path = _resolved_result_path(capture) if returncode == 0 else None
        deferred_stderr = ""
    except Exception as exc:
        records = []
        status = "failed"
        effective_returncode = returncode or 1
        result_path = None
        failure_message = f"llmbench failed to load benchmark results: {exc}"
        stderr_text = f"{stderr}\n{failure_message}" if stderr else failure_message
        deferred_stderr = failure_message

    return BenchmarkExecution(
        job_id=job_id,
        subcommand=" ".join(bench_path) or "bench",
        bench_path=list(bench_path),
        raw_args=list(raw_args),
        command=command,
        status=status,
        returncode=effective_returncode,
        stdout=stdout,
        stderr=stderr_text,
        started_at=started_at,
        finished_at=time.time(),
        artifact_dir=str(artifact_dir),
        result_path=result_path,
        records=records,
        streamed_output=stream_output,
        deferred_stderr=deferred_stderr,
    )


def load_result_records(capture: CaptureConfig, stdout: str = "", *, allow_stdout_fallback: bool = True) -> list[dict[str, Any]]:
    if _capture_uses_stdout(capture):
        # Upstream may interleave human-readable lines on stdout even when emitting JSON
        # records. Ignore plainly non-JSON diagnostics, but fail if JSON-shaped records are
        # malformed so partial exports are never treated as success.
        return _parse_record_content(
            stdout,
            strict=True,
            source_label="stdout result stream",
            ignore_non_json_lines=True,
        )
    if capture.mode in {"serve_result", "output_json"} and capture.target_path and capture.target_path.exists():
        if _is_stale_signature(capture, capture.target_path):
            return []
        return _parse_record_content(
            capture.target_path.read_text(encoding="utf-8"),
            strict=True,
            source_label=str(capture.target_path),
        )
    if capture.mode == "serve_result_dir" and capture.target_path and capture.target_path.exists():
        records: list[dict[str, Any]] = []
        for path in _iter_serve_result_json_paths(capture.target_path):
            if _is_stale_signature(capture, path):
                continue
            records.extend(
                _parse_record_content(
                    path.read_text(encoding="utf-8"),
                    strict=True,
                    source_label=str(path),
                )
            )
        return records
    if capture.mode == "output_dir" and capture.target_path and capture.target_path.exists():
        records: list[dict[str, Any]] = []
        for path in _iter_output_dir_json_paths(capture.target_path):
            if _is_stale_signature(capture, path):
                continue
            records.extend(
                _parse_record_content(
                    path.read_text(encoding="utf-8"),
                    strict=True,
                    source_label=str(path),
                )
            )
        return records
    if not allow_stdout_fallback:
        return []
    return _parse_record_content(stdout)


def _classify_execution_outcome(
    *,
    returncode: int,
    records: list[dict[str, Any]],
    stdout: str,
    stderr: str,
) -> tuple[str, int, str]:
    status = "completed" if returncode == 0 else "failed"
    effective_returncode = returncode
    stderr_text = stderr
    if returncode == 0 and _benchmark_requests_failed(records, stdout=stdout, stderr=stderr):
        status = "failed"
        effective_returncode = 1
        if ALL_REQUESTS_FAILED_MESSAGE not in stderr_text:
            stderr_text = f"{stderr_text}\n{ALL_REQUESTS_FAILED_MESSAGE}" if stderr_text else ALL_REQUESTS_FAILED_MESSAGE
    return status, effective_returncode, stderr_text


def _benchmark_requests_failed(records: list[dict[str, Any]], *, stdout: str, stderr: str) -> bool:
    return _all_benchmark_requests_failed(records) or _text_shows_all_requests_failed(stdout, stderr)


def _all_benchmark_requests_failed(records: list[dict[str, Any]]) -> bool:
    return any(_payload_shows_all_requests_failed(record) for record in records)


def _payload_shows_all_requests_failed(payload: Any) -> bool:
    if isinstance(payload, dict):
        numeric_by_key: dict[str, float] = {}
        for key, value in payload.items():
            normalized_key = _normalize_request_metric_key(key)
            numeric_value = _coerce_numeric_metric(value)
            if normalized_key and numeric_value is not None:
                numeric_by_key[normalized_key] = numeric_value
        successful_requests = _first_numeric_metric(numeric_by_key, SUCCESSFUL_REQUEST_KEYS)
        failed_requests = _first_numeric_metric(numeric_by_key, FAILED_REQUEST_KEYS)
        if successful_requests is not None and failed_requests is not None:
            return successful_requests <= 0 and failed_requests > 0
        return any(_payload_shows_all_requests_failed(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_shows_all_requests_failed(item) for item in payload)
    return False


def _normalize_request_metric_key(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _coerce_numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _first_numeric_metric(values: dict[str, float], candidate_keys: set[str]) -> float | None:
    for key in candidate_keys:
        if key in values:
            return values[key]
    return None


def _text_shows_all_requests_failed(stdout: str, stderr: str) -> bool:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    if not combined:
        return False
    successful = _extract_request_metric_from_text(SUCCESSFUL_REQUESTS_RE, combined)
    failed = _extract_request_metric_from_text(FAILED_REQUESTS_RE, combined)
    if successful is not None and failed is not None:
        return successful <= 0 and failed > 0
    return bool(ALL_REQUESTS_FAILED_RE.search(combined))


def _extract_request_metric_from_text(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_record_content(
    content: str,
    *,
    strict: bool = False,
    source_label: str = "benchmark output",
    ignore_non_json_lines: bool = False,
) -> list[dict[str, Any]]:
    content = content.strip()
    if not content:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        records = []
        ignored_non_json_line = False
        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed_line = json.loads(line)
            except json.JSONDecodeError:
                if ignore_non_json_lines and not _looks_like_json_record_fragment(line):
                    ignored_non_json_line = True
                    continue
                if strict:
                    raise ValueError(f"{source_label} contained invalid JSON on line {line_number}")
                continue
            if isinstance(parsed_line, dict):
                records.append(parsed_line)
            elif strict:
                raise ValueError(f"{source_label} contained a non-object JSON record on line {line_number}")
        if records or not strict or ignored_non_json_line:
            return records
        raise ValueError(f"{source_label} did not contain valid JSON benchmark records")
    if isinstance(parsed, list):
        dict_items = [item for item in parsed if isinstance(item, dict)]
        if strict and len(dict_items) != len(parsed):
            raise ValueError(f"{source_label} did not contain valid JSON benchmark records")
        return dict_items
    if isinstance(parsed, dict):
        return [parsed]
    if strict:
        raise ValueError(f"{source_label} did not contain valid JSON benchmark records")
    return []


def _looks_like_json_record_fragment(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped[0] in {'{', '[', '"'}:
        return True
    if stripped[0].isdigit():
        return True
    if stripped[0] == "-" and len(stripped) > 1 and stripped[1].isdigit():
        return True
    return stripped.startswith("true") or stripped.startswith("false") or stripped.startswith("null")


def _build_capture_config(bench_path: list[str], raw_args: list[str], artifact_dir: Path, capture_results: bool) -> CaptureConfig:
    if not capture_results:
        return CaptureConfig(mode="none", target_path=None, command_args=[], baseline_signatures={})
    if not bench_path:
        raise ValueError("A vLLM bench command path is required when collecting results.")

    options, flags = _extract_cli_options(raw_args)
    head = bench_path[0]
    if head == "serve":
        result_dir = Path(options.get("--result-dir", str(artifact_dir)))
        result_name = options.get("--result-filename")
        command_args: list[str] = []
        if "--save-result" not in flags:
            command_args.append("--save-result")
        if "--result-dir" not in options:
            command_args.extend(["--result-dir", str(result_dir)])
        if result_name:
            capture = CaptureConfig(
                mode="serve_result",
                target_path=result_dir / result_name,
                command_args=command_args,
                baseline_signatures={},
            )
        else:
            capture = CaptureConfig(
                mode="serve_result_dir",
                target_path=result_dir,
                command_args=command_args,
                baseline_signatures={},
            )
        return _with_capture_baseline(capture)
    if head in OUTPUT_JSON_BENCH_COMMANDS:
        result_path = Path(options.get("--output-json", str(artifact_dir / "vllm-result.json")))
        capture = CaptureConfig(
            mode="output_json",
            target_path=result_path,
            command_args=[] if "--output-json" in options else ["--output-json", str(result_path)],
            baseline_signatures={},
        )
        return _with_capture_baseline(capture)
    if head == "sweep" and len(bench_path) > 1 and bench_path[1] in SWEEP_OUTPUT_DIR_COMMANDS:
        output_dir = Path(options.get("--output-dir", str(artifact_dir)))
        capture = CaptureConfig(
            mode="output_dir",
            target_path=output_dir,
            command_args=[] if "--output-dir" in options else ["--output-dir", str(output_dir)],
            baseline_signatures={},
        )
        return _with_capture_baseline(capture)
    return CaptureConfig(mode="none", target_path=None, command_args=[], baseline_signatures={})


def _resolved_result_path(capture: CaptureConfig) -> str | None:
    if not capture.target_path:
        return None
    if _capture_uses_stdout(capture):
        return str(capture.target_path)
    if capture.mode in {"serve_result", "output_json"}:
        if not capture.target_path.exists():
            return None
        return None if _is_stale_signature(capture, capture.target_path) else str(capture.target_path)
    if capture.mode == "serve_result_dir":
        if not capture.target_path.exists():
            return None
        for path in _iter_serve_result_json_paths(capture.target_path):
            if not _is_stale_signature(capture, path):
                return str(path)
        return None
    if capture.mode == "output_dir":
        if not capture.target_path.exists():
            return None
        for path in _iter_output_dir_json_paths(capture.target_path):
            if not _is_stale_signature(capture, path):
                return str(capture.target_path)
        return None
    return None


def _iter_output_dir_json_paths(target_path: Path) -> list[Path]:
    json_paths: list[Path] = []
    candidate_dirs: list[Path] = []
    for subcommand in SWEEP_OUTPUT_DIR_COMMANDS:
        subdir = target_path / subcommand
        if not subdir.exists() or not subdir.is_dir():
            continue
        candidate_dirs.append(subdir)
    for experiment_dir in sorted(path for path in target_path.iterdir() if path.is_dir() and path.name not in SWEEP_OUTPUT_DIR_COMMANDS):
        candidate_dirs.extend(path for path in sorted(experiment_dir.iterdir()) if path.is_dir())
    for candidate in candidate_dirs:
        json_paths.extend(path for path in sorted(candidate.glob("run*.json")) if path.is_file())
        summary_path = candidate / "summary.json"
        if summary_path.exists() and summary_path.is_file():
            json_paths.append(summary_path)
    if json_paths:
        # Preserve deterministic order while deduplicating.
        deduped = list(dict.fromkeys(json_paths))
        return deduped
    root_summary = target_path / "summary.json"
    if root_summary.exists() and root_summary.is_file():
        return [root_summary]
    return []


def _iter_serve_result_json_paths(target_path: Path) -> list[Path]:
    if not target_path.exists() or not target_path.is_dir():
        return []
    return [path for path in sorted(target_path.glob("*.json")) if path.is_file()]


def _path_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _snapshot_capture_signatures(capture: CaptureConfig) -> dict[str, tuple[int, int]]:
    target = capture.target_path
    if target is None or _capture_uses_stdout(capture):
        return {}
    signatures: dict[str, tuple[int, int]] = {}
    if capture.mode in {"serve_result", "output_json"}:
        if target.exists() and target.is_file():
            signatures[str(target)] = _path_signature(target)
        return signatures
    if capture.mode == "serve_result_dir" and target.exists() and target.is_dir():
        for path in _iter_serve_result_json_paths(target):
            signatures[str(path)] = _path_signature(path)
        return signatures
    if capture.mode == "output_dir" and target.exists() and target.is_dir():
        for path in _iter_output_dir_json_paths(target):
            if path.is_file():
                signatures[str(path)] = _path_signature(path)
    return signatures


def _with_capture_baseline(capture: CaptureConfig) -> CaptureConfig:
    baseline = _snapshot_capture_signatures(capture)
    return CaptureConfig(
        mode=capture.mode,
        target_path=capture.target_path,
        command_args=list(capture.command_args),
        baseline_signatures=baseline,
    )


def _is_stale_signature(capture: CaptureConfig, path: Path) -> bool:
    key = str(path)
    if key not in capture.baseline_signatures:
        return False
    try:
        current = _path_signature(path)
    except OSError:
        return False
    return current == capture.baseline_signatures[key]


def _extract_cli_options(raw_args: list[str]) -> tuple[dict[str, str], set[str]]:
    options: dict[str, str] = {}
    flags: set[str] = set()
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token == "--":
            index += 1
            continue
        if token.startswith("--"):
            if "=" in token:
                name, value = token.split("=", 1)
                if name in VALUE_OPTIONS:
                    options[name] = value
                elif value:
                    options[name] = value
                else:
                    flags.add(name)
                index += 1
                continue
            if index + 1 < len(raw_args) and (not raw_args[index + 1].startswith("-") or (token in VALUE_OPTIONS and raw_args[index + 1] == "-")):
                options[token] = raw_args[index + 1]
                index += 2
                continue
            flags.add(token)
            index += 1
            continue
        if token.startswith("-"):
            canonical = SHORT_OPTION_ALIASES.get(token)
            if canonical:
                if index + 1 < len(raw_args) and (not raw_args[index + 1].startswith("-") or raw_args[index + 1] == "-"):
                    options[canonical] = raw_args[index + 1]
                    index += 2
                    continue
                flags.add(canonical)
                index += 1
                continue
            if len(token) > 2:
                short_flag = token[:2]
                canonical = SHORT_OPTION_ALIASES.get(short_flag)
                if canonical:
                    attached_value = token[2:]
                    if attached_value.startswith("="):
                        attached_value = attached_value[1:]
                    if canonical in VALUE_OPTIONS:
                        options[canonical] = attached_value
                        index += 1
                        continue
                    if attached_value:
                        options[canonical] = attached_value
                        index += 1
                        continue
            index += 1
            continue
        if not token.startswith("-"):
            index += 1
            continue
    return options, flags


def _normalize_capture_path_args(raw_args: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token.startswith("--"):
            if "=" in token:
                name, value = token.split("=", 1)
                if name in CAPTURE_PATH_FLAGS:
                    value = _expand_capture_path_value(value)
                normalized.append(f"{name}={value}")
                index += 1
                continue
            normalized.append(token)
            if token in CAPTURE_PATH_FLAGS and index + 1 < len(raw_args):
                normalized.append(_expand_capture_path_value(raw_args[index + 1]))
                index += 2
                continue
            index += 1
            continue
        if token == "-o":
            normalized.append(token)
            if index + 1 < len(raw_args):
                normalized.append(_expand_capture_path_value(raw_args[index + 1]))
                index += 2
            else:
                index += 1
            continue
        if token.startswith("-o="):
            value = token.split("=", 1)[1]
            normalized.append(f"-o={_expand_capture_path_value(value)}")
            index += 1
            continue
        if token.startswith("-o") and len(token) > 2 and token[2] != "=":
            normalized.append(f"-o{_expand_capture_path_value(token[2:])}")
            index += 1
            continue
        normalized.append(token)
        index += 1
    return normalized


def _rewrite_output_json_dash_args(bench_path: list[str], raw_args: list[str], artifact_dir: Path) -> list[str]:
    if not bench_path or bench_path[0] not in OUTPUT_JSON_BENCH_COMMANDS:
        return raw_args
    rewritten: list[str] = []
    fallback_path = str(artifact_dir / "vllm-result.json")
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token == "--output-json" and index + 1 < len(raw_args) and raw_args[index + 1] == "-":
            rewritten.extend(["--output-json", fallback_path])
            index += 2
            continue
        if token.startswith("--output-json=") and token.split("=", 1)[1] == "-":
            rewritten.append(f"--output-json={fallback_path}")
            index += 1
            continue
        rewritten.append(token)
        index += 1
    return rewritten


def _expand_capture_path_value(value: str) -> str:
    if value in {"", "-"}:
        return value
    return str(Path(value).expanduser())


def _validate_explicit_result_destination_values(options: dict[str, str]) -> None:
    for flag in VALUE_OPTIONS:
        if flag in options and options[flag] == "":
            raise ValueError(f"{flag} requires a non-empty value.")


def _build_proc_children_index() -> dict[int, set[int]]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return {}
    children_by_parent: dict[int, set[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        parent_pid: int | None = None
        for line in status.splitlines():
            if not line.startswith("PPid:"):
                continue
            try:
                parent_pid = int(line.split(":", 1)[1].strip())
            except ValueError:
                parent_pid = None
            break
        if parent_pid is None:
            continue
        pid = int(entry.name)
        children_by_parent.setdefault(parent_pid, set()).add(pid)
    return children_by_parent


def _collect_process_tree_pids(*root_pids: int) -> set[int]:
    frontier = {pid for pid in root_pids if pid > 0}
    if not frontier:
        return set()
    collected = set(frontier)
    children_by_parent = _build_proc_children_index()
    while frontier:
        next_frontier: set[int] = set()
        for pid in frontier:
            for child_pid in children_by_parent.get(pid, ()):
                if child_pid in collected:
                    continue
                collected.add(child_pid)
                next_frontier.add(child_pid)
        frontier = next_frontier
    return collected


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live_processes(pids: set[int]) -> set[int]:
    return {pid for pid in pids if _process_exists(pid)}


def _signal_process_tree(root_pid: int, sig: int) -> set[int]:
    tracked_pids = _collect_process_tree_pids(root_pid)
    _signal_process_group(root_pid, sig)
    _signal_pids(tracked_pids - {root_pid}, sig)
    return tracked_pids


def _signal_pids(pids: set[int], sig: int) -> None:
    for pid in sorted(pids):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _terminate_process_tree_sync(
    process: subprocess.Popen[str],
    *,
    graceful_timeout: float = 3,
    force_timeout: float = 3,
) -> None:
    tracked_pids = _signal_process_tree(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        pass
    remaining = _live_processes(_collect_process_tree_pids(*tracked_pids))
    if process.returncode is None and _process_exists(process.pid):
        remaining.add(process.pid)
    if not remaining:
        return
    _signal_process_group(process.pid, signal.SIGKILL)
    _signal_pids(remaining - {process.pid}, signal.SIGKILL)
    try:
        process.wait(timeout=force_timeout)
    except subprocess.TimeoutExpired:
        pass


async def _terminate_process_tree_async(
    process: asyncio.subprocess.Process,
    *,
    graceful_timeout: float = 3,
    force_timeout: float = 3,
) -> None:
    tracked_pids = _signal_process_tree(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=graceful_timeout)
    except asyncio.TimeoutError:
        pass
    remaining = _live_processes(_collect_process_tree_pids(*tracked_pids))
    if process.returncode is None and _process_exists(process.pid):
        remaining.add(process.pid)
    if not remaining:
        return
    _signal_process_group(process.pid, signal.SIGKILL)
    _signal_pids(remaining - {process.pid}, signal.SIGKILL)
    try:
        await asyncio.wait_for(process.wait(), timeout=force_timeout)
    except asyncio.TimeoutError:
        pass


def _run_and_capture_streaming(
    command: list[str],
    *,
    env: dict[str, str],
    live_stdout: TextIO | None,
    live_stderr: TextIO | None,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
        env=env,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    stdout_thread = threading.Thread(
        target=_tee_stream,
        args=(process.stdout, stdout_chunks, live_stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_tee_stream,
        args=(process.stderr, stderr_chunks, live_stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        _terminate_process_tree_sync(process)
        returncode = 130
    _join_tee_threads(
        stdout_thread,
        stderr_thread,
        process.stdout,
        process.stderr,
    )
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def _tee_stream(stream: BinaryIO, sink: list[str], live_target: TextIO | None) -> None:
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = _read_stream_chunk(stream)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if not text:
                continue
            sink.append(text)
            if live_target is not None:
                live_target.write(text)
                live_target.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            sink.append(tail)
            if live_target is not None:
                live_target.write(tail)
                live_target.flush()
    except (OSError, ValueError):
        # Stream may be closed by shutdown logic to unblock stuck descendants.
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _read_stream_chunk(stream: BinaryIO, size: int = 4096) -> bytes:
    read1 = getattr(stream, "read1", None)
    if callable(read1):
        return read1(size)
    return stream.read(size)


def _join_tee_threads(
    stdout_thread: threading.Thread,
    stderr_thread: threading.Thread,
    stdout_stream,
    stderr_stream,
) -> None:
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    if stdout_thread.is_alive() and stdout_stream is not None:
        try:
            stdout_stream.close()
        except OSError:
            pass
    if stderr_thread.is_alive() and stderr_stream is not None:
        try:
            stderr_stream.close()
        except OSError:
            pass
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)


def _signal_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (AttributeError, ProcessLookupError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _split_launch_command(vllm_binary: str) -> list[str]:
    command = vllm_binary.strip()
    if not command:
        raise ValueError("The vLLM binary cannot be empty.")

    literal_path = Path(command).expanduser()
    if literal_path.exists():
        return [str(literal_path)]

    parts = shlex.split(command)
    if not parts:
        raise ValueError("The vLLM binary cannot be empty.")

    parts[0] = str(Path(parts[0]).expanduser())
    if len(parts) <= 1:
        resolved = _resolve_default_vllm_binary(parts[0])
        return [resolved] if resolved else parts

    first = Path(parts[0])
    if first.exists() or shutil.which(parts[0]):
        return parts

    return [command]


def _resolve_default_vllm_binary(binary_name: str) -> str | None:
    if Path(binary_name).name != binary_name:
        return None
    if shutil.which(binary_name):
        return binary_name
    sibling_candidates: list[Path] = []
    seen_candidates: set[str] = set()
    for origin in (sys.executable, sys.argv[0] if sys.argv else ""):
        origin_text = str(origin or "").strip()
        if not origin_text:
            continue
        origin_path = Path(origin_text).expanduser()
        has_separator = os.sep in origin_text or (os.altsep is not None and os.altsep in origin_text)
        if origin_path.is_absolute():
            absolute_origin = origin_path
        elif has_separator:
            absolute_origin = (Path.cwd() / origin_path).absolute()
        else:
            continue
        for candidate in (absolute_origin.with_name(binary_name),):
            candidate_key = str(candidate)
            if candidate_key not in seen_candidates:
                seen_candidates.add(candidate_key)
                sibling_candidates.append(candidate)
        try:
            resolved_candidate = absolute_origin.resolve().with_name(binary_name)
        except OSError:
            continue
        resolved_key = str(resolved_candidate)
        if resolved_key not in seen_candidates:
            seen_candidates.add(resolved_key)
            sibling_candidates.append(resolved_candidate)
    for candidate in sibling_candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _prepare_capture_target(capture: CaptureConfig) -> None:
    if not capture.target_path:
        return
    if _capture_uses_stdout(capture):
        return
    _validate_capture_target(capture)
    if capture.mode in {"serve_result", "output_json"}:
        capture.target_path.parent.mkdir(parents=True, exist_ok=True)
    elif capture.mode in {"serve_result_dir", "output_dir"}:
        capture.target_path.mkdir(parents=True, exist_ok=True)


def _validate_capture_target(capture: CaptureConfig) -> None:
    target = capture.target_path
    if target is None:
        return
    if capture.mode in {"serve_result", "output_json"}:
        if target.exists():
            if target.is_dir():
                raise IsADirectoryError(f"{target} is a directory")
            if not target.is_file():
                raise OSError(f"{target} is not a regular file")
            with target.open("rb"):
                pass
        return
    if capture.mode == "serve_result_dir":
        if target.exists() and not target.is_dir():
            raise NotADirectoryError(f"{target} is not a directory")
        return
    if capture.mode == "output_dir" and target.exists() and not target.is_dir():
        raise NotADirectoryError(f"{target} is not a directory")


def _capture_uses_stdout(capture: CaptureConfig) -> bool:
    return capture.mode == "output_json" and capture.target_path == Path("-")


class JobManager:
    def __init__(self, artifact_root: Path, default_vllm_binary: str) -> None:
        self._artifact_root = artifact_root
        self._default_vllm_binary = default_vllm_binary
        self._jobs: dict[str, BenchmarkExecution] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._requested_stop: set[str] = set()
        artifact_root.mkdir(parents=True, exist_ok=True)

    async def start_job(self, *, bench_path: list[str], raw_args: list[str], vllm_binary: str | None = None) -> BenchmarkExecution:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        artifact_dir = self._artifact_root / job_id
        started_at = time.time()
        try:
            command, capture = build_vllm_command(
                vllm_binary=vllm_binary or self._default_vllm_binary,
                bench_path=bench_path,
                raw_args=raw_args,
                artifact_dir=artifact_dir,
            )
        except (OSError, ValueError) as exc:
            execution = BenchmarkExecution(
                job_id=job_id,
                subcommand=" ".join(bench_path) or "bench",
                bench_path=list(bench_path),
                raw_args=list(raw_args),
                command=[],
                status="failed",
                returncode=2,
                stdout="",
                stderr=f"Failed to prepare vLLM benchmark invocation: {exc}",
                started_at=started_at,
                finished_at=time.time(),
                artifact_dir=str(artifact_dir),
                result_path=None,
                records=[],
            )
            self._jobs[job_id] = execution
            return execution
        try:
            launch_env = build_launch_env(command)
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                env=launch_env,
            )
        except Exception as exc:
            execution = BenchmarkExecution(
                job_id=job_id,
                subcommand=" ".join(bench_path) or "bench",
                bench_path=list(bench_path),
                raw_args=list(raw_args),
                command=command,
                status="failed",
                returncode=127,
                stdout="",
                stderr=f"Failed to launch vLLM binary '{vllm_binary or self._default_vllm_binary}': {exc}",
                started_at=started_at,
                finished_at=time.time(),
                artifact_dir=str(artifact_dir),
                result_path=str(capture.target_path) if capture.target_path else None,
                records=[],
            )
            self._jobs[job_id] = execution
            return execution

        execution = BenchmarkExecution(
            job_id=job_id,
            subcommand=" ".join(bench_path) or "bench",
            bench_path=list(bench_path),
            raw_args=list(raw_args),
            command=command,
            status="running",
            returncode=None,
            stdout="",
            stderr="",
            started_at=started_at,
            finished_at=None,
            artifact_dir=str(artifact_dir),
            result_path=str(capture.target_path) if capture.target_path else None,
            records=[],
        )
        self._jobs[job_id] = execution
        self._processes[job_id] = process
        asyncio.create_task(self._wait_for_job(job_id, capture))
        return execution

    async def _wait_for_job(self, job_id: str, capture: CaptureConfig) -> None:
        process = self._processes[job_id]
        execution = self._jobs[job_id]

        stdout_task = asyncio.create_task(self._stream_process_output(process.stdout, execution, "stdout"))
        stderr_task = asyncio.create_task(self._stream_process_output(process.stderr, execution, "stderr"))
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task)

        try:
            records = (
                load_result_records(
                    capture,
                    execution.stdout,
                    allow_stdout_fallback=(_capture_uses_stdout(capture) and process.returncode == 0),
                )
                if process.returncode == 0
                else []
            )
            status, effective_returncode, stderr_text = _classify_execution_outcome(
                returncode=process.returncode or 0,
                records=records,
                stdout=execution.stdout,
                stderr=execution.stderr,
            )
            execution.returncode = effective_returncode
            execution.finished_at = time.time()
            execution.records = records
            execution.result_path = _resolved_result_path(capture) if process.returncode == 0 else None
            execution.stderr = stderr_text
            if job_id in self._requested_stop:
                execution.status = "stopped"
            else:
                execution.status = status
        except Exception as exc:
            execution.returncode = process.returncode
            execution.finished_at = time.time()
            execution.records = []
            execution.result_path = None
            if execution.stderr:
                execution.stderr = f"{execution.stderr}\nllmbench failed to load benchmark results: {exc}"
            else:
                execution.stderr = f"llmbench failed to load benchmark results: {exc}"
            execution.status = "failed"
        finally:
            self._processes.pop(job_id, None)

    async def _stream_process_output(
        self,
        stream: asyncio.StreamReader | None,
        execution: BenchmarkExecution,
        field: str,
    ) -> None:
        if stream is None:
            return
        def append_text(target: str, text: str) -> None:
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            if target == "stdout":
                execution.stdout += normalized
            else:
                execution.stderr += normalized
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            append_text(field, chunk.decode("utf-8", errors="replace"))

    async def stop_job(self, job_id: str) -> BenchmarkExecution:
        execution = self._jobs[job_id]
        process = self._processes.get(job_id)
        if not process:
            return execution
        if process.returncode is not None:
            execution.returncode = process.returncode
            execution.finished_at = execution.finished_at or time.time()
            if execution.status in {"running", "stopping"}:
                execution.status = "completed" if process.returncode == 0 else "failed"
            self._processes.pop(job_id, None)
            return execution
        self._requested_stop.add(job_id)
        execution.status = "stopping"
        await _terminate_process_tree_async(process)
        return execution

    def get_job(self, job_id: str) -> BenchmarkExecution | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BenchmarkExecution]:
        return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)

    def export_records(self, job_id: str) -> list[dict[str, Any]]:
        execution = self._jobs[job_id]
        return execution.records
