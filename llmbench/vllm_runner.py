from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


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
    "serve_sla",
    "serve_workload",
    "startup",
)

OUTPUT_JSON_BENCH_COMMANDS = (
    "latency",
    "mm-processor",
    "startup",
    "throughput",
)

SWEEP_OUTPUT_DIR_COMMANDS = (
    "serve",
    "serve_sla",
    "serve_workload",
    "startup",
)

SHORT_OPTION_ALIASES = {
    "-o": "--output-dir",
}


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
    capture = _build_capture_config(bench_path, raw_args, artifact_dir, capture_results)
    _prepare_capture_target(capture)
    command = _split_launch_command(vllm_binary) + ["bench", *bench_path] + list(raw_args)
    command.extend(capture.command_args)
    return command, capture


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
        if stream_output:
            returncode, stdout, stderr = _run_and_capture_streaming(
                command,
                live_stdout=live_stdout,
                live_stderr=live_stderr,
            )
        else:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
    except OSError as exc:
        return BenchmarkExecution(
            job_id=job_id,
            subcommand=" ".join(bench_path) or "bench",
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
        status = "completed" if returncode == 0 else "failed"
        stderr_text = stderr
        effective_returncode = returncode
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
        return _parse_record_content(stdout, strict=True, source_label="stdout result stream")
    if capture.mode in {"serve_result", "output_json"} and capture.target_path and capture.target_path.exists():
        if _is_stale_signature(capture, capture.target_path):
            return []
        return _parse_record_content(
            capture.target_path.read_text(encoding="utf-8"),
            strict=True,
            source_label=str(capture.target_path),
        )
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


def _parse_record_content(content: str, *, strict: bool = False, source_label: str = "benchmark output") -> list[dict[str, Any]]:
    content = content.strip()
    if not content:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed_line = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_line, dict):
                records.append(parsed_line)
        if records or not strict:
            return records
        raise ValueError(f"{source_label} did not contain valid JSON benchmark records")
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    if strict:
        raise ValueError(f"{source_label} did not contain valid JSON benchmark records")
    return []


def _build_capture_config(bench_path: list[str], raw_args: list[str], artifact_dir: Path, capture_results: bool) -> CaptureConfig:
    if not capture_results:
        return CaptureConfig(mode="none", target_path=None, command_args=[], baseline_signatures={})
    if not bench_path:
        raise ValueError("A vLLM bench command path is required when collecting results.")

    options, flags = _extract_cli_options(raw_args)
    head = bench_path[0]
    if head == "serve":
        result_dir = Path(options.get("--result-dir", str(artifact_dir)))
        result_name = options.get("--result-filename", "vllm-result.json")
        result_path = result_dir / result_name
        command_args: list[str] = []
        if "--save-result" not in flags:
            command_args.append("--save-result")
        if "--result-dir" not in options:
            command_args.extend(["--result-dir", str(result_dir)])
        if "--result-filename" not in options:
            command_args.extend(["--result-filename", result_name])
        capture = CaptureConfig(
            mode="serve_result",
            target_path=result_path,
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
    for subcommand in SWEEP_OUTPUT_DIR_COMMANDS:
        subdir = target_path / subcommand
        if not subdir.exists() or not subdir.is_dir():
            continue
        json_paths.extend(path for path in sorted(subdir.glob("run-*.json")) if path.is_file())
        summary_path = subdir / "summary.json"
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
    value_options = {"--result-dir", "--result-filename", "--output-json", "--output-dir"}
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token == "--":
            index += 1
            continue
        if token.startswith("--"):
            if "=" in token:
                name, value = token.split("=", 1)
                if value:
                    options[name] = value
                else:
                    flags.add(name)
                index += 1
                continue
            if index + 1 < len(raw_args) and (not raw_args[index + 1].startswith("-") or (token in value_options and raw_args[index + 1] == "-")):
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


def _run_and_capture_streaming(
    command: list[str],
    *,
    live_stdout: TextIO | None,
    live_stderr: TextIO | None,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
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
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def _tee_stream(stream, sink: list[str], live_target: TextIO | None) -> None:
    try:
        for line in iter(stream.readline, ""):
            sink.append(line)
            if live_target is not None:
                live_target.write(line)
                live_target.flush()
    finally:
        stream.close()


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
        return parts

    first = Path(parts[0])
    if first.exists() or shutil.which(parts[0]):
        return parts

    return [command]


def _prepare_capture_target(capture: CaptureConfig) -> None:
    if not capture.target_path:
        return
    if _capture_uses_stdout(capture):
        return
    _validate_capture_target(capture)
    if capture.mode in {"serve_result", "output_json"}:
        capture.target_path.parent.mkdir(parents=True, exist_ok=True)
    elif capture.mode == "output_dir":
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
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            execution = BenchmarkExecution(
                job_id=job_id,
                subcommand=" ".join(bench_path) or "bench",
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
            execution.returncode = process.returncode
            execution.finished_at = time.time()
            execution.records = (
                load_result_records(
                    capture,
                    execution.stdout,
                    allow_stdout_fallback=(_capture_uses_stdout(capture) and process.returncode == 0),
                )
                if process.returncode == 0
                else []
            )
            execution.result_path = _resolved_result_path(capture) if process.returncode == 0 else None
            if job_id in self._requested_stop:
                execution.status = "stopped"
            else:
                execution.status = "completed" if process.returncode == 0 else "failed"
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
        while True:
            chunk = await stream.readline()
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            if field == "stdout":
                execution.stdout += text
            else:
                execution.stderr += text

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
        try:
            process.terminate()
        except ProcessLookupError:
            execution.returncode = process.returncode
            execution.finished_at = execution.finished_at or time.time()
            execution.status = "completed" if process.returncode == 0 else "failed"
            self._processes.pop(job_id, None)
            return execution
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        return execution

    def get_job(self, job_id: str) -> BenchmarkExecution | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BenchmarkExecution]:
        return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)

    def export_records(self, job_id: str) -> list[dict[str, Any]]:
        execution = self._jobs[job_id]
        return execution.records
