from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import TextIO

from llmbench.exporters import records_to_csv, records_to_jsonl, records_to_stdout
from llmbench.vllm_runner import (
    SWEEP_BENCH_COMMANDS,
    TOP_LEVEL_BENCH_COMMANDS,
    run_benchmark_sync,
)


REDACTED_BEARER_TOKEN = "<redacted-bearer-token>"
REDACTED_AUTH_VALUE = "<redacted-auth-value>"
_BEARER_INLINE_TOKEN_RE = re.compile(r"(?i)\b(Bearer)(\s+)([^\s'\"`]+)")
_AUTH_HEADER_VALUE_RE = re.compile(r"(?i)\b(Authorization\s*[:=]\s*)([^\s'\"`,\]\)]+(?:\s+[^\s'\"`,\]\)]+)?)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmbench", description="CLI and web wrapper around vLLM bench.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    cli_parser = subparsers.add_parser(
        "cli",
        add_help=False,
        help="Run vLLM bench from the command line.",
        description="Run a vLLM bench command path and pass through all unknown upstream arguments.",
        epilog=(
            "Wrapper help: llmbench cli --app-help\n"
            "Upstream help: llmbench cli serve --help\n"
            "Nested benchmark paths: llmbench cli sweep serve ..."
        ),
    )
    cli_parser.add_argument("--app-vllm-binary", default="vllm")
    cli_parser.add_argument("--app-artifact-root")
    cli_parser.add_argument("--app-output", choices=["stdout", "csv", "jsonl"], default="stdout")
    cli_parser.add_argument("--app-output-path")
    cli_parser.add_argument("--app-raw-args", default="")
    cli_parser.add_argument("--app-bench-path", default="", help="Explicit vLLM bench path, such as 'serve' or 'sweep serve'.")
    cli_parser.add_argument("--app-help", action="store_true", help="Show wrapper help instead of forwarding help to vLLM.")

    web_parser = subparsers.add_parser("web", help="Start the browser UI.")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8080)
    web_parser.add_argument("--vllm-binary", default="vllm")
    web_parser.add_argument("--artifact-root", default=".llmbench_runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.mode == "cli":
        return _run_cli(args, unknown)
    if args.mode == "web":
        return _run_web(args)
    parser.error("Unsupported mode")
    return 2


def _run_cli(args: argparse.Namespace, unknown: list[str]) -> int:
    if args.app_help:
        print(_render_cli_help())
        return 0

    raw_args = list(unknown)
    if args.app_raw_args:
        try:
            raw_args.extend(shlex.split(args.app_raw_args))
        except ValueError as exc:
            print(f"llmbench cli: invalid --app-raw-args: {exc}", file=sys.stderr)
            return 2
    if _is_wrapper_help_request(raw_args, args.app_bench_path):
        print(_render_cli_help())
        return 0
    try:
        bench_path, raw_args = _extract_cli_invocation(raw_args, args.app_bench_path)
    except ValueError as exc:
        print(f"llmbench cli: {exc}", file=sys.stderr)
        print("Use `llmbench cli --app-help` for wrapper help.", file=sys.stderr)
        return 2
    raw_args = _rewrite_random_length_args_for_throughput(bench_path, raw_args)

    forward_help = any(token in {"-h", "--help"} for token in raw_args)
    forward_help = forward_help or any(token.startswith("--help=") for token in raw_args)
    if not bench_path and not forward_help:
        print("llmbench cli: missing vLLM bench command path", file=sys.stderr)
        print("Use `llmbench cli --app-help` for wrapper help.", file=sys.stderr)
        return 2

    structured_output_mode = args.app_output in {"csv", "jsonl"}
    if args.app_output_path and not structured_output_mode:
        print(
            "llmbench cli: --app-output-path requires --app-output csv or jsonl.",
            file=sys.stderr,
        )
        return 2

    output_path: Path | None = None
    if args.app_output_path:
        try:
            output_path = _prepare_output_path(Path(args.app_output_path))
        except OSError as exc:
            print(f"llmbench cli: failed to prepare output file '{args.app_output_path}': {exc}", file=sys.stderr)
            return 1

    artifact_root = _resolve_cli_artifact_root(args.app_artifact_root)

    stdout_result_stream = _uses_stdout_result_stream(bench_path, raw_args) and not forward_help
    # File exports are only supported for structured modes and should keep stdout
    # machine-safe for shell pipelines and redirection workflows.
    stream_child_stdout_to_stdout = not structured_output_mode and output_path is None
    # Keep bearer/auth fragments redacted in both live forwarding and wrapper
    # summaries without changing the stdout/stderr destination contract.
    live_stdout: TextIO = _SensitiveRedactingStream(sys.stdout if stream_child_stdout_to_stdout else sys.stderr)
    live_stderr: TextIO = _SensitiveRedactingStream(sys.stderr)
    execution = run_benchmark_sync(
        vllm_binary=args.app_vllm_binary,
        bench_path=bench_path,
        raw_args=raw_args,
        artifact_root=artifact_root,
        capture_results=not forward_help,
        stream_output=not forward_help,
        live_stdout=live_stdout,
        live_stderr=live_stderr,
    )

    if forward_help:
        if execution.stdout:
            print(execution.stdout, end="" if execution.stdout.endswith("\n") else "\n")
        if execution.stderr:
            print(execution.stderr, end="" if execution.stderr.endswith("\n") else "\n", file=sys.stderr)
        return execution.returncode or 0

    formatter = {
        "stdout": lambda records: _render_stdout_report(execution),
        "csv": records_to_csv,
        "jsonl": records_to_jsonl,
    }[args.app_output]
    body = formatter(execution.records)
    preserve_output_file = (
        output_path is not None
        and args.app_output in {"csv", "jsonl"}
        and not execution.records
        and (execution.status != "completed" or execution.returncode != 0)
    )
    write_empty_success_export = (
        output_path is not None
        and args.app_output in {"csv", "jsonl"}
        and not execution.records
        and execution.status == "completed"
        and execution.returncode == 0
    )
    if output_path is not None:
        if preserve_output_file:
            if output_path.exists():
                print(
                    f"llmbench cli: preserving existing output file '{args.app_output_path}' because benchmark records were unavailable.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"llmbench cli: skipping output file '{args.app_output_path}' because benchmark records were unavailable.",
                    file=sys.stderr,
                )
        else:
            try:
                output_path.write_text(body, encoding="utf-8")
            except OSError as exc:
                print(f"llmbench cli: failed to write output file '{args.app_output_path}': {exc}", file=sys.stderr)
                return execution.returncode or 1
            if write_empty_success_export:
                print(
                    f"llmbench cli: wrote empty output file '{args.app_output_path}' because benchmark records were unavailable.",
                    file=sys.stderr,
                )
    else:
        if stdout_result_stream and args.app_output == "stdout":
            # Keep the upstream stdout result stream intact for shell pipelines.
            _emit_text(body, stream=sys.stderr)
        else:
            _emit_text(body, stream=sys.stdout)
    if execution.deferred_stderr:
        _emit_text(execution.deferred_stderr, stream=sys.stderr)
    elif execution.stderr and not execution.streamed_output:
        _emit_text(execution.stderr, stream=sys.stderr, ensure_trailing_newline=False)
    return 0 if execution.status == "completed" and execution.returncode == 0 else (execution.returncode or 1)


def _run_web(args: argparse.Namespace) -> int:
    from aiohttp import web

    from llmbench.webapp import create_app

    app = create_app(
        artifact_root=Path(args.artifact_root).expanduser().resolve(),
        default_vllm_binary=args.vllm_binary,
    )
    web.run_app(app, host=args.host, port=args.port)
    return 0


def _resolve_cli_artifact_root(raw_value: str | None) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return _default_cli_artifact_root().resolve()


def _default_cli_artifact_root() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "llmbench" / "runs"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home) / "llmbench" / "runs"
    return Path.home() / ".cache" / "llmbench" / "runs"


def _redact_live_output_sensitive_text(text: str) -> str:
    if not text:
        return text
    redacted = _BEARER_INLINE_TOKEN_RE.sub(r"\1\2" + REDACTED_BEARER_TOKEN, text)
    return _AUTH_HEADER_VALUE_RE.sub(r"\1" + REDACTED_AUTH_VALUE, redacted)


class _SensitiveRedactingStream:
    def __init__(self, wrapped: TextIO) -> None:
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        return self._wrapped.write(_redact_live_output_sensitive_text(text))

    def flush(self) -> None:
        self._wrapped.flush()


def _render_stdout_report(execution) -> str:
    lines = [
        f"status   : {execution.status}",
        f"code     : {execution.returncode}",
        f"command  : {_redact_live_output_sensitive_text(shlex.join(execution.command))}",
        f"run_dir  : {execution.artifact_dir}",
        f"results  : {execution.result_path or '-'}",
        "",
        "records",
        "-------",
        records_to_stdout(execution.records),
    ]
    if execution.stdout.strip() and not execution.streamed_output:
        lines.extend(["", "vllm stdout", "-----------", _redact_live_output_sensitive_text(execution.stdout.strip())])
    return "\n".join(lines)


def _emit_text(text: str, *, stream, ensure_trailing_newline: bool = True) -> None:
    if not text:
        return
    stream.write(text)
    if ensure_trailing_newline and not text.endswith("\n"):
        stream.write("\n")


def _render_cli_help() -> str:
    supported = ", ".join(TOP_LEVEL_BENCH_COMMANDS)
    return "\n".join(
        [
            "llmbench cli",
            "",
            "Wrapper options:",
            "  --app-vllm-binary PATH",
            "  --app-artifact-root PATH",
            "  --app-output stdout|csv|jsonl",
            "  --app-output-path PATH (csv/jsonl only)",
            "  --app-raw-args '...'",
            "  --app-bench-path 'serve'|'sweep serve'",
            "  --app-help",
            "",
            f"Top-level vLLM bench commands: {supported}",
            "Nested command examples:",
            "  llmbench cli serve --help",
            "  llmbench cli startup --help",
            "  llmbench cli mm-processor --help",
            "  llmbench cli sweep serve --help",
            "",
            "Any unknown arguments after the bench path are passed through to vLLM.",
        ]
    )


def _extract_cli_invocation(raw_args: list[str], bench_path_override: str) -> tuple[list[str], list[str]]:
    parse_tokens = _strip_wrapper_separator(raw_args)
    if bench_path_override.strip():
        bench_path = shlex.split(bench_path_override)
        if not bench_path:
            raise ValueError("`--app-bench-path` cannot be empty.")
        return bench_path, _strip_wrapper_separator(raw_args)

    if not parse_tokens:
        return [], parse_tokens

    first_token = parse_tokens[0]
    if first_token.startswith("-"):
        if _contains_candidate_bench_path(parse_tokens[1:]):
            raise ValueError(
                "ambiguous vLLM bench invocation: forwarded options appeared before the bench path. "
                "Put the bench path immediately after `llmbench cli`, or pass it explicitly with `--app-bench-path`."
            )
        return [], parse_tokens

    bench_path = _consume_bench_path(parse_tokens)
    if not bench_path:
        raise ValueError(
            f"unknown vLLM bench command path start '{first_token}'. "
            f"Supported top-level commands: {', '.join(TOP_LEVEL_BENCH_COMMANDS)}."
        )
    return bench_path, _strip_wrapper_separator(parse_tokens[len(bench_path) :])


def _consume_bench_path(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    head = tokens[0]
    if head not in TOP_LEVEL_BENCH_COMMANDS:
        return []
    if head == "sweep" and len(tokens) > 1 and not tokens[1].startswith("-") and tokens[1] in SWEEP_BENCH_COMMANDS:
        return [head, tokens[1]]
    return [head]


def _contains_candidate_bench_path(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token in TOP_LEVEL_BENCH_COMMANDS:
            if token != "sweep":
                return True
            if index + 1 < len(tokens):
                next_token = tokens[index + 1]
                if not next_token.startswith("-") and next_token in SWEEP_BENCH_COMMANDS:
                    return True
            return True
    return False


def _is_wrapper_help_request(raw_args: list[str], bench_path_override: str) -> bool:
    if bench_path_override.strip():
        return False
    if not raw_args:
        return False
    return all(token in {"-h", "--help"} for token in raw_args)


def _prepare_output_path(path: Path) -> Path:
    resolved = path.expanduser()
    if resolved.exists() and resolved.is_dir():
        raise IsADirectoryError(f"{resolved} is a directory")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _rewrite_random_length_args_for_throughput(bench_path: list[str], raw_args: list[str]) -> list[str]:
    if not _uses_random_length_flags(bench_path, raw_args):
        return list(raw_args)
    rewritten: list[str] = []
    for token in raw_args:
        if token == "--input-len":
            rewritten.append("--random-input-len")
        elif token.startswith("--input-len="):
            rewritten.append(f"--random-input-len={token.split('=', 1)[1]}")
        elif token == "--output-len":
            rewritten.append("--random-output-len")
        elif token.startswith("--output-len="):
            rewritten.append(f"--random-output-len={token.split('=', 1)[1]}")
        else:
            rewritten.append(token)
    return rewritten


def _uses_random_length_flags(bench_path: list[str], raw_args: list[str]) -> bool:
    if not bench_path or bench_path[0] != "throughput":
        return False
    dataset_name = ""
    for index, token in enumerate(raw_args):
        if token.startswith("--dataset-name="):
            dataset_name = token.split("=", 1)[1]
            continue
        if token == "--dataset-name" and index + 1 < len(raw_args):
            dataset_name = raw_args[index + 1]
    return dataset_name == "random"


def _strip_wrapper_separator(tokens: list[str]) -> list[str]:
    stripped = list(tokens)
    if stripped and stripped[0] == "--":
        stripped = stripped[1:]
    return stripped


def _uses_stdout_result_stream(bench_path: list[str], raw_args: list[str]) -> bool:
    # Real vLLM 0.19.x does not emit structured JSON records to stdout for
    # `--output-json -`; llmbench rewrites that hint to a wrapper-managed
    # capture file instead of relying on a nonexistent upstream stdout contract.
    return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
