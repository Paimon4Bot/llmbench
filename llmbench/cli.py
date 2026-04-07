from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from llmbench.exporters import records_to_csv, records_to_jsonl, records_to_stdout
from llmbench.vllm_runner import (
    OUTPUT_JSON_BENCH_COMMANDS,
    SWEEP_BENCH_COMMANDS,
    TOP_LEVEL_BENCH_COMMANDS,
    run_benchmark_sync,
)


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
            "Nested benchmark paths: llmbench cli sweep serve_sla ..."
        ),
    )
    cli_parser.add_argument("--app-vllm-binary", default="vllm")
    cli_parser.add_argument("--app-artifact-root", default=".llmbench_runs")
    cli_parser.add_argument("--app-output", choices=["stdout", "csv", "jsonl"], default="stdout")
    cli_parser.add_argument("--app-output-path")
    cli_parser.add_argument("--app-raw-args", default="")
    cli_parser.add_argument("--app-bench-path", default="", help="Explicit vLLM bench path, such as 'serve' or 'sweep serve_sla'.")
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

    artifact_root = Path(args.app_artifact_root).resolve()
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

    forward_help = any(token in {"-h", "--help"} for token in raw_args)
    forward_help = forward_help or any(token.startswith("--help=") for token in raw_args)
    if not bench_path and not forward_help:
        print("llmbench cli: missing vLLM bench command path", file=sys.stderr)
        print("Use `llmbench cli --app-help` for wrapper help.", file=sys.stderr)
        return 2

    output_path: Path | None = None
    if args.app_output_path:
        try:
            output_path = _prepare_output_path(Path(args.app_output_path))
        except OSError as exc:
            print(f"llmbench cli: failed to prepare output file '{args.app_output_path}': {exc}", file=sys.stderr)
            return 1

    stdout_result_stream = _uses_stdout_result_stream(bench_path, raw_args) and not forward_help
    stream_child_stdout_to_stdout = stdout_result_stream and output_path is None and args.app_output == "stdout"
    execution = run_benchmark_sync(
        vllm_binary=args.app_vllm_binary,
        bench_path=bench_path,
        raw_args=raw_args,
        artifact_root=artifact_root,
        capture_results=not forward_help,
        stream_output=not forward_help,
        live_stdout=sys.stdout if stream_child_stdout_to_stdout else sys.stderr,
        live_stderr=sys.stderr,
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
    skip_export_write = (
        output_path is not None
        and args.app_output in {"csv", "jsonl"}
        and not execution.records
    )
    if output_path is not None:
        if skip_export_write:
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
    else:
        if stream_child_stdout_to_stdout:
            # Keep the upstream stdout result stream intact for shell pipelines.
            print(body, file=sys.stderr)
        else:
            print(body)
    if execution.deferred_stderr:
        print(execution.deferred_stderr, end="" if execution.deferred_stderr.endswith("\n") else "\n", file=sys.stderr)
    elif execution.stderr and not execution.streamed_output:
        print(execution.stderr, end="", file=sys.stderr)
    return 0 if execution.status == "completed" and execution.returncode == 0 else (execution.returncode or 1)


def _run_web(args: argparse.Namespace) -> int:
    from aiohttp import web

    from llmbench.webapp import create_app

    app = create_app(
        artifact_root=Path(args.artifact_root).resolve(),
        default_vllm_binary=args.vllm_binary,
    )
    web.run_app(app, host=args.host, port=args.port)
    return 0


def _render_stdout_report(execution) -> str:
    lines = [
        f"status   : {execution.status}",
        f"code     : {execution.returncode}",
        f"command  : {shlex.join(execution.command)}",
        f"run_dir  : {execution.artifact_dir}",
        f"results  : {execution.result_path or '-'}",
        "",
        "records",
        "-------",
        records_to_stdout(execution.records),
    ]
    if execution.stdout.strip() and not execution.streamed_output:
        lines.extend(["", "vllm stdout", "-----------", execution.stdout.strip()])
    return "\n".join(lines)


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
            "  --app-output-path PATH",
            "  --app-raw-args '...'",
            "  --app-bench-path 'serve'|'sweep serve_sla'",
            "  --app-help",
            "",
            f"Top-level vLLM bench commands: {supported}",
            "Nested command examples:",
            "  llmbench cli serve --help",
            "  llmbench cli startup --help",
            "  llmbench cli mm-processor --help",
            "  llmbench cli sweep serve_sla --help",
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
        return bench_path, list(raw_args)

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


def _strip_wrapper_separator(tokens: list[str]) -> list[str]:
    stripped = list(tokens)
    if stripped and stripped[0] == "--":
        stripped = stripped[1:]
    return stripped


def _uses_stdout_result_stream(bench_path: list[str], raw_args: list[str]) -> bool:
    if not bench_path or bench_path[0] not in OUTPUT_JSON_BENCH_COMMANDS:
        return False
    for index, token in enumerate(raw_args):
        if token.startswith("--output-json="):
            return token.split("=", 1)[1] == "-"
        if token == "--output-json" and index + 1 < len(raw_args):
            return raw_args[index + 1] == "-"
    return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
