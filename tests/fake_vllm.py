#!/usr/bin/env python3
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

SHORT_OPTION_ALIASES = {
    "-o": "--output-dir",
}

VALUE_OPTIONS = {"--result-dir", "--result-filename", "--output-json", "--output-dir", "--fake-sleep"}


def _parse_args(argv: list[str]) -> tuple[list[str], dict[str, str], set[str]]:
    if len(argv) < 2 or argv[1] != "bench":
        raise SystemExit(2)
    bench_path: list[str] = []
    options: dict[str, str] = {}
    flags: set[str] = set()
    index = 2
    while index < len(argv) and not argv[index].startswith("-"):
        bench_path.append(argv[index])
        index += 1
    while index < len(argv):
        token = argv[index]
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
            if index + 1 < len(argv) and (not argv[index + 1].startswith("-") or (token in VALUE_OPTIONS and argv[index + 1] == "-")):
                options[token] = argv[index + 1]
                index += 2
            else:
                flags.add(token)
                index += 1
        elif token.startswith("-"):
            canonical = SHORT_OPTION_ALIASES.get(token)
            if canonical and index + 1 < len(argv):
                options[canonical] = argv[index + 1]
                index += 2
            elif canonical:
                flags.add(canonical)
                index += 1
            else:
                index += 1
        else:
            index += 1
    return bench_path, options, flags


def _handle_term(signum, frame):  # pragma: no cover
    print("fake vllm received terminate", file=sys.stderr)
    raise SystemExit(143)


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGTERM, _handle_term)
    bench_path, options, flags = _parse_args(argv)
    command_name = " ".join(bench_path) or "bench"
    top_level = bench_path[0] if bench_path else ""
    sweep_subcommand = bench_path[1] if len(bench_path) > 1 else ""

    if "--help" in flags or "-h" in flags or "--help" in options:
        print(f"fake vllm help for {command_name}")
        return 0

    sleep_seconds = float(options.get("--fake-sleep", "0"))
    if "--fake-live-output" in flags:
        print("fake live stdout chunk 1", flush=True)
        print("fake live stderr chunk 1", file=sys.stderr, flush=True)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    result = {
        "subcommand": command_name,
        "config": {
            "model": options.get("--model", ""),
            "backend": options.get("--backend", ""),
            "dataset_name": options.get("--dataset-name", ""),
            "dataset_path": options.get("--dataset-path", ""),
            "base_url": options.get("--base-url", ""),
            "request_rate": options.get("--request-rate", ""),
            "max_concurrency": options.get("--max-concurrency", ""),
        },
        "metrics": {
            "ttft_ms": 41.2,
            "itl_ms": 8.4,
            "request_throughput": 11.7,
            "output_token_throughput": 189.3,
        },
        "flags": sorted(flags),
    }

    if top_level == "serve":
        if "--result-dir" not in options or "--save-result" not in flags:
            print("serve expects --save-result and --result-dir", file=sys.stderr)
            return 2
    elif top_level in {"latency", "mm-processor", "startup", "throughput"}:
        if "--output-json" not in options:
            print(f"{top_level} expects --output-json", file=sys.stderr)
            return 2
        if "--save-result" in flags or "--result-dir" in options:
            print(f"{top_level} rejects serve-style result flags", file=sys.stderr)
            return 2
    elif top_level == "sweep" and sweep_subcommand in {"serve", "serve_sla", "serve_workload", "startup"}:
        if "--output-dir" not in options:
            print(f"{command_name} expects --output-dir", file=sys.stderr)
            return 2
        if "--save-result" in flags or "--result-dir" in options:
            print(f"{command_name} rejects serve-style result flags", file=sys.stderr)
            return 2
    elif top_level == "sweep" and sweep_subcommand in {"plot", "plot_pareto"}:
        if "--save-result" in flags or "--result-dir" in options or "--output-json" in options or "--output-dir" in options:
            print(f"{command_name} should not receive structured result-capture flags", file=sys.stderr)
            return 2

    if top_level == "serve":
        result_dir = Path(options["--result-dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / options.get("--result-filename", "result.json")
        result_path.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    elif top_level in {"latency", "mm-processor", "startup", "throughput"}:
        output_json = options.get("--output-json")
        if output_json:
            if output_json == "-":
                print(json.dumps(result, ensure_ascii=True), flush=True)
                print(f"fake vllm bench {command_name} completed", file=sys.stderr, flush=True)
                return 0
            result_path = Path(output_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    elif top_level == "sweep" and sweep_subcommand in {"serve", "serve_sla", "serve_workload", "startup"}:
        output_dir = options.get("--output-dir")
        if output_dir:
            result_dir = Path(output_dir) / sweep_subcommand
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "run-0.json").write_text(json.dumps({"subcommand": f"{command_name} run-0"}, ensure_ascii=True), encoding="utf-8")
            result_path = result_dir / "summary.json"
            result_path.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")

    print(f"fake vllm bench {command_name} completed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
