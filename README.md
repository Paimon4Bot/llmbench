# llmbench

`llmbench` is a thin wrapper around `vllm bench` with two entry modes:

- `CLI` mode for scripting and CI
- `web` mode for launching benchmarks from a browser

The tool does not reimplement benchmarking logic. It delegates execution to `vllm bench` and adds:

- pass-through support for all vLLM bench arguments
- support for multi-part bench paths such as `sweep serve`
- wrapper-managed result capture that adapts to each bench command family
- stdout / CSV / JSONL export
- browser-based launch, stop, inspect, and export workflow

## Install

`llmbench` currently supports Python `3.11` through `3.13`. The `bench` extra follows the current upstream `vllm[bench]` support window and is not published for Python `3.14`.

```bash
uv venv --python 3.13 .venv
uv sync --python .venv/bin/python --extra bench
```

If you want to manage `vllm` separately, keep the repo environment under `uv` and install the benchmark extra in a dedicated uv-managed environment. Point `--app-vllm-binary` at that environment explicitly unless you have already added it to `PATH`:

```bash
uv venv --python 3.13 .venv
uv sync --python .venv/bin/python
uv venv --python 3.13 .venv-vllm
uv pip install --python .venv-vllm/bin/python "pyarrow<21" "vllm[bench]"
```

That extra `pyarrow<21` pin matches the current `vllm 0.19.x` compatibility window. If you install `vllm[bench]` into a separate environment without it, `vllm bench` can fail during startup before `llmbench` ever launches it.

## CLI Mode

Run any `vllm bench` command path and pass any unknown arguments directly to `vllm bench`.

```bash
# `serve` benchmarks an already-running endpoint. Start the server first.
# If the endpoint is missing or unreachable, llmbench now treats the run as failed.
.venv-vllm/bin/vllm serve ~/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 8000

.venv/bin/llmbench cli serve \
  --app-vllm-binary ".venv-vllm/bin/vllm" \
  --app-output stdout \
  --model ~/Qwen3.5-0.8B \
  --tokenizer ~/Qwen3.5-0.8B \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --request-rate 4
```

For `throughput --dataset-name random`, keep using the familiar `--input-len` and `--output-len` flags in `llmbench cli`. The wrapper rewrites them to vLLM's current `--random-input-len` / `--random-output-len` surface before launch.

For a standalone command that does not need a separate API server, use a non-`serve` benchmark such as `throughput` or `latency`. If you want to benchmark a remote or pre-existing serving endpoint, keep using `serve` and provide the endpoint target explicitly with `--base-url` or matching `--host`/`--port`.

Nested benchmark paths work too:

```bash
.venv/bin/llmbench cli sweep serve \
  --app-vllm-binary ".venv-vllm/bin/vllm" \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Show wrapper help without requiring `vllm` to be installed:

```bash
.venv/bin/llmbench cli --help
.venv/bin/llmbench cli -h
```

Inspect upstream vLLM help through the wrapper:

```bash
.venv/bin/llmbench cli serve --help
.venv/bin/llmbench cli sweep serve --help
```

When `llmbench` itself is installed into a single uv-managed environment with `vllm`, the default `--app-vllm-binary vllm` now auto-resolves a sibling `vllm` executable next to the active Python or `llmbench` script even if that environment's `bin` directory is not already on `PATH`.

Write flattened results to CSV:

```bash
.venv/bin/llmbench cli serve \
  --app-vllm-binary ".venv-vllm/bin/vllm" \
  --app-output csv \
  --app-output-path results.csv \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend openai \
  --dataset-name random \
  --input-len 512 \
  --output-len 128
```

Streaming behavior is shell-oriented by default:

- live upstream `stdout` remains on `stdout` for `--app-output stdout`
- live upstream `stderr` remains on `stderr`
- in `--app-output stdout`, both live forwarded output and the wrapper's summary redact bearer and `Authorization` header values before they are printed
- for `--app-output csv|jsonl`, live upstream `stdout` is mirrored to `stderr` so `stdout` stays machine-safe
- in that mirrored path, bearer and `Authorization` header values are redacted before they hit `stderr`
- `--app-output-path` is accepted only with `--app-output csv|jsonl`; using it with `--app-output stdout` is rejected
- when `--app-output-path` is set, structured output is written only to that file and wrapper output remains off `stdout`
- when `--app-artifact-root` is omitted, wrapper-managed run artifacts default to your user cache directory (for example `$XDG_CACHE_HOME/llmbench/runs` or `~/.cache/llmbench/runs`) instead of a hidden `.llmbench_runs` directory under the current working tree

Current `vllm 0.19.x` does not emit JSON records to `stdout` for `--output-json -`; it writes to a literal file named `-`. `llmbench` treats `--output-json -` as a wrapper hint, rewrites it to a server-managed capture file, and then loads records from that file. Use `--app-output csv|jsonl` without `--app-output-path` when you need machine-safe structured output on `stdout`.

Result capture is command-aware:

- `serve` uses vLLM's `--save-result`
- `latency`, `throughput`, `mm-processor`, and `startup` use `--output-json`
- sweep run commands such as `sweep serve`, `sweep serve_workload`, and `sweep startup` use `--output-dir`
- plot-style commands that do not expose machine-readable result files still run, but structured CSV/JSONL output may be empty

In CLI mode, if you explicitly pass upstream result-destination flags such as `--result-dir`, `--result-filename`, `--output-json`, or `--output-dir`, `llmbench` preserves them instead of silently overriding them, with one exception: `--output-json -` is normalized to a wrapper-managed capture file because current upstream vLLM does not stream JSON records to `stdout`. Empty destination values such as `--result-dir=` or `--output-json=` are rejected instead of being rewritten to the current working directory.

If a run exits successfully but produces no structured benchmark records, `--app-output-path` is refreshed with an empty file to avoid stale exports from previous runs.
If structured result loading fails (for example, malformed JSON shape), the run is treated as failed and existing `--app-output-path` exports are preserved.

## Web Mode

```bash
.venv/bin/llmbench web --host 127.0.0.1 --port 8080 --vllm-binary ".venv-vllm/bin/vllm"
```

Open `http://127.0.0.1:8080`.

The page supports:

- selecting current top-level bench commands including `serve`, `latency`, `throughput`, `mm-processor`, `startup`, and `sweep`
- selecting sweep subcommands or overriding the full bench path manually, with an explicit override-active cue
- entering common vLLM bench parameters via form controls
- using throughput-focused low-memory launch controls directly in the form; when `throughput` is selected, blank `GPU Memory Utilization`, `Max Model Len`, and `Enforce Eager` inputs default to `0.60`, `1024`, and enabled for a safer first run on common 8 GB GPUs
- appending extra raw vLLM arguments (except server-side result-destination overrides such as `--result-dir`, `--result-filename`, `--output-dir`, or `--output-json PATH`; `--output-json -` remains allowed as a safe shorthand that `llmbench` rewrites to a server-managed capture file)
- expanding common local `~/...` model/tokenizer paths before launch so browser-submitted local paths resolve on host
- using the server-configured vLLM binary only; the browser cannot override host commands
- keeping draft/result command surfaces redacted with stable browser-visible identifiers so server binary, host-local model/tokenizer paths, raw runtime base URLs, and bearer credentials in additional raw args are not exposed to the browser
- storing browser session form snapshots with bearer credentials redacted (for example, in `sessionStorage`)
- omitting host artifact paths from job API payloads and redacting server-path diagnostics plus backend runtime endpoints in browser-visible output
- starting and stopping a benchmark
- recovering jobs that were released by the same browser profile (automatic tab-close release or manual release), including runs that already reached a terminal status
- keeping attached running jobs private to the owning browser tab, giving duplicated tabs a fresh tab identity, and preventing released jobs from being adopted by unrelated browser profiles
- promoting stale tab-owned jobs to recoverable for that same browser profile after a server-side lease timeout so hard browser crashes do not strand long-running benchmarks
- viewing results in an HTML table with horizontal scrolling for wide result sets, including keyboard scrolling via arrow keys, Home, and End when the table overflows
- surfacing request and parsing errors in-page instead of failing silently
- surfacing runtime benchmark failure summaries in-page while keeping full stdout/stderr diagnostics visible below
- preserving the previously displayed results if a new submission fails validation
- keeping the draft launch command distinct from the currently displayed result/export surface
- exporting the latest run as CSV or JSONL

## Wrapper Options

CLI wrapper options are prefixed with `--app-` so they do not collide with vLLM arguments:

- `--app-vllm-binary`
- `--app-output`
- `--app-output-path`
- `--app-raw-args`
- `--app-bench-path`
- `--app-help`

Everything else is passed to `vllm bench`.

If forwarded vLLM options must appear before the bench path, pass the path explicitly:

```bash
.venv/bin/llmbench cli \
  --app-bench-path "serve" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend openai
```

With `--app-bench-path`, a single leading `--` separator is still treated as wrapper syntax and stripped before forwarding to vLLM. For example, `llmbench cli --app-bench-path throughput -- --help` behaves like `vllm bench throughput --help`.
