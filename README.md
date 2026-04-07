# llmbench

`llmbench` is a thin wrapper around `vllm bench` with two entry modes:

- `CLI` mode for scripting and CI
- `web` mode for launching benchmarks from a browser

The tool does not reimplement benchmarking logic. It delegates execution to `vllm bench` and adds:

- pass-through support for all vLLM bench arguments
- support for multi-part bench paths such as `sweep serve_sla`
- wrapper-managed result capture that adapts to each bench command family
- stdout / CSV / JSONL export
- browser-based launch, stop, inspect, and export workflow

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[bench]"
```

If you want to manage `vllm` separately, the benchmark prerequisite is still required:

```bash
.venv/bin/pip install -e .
.venv/bin/pip install "vllm[bench]"
```

## CLI Mode

Run any `vllm bench` command path and pass any unknown arguments directly to `vllm bench`.

```bash
.venv/bin/llmbench cli serve \
  --app-vllm-binary "vllm" \
  --app-output stdout \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend openai \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --request-rate 4
```

Nested benchmark paths work too:

```bash
.venv/bin/llmbench cli sweep serve_sla \
  --app-vllm-binary "vllm" \
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
.venv/bin/llmbench cli sweep serve_sla --help
```

Write flattened results to CSV:

```bash
.venv/bin/llmbench cli serve \
  --app-vllm-binary "vllm" \
  --app-output csv \
  --app-output-path results.csv \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend openai \
  --dataset-name random \
  --input-len 512 \
  --output-len 128
```

Result capture is command-aware:

- `serve` uses vLLM's `--save-result`
- `latency`, `throughput`, `mm-processor`, and `startup` use `--output-json`
- sweep run commands such as `sweep serve_sla` use `--output-dir`
- plot-style commands that do not expose machine-readable result files still run, but structured CSV/JSONL output may be empty

If you explicitly pass upstream result-destination flags such as `--result-dir`, `--result-filename`, `--output-json`, or `--output-dir`, `llmbench` preserves them instead of silently overriding them.

## Web Mode

```bash
.venv/bin/llmbench web --host 127.0.0.1 --port 8080 --vllm-binary "vllm"
```

Open `http://127.0.0.1:8080`.

The page supports:

- selecting current top-level bench commands including `serve`, `latency`, `throughput`, `mm-processor`, `startup`, and `sweep`
- selecting sweep subcommands or overriding the full bench path manually, with an explicit override-active cue
- entering common vLLM bench parameters via form controls
- appending any extra raw vLLM arguments
- using the server-configured vLLM binary only; the browser cannot override host commands
- starting and stopping a benchmark
- recovering the current tab's active job after a page reload
- viewing results in an HTML table with horizontal scrolling for wide result sets
- surfacing request and parsing errors in-page instead of failing silently
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
