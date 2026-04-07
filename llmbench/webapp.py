from __future__ import annotations

import html
import json
import shlex
from pathlib import Path
from typing import Any

from aiohttp import web

from llmbench.exporters import normalize_records, records_to_csv, records_to_jsonl
from llmbench.vllm_runner import JobManager

JOBS_KEY = web.AppKey("jobs", JobManager)
DEFAULT_VLLM_BINARY_KEY = web.AppKey("default_vllm_binary", str)


def create_app(*, artifact_root: Path, default_vllm_binary: str) -> web.Application:
    app = web.Application()
    app[JOBS_KEY] = JobManager(artifact_root=artifact_root, default_vllm_binary=default_vllm_binary)
    app[DEFAULT_VLLM_BINARY_KEY] = default_vllm_binary
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/jobs", handle_list_jobs)
    app.router.add_post("/api/jobs", handle_start_job)
    app.router.add_get("/api/jobs/{job_id}", handle_get_job)
    app.router.add_post("/api/jobs/{job_id}/stop", handle_stop_job)
    app.router.add_get("/api/jobs/{job_id}/export.csv", handle_export_csv)
    app.router.add_get("/api/jobs/{job_id}/export.jsonl", handle_export_jsonl)
    return app


async def handle_index(request: web.Request) -> web.Response:
    page = INDEX_HTML.replace("__SERVER_BINARY_LABEL__", html.escape("Server-configured vLLM binary", quote=True))
    page = page.replace("__SERVER_BINARY_PREVIEW__", json.dumps(request.app[DEFAULT_VLLM_BINARY_KEY]))
    return web.Response(text=page, content_type="text/html")


async def handle_start_job(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        bench_path = build_bench_path(payload)
        raw_args = build_web_args(payload)
    except ValueError as exc:
        return _json_error(str(exc), status=400)
    manager: JobManager = request.app[JOBS_KEY]
    execution = await manager.start_job(
        bench_path=bench_path,
        raw_args=raw_args,
    )
    return web.json_response(execution.to_dict())


async def handle_list_jobs(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    jobs = manager.list_jobs()
    status_filter_raw = request.query.get("status", "")
    status_filter = {item.strip() for item in status_filter_raw.split(",") if item.strip()}
    if status_filter:
        jobs = [job for job in jobs if job.status in status_filter]
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
    payload = execution.to_dict()
    payload["rows"] = normalize_records(execution.records)
    payload["columns"] = sorted({key for row in payload["rows"] for key in row.keys()})
    return web.json_response(payload)


async def handle_stop_job(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        return _json_error("Unknown job id", status=404)
    await manager.stop_job(execution.job_id)
    refreshed = manager.get_job(execution.job_id)
    if refreshed is None:
        return _json_error("Unknown job id", status=404)
    payload = refreshed.to_dict()
    payload["rows"] = normalize_records(refreshed.records)
    payload["columns"] = sorted({key for row in payload["rows"] for key in row.keys()})
    return web.json_response(payload)


async def handle_export_csv(request: web.Request) -> web.Response:
    manager: JobManager = request.app[JOBS_KEY]
    execution = manager.get_job(request.match_info["job_id"])
    if execution is None:
        raise web.HTTPNotFound(text="Unknown job id")
    records = manager.export_records(request.match_info["job_id"])
    body = records_to_csv(records)
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
    records = manager.export_records(request.match_info["job_id"])
    body = records_to_jsonl(records)
    filename = f"llmbench-{execution.job_id}.jsonl"
    return web.Response(
        text=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        content_type="application/x-ndjson",
    )


def build_web_args(payload: dict[str, Any]) -> list[str]:
    args: list[str] = []
    mappings = [
        ("model", "--model"),
        ("backend", "--backend"),
        ("endpoint", "--endpoint"),
        ("base_url", "--base-url"),
        ("host", "--host"),
        ("port", "--port"),
        ("dataset_name", "--dataset-name"),
        ("dataset_path", "--dataset-path"),
        ("tokenizer", "--tokenizer"),
        ("input_len", "--input-len"),
        ("output_len", "--output-len"),
        ("num_prompts", "--num-prompts"),
        ("request_rate", "--request-rate"),
        ("max_concurrency", "--max-concurrency"),
        ("percentile_metrics", "--percentile-metrics"),
        ("metric_percentiles", "--metric-percentiles"),
    ]
    for key, flag in mappings:
        value = payload.get(key)
        if value not in (None, ""):
            args.extend([flag, str(value)])
    for key, flag in [("save_detailed", "--save-detailed"), ("ignore_eos", "--ignore-eos"), ("disable_tqdm", "--disable-tqdm")]:
        if payload.get(key):
            args.append(flag)
    extra_args = payload.get("extra_args", "")
    if extra_args:
        args.extend(shlex.split(extra_args))
    return args


def build_bench_path(payload: dict[str, Any]) -> list[str]:
    override = str(payload.get("bench_path_override", "") or "").strip()
    if override:
        bench_path = shlex.split(override)
        if not bench_path:
            raise ValueError("Bench path override cannot be empty.")
        return bench_path

    subcommand = str(payload.get("subcommand", "serve") or "serve").strip()
    if not subcommand:
        raise ValueError("A vLLM bench subcommand is required.")

    bench_path = [subcommand]
    sweep_subcommand = str(payload.get("sweep_subcommand", "") or "").strip()
    if subcommand == "sweep" and sweep_subcommand:
        bench_path.append(sweep_subcommand)
    return bench_path


def _json_error(message: str, *, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


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
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="eyebrow">vLLM Bench Control Plane</div>
        <h1>Launch, stop, inspect, export.</h1>
        <p>
          This console wraps <code>vllm bench</code> instead of replacing it.
          Use structured controls for common parameters, then append any extra raw vLLM arguments when you need full coverage.
        </p>
      </div>
      <div class="command-card">
        <span class="label" id="commandPreviewLabel">Draft launch command</span>
        <div id="commandPreview">vllm bench serve --model ...</div>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <h2>Benchmark Inputs</h2>
        <form id="benchForm">
          <div class="actions">
            <button type="submit" class="primary" id="startButton">Start Benchmark</button>
            <button type="button" class="danger" id="stopButton" disabled>Stop Benchmark</button>
          </div>
          <div class="grid">
            <label>Bench Subcommand
              <select name="subcommand">
                <option value="serve">serve</option>
                <option value="latency">latency</option>
                <option value="mm-processor">mm-processor</option>
                <option value="startup">startup</option>
                <option value="sweep">sweep</option>
                <option value="throughput">throughput</option>
              </select>
            </label>
            <label>Sweep Subcommand
              <select name="sweep_subcommand">
                <option value="serve">serve</option>
                <option value="serve_sla">serve_sla</option>
                <option value="serve_workload">serve_workload</option>
                <option value="startup">startup</option>
              </select>
            </label>
            <div class="hint-card">
              <strong>Execution Binary</strong>
              This browser UI always uses the server-configured binary to avoid client-side command injection.
              <code>__SERVER_BINARY_LABEL__</code>
            </div>
            <label>Model
              <input name="model" placeholder="meta-llama/Llama-3.1-8B-Instruct">
            </label>
            <label>Backend
              <select name="backend">
                <option value="">(optional)</option>
                <option value="openai">openai</option>
                <option value="vllm">vllm</option>
                <option value="sglang">sglang</option>
                <option value="tgi">tgi</option>
              </select>
            </label>
            <label>Base URL
              <input name="base_url" placeholder="http://127.0.0.1:8000">
            </label>
            <label>Endpoint
              <input name="endpoint" placeholder="/v1/chat/completions">
            </label>
            <label>Host
              <input name="host" placeholder="127.0.0.1">
            </label>
            <label>Port
              <input name="port" placeholder="8000">
            </label>
            <label>Dataset
              <select name="dataset_name">
                <option value="">(optional)</option>
                <option value="random">random</option>
                <option value="sharegpt">sharegpt</option>
                <option value="custom">custom</option>
                <option value="hf">hf</option>
                <option value="burstgpt">burstgpt</option>
              </select>
            </label>
            <label>Dataset Path
              <input name="dataset_path" placeholder="/abs/path/to/dataset">
            </label>
            <label>Tokenizer
              <input name="tokenizer" placeholder="meta-llama/Llama-3.1-8B-Instruct">
            </label>
            <label>Num Prompts
              <input name="num_prompts" placeholder="1000">
            </label>
            <label>Input Len
              <input name="input_len" placeholder="512">
            </label>
            <label>Output Len
              <input name="output_len" placeholder="128">
            </label>
            <label>Request Rate
              <input name="request_rate" placeholder="4">
            </label>
            <label>Max Concurrency
              <input name="max_concurrency" placeholder="32">
            </label>
            <label>Percentile Metrics
              <input name="percentile_metrics" placeholder="ttft,tpot,itl">
            </label>
            <label>Metric Percentiles
              <input name="metric_percentiles" placeholder="50,90,95,99">
            </label>
            <label>Bench Path Override
              <input name="bench_path_override" placeholder="sweep serve_sla">
            </label>
          </div>
          <div class="hint-card" id="overrideNotice" hidden>
            <strong>Override Active</strong>
            Override is controlling the command path right now. The basic selector is temporarily locked so you do not launch a different benchmark by accident.
          </div>

          <div class="checks">
            <label><input type="checkbox" name="save_detailed"> Save Detailed</label>
            <label><input type="checkbox" name="ignore_eos"> Ignore EOS</label>
            <label><input type="checkbox" name="disable_tqdm"> Disable TQDM</label>
          </div>

          <label>Additional Raw vLLM Args
            <textarea name="extra_args" placeholder="--request-id-prefix bench- --save-detailed --metadata version=exp1"></textarea>
          </label>

        </form>
      </section>

      <section class="panel">
        <h2>Result Surface</h2>
        <div class="error-banner" id="errorBanner" role="alert" aria-live="assertive" tabindex="-1" hidden></div>
        <div class="info-banner" id="infoBanner" role="status" aria-live="polite" tabindex="-1" hidden></div>
        <div class="status-grid">
          <div class="status-card"><div class="label">Job</div><div class="value" id="jobId">-</div></div>
          <div class="status-card"><div class="label">Status</div><div class="value" id="jobStatus" aria-live="polite">idle</div></div>
          <div class="status-card"><div class="label">Return Code</div><div class="value" id="jobCode">-</div></div>
          <div class="status-card"><div class="label">Records</div><div class="value" id="jobRecords">0</div></div>
          <div class="status-card"><div class="label">Last Update</div><div class="value" id="jobUpdated">-</div></div>
          <div class="status-card"><div class="label">Elapsed</div><div class="value" id="jobElapsed">-</div></div>
        </div>
        <div class="hint-card" id="resultContextCard">
          <strong id="resultContextTitle">Displayed Results</strong>
          <div id="resultContextText">No benchmark results are loaded.</div>
          <code id="resultCommand">-</code>
        </div>
        <div class="hint-card recovery-panel">
          <strong>Recover Running Jobs</strong>
          Fresh tabs do not auto-attach to server jobs. Select one to adopt into this tab.
          <div class="recovery-grid">
            <label>Active Jobs
              <select id="recoverJobSelect">
                <option value="">No active jobs discovered</option>
              </select>
            </label>
            <button type="button" class="secondary" id="refreshJobsButton">Refresh Jobs</button>
            <button type="button" class="secondary" id="adoptJobButton" disabled>Adopt Job</button>
          </div>
        </div>
        <div class="exports">
          <button type="button" class="export-link" id="csvExport" disabled aria-disabled="true" tabindex="-1">Export CSV</button>
          <button type="button" class="export-link" id="jsonlExport" disabled aria-disabled="true" tabindex="-1">Export JSONL</button>
        </div>
        <div id="resultsTableWrap"></div>
        <h3>STDOUT</h3>
        <div class="mono" id="stdoutBox"></div>
        <h3>STDERR</h3>
        <div class="mono" id="stderrBox"></div>
      </section>
    </div>
  </div>

  <script>
    const serverBenchBinary = __SERVER_BINARY_PREVIEW__;
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
    const subcommandField = form.elements.namedItem("subcommand");
    const sweepSubcommandField = form.elements.namedItem("sweep_subcommand");
    const overrideField = form.elements.namedItem("bench_path_override");
    const JOB_STORAGE_KEY = "llmbench.currentJobId";
    const JOB_FORM_STORAGE_KEY = "llmbench.currentJobForm";
    const JOB_CONTEXT_FORM_KEY = "llmbench.contextJobForm";
    const storage = window.sessionStorage;
    const RECOVERY_DELAY_MS = 1000;
    let lockedFormData = null;
    let contextFormData = null;
    let currentJobId = null;
    let displayedJob = null;
    let pollTimer = null;
    let recoveryTimer = null;
    let startRequestInFlight = false;
    let discoveredRecoverableJobs = [];

    const fields = Array.from(form.elements).filter((element) => element.name);
    for (const field of fields) {
      field.addEventListener("input", updatePreview);
      field.addEventListener("change", updatePreview);
    }
    recoverJobSelect.addEventListener("change", () => {
      adoptJobButton.disabled = !recoverJobSelect.value;
    });
    refreshJobsButton.addEventListener("click", async () => {
      try {
        setError("");
        setInfo("Refreshing active benchmark jobs...");
        await refreshRecoverableJobs(true);
      } catch (error) {
        setError(error.message || "Failed to discover active benchmark jobs.");
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

    function updatePreview() {
      const data = lockedFormData || formPayload();
      const args = [];
      const benchPath = benchPathFromForm(data);
      const binaryTokens = splitShellWords(serverBenchBinary) || [String(serverBenchBinary || "vllm")];
      syncOverrideState(data);
      const append = (flag, value) => {
        if (value) args.push(flag, value);
      };
      append("--model", data.model);
      append("--backend", data.backend);
      append("--base-url", data.base_url);
      append("--endpoint", data.endpoint);
      append("--host", data.host);
      append("--port", data.port);
      append("--dataset-name", data.dataset_name);
      append("--dataset-path", data.dataset_path);
      append("--tokenizer", data.tokenizer);
      append("--num-prompts", data.num_prompts);
      append("--input-len", data.input_len);
      append("--output-len", data.output_len);
      append("--request-rate", data.request_rate);
      append("--max-concurrency", data.max_concurrency);
      append("--percentile-metrics", data.percentile_metrics);
      append("--metric-percentiles", data.metric_percentiles);
      if (data.save_detailed) args.push("--save-detailed");
      if (data.ignore_eos) args.push("--ignore-eos");
      if (data.disable_tqdm) args.push("--disable-tqdm");
      if (data.extra_args) {
        const parsedExtraArgs = splitShellWords(data.extra_args);
        if (parsedExtraArgs) {
          args.push(...parsedExtraArgs);
        } else {
          args.push(data.extra_args);
        }
      }
      commandPreview.textContent = shellJoinArgs([...binaryTokens, "bench", ...benchPath, ...args]);
      commandPreviewLabel.textContent = lockedFormData ? "Locked launch command" : "Draft launch command";
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
        storage.setItem(JOB_CONTEXT_FORM_KEY, JSON.stringify(contextFormData));
      } else {
        storage.removeItem(JOB_CONTEXT_FORM_KEY);
      }
    }

    function setContextFormData(data) {
      contextFormData = data ? {...data} : null;
      persistContextFormData();
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
      lockedFormData = data ? {...data} : null;
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

    function buildResultContext(job) {
      if (!job || !job.job_id) {
        return {
          title: "Displayed Results",
          text: "No benchmark results are loaded.",
          command: "-",
        };
      }
      const commandText = Array.isArray(job.command) && job.command.length
        ? shellJoinArgs(job.command)
        : shellQuoteArg(job.subcommand || "-");
      if (job.status === "running" || job.status === "stopping" || job.status === "reconnecting") {
        return {
          title: "Current Benchmark",
          text: `Job ${job.job_id} is ${job.status}. Results and exports will refresh when the command emits records or exits.`,
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
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
      enterRecoveryState(message);
      if (recoveryTimer) {
        return;
      }
      recoveryTimer = window.setTimeout(async () => {
        recoveryTimer = null;
        try {
          const job = await fetchJob(currentJobId);
          setError("");
          setJob(job);
          if (job.status === "running") {
            startPolling();
          }
        } catch (error) {
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
          lockedFormData = parsed;
          applyFormData(parsed);
        } catch (error) {
          storage.removeItem(JOB_FORM_STORAGE_KEY);
          lockedFormData = null;
        }
      }
      if (storedContextForm) {
        try {
          const parsedContext = JSON.parse(storedContextForm);
          setContextFormData(parsedContext);
          applyFormData(parsedContext);
        } catch (error) {
          storage.removeItem(JOB_CONTEXT_FORM_KEY);
          setContextFormData(null);
        }
      }
      if (!storedJobId) {
        currentJobId = null;
        persistCurrentJobId();
        setFormLocked(null);
        try {
          const jobs = await refreshRecoverableJobs(false);
          if (jobs.length) {
            setInfo(`Found ${jobs.length} active benchmark job(s). Select one and click Adopt Job to inspect and control it in this tab.`);
          } else {
            setInfo("");
          }
        } catch (error) {
          renderRecoverableJobs([]);
          setInfo("Unable to discover active benchmarks right now. You can retry with Refresh Jobs.");
        }
        return;
      }
      try {
        currentJobId = storedJobId;
        const job = await fetchJob(storedJobId);
        setJob(job);
        if (job.status === "running" || job.status === "stopping") {
          startPolling();
        }
      } catch (error) {
        currentJobId = storedJobId;
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

    async function fetchJobs(statuses, limit) {
      const params = new URLSearchParams();
      if (statuses && statuses.length) {
        params.set("status", statuses.join(","));
      }
      if (limit) {
        params.set("limit", String(limit));
      }
      const query = params.toString();
      const response = await fetch(`/api/jobs${query ? "?" + query : ""}`);
      return parseResponse(response, "Failed to discover benchmark jobs");
    }

    function formatRecoverableJobLabel(job) {
      const started = formatTimestamp(job.started_at);
      const status = job.status || "unknown";
      const records = Number(job.record_count || 0);
      return `${job.job_id} | ${status} | started ${started} | records ${records}`;
    }

    function renderRecoverableJobs(jobs) {
      discoveredRecoverableJobs = Array.isArray(jobs) ? jobs.filter((job) => job && job.job_id) : [];
      recoverJobSelect.innerHTML = "";
      if (!discoveredRecoverableJobs.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No active jobs discovered";
        recoverJobSelect.appendChild(option);
        adoptJobButton.disabled = true;
        return;
      }
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select a running/stopping job...";
      recoverJobSelect.appendChild(placeholder);
      for (const job of discoveredRecoverableJobs) {
        const option = document.createElement("option");
        option.value = job.job_id;
        option.textContent = formatRecoverableJobLabel(job);
        recoverJobSelect.appendChild(option);
      }
      recoverJobSelect.value = "";
      adoptJobButton.disabled = true;
    }

    async function discoverRecoverableJobs() {
      const payload = await fetchJobs(["running", "stopping"], 200);
      return Array.isArray(payload.jobs) ? payload.jobs : [];
    }

    async function refreshRecoverableJobs(announceSummary) {
      const jobs = await discoverRecoverableJobs();
      renderRecoverableJobs(jobs);
      if (announceSummary) {
        if (!jobs.length) {
          setInfo("No active server-side benchmark jobs are waiting for adoption.");
        } else {
          setInfo(`Found ${jobs.length} active benchmark job(s). Select one and click Adopt Job to inspect and control it in this tab.`);
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
      const response = await fetch(`/api/jobs/${jobId}`);
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
        const response = await fetch(`/api/jobs/${jobId}/export.${format}`);
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
      const snapshot = formPayload();
      startRequestInFlight = true;
      syncButtons();
      try {
        const response = await fetch("/api/jobs", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(snapshot)
        });
        const job = await parseResponse(response, "Failed to start benchmark");
        clearPoll();
        setContextFormData(snapshot);
        applyFormData(snapshot);
        setFormLocked(snapshot);
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
        const response = await fetch(`/api/jobs/${currentJobId}/stop`, {method: "POST"});
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
      clearRecoveryTimer();
      syncButtons();
      pollTimer = setInterval(async () => {
        try {
          const job = await fetchJob(currentJobId);
          setJob(job);
          if (job.status !== "running") {
            clearPoll();
          }
        } catch (error) {
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
      }, 1000);
    }

    function clearPoll() {
      if (pollTimer) {
        clearInterval(pollTimer);
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
      const derivedContext = deriveFormContextFromJob(job);
      if (derivedContext) {
        setContextFormData(derivedContext);
      }
      if (job.status === "running" || job.status === "stopping") {
        const lockSource = derivedContext || contextFormData || lockedFormData || formPayload();
        applyFormData(lockSource);
        setFormLocked(lockSource);
      } else {
        setFormLocked(null);
        if (contextFormData) {
          applyFormData(contextFormData);
        }
      }
      setError("");
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
      if (!rows.length || !columns.length) {
        if (job.status === "completed") {
          setInfo("This benchmark completed successfully, but this command path did not emit structured rows for the results table or file exports. Inspect stdout and stderr for the primary output.");
          wrap.innerHTML = "<p>No structured result rows were emitted for this benchmark command.</p>";
          return;
        }
        setInfo("");
        wrap.innerHTML = "<p>No completed benchmark rows yet.</p>";
        return;
      }
      setInfo("");
      const header = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
      const body = rows.map((row) => {
        return `<tr>${columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("")}</tr>`;
      }).join("");
      wrap.innerHTML = `<table><thead>${header}</thead><tbody>${body}</tbody></table>`;
    }

    function deriveFormContextFromJob(job) {
      if (!job || !Array.isArray(job.command) || !job.command.length) {
        return null;
      }
      const benchIndex = job.command.indexOf("bench");
      if (benchIndex < 0 || benchIndex + 1 >= job.command.length) {
        return null;
      }
      const tokens = job.command.slice(benchIndex + 1).map((token) => String(token));
      let index = 0;
      const pathTokens = [];
      while (index < tokens.length && !tokens[index].startsWith("-")) {
        pathTokens.push(tokens[index]);
        index += 1;
      }
      if (!pathTokens.length) {
        return null;
      }

      const topLevel = new Set(["serve", "latency", "mm-processor", "startup", "sweep", "throughput"]);
      const sweepChoices = new Set(["serve", "serve_sla", "serve_workload", "startup"]);
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
        "--dataset-name": "dataset_name",
        "--dataset-path": "dataset_path",
        "--tokenizer": "tokenizer",
        "--input-len": "input_len",
        "--output-len": "output_len",
        "--num-prompts": "num_prompts",
        "--request-rate": "request_rate",
        "--max-concurrency": "max_concurrency",
        "--percentile-metrics": "percentile_metrics",
        "--metric-percentiles": "metric_percentiles",
      };
      const boolFlags = {
        "--save-detailed": "save_detailed",
        "--ignore-eos": "ignore_eos",
        "--disable-tqdm": "disable_tqdm",
      };
      const ignoredValueFlags = new Set(["--result-dir", "--result-filename", "--output-json", "--output-dir"]);
      const ignoredFlags = new Set(["--save-result"]);
      const extraTokens = [];
      while (index < tokens.length) {
        const token = tokens[index];
        if (ignoredValueFlags.has(token)) {
          index += 1;
          if (index < tokens.length && !tokens[index].startsWith("-")) {
            index += 1;
          }
          continue;
        }
        if (ignoredFlags.has(token)) {
          index += 1;
          continue;
        }
        if (Object.prototype.hasOwnProperty.call(boolFlags, token)) {
          data[boolFlags[token]] = true;
          index += 1;
          continue;
        }
        if (Object.prototype.hasOwnProperty.call(valueFlags, token)) {
          if (index + 1 < tokens.length && !tokens[index + 1].startsWith("-")) {
            data[valueFlags[token]] = tokens[index + 1];
            index += 2;
            continue;
          }
          extraTokens.push(token);
          index += 1;
          continue;
        }
        if (token.startsWith("-") && index + 1 < tokens.length && !tokens[index + 1].startsWith("-")) {
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
