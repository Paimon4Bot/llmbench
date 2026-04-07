from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession, web

from llmbench.webapp import create_app


class WebModeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        self.app = create_app(
            artifact_root=Path(self.tempdir.name),
            default_vllm_binary=self.fake_binary,
        )
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets  # type: ignore[attr-defined]
        self.port = sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.session = ClientSession()

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.runner.cleanup()
        self.tempdir.cleanup()

    async def test_web_job_completion_and_exports(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "web-model",
                "backend": "openai",
                "request_rate": "3",
            },
        )
        job = await response.json()
        job_id = job["job_id"]

        final = await self._wait_for_terminal_state(job_id)
        self.assertEqual("completed", final["status"])
        self.assertEqual(1, final["record_count"])
        self.assertIn("metrics.ttft_ms", final["columns"])

        csv_response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}/export.csv")
        csv_text = await csv_response.text()
        self.assertIn("metrics.ttft_ms", csv_text)
        self.assertIn(f'llmbench-{job_id}.csv', csv_response.headers.get("Content-Disposition", ""))

        jsonl_response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}/export.jsonl")
        jsonl_text = await jsonl_response.text()
        self.assertIn('"subcommand": "serve"', jsonl_text)
        self.assertIn(f'llmbench-{job_id}.jsonl', jsonl_response.headers.get("Content-Disposition", ""))

    async def test_web_supports_nested_bench_paths(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "sweep",
                "sweep_subcommand": "serve_sla",
                "model": "web-model",
            },
        )
        self.assertEqual(200, response.status)
        job = await response.json()
        final = await self._wait_for_terminal_state(job["job_id"])
        self.assertEqual("completed", final["status"])
        subcommands = {row["subcommand"] for row in final["rows"]}
        self.assertIn("sweep serve_sla", subcommands)
        self.assertIn("sweep serve_sla run-0", subcommands)

    async def test_web_ignores_browser_supplied_binary_override(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "web-model",
                "vllm_binary": "definitely-not-a-command",
            },
        )
        self.assertEqual(200, response.status)
        job = await response.json()
        final = await self._wait_for_terminal_state(job["job_id"])
        self.assertEqual("completed", final["status"])
        self.assertIn("fake vllm bench serve completed", final["stdout"])

    async def test_web_job_can_be_stopped(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "slow-model",
                "extra_args": "--fake-sleep 10",
            },
        )
        job = await response.json()
        job_id = job["job_id"]

        stop_response = await self.session.post(f"{self.base_url}/api/jobs/{job_id}/stop")
        self.assertEqual(200, stop_response.status)
        final = await self._wait_for_terminal_state(job_id)
        self.assertEqual("stopped", final["status"])

    async def test_web_marks_prepare_failures_as_failed_instead_of_running_forever(self) -> None:
        bad_output_path = Path(self.tempdir.name) / "outdir"
        bad_output_path.mkdir()
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "extra_args": f"--output-json {bad_output_path}",
            },
        )
        self.assertEqual(200, response.status)
        job = await response.json()
        final = await self._wait_for_terminal_state(job["job_id"])
        self.assertEqual("failed", final["status"])
        self.assertIsNotNone(final["finished_at"])
        self.assertIn("Failed to prepare vLLM benchmark invocation", final["stderr"])
        self.assertIn("is a directory", final["stderr"])

    async def test_web_reads_output_json_dash_destination_from_stdout(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "extra_args": "--output-json -",
            },
        )
        self.assertEqual(200, response.status)
        job = await response.json()
        final = await self._wait_for_terminal_state(job["job_id"])
        self.assertEqual("completed", final["status"])
        self.assertEqual("-", final["result_path"])
        self.assertEqual(1, final["record_count"])
        self.assertEqual("throughput", final["rows"][0]["subcommand"])

    async def test_web_returns_json_error_for_bad_input(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "extra_args": '--fake-sleep "2',
            },
        )
        self.assertEqual(400, response.status)
        payload = await response.json()
        self.assertIn("No closing quotation", payload["error"])

    async def test_web_exposes_live_output_while_job_is_running(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "live-output-model",
                "extra_args": "--fake-live-output --fake-sleep 5",
            },
        )
        self.assertEqual(200, response.status)
        payload = await response.json()
        job_id = payload["job_id"]

        running_with_output = None
        for _ in range(40):
            job_response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}")
            job_payload = await job_response.json()
            if (
                job_payload["status"] == "running"
                and "fake live stdout chunk 1" in (job_payload.get("stdout") or "")
                and "fake live stderr chunk 1" in (job_payload.get("stderr") or "")
            ):
                running_with_output = job_payload
                break
            await asyncio.sleep(0.15)
        self.assertIsNotNone(running_with_output)
        assert running_with_output is not None
        self.assertEqual("running", running_with_output["status"])

        stop_response = await self.session.post(f"{self.base_url}/api/jobs/{job_id}/stop")
        self.assertEqual(200, stop_response.status)
        final = await self._wait_for_terminal_state(job_id)
        self.assertEqual("stopped", final["status"])

    async def _wait_for_terminal_state(self, job_id: str) -> dict:
        for _ in range(60):
            response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}")
            payload = await response.json()
            if payload["status"] != "running":
                return payload
            await asyncio.sleep(0.1)
        self.fail("Job did not reach a terminal state")
