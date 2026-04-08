from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession, CookieJar, web

from llmbench.webapp import JOB_LAST_TOUCH_KEY, JOBS_KEY, RECOVERY_LEASE_TIMEOUT_SECONDS, create_app


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
        self.session = ClientSession(cookie_jar=CookieJar(unsafe=True))

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
                "tokenizer": "web-model",
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

    async def test_web_index_lists_current_sweep_subcommands(self) -> None:
        response = await self.session.get(f"{self.base_url}/")
        self.assertEqual(200, response.status)
        html = await response.text()
        self.assertIn('<option value="serve">serve</option>', html)
        self.assertIn('<option value="serve_workload">serve_workload</option>', html)
        self.assertIn('<option value="startup">startup</option>', html)
        self.assertIn('<option value="plot">plot</option>', html)
        self.assertIn('<option value="plot_pareto">plot_pareto</option>', html)
        self.assertNotIn("serve_sla", html)

    async def test_web_supports_nested_bench_paths(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "sweep",
                "sweep_subcommand": "serve",
                "model": "web-model",
            },
        )
        self.assertEqual(200, response.status)
        job = await response.json()
        final = await self._wait_for_terminal_state(job["job_id"])
        self.assertEqual("completed", final["status"])
        subcommands = {row["subcommand"] for row in final["rows"]}
        self.assertIn("sweep serve", subcommands)
        self.assertIn("sweep serve run-0", subcommands)

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

    async def test_web_session_isolation_blocks_cross_session_job_access(self) -> None:
        owner = ClientSession(cookie_jar=CookieJar(unsafe=True))
        other = ClientSession(cookie_jar=CookieJar(unsafe=True))
        try:
            start_response = await owner.post(
                f"{self.base_url}/api/jobs",
                json={
                    "subcommand": "serve",
                    "model": "owner-model",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, start_response.status)
            job = await start_response.json()
            job_id = job["job_id"]

            list_response = await other.get(f"{self.base_url}/api/jobs?status=running,stopping&limit=50")
            self.assertEqual(200, list_response.status)
            listed = await list_response.json()
            listed_ids = {item["job_id"] for item in listed.get("jobs", [])}
            self.assertNotIn(job_id, listed_ids)

            foreign_get = await other.get(f"{self.base_url}/api/jobs/{job_id}")
            self.assertEqual(404, foreign_get.status)
            foreign_stop = await other.post(f"{self.base_url}/api/jobs/{job_id}/stop")
            self.assertEqual(404, foreign_stop.status)

            owner_stop = await owner.post(f"{self.base_url}/api/jobs/{job_id}/stop")
            self.assertEqual(200, owner_stop.status)
            final = None
            for _ in range(60):
                owner_poll = await owner.get(f"{self.base_url}/api/jobs/{job_id}")
                owner_payload = await owner_poll.json()
                if owner_payload["status"] != "running":
                    final = owner_payload
                    break
                await asyncio.sleep(0.1)
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual("stopped", final["status"])
        finally:
            await owner.close()
            await other.close()

    async def test_web_rejects_capture_destination_overrides_from_browser(self) -> None:
        bad_output_path = Path(self.tempdir.name) / "browser-proof.json"
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "extra_args": f"--output-json {bad_output_path}",
            },
        )
        self.assertEqual(400, response.status)
        payload = await response.json()
        self.assertIn("only allows --output-json -", payload.get("error", ""))
        self.assertFalse(bad_output_path.exists())

        disallowed_result_dir = Path(self.tempdir.name) / "custom-result-dir"
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "demo-model",
                "extra_args": f"--result-dir {disallowed_result_dir}",
            },
        )
        self.assertEqual(400, response.status)
        payload = await response.json()
        self.assertIn("does not allow overriding result destinations", payload.get("error", ""))

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
        self.assertNotIn("result_path", final)
        self.assertEqual(1, final["record_count"])
        self.assertEqual("throughput", final["rows"][0]["subcommand"])

    async def test_web_throughput_random_maps_length_fields_to_random_flags(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "dataset_name": "random",
                "input_len": "8",
                "output_len": "8",
                "num_prompts": "1",
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        self.assertIn("--random-input-len", started["raw_args"])
        self.assertIn("--random-output-len", started["raw_args"])
        self.assertNotIn("--input-len", started["raw_args"])
        self.assertNotIn("--output-len", started["raw_args"])
        self.assertIn("--gpu-memory-utilization", started["raw_args"])
        self.assertIn("--max-model-len", started["raw_args"])
        gpu_index = started["raw_args"].index("--gpu-memory-utilization")
        max_len_index = started["raw_args"].index("--max-model-len")
        self.assertEqual("0.60", started["raw_args"][gpu_index + 1])
        self.assertEqual("1024", started["raw_args"][max_len_index + 1])
        self.assertIn("--enforce-eager", started["raw_args"])

        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertEqual("completed", final["status"])

    async def test_web_backfills_blank_low_memory_fields_with_safe_defaults(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "dataset_name": "random",
                "input_len": "8",
                "output_len": "8",
                "num_prompts": "1",
                "gpu_memory_utilization": "",
                "max_model_len": "",
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        self.assertIn("--gpu-memory-utilization", started["raw_args"])
        self.assertIn("--max-model-len", started["raw_args"])
        gpu_index = started["raw_args"].index("--gpu-memory-utilization")
        max_len_index = started["raw_args"].index("--max-model-len")
        self.assertEqual("0.60", started["raw_args"][gpu_index + 1])
        self.assertEqual("1024", started["raw_args"][max_len_index + 1])
        self.assertIn("--enforce-eager", started["raw_args"])
        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertEqual("completed", final["status"])

    async def test_web_throughput_ignores_unsupported_vllm_019_flags(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "dataset_name": "random",
                "input_len": "8",
                "output_len": "8",
                "num_prompts": "1",
                "save_detailed": True,
                "ignore_eos": True,
                "disable_tqdm": True,
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        self.assertNotIn("--save-detailed", started["raw_args"])
        self.assertNotIn("--ignore-eos", started["raw_args"])
        self.assertNotIn("--disable-tqdm", started["raw_args"])
        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertEqual("completed", final["status"])
        self.assertNotIn("--save-detailed", final["raw_args"])
        self.assertNotIn("--ignore-eos", final["raw_args"])
        self.assertNotIn("--disable-tqdm", final["raw_args"])

    async def test_web_serve_omits_throughput_only_low_memory_launch_flags(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "demo-model",
                "gpu_memory_utilization": "0.60",
                "max_model_len": "1024",
                "enforce_eager": True,
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        self.assertNotIn("--gpu-memory-utilization", started["raw_args"])
        self.assertNotIn("--max-model-len", started["raw_args"])
        self.assertNotIn("--enforce-eager", started["raw_args"])
        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertEqual("completed", final["status"])

    async def test_web_openai_blank_endpoint_defaults_to_completions(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "backend": "openai",
                "model": "demo-model",
                "tokenizer": "demo-model",
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        self.assertIn("--endpoint", started["raw_args"])
        endpoint_index = started["raw_args"].index("--endpoint")
        self.assertEqual("/v1/completions", started["raw_args"][endpoint_index + 1])
        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertEqual("completed", final["status"])

    async def test_web_openai_alias_without_tokenizer_rejects_without_server_probe(self) -> None:
        models_app = web.Application()
        seen_requests: list[str] = []

        async def handle_models(_request: web.Request) -> web.Response:
            seen_requests.append("models")
            return web.json_response(
                {
                    "data": [
                        {
                            "id": "qwen-local",
                            "root": "/home/leo/Qwen3.5-0.8B",
                        }
                    ]
                }
            )

        models_app.router.add_get("/v1/models", handle_models)
        models_runner = web.AppRunner(models_app)
        await models_runner.setup()
        models_site = web.TCPSite(models_runner, "127.0.0.1", 0)
        await models_site.start()
        sockets = models_site._server.sockets  # type: ignore[attr-defined]
        models_port = sockets[0].getsockname()[1]
        try:
            response = await self.session.post(
                f"{self.base_url}/api/jobs",
                json={
                    "subcommand": "throughput",
                    "backend": "openai",
                    "base_url": f"http://127.0.0.1:{models_port}",
                    "endpoint": "/v1/completions",
                    "model": "qwen-local",
                    "dataset_name": "random",
                    "input_len": "4",
                    "output_len": "2",
                    "num_prompts": "1",
                },
            )
            self.assertEqual(400, response.status)
            payload = await response.json()
            self.assertIn("Tokenizer is required", payload["error"])
            self.assertEqual([], seen_requests)
        finally:
            await models_runner.cleanup()

    async def test_web_openai_alias_without_tokenizer_returns_actionable_error_when_unresolved(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "backend": "openai",
                "base_url": "http://127.0.0.1:9",
                "endpoint": "/v1/completions",
                "model": "qwen-local",
                "dataset_name": "random",
                "input_len": "4",
                "output_len": "2",
                "num_prompts": "1",
            },
        )
        self.assertEqual(400, response.status)
        payload = await response.json()
        self.assertIn("Tokenizer is required", payload["error"])

    async def test_web_released_job_can_be_recovered_by_fresh_session(self) -> None:
        owner = ClientSession(cookie_jar=CookieJar(unsafe=True))
        same_browser_fresh = ClientSession(cookie_jar=CookieJar(unsafe=True))
        foreign_browser = ClientSession(cookie_jar=CookieJar(unsafe=True))
        owner_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-owner",
            "X-llmbench-tab-instance-id": "instance-owner",
        }
        same_browser_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-fresh",
            "X-llmbench-tab-instance-id": "instance-fresh",
        }
        foreign_browser_headers = {
            "X-llmbench-browser-id": "browser-foreign",
            "X-llmbench-tab-id": "tab-foreign",
            "X-llmbench-tab-instance-id": "instance-foreign",
        }
        try:
            start_response = await owner.post(
                f"{self.base_url}/api/jobs",
                headers=owner_headers,
                json={
                    "subcommand": "serve",
                    "model": "recoverable-model",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, start_response.status)
            started = await start_response.json()
            job_id = started["job_id"]

            release_response = await owner.post(f"{self.base_url}/api/jobs/{job_id}/release", headers=owner_headers)
            self.assertEqual(200, release_response.status)
            release_payload = await release_response.json()
            self.assertTrue(release_payload["recoverable"])

            listed_response = await same_browser_fresh.get(
                f"{self.base_url}/api/jobs?status=running,stopping&limit=50",
                headers=same_browser_headers,
            )
            self.assertEqual(200, listed_response.status)
            listed_payload = await listed_response.json()
            listed_ids = {item["job_id"] for item in listed_payload.get("jobs", [])}
            self.assertIn(job_id, listed_ids)

            foreign_listed_response = await foreign_browser.get(
                f"{self.base_url}/api/jobs?status=running,stopping&limit=50",
                headers=foreign_browser_headers,
            )
            self.assertEqual(200, foreign_listed_response.status)
            foreign_listed_payload = await foreign_listed_response.json()
            foreign_listed_ids = {item["job_id"] for item in foreign_listed_payload.get("jobs", [])}
            self.assertNotIn(job_id, foreign_listed_ids)

            foreign_get = await foreign_browser.get(f"{self.base_url}/api/jobs/{job_id}", headers=foreign_browser_headers)
            self.assertEqual(404, foreign_get.status)

            adopted_response = await same_browser_fresh.get(f"{self.base_url}/api/jobs/{job_id}", headers=same_browser_headers)
            self.assertEqual(200, adopted_response.status)
            adopted_payload = await adopted_response.json()
            self.assertEqual(job_id, adopted_payload["job_id"])

            stop_response = await same_browser_fresh.post(f"{self.base_url}/api/jobs/{job_id}/stop", headers=same_browser_headers)
            self.assertEqual(200, stop_response.status)
            final = await self._wait_for_terminal_state(job_id, session=same_browser_fresh, headers=same_browser_headers)
            self.assertEqual("stopped", final["status"])
        finally:
            await owner.close()
            await same_browser_fresh.close()
            await foreign_browser.close()

    async def test_web_completed_job_can_be_released_and_recovered_by_fresh_session(self) -> None:
        owner = ClientSession(cookie_jar=CookieJar(unsafe=True))
        same_browser_fresh = ClientSession(cookie_jar=CookieJar(unsafe=True))
        owner_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-owner",
            "X-llmbench-tab-instance-id": "instance-owner",
        }
        same_browser_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-fresh",
            "X-llmbench-tab-instance-id": "instance-fresh",
        }
        try:
            start_response = await owner.post(
                f"{self.base_url}/api/jobs",
                headers=owner_headers,
                json={
                    "subcommand": "serve",
                    "model": "completed-recoverable-model",
                },
            )
            self.assertEqual(200, start_response.status)
            started = await start_response.json()
            job_id = started["job_id"]

            final = await self._wait_for_terminal_state(job_id, session=owner, headers=owner_headers)
            self.assertEqual("completed", final["status"])

            release_response = await owner.post(f"{self.base_url}/api/jobs/{job_id}/release", headers=owner_headers)
            self.assertEqual(200, release_response.status)
            release_payload = await release_response.json()
            self.assertTrue(release_payload["recoverable"])

            listed_response = await same_browser_fresh.get(f"{self.base_url}/api/jobs?limit=50", headers=same_browser_headers)
            self.assertEqual(200, listed_response.status)
            listed_payload = await listed_response.json()
            listed_ids = {item["job_id"] for item in listed_payload.get("jobs", [])}
            self.assertIn(job_id, listed_ids)

            adopted_response = await same_browser_fresh.get(f"{self.base_url}/api/jobs/{job_id}", headers=same_browser_headers)
            self.assertEqual(200, adopted_response.status)
            adopted_payload = await adopted_response.json()
            self.assertEqual(job_id, adopted_payload["job_id"])
            self.assertEqual("completed", adopted_payload["status"])
        finally:
            await owner.close()
            await same_browser_fresh.close()

    async def test_web_release_owned_endpoint_releases_tab_owned_jobs_without_explicit_job_id(self) -> None:
        owner = ClientSession(cookie_jar=CookieJar(unsafe=True))
        same_browser_fresh = ClientSession(cookie_jar=CookieJar(unsafe=True))
        owner_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-owner",
            "X-llmbench-tab-instance-id": "instance-owner",
        }
        same_browser_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-fresh",
            "X-llmbench-tab-instance-id": "instance-fresh",
        }
        try:
            start_response = await owner.post(
                f"{self.base_url}/api/jobs",
                headers=owner_headers,
                json={
                    "subcommand": "serve",
                    "model": "recoverable-without-job-id",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, start_response.status)
            started = await start_response.json()
            job_id = started["job_id"]

            release_response = await owner.post(f"{self.base_url}/api/session/release-owned-jobs", headers=owner_headers)
            self.assertEqual(200, release_response.status)
            release_payload = await release_response.json()
            self.assertIn(job_id, release_payload["released_job_ids"])

            listed_response = await same_browser_fresh.get(
                f"{self.base_url}/api/jobs?status=running,stopping&limit=50",
                headers=same_browser_headers,
            )
            self.assertEqual(200, listed_response.status)
            listed_payload = await listed_response.json()
            listed_ids = {item["job_id"] for item in listed_payload.get("jobs", [])}
            self.assertIn(job_id, listed_ids)

            adopted_response = await same_browser_fresh.get(f"{self.base_url}/api/jobs/{job_id}", headers=same_browser_headers)
            self.assertEqual(200, adopted_response.status)
            stop_response = await same_browser_fresh.post(f"{self.base_url}/api/jobs/{job_id}/stop", headers=same_browser_headers)
            self.assertEqual(200, stop_response.status)
            final = await self._wait_for_terminal_state(job_id, session=same_browser_fresh, headers=same_browser_headers)
            self.assertEqual("stopped", final["status"])
        finally:
            await owner.close()
            await same_browser_fresh.close()

    async def test_web_hard_crash_fallback_recovers_stale_running_job_for_same_browser_only(self) -> None:
        owner = ClientSession(cookie_jar=CookieJar(unsafe=True))
        same_browser_fresh = ClientSession(cookie_jar=CookieJar(unsafe=True))
        foreign_browser = ClientSession(cookie_jar=CookieJar(unsafe=True))
        owner_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-owner",
            "X-llmbench-tab-instance-id": "instance-owner",
        }
        same_browser_headers = {
            "X-llmbench-browser-id": "browser-owner",
            "X-llmbench-tab-id": "tab-fresh",
            "X-llmbench-tab-instance-id": "instance-fresh",
        }
        foreign_browser_headers = {
            "X-llmbench-browser-id": "browser-foreign",
            "X-llmbench-tab-id": "tab-foreign",
            "X-llmbench-tab-instance-id": "instance-foreign",
        }
        try:
            start_response = await owner.post(
                f"{self.base_url}/api/jobs",
                headers=owner_headers,
                json={
                    "subcommand": "serve",
                    "model": "crash-recovery-model",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, start_response.status)
            started = await start_response.json()
            job_id = started["job_id"]

            pre_list = await same_browser_fresh.get(
                f"{self.base_url}/api/jobs?status=running,stopping&recoverable_only=1&limit=50",
                headers=same_browser_headers,
            )
            self.assertEqual(200, pre_list.status)
            pre_payload = await pre_list.json()
            pre_ids = {item["job_id"] for item in pre_payload.get("jobs", [])}
            self.assertNotIn(job_id, pre_ids)

            self.app[JOB_LAST_TOUCH_KEY][job_id] -= (RECOVERY_LEASE_TIMEOUT_SECONDS + 1.0)

            same_browser_list = await same_browser_fresh.get(
                f"{self.base_url}/api/jobs?status=running,stopping&recoverable_only=1&limit=50",
                headers=same_browser_headers,
            )
            self.assertEqual(200, same_browser_list.status)
            same_browser_payload = await same_browser_list.json()
            same_browser_ids = {item["job_id"] for item in same_browser_payload.get("jobs", [])}
            self.assertIn(job_id, same_browser_ids)

            foreign_list = await foreign_browser.get(
                f"{self.base_url}/api/jobs?status=running,stopping&recoverable_only=1&limit=50",
                headers=foreign_browser_headers,
            )
            self.assertEqual(200, foreign_list.status)
            foreign_payload = await foreign_list.json()
            foreign_ids = {item["job_id"] for item in foreign_payload.get("jobs", [])}
            self.assertNotIn(job_id, foreign_ids)

            adopted_response = await same_browser_fresh.get(f"{self.base_url}/api/jobs/{job_id}", headers=same_browser_headers)
            self.assertEqual(200, adopted_response.status)
            stop_response = await same_browser_fresh.post(f"{self.base_url}/api/jobs/{job_id}/stop", headers=same_browser_headers)
            self.assertEqual(200, stop_response.status)
            final = await self._wait_for_terminal_state(job_id, session=same_browser_fresh, headers=same_browser_headers)
            self.assertEqual("stopped", final["status"])
        finally:
            await owner.close()
            await same_browser_fresh.close()
            await foreign_browser.close()

    async def test_web_owned_running_job_is_not_marked_recoverable(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "owned-model",
                "extra_args": "--fake-sleep 5",
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        job_id = started["job_id"]

        listed_response = await self.session.get(f"{self.base_url}/api/jobs?status=running,stopping&limit=50")
        self.assertEqual(200, listed_response.status)
        listed_payload = await listed_response.json()
        listed_job = next(item for item in listed_payload.get("jobs", []) if item["job_id"] == job_id)
        self.assertFalse(listed_job["recoverable"])

        stop_response = await self.session.post(f"{self.base_url}/api/jobs/{job_id}/stop")
        self.assertEqual(200, stop_response.status)
        final = await self._wait_for_terminal_state(job_id)
        self.assertEqual("stopped", final["status"])

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

    async def test_web_redacts_runtime_endpoints_from_public_stdout(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "sensitive-stdout-model",
                "extra_args": "--fake-sensitive-stdout",
            },
        )
        self.assertEqual(200, response.status)
        payload = await response.json()
        final = await self._wait_for_terminal_state(payload["job_id"])
        self.assertEqual("completed", final["status"])
        self.assertIn("tcp://<redacted>", final["stdout"])
        self.assertNotIn("10.10.10.6", final["stdout"])

    async def test_web_surfaces_carriage_return_live_output_while_running(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "demo-model",
                "extra_args": "--fake-live-cr-output --fake-sleep 3",
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
                and "fake live stdout carriage 1" in (job_payload.get("stdout") or "")
                and "fake live stderr carriage 1" in (job_payload.get("stderr") or "")
            ):
                running_with_output = job_payload
                break
            await asyncio.sleep(0.15)
        self.assertIsNotNone(running_with_output)
        assert running_with_output is not None
        self.assertEqual("running", running_with_output["status"])

    async def test_web_redacts_host_local_model_and_tokenizer_paths_from_public_job_payload(self) -> None:
        local_model = "/home/leo/Qwen3.5-0.8B"
        local_tokenizer = "~/Qwen3.5-0.8B"
        local_base_url = "http://127.0.0.1:18011/v1"
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": local_model,
                "tokenizer": local_tokenizer,
                "base_url": local_base_url,
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        started_args_text = " ".join(str(token) for token in started.get("raw_args", []))
        started_payload = json.dumps(started)
        self.assertNotIn(local_model, started_payload)
        self.assertNotIn(local_tokenizer, started_payload)
        self.assertNotIn(local_base_url, started_payload)
        self.assertRegex(started_args_text, r"--model\s+browser-configured-model-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(started_args_text, r"--tokenizer\s+browser-configured-tokenizer-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(started_args_text, r"--base-url\s+browser-configured-base-url-[A-Za-z0-9._-]+-[A-Za-z0-9._-]+-[0-9a-f]{8}")

        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertEqual("completed", final["status"])
        final_payload = json.dumps(final)
        self.assertNotIn(local_model, final_payload)
        self.assertNotIn(local_tokenizer, final_payload)
        self.assertNotIn(local_base_url, final_payload)
        self.assertRegex(final_payload, r"browser-configured-model-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(final_payload, r"browser-configured-tokenizer-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(final_payload, r"browser-configured-base-url-[A-Za-z0-9._-]+-[A-Za-z0-9._-]+-[0-9a-f]{8}")

    async def test_web_expands_tilde_model_tokenizer_paths_and_redacts_bearer_tokens_in_public_raw_args(self) -> None:
        local_model = "~/Qwen3.5-0.8B"
        local_tokenizer = "~/Qwen3.5-0.8B"
        bearer_token = "sk-live-secret-token"
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": local_model,
                "tokenizer": local_tokenizer,
                "extra_args": f'--header "Authorization: Bearer {bearer_token}"',
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        started_payload = json.dumps(started)
        self.assertNotIn(bearer_token, started_payload)
        self.assertIn("<redacted-bearer-token>", started_payload)
        self.assertNotIn(local_model, started_payload)
        self.assertNotIn(local_tokenizer, started_payload)

        execution = self.app[JOBS_KEY].get_job(started["job_id"])
        self.assertIsNotNone(execution)
        assert execution is not None
        expanded_path = str(Path(local_model).expanduser())
        self.assertIn("--model", execution.raw_args)
        self.assertEqual(expanded_path, execution.raw_args[execution.raw_args.index("--model") + 1])
        self.assertIn("--tokenizer", execution.raw_args)
        self.assertEqual(expanded_path, execution.raw_args[execution.raw_args.index("--tokenizer") + 1])

        final = await self._wait_for_terminal_state(started["job_id"])
        final_payload = json.dumps(final)
        self.assertNotIn(bearer_token, final_payload)
        self.assertIn("<redacted-bearer-token>", final_payload)

    async def test_web_redacts_completed_record_model_and_tokenizer_paths_from_payload_and_exports(self) -> None:
        local_path = "/home/leo/Qwen3.5-0.8B"
        local_base_url = "https://internal.example.local:8443/v1"
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "serve",
                "model": "web-model",
                "backend": "openai",
                "tokenizer": "web-model",
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        job_id = started["job_id"]

        final = await self._wait_for_terminal_state(job_id)
        self.assertEqual("completed", final["status"])

        execution = self.app[JOBS_KEY].get_job(job_id)
        assert execution is not None
        execution.records = [
            {
                "tokenizer_id": local_path,
                "model_id": "~/Qwen3.5-0.8B",
                "base_url": local_base_url,
                "nested": {"tokenizerId": local_path},
            }
        ]

        job_response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}")
        self.assertEqual(200, job_response.status)
        payload = await job_response.json()
        payload_text = json.dumps(payload)
        self.assertRegex(payload_text, r"browser-configured-tokenizer-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(payload_text, r"browser-configured-model-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(payload_text, r"browser-configured-base-url-[A-Za-z0-9._-]+-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertNotIn(local_path, payload_text)
        self.assertNotIn("~/Qwen3.5-0.8B", payload_text)
        self.assertNotIn(local_base_url, payload_text)
        self.assertIn("Qwen3.5-0.8B", payload_text)

        csv_response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}/export.csv")
        self.assertEqual(200, csv_response.status)
        csv_text = await csv_response.text()
        self.assertRegex(csv_text, r"browser-configured-tokenizer-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(csv_text, r"browser-configured-model-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(csv_text, r"browser-configured-base-url-[A-Za-z0-9._-]+-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertNotIn(local_path, csv_text)
        self.assertNotIn("~/Qwen3.5-0.8B", csv_text)
        self.assertNotIn(local_base_url, csv_text)
        self.assertIn("Qwen3.5-0.8B", csv_text)

        jsonl_response = await self.session.get(f"{self.base_url}/api/jobs/{job_id}/export.jsonl")
        self.assertEqual(200, jsonl_response.status)
        jsonl_text = await jsonl_response.text()
        self.assertRegex(jsonl_text, r"browser-configured-tokenizer-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(jsonl_text, r"browser-configured-model-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertRegex(jsonl_text, r"browser-configured-base-url-[A-Za-z0-9._-]+-[A-Za-z0-9._-]+-[0-9a-f]{8}")
        self.assertNotIn(local_path, jsonl_text)
        self.assertNotIn("~/Qwen3.5-0.8B", jsonl_text)
        self.assertNotIn(local_base_url, jsonl_text)
        self.assertIn("Qwen3.5-0.8B", jsonl_text)

    async def test_web_hides_host_command_and_exposes_user_invocation_tokens(self) -> None:
        response = await self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "subcommand": "throughput",
                "model": "demo-model",
                "extra_args": "--output-json - --fake-live-output",
            },
        )
        self.assertEqual(200, response.status)
        started = await response.json()
        self.assertNotIn("command", started)
        self.assertNotIn("artifact_dir", started)
        self.assertNotIn("result_path", started)
        self.assertEqual(["throughput"], started["bench_path"])
        self.assertIn("--output-json", started["raw_args"])
        self.assertIn("-", started["raw_args"])
        self.assertIn("--fake-live-output", started["raw_args"])

        final = await self._wait_for_terminal_state(started["job_id"])
        self.assertNotIn("command", final)
        self.assertNotIn("artifact_dir", final)
        self.assertNotIn("result_path", final)
        self.assertEqual(["throughput"], final["bench_path"])

    async def test_web_redacts_server_binary_path_in_public_errors(self) -> None:
        isolated_app = create_app(
            artifact_root=Path(self.tempdir.name) / "isolated-artifacts",
            default_vllm_binary="/tmp/definitely-missing-vllm-binary",
        )
        isolated_runner = web.AppRunner(isolated_app)
        await isolated_runner.setup()
        isolated_site = web.TCPSite(isolated_runner, "127.0.0.1", 0)
        await isolated_site.start()
        sockets = isolated_site._server.sockets  # type: ignore[attr-defined]
        isolated_port = sockets[0].getsockname()[1]
        isolated_base = f"http://127.0.0.1:{isolated_port}"
        isolated_session = ClientSession(cookie_jar=CookieJar(unsafe=True))
        try:
            start = await isolated_session.post(
                f"{isolated_base}/api/jobs",
                json={"subcommand": "serve", "model": "demo-model"},
            )
            self.assertEqual(200, start.status)
            started = await start.json()
            job_id = started["job_id"]

            final = None
            for _ in range(60):
                response = await isolated_session.get(f"{isolated_base}/api/jobs/{job_id}")
                payload = await response.json()
                if payload["status"] != "running":
                    final = payload
                    break
                await asyncio.sleep(0.1)
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual("failed", final["status"])
            self.assertIn("Failed to launch server-configured vLLM binary.", final["stderr"])
            self.assertNotIn("/tmp/definitely-missing-vllm-binary", final["stderr"])
            self.assertNotIn("artifact_dir", final)
            self.assertNotIn("result_path", final)
        finally:
            await isolated_session.close()
            await isolated_runner.cleanup()

    async def _wait_for_terminal_state(
        self,
        job_id: str,
        session: ClientSession | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        active_session = session or self.session
        for _ in range(60):
            response = await active_session.get(f"{self.base_url}/api/jobs/{job_id}", headers=headers)
            payload = await response.json()
            if payload["status"] != "running":
                return payload
            await asyncio.sleep(0.1)
        self.fail("Job did not reach a terminal state")
