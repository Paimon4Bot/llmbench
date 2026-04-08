from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from playwright.async_api import async_playwright

from llmbench.webapp import JOB_LAST_TOUCH_KEY, JOBS_KEY, RECOVERY_LEASE_TIMEOUT_SECONDS, create_app


class WebUiPlaywrightTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.fake_binary = f"python3 {Path(__file__).with_name('fake_vllm.py')}"
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

    async def asyncTearDown(self) -> None:
        await self.runner.cleanup()
        self.tempdir.cleanup()

    async def test_bad_input_surfaces_visible_error_without_page_crash(self) -> None:
        page_errors: list[str] = []
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            await page.goto(self.base_url)
            await page.locator('textarea[name="extra_args"]').fill('--fake-sleep "2')
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")
            error_text = await page.locator("#errorBanner").text_content()
            await browser.close()

        self.assertEqual([], page_errors)
        self.assertIsNotNone(error_text)
        self.assertIn("No closing quotation", error_text)

    async def test_bad_input_preserves_previous_success_state(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            await page.wait_for_function("document.getElementById('jobRecords').textContent === '1'")
            first_job_id = await page.locator("#jobId").text_content()

            await page.locator('textarea[name="extra_args"]').fill('--fake-sleep "2')
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")

            self.assertEqual(first_job_id, await page.locator("#jobId").text_content())
            self.assertEqual("completed", await page.locator("#jobStatus").text_content())
            self.assertEqual("1", await page.locator("#jobRecords").text_content())
            self.assertIn(f"job {first_job_id}", (await page.locator("#resultContextText").text_content()) or "")
            self.assertTrue(await page.locator("#csvExport").is_enabled())
            self.assertIn("serve", await page.locator("#resultsTableWrap").text_content())

            await browser.close()

    async def test_reload_restores_running_job(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 5")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            first_job_id = await page.locator("#jobId").text_content()
            self.assertFalse(await page.locator('input[name="model"]').is_enabled())

            await page.reload()
            await page.wait_for_function("document.getElementById('jobId').textContent !== '-'")
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")

            self.assertEqual(first_job_id, await page.locator("#jobId").text_content())
            self.assertEqual("demo-model", await page.locator('input[name="model"]').input_value())
            self.assertIn("--model demo-model", await page.locator("#commandPreview").text_content())
            self.assertTrue(await page.locator("#stopButton").is_enabled())

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopping'")
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")

            await browser.close()

    async def test_reload_recovers_running_job_when_start_response_is_lost(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            local_model = "/home/leo/Qwen3.5-0.8B"
            local_tokenizer = "~/Qwen3.5-0.8B"
            local_base_url = "http://127.0.0.1:18011"
            await page.locator('input[name="model"]').fill(local_model)
            await page.locator('input[name="tokenizer"]').fill(local_tokenizer)
            await page.locator('input[name="base_url"]').fill(local_base_url)
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")

            dropped = {"pending": 1}

            async def drop_start_response(route):
                request = route.request
                if request.method == "POST" and dropped["pending"] > 0:
                    dropped["pending"] -= 1
                    await route.fetch()
                    await route.fulfill(
                        status=503,
                        content_type="application/json",
                        body='{"error":"simulated response loss"}',
                    )
                    return
                await route.continue_()

            await page.route("**/api/jobs", drop_start_response)
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")
            self.assertIn("simulated response loss", (await page.locator("#errorBanner").text_content()) or "")
            await page.unroute("**/api/jobs", drop_start_response)

            await page.reload()
            self.assertEqual("-", await page.locator("#jobId").text_content())
            self.assertEqual("idle", await page.locator("#jobStatus").text_content())
            await page.wait_for_function(
                """
                () => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value);
                }
                """
            )
            recovered_job_id = await page.evaluate(
                """
                () => {
                  const select = document.getElementById('recoverJobSelect');
                  const option = Array.from(select.options).find((item) => item.value);
                  return option ? option.value : '';
                }
                """
            )
            self.assertNotEqual("", recovered_job_id)
            await page.select_option("#recoverJobSelect", recovered_job_id)
            await page.get_by_role("button", name="Adopt Job").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertEqual(recovered_job_id, await page.locator("#jobId").text_content())
            recovered_model = await page.locator('input[name="model"]').input_value()
            recovered_tokenizer = await page.locator('input[name="tokenizer"]').input_value()
            recovered_base_url = await page.locator('input[name="base_url"]').input_value()
            self.assertNotEqual("", recovered_model)
            self.assertNotEqual("", recovered_tokenizer)
            self.assertNotEqual("", recovered_base_url)
            self.assertTrue(recovered_model.startswith("browser-configured-model-"))
            self.assertTrue(recovered_tokenizer.startswith("browser-configured-tokenizer-"))
            self.assertTrue(recovered_base_url.startswith("browser-configured-base-url-"))

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("browser-configured-model-", preview)
            self.assertIn("browser-configured-tokenizer-", preview)
            self.assertIn("browser-configured-base-url-", preview)
            self.assertNotIn(local_model, preview)
            self.assertNotIn(local_tokenizer, preview)
            self.assertNotIn(local_base_url, preview)

            stored_context = await page.evaluate("window.sessionStorage.getItem('llmbench.contextJobForm') || ''")
            self.assertIn("browser-configured-model-", stored_context)
            self.assertIn("browser-configured-tokenizer-", stored_context)
            self.assertIn("browser-configured-base-url-", stored_context)
            self.assertNotIn(local_model, stored_context)
            self.assertNotIn(local_tokenizer, stored_context)
            self.assertNotIn(local_base_url, stored_context)

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await browser.close()

    async def test_closing_page_after_lost_start_response_releases_running_job_for_fresh_session(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            owner_context = await browser.new_context()
            owner_page = await owner_context.new_page()
            await owner_page.goto(self.base_url)

            local_model = "/home/leo/Qwen3.5-0.8B"
            local_tokenizer = "~/Qwen3.5-0.8B"
            local_base_url = "http://127.0.0.1:18011"
            await owner_page.locator('input[name="model"]').fill(local_model)
            await owner_page.locator('input[name="tokenizer"]').fill(local_tokenizer)
            await owner_page.locator('input[name="base_url"]').fill(local_base_url)
            await owner_page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")

            dropped = {"pending": 1}

            async def drop_start_response(route):
                request = route.request
                if request.method == "POST" and dropped["pending"] > 0:
                    dropped["pending"] -= 1
                    await route.fetch()
                    await route.fulfill(
                        status=503,
                        content_type="application/json",
                        body='{"error":"simulated response loss"}',
                    )
                    return
                await route.continue_()

            await owner_page.route("**/api/jobs", drop_start_response)
            await owner_page.get_by_role("button", name="Start Benchmark").click()
            await owner_page.wait_for_selector("#errorBanner:not([hidden])")
            await owner_page.unroute("**/api/jobs", drop_start_response)

            await owner_page.close(run_before_unload=True)
            recovered_page = await owner_context.new_page()
            await recovered_page.goto(self.base_url)
            await recovered_page.get_by_role("button", name="Refresh Jobs").click()
            await recovered_page.wait_for_function(
                """
                () => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value);
                }
                """
            )
            recovered_job_id = await recovered_page.evaluate(
                """
                () => {
                  const select = document.getElementById('recoverJobSelect');
                  const option = Array.from(select.options).find((item) => item.value);
                  return option ? option.value : '';
                }
                """
            )
            self.assertNotEqual("", recovered_job_id)
            await recovered_page.select_option("#recoverJobSelect", recovered_job_id)
            await recovered_page.get_by_role("button", name="Adopt Job").click()
            await recovered_page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertEqual(recovered_job_id, await recovered_page.locator("#jobId").text_content())
            recovered_model = await recovered_page.locator('input[name="model"]').input_value()
            recovered_tokenizer = await recovered_page.locator('input[name="tokenizer"]').input_value()
            recovered_base_url = await recovered_page.locator('input[name="base_url"]').input_value()
            self.assertNotEqual("", recovered_model)
            self.assertNotEqual("", recovered_tokenizer)
            self.assertNotEqual("", recovered_base_url)
            self.assertTrue(recovered_model.startswith("browser-configured-model-"))
            self.assertTrue(recovered_tokenizer.startswith("browser-configured-tokenizer-"))
            self.assertTrue(recovered_base_url.startswith("browser-configured-base-url-"))

            preview = (await recovered_page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--model browser-configured-model-", preview)
            self.assertIn("--tokenizer browser-configured-tokenizer-", preview)
            self.assertIn("--base-url browser-configured-base-url-", preview)
            self.assertNotIn(local_model, preview)
            self.assertNotIn(local_tokenizer, preview)
            self.assertNotIn(local_base_url, preview)

            result_command = (await recovered_page.locator("#resultCommand").text_content()) or ""
            self.assertIn("--model browser-configured-model-", result_command)
            self.assertIn("--tokenizer browser-configured-tokenizer-", result_command)
            self.assertIn("--base-url browser-configured-base-url-", result_command)
            self.assertNotIn(local_model, result_command)
            self.assertNotIn(local_tokenizer, result_command)
            self.assertNotIn(local_base_url, result_command)

            await recovered_page.get_by_role("button", name="Stop Benchmark").click()
            await recovered_page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await owner_context.close()
            await browser.close()

    async def test_reload_recovers_completed_job_when_start_response_is_lost(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("completed-recovery-model")

            dropped = {"pending": 1}

            async def drop_start_response(route):
                request = route.request
                if request.method == "POST" and dropped["pending"] > 0:
                    dropped["pending"] -= 1
                    await route.fetch()
                    await route.fulfill(
                        status=503,
                        content_type="application/json",
                        body='{"error":"simulated response loss"}',
                    )
                    return
                await route.continue_()

            await page.route("**/api/jobs", drop_start_response)
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")
            self.assertIn("simulated response loss", (await page.locator("#errorBanner").text_content()) or "")
            await page.unroute("**/api/jobs", drop_start_response)

            await page.reload()
            await page.wait_for_function(
                """
                () => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value);
                }
                """
            )
            recovered_job_id = await page.evaluate(
                """
                () => {
                  const select = document.getElementById('recoverJobSelect');
                  const option = Array.from(select.options).find((item) => item.value);
                  return option ? option.value : '';
                }
                """
            )
            self.assertNotEqual("", recovered_job_id)
            await page.select_option("#recoverJobSelect", recovered_job_id)
            await page.get_by_role("button", name="Adopt Job").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            self.assertEqual(recovered_job_id, await page.locator("#jobId").text_content())
            self.assertTrue(await page.locator("#csvExport").is_enabled())

            await browser.close()

    async def test_closing_page_releases_running_job_for_fresh_session_recovery(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            owner_context = await browser.new_context()
            owner_page = await owner_context.new_page()
            await owner_page.goto(self.base_url)
            local_model = "/home/leo/Qwen3.5-0.8B"
            local_tokenizer = "~/Qwen3.5-0.8B"
            local_base_url = "http://127.0.0.1:18011"
            await owner_page.locator('input[name="model"]').fill(local_model)
            await owner_page.locator('input[name="tokenizer"]').fill(local_tokenizer)
            await owner_page.locator('input[name="base_url"]').fill(local_base_url)
            await owner_page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await owner_page.get_by_role("button", name="Start Benchmark").click()
            await owner_page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_id = (await owner_page.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

            await owner_page.close(run_before_unload=True)
            recovered_page = await owner_context.new_page()
            await recovered_page.goto(self.base_url)
            await recovered_page.get_by_role("button", name="Refresh Jobs").click()
            await recovered_page.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )
            await recovered_page.evaluate(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return;
                  select.value = targetJobId;
                  select.dispatchEvent(new Event('change', {bubbles: true}));
                }
                """,
                job_id,
            )
            await recovered_page.wait_for_function("!document.getElementById('adoptJobButton').disabled")
            await recovered_page.get_by_role("button", name="Adopt Job").click()
            await recovered_page.wait_for_function("document.getElementById('jobId').textContent !== '-'")
            self.assertEqual(job_id, await recovered_page.locator("#jobId").text_content())
            recovered_model = await recovered_page.locator('input[name="model"]').input_value()
            recovered_tokenizer = await recovered_page.locator('input[name="tokenizer"]').input_value()
            recovered_base_url = await recovered_page.locator('input[name="base_url"]').input_value()
            self.assertNotEqual("", recovered_model)
            self.assertNotEqual("", recovered_tokenizer)
            self.assertNotEqual("", recovered_base_url)
            self.assertTrue(recovered_model.startswith("browser-configured-model-"))
            self.assertTrue(recovered_tokenizer.startswith("browser-configured-tokenizer-"))
            self.assertTrue(recovered_base_url.startswith("browser-configured-base-url-"))

            preview = (await recovered_page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--model browser-configured-model-", preview)
            self.assertIn("--tokenizer browser-configured-tokenizer-", preview)
            self.assertIn("--base-url browser-configured-base-url-", preview)
            self.assertNotIn(local_model, preview)
            self.assertNotIn(local_tokenizer, preview)
            self.assertNotIn(local_base_url, preview)

            result_command = (await recovered_page.locator("#resultCommand").text_content()) or ""
            self.assertIn("--model browser-configured-model-", result_command)
            self.assertIn("--tokenizer browser-configured-tokenizer-", result_command)
            self.assertIn("--base-url browser-configured-base-url-", result_command)
            self.assertNotIn(local_model, result_command)
            self.assertNotIn(local_tokenizer, result_command)
            self.assertNotIn(local_base_url, result_command)

            await recovered_page.get_by_role("button", name="Stop Benchmark").click()
            await recovered_page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await owner_context.close()
            await browser.close()

    async def test_closing_page_releases_completed_job_for_fresh_session_recovery(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            owner_context = await browser.new_context()
            owner_page = await owner_context.new_page()
            await owner_page.goto(self.base_url)
            await owner_page.locator('input[name="model"]').fill("completed-recover-after-close")
            await owner_page.get_by_role("button", name="Start Benchmark").click()
            await owner_page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_id = (await owner_page.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

            await owner_page.close(run_before_unload=True)
            recovered_page = await owner_context.new_page()
            await recovered_page.goto(self.base_url)
            await recovered_page.get_by_role("button", name="Refresh Jobs").click()
            await recovered_page.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )
            await recovered_page.evaluate(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return;
                  select.value = targetJobId;
                  select.dispatchEvent(new Event('change', {bubbles: true}));
                }
                """,
                job_id,
            )
            await recovered_page.wait_for_function("!document.getElementById('adoptJobButton').disabled")
            await recovered_page.get_by_role("button", name="Adopt Job").click()
            await recovered_page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            self.assertEqual(job_id, await recovered_page.locator("#jobId").text_content())
            self.assertTrue(await recovered_page.locator("#csvExport").is_enabled())

            await owner_context.close()
            await browser.close()

    async def test_released_job_is_not_recoverable_from_foreign_browser_session(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            owner_context = await browser.new_context()
            owner_page = await owner_context.new_page()
            await owner_page.goto(self.base_url)
            await owner_page.locator('input[name="model"]').fill("release-boundary-model")
            await owner_page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await owner_page.get_by_role("button", name="Start Benchmark").click()
            await owner_page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_id = (await owner_page.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

            await owner_page.close(run_before_unload=True)
            await owner_context.close()

            foreign_context = await browser.new_context()
            foreign_page = await foreign_context.new_page()
            await foreign_page.goto(self.base_url)
            await foreign_page.get_by_role("button", name="Refresh Jobs").click()
            await foreign_page.wait_for_timeout(350)
            recoverable_values = await foreign_page.evaluate(
                """
                () => Array.from(document.getElementById('recoverJobSelect').options).map((option) => option.value)
                """
            )
            self.assertEqual([""], recoverable_values)
            foreign_status = await foreign_page.evaluate(
                """
                async (targetJobId) => {
                  const response = await fetch(`/api/jobs/${targetJobId}`);
                  return response.status;
                }
                """,
                job_id,
            )
            self.assertEqual(404, foreign_status)

            await foreign_context.close()
            await browser.close()

    async def test_stale_owner_lease_allows_same_browser_crash_recovery(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            owner_page = await context.new_page()
            await owner_page.goto(self.base_url)
            await owner_page.locator('input[name="model"]').fill("crash-lease-model")
            await owner_page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await owner_page.get_by_role("button", name="Start Benchmark").click()
            await owner_page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_id = (await owner_page.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

            # Simulate a hard crash path: no pagehide release is delivered.
            self.app[JOB_LAST_TOUCH_KEY][job_id] -= (RECOVERY_LEASE_TIMEOUT_SECONDS + 1.0)
            await owner_page.close()

            recovery_page = await context.new_page()
            await recovery_page.goto(self.base_url)
            await recovery_page.get_by_role("button", name="Refresh Jobs").click()
            await recovery_page.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )
            await recovery_page.select_option("#recoverJobSelect", job_id)
            await recovery_page.get_by_role("button", name="Adopt Job").click()
            await recovery_page.wait_for_function("document.getElementById('jobId').textContent !== '-'")
            self.assertEqual(job_id, await recovery_page.locator("#jobId").text_content())

            await recovery_page.get_by_role("button", name="Stop Benchmark").click()
            await recovery_page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context.close()
            await browser.close()

    async def test_fresh_session_cannot_adopt_foreign_running_job(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context_a = await browser.new_context()
            page_a = await context_a.new_page()
            await page_a.goto(self.base_url)
            await page_a.locator('input[name="model"]').fill("session-a-model")
            await page_a.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page_a.get_by_role("button", name="Start Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_id = (await page_a.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

            context_b = await browser.new_context()
            page_b = await context_b.new_page()
            await page_b.goto(self.base_url)
            await page_b.get_by_role("button", name="Refresh Jobs").click()
            await page_b.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return !Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )
            self.assertEqual("idle", await page_b.locator("#jobStatus").text_content())
            self.assertEqual("-", await page_b.locator("#jobId").text_content())
            self.assertTrue(await page_b.locator("#adoptJobButton").is_disabled())
            foreign_status = await page_b.evaluate(
                """
                async (targetJobId) => {
                  const response = await fetch(`/api/jobs/${targetJobId}`);
                  return response.status;
                }
                """,
                job_id,
            )
            self.assertEqual(404, foreign_status)

            await page_a.get_by_role("button", name="Stop Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context_b.close()
            await context_a.close()
            await browser.close()

    async def test_recovery_selector_lists_multiple_jobs_and_can_adopt_older_one(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.base_url)
            older_job_id = await page.evaluate(
                """
                async (baseUrl) => {
                  const response = await fetch(`${baseUrl}/api/jobs`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                      subcommand: "serve",
                      model: "older-model",
                      extra_args: "--fake-sleep 8"
                    }),
                  });
                  const payload = await response.json();
                  return payload.job_id || "";
                }
                """,
                self.base_url,
            )
            newer_job_id = await page.evaluate(
                """
                async (baseUrl) => {
                  const response = await fetch(`${baseUrl}/api/jobs`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                      subcommand: "serve",
                      model: "newer-model",
                      extra_args: "--fake-sleep 8"
                    }),
                  });
                  const payload = await response.json();
                  return payload.job_id || "";
                }
                """,
                self.base_url,
            )
            self.assertNotEqual("", older_job_id)
            self.assertNotEqual("", newer_job_id)
            await page.get_by_role("button", name="Refresh Jobs").click()
            await page.wait_for_function(
                f"""
                () => {{
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  const values = Array.from(select.options).map((option) => option.value);
                  return values.includes({json.dumps(older_job_id)}) && values.includes({json.dumps(newer_job_id)});
                }}
                """
            )

            await page.evaluate(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return;
                  select.value = targetJobId;
                  select.dispatchEvent(new Event('change', {bubbles: true}));
                }
                """,
                older_job_id,
            )
            await page.wait_for_function("!document.getElementById('adoptJobButton').disabled")
            await page.get_by_role("button", name="Adopt Job").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertEqual(older_job_id, await page.locator("#jobId").text_content())
            self.assertEqual("older-model", await page.locator('input[name="model"]').input_value())

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await page.evaluate(
                """
                async ({baseUrl, targetJobId}) => {
                  await fetch(`${baseUrl}/api/jobs/${targetJobId}/stop`, {method: "POST"});
                }
                """,
                {"baseUrl": self.base_url, "targetJobId": newer_job_id},
            )
            await context.close()
            await browser.close()

    async def test_recovery_selector_drops_adopted_job_without_manual_refresh(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.base_url)

            job_id = await page.evaluate(
                """
                async (baseUrl) => {
                  const response = await fetch(`${baseUrl}/api/jobs`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                      subcommand: "serve",
                      model: "selector-sync-model",
                      extra_args: "--fake-sleep 8"
                    }),
                  });
                  const payload = await response.json();
                  return payload.job_id || "";
                }
                """,
                self.base_url,
            )
            self.assertNotEqual("", job_id)

            await page.get_by_role("button", name="Refresh Jobs").click()
            await page.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )
            await page.select_option("#recoverJobSelect", job_id)
            await page.get_by_role("button", name="Adopt Job").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            await page.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return !Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await page.wait_for_function(
                """
                (targetJobId) => {
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return !Array.from(select.options).some((option) => option.value === targetJobId);
                }
                """,
                arg=job_id,
            )

            await context.close()
            await browser.close()

    async def test_reload_transient_fetch_failure_keeps_running_job_identity(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            first_job_id = await page.locator("#jobId").text_content()

            blocked = {"remaining": 1}

            async def maybe_abort(route):
                if blocked["remaining"] > 0:
                    blocked["remaining"] -= 1
                    await route.abort()
                    return
                await route.continue_()

            await page.route("**/api/jobs/*", maybe_abort)
            await page.reload()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'reconnecting'")

            self.assertEqual(first_job_id, await page.locator("#jobId").text_content())
            self.assertEqual(first_job_id, await page.evaluate("window.sessionStorage.getItem('llmbench.currentJobId')"))

            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertTrue(await page.locator("#stopButton").is_enabled())

            await page.unroute("**/api/jobs/*", maybe_abort)
            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await browser.close()

    async def test_transient_poll_failure_recovers_without_stranding_controls(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")

            blocked = {"remaining": 1}

            async def maybe_abort(route):
                if blocked["remaining"] > 0:
                    blocked["remaining"] -= 1
                    await route.abort()
                    return
                await route.continue_()

            await page.route("**/api/jobs/*", maybe_abort)
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'reconnecting'")
            self.assertTrue(await page.locator("#stopButton").is_enabled())
            self.assertFalse(await page.locator("#startButton").is_enabled())

            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            await page.unroute("**/api/jobs/*", maybe_abort)
            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await browser.close()

    async def test_delayed_stale_poll_response_cannot_override_stop_terminal_state(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_id = ((await page.locator("#jobId").text_content()) or "").strip()
            self.assertNotEqual("", job_id)
            stale_response = await page.evaluate(
                """
                async (targetJobId) => {
                  const tabId = window.sessionStorage.getItem('llmbench.tabId') || '';
                  const response = await fetch(`/api/jobs/${targetJobId}`, {
                    headers: tabId ? {'X-llmbench-tab-id': tabId} : {},
                  });
                  return {
                    status: response.status,
                    payload: await response.json(),
                  };
                }
                """,
                job_id,
            )
            self.assertEqual(200, stale_response["status"])
            stale_payload = stale_response["payload"]
            self.assertEqual("running", stale_payload["status"])

            delayed_poll_started = asyncio.Event()
            first_request = {"pending": 1}

            async def serve_stale_poll(route):
                request = route.request
                if (
                    request.method == "GET"
                    and first_request["pending"] > 0
                    and request.url.endswith(f"/api/jobs/{job_id}")
                ):
                    first_request["pending"] -= 1
                    delayed_poll_started.set()
                    await asyncio.sleep(1.5)
                    await route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(stale_payload),
                    )
                    return
                await route.continue_()

            await page.route("**/api/jobs/*", serve_stale_poll)
            await asyncio.wait_for(delayed_poll_started.wait(), timeout=5)
            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await page.wait_for_timeout(1700)

            self.assertEqual("stopped", await page.locator("#jobStatus").text_content())
            self.assertFalse(await page.locator("#stopButton").is_enabled())

            await page.unroute("**/api/jobs/*", serve_stale_poll)
            await browser.close()

    async def test_transient_stop_failure_leaves_retryable_controls(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")

            blocked = {"remaining": 1}

            async def maybe_abort(route):
                if blocked["remaining"] > 0:
                    blocked["remaining"] -= 1
                    await route.abort()
                    return
                await route.continue_()

            await page.route("**/api/jobs/*/stop", maybe_abort)
            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")

            self.assertTrue(await page.locator("#stopButton").is_enabled())
            self.assertFalse(await page.locator("#startButton").is_enabled())
            self.assertEqual("running", await page.locator("#jobStatus").text_content())

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await page.unroute("**/api/jobs/*/stop", maybe_abort)
            await browser.close()

    async def test_override_notice_locks_basic_selector(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("latency")
            await page.locator("summary", has_text="Runtime And Connection Controls").click()
            await page.locator('input[name="bench_path_override"]').fill("serve")
            await page.wait_for_selector("#overrideNotice:not([hidden])")

            self.assertIn("Override is controlling the command path", await page.locator("#overrideNotice").text_content())
            self.assertFalse(await page.locator('select[name="subcommand"]').is_enabled())
            self.assertIn("bench serve", await page.locator("#commandPreview").text_content())

            await browser.close()

    async def test_preview_uses_server_configured_binary_command(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("server-configured-vllm", preview)
            self.assertIn("bench serve", preview)
            self.assertNotIn("fake_vllm.py", preview)
            self.assertNotIn(str(Path(self.tempdir.name)), preview)

            await browser.close()

    async def test_openai_blank_endpoint_preview_shows_effective_default(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator("summary", has_text="Secondary Scenario Controls").click()
            await page.locator('select[name="backend"]').select_option("openai")
            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('input[name="tokenizer"]').fill("demo-model")
            await page.locator('input[name="endpoint"]').fill("")
            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--endpoint /v1/completions", preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("--endpoint /v1/completions", result_command)

            await browser.close()

    async def test_default_serve_preview_omits_throughput_only_low_memory_flags(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertNotIn("--gpu-memory-utilization", preview)
            self.assertNotIn("--max-model-len", preview)
            self.assertNotIn("--enforce-eager", preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertNotIn("--gpu-memory-utilization", result_command)
            self.assertNotIn("--max-model-len", result_command)
            self.assertNotIn("--enforce-eager", result_command)

            await browser.close()

    async def test_throughput_form_hides_unsupported_vllm_019_flags(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("throughput")
            self.assertEqual(0, await page.locator('input[name="save_detailed"]').count())
            self.assertEqual(0, await page.locator('input[name="ignore_eos"]').count())
            self.assertEqual(0, await page.locator('input[name="disable_tqdm"]').count())

            extra_args_placeholder = (await page.locator('textarea[name="extra_args"]').get_attribute("placeholder")) or ""
            self.assertNotIn("--save-detailed", extra_args_placeholder)
            self.assertNotIn("--ignore-eos", extra_args_placeholder)
            self.assertNotIn("--disable-tqdm", extra_args_placeholder)

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertNotIn("--save-detailed", preview)
            self.assertNotIn("--ignore-eos", preview)
            self.assertNotIn("--disable-tqdm", preview)

            await browser.close()

    async def test_blank_structured_low_memory_inputs_still_launch_with_safe_defaults(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("throughput")
            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator("summary", has_text="Secondary Scenario Controls").click()
            await page.locator('select[name="dataset_name"]').select_option("random")
            await page.locator('input[name="input_len"]').fill("8")
            await page.locator('input[name="output_len"]').fill("8")
            await page.locator("summary", has_text="Runtime And Connection Controls").click()
            await page.locator('input[name="num_prompts"]').fill("1")
            await page.locator('input[name="gpu_memory_utilization"]').fill("")
            await page.locator('input[name="max_model_len"]').fill("")

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--gpu-memory-utilization 0.60", preview)
            self.assertIn("--max-model-len 1024", preview)
            self.assertIn("--enforce-eager", preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("--gpu-memory-utilization 0.60", result_command)
            self.assertIn("--max-model-len 1024", result_command)
            self.assertIn("--enforce-eager", result_command)

            await browser.close()

    async def test_command_trust_cues_quote_arguments_with_spaces(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo model")
            await page.locator("summary", has_text="Advanced Inputs And Metrics").click()
            await page.locator('input[name="dataset_path"]').fill("/tmp/data dir/input.json")
            await page.locator('textarea[name="extra_args"]').fill("--request-id-prefix 'bench run'")
            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--model 'demo model'", preview)
            self.assertIn("--dataset-path '/tmp/data dir/input.json'", preview)
            self.assertIn("--request-id-prefix 'bench run'", preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("server-configured-vllm", result_command)
            self.assertIn("--model 'demo model'", result_command)
            self.assertIn("--dataset-path '/tmp/data dir/input.json'", result_command)
            self.assertIn("--request-id-prefix 'bench run'", result_command)
            self.assertNotIn("fake_vllm.py", result_command)
            self.assertNotIn(str(Path(self.tempdir.name)), result_command)

            await browser.close()

    async def test_command_surfaces_and_session_storage_redact_bearer_tokens_from_extra_args(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            bearer_token = "sk-live-secret-token"
            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill(
                f'--header "Authorization: Bearer {bearer_token}" --fake-sleep 5'
            )

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("Authorization: Bearer <redacted-bearer-token>", preview)
            self.assertNotIn(bearer_token, preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function(
                """
                () => {
                  const status = document.getElementById('jobStatus').textContent || '';
                  return status === 'running' || status === 'stopping' || status === 'completed';
                }
                """
            )
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("Authorization: Bearer <redacted-bearer-token>", result_command)
            self.assertNotIn(bearer_token, result_command)

            storage_dump = await page.evaluate(
                """
                () => {
                  const entries = {};
                  for (let index = 0; index < window.sessionStorage.length; index += 1) {
                    const key = window.sessionStorage.key(index);
                    if (key) {
                      entries[key] = window.sessionStorage.getItem(key);
                    }
                  }
                  return JSON.stringify(entries);
                }
                """
            )
            self.assertNotIn(bearer_token, storage_dump)
            self.assertIn("<redacted-bearer-token>", storage_dump)

            if await page.locator("#stopButton").is_enabled():
                await page.get_by_role("button", name="Stop Benchmark").click()
                await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")

            await browser.close()

    async def test_result_command_uses_generic_server_preview_after_failed_run(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("throughput")
            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--save-result")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'failed'")
            await page.wait_for_selector("#errorBanner:not([hidden])")
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("server-configured-vllm", result_command)
            self.assertIn("bench throughput", result_command)
            self.assertIn("--save-result", result_command)
            self.assertNotIn("fake_vllm.py", result_command)
            self.assertIn("Benchmark failed", (await page.locator("#errorBanner").text_content()) or "")
            self.assertIn("throughput rejects serve-style result flags", (await page.locator("#errorBanner").text_content()) or "")
            self.assertIn("Benchmark Failed", (await page.locator("#resultContextTitle").text_content()) or "")
            self.assertIn("failed before structured result rows were captured", (await page.locator("#resultContextText").text_content()) or "")
            self.assertIn("failure summary", (await page.locator("#resultsTableWrap").text_content()) or "")

            await browser.close()

    async def test_runtime_all_requests_failed_is_summarized_in_page(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("throughput")
            await page.locator('input[name="model"]').fill("runtime-failure-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-all-requests-fail")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'failed'")
            await page.wait_for_selector("#errorBanner:not([hidden])")

            self.assertIn("Benchmark failed", (await page.locator("#errorBanner").text_content()) or "")
            self.assertIn("All benchmark requests failed", (await page.locator("#errorBanner").text_content()) or "")
            self.assertIn("Benchmark Failed", (await page.locator("#resultContextTitle").text_content()) or "")
            self.assertIn("recorded zero successful requests", (await page.locator("#resultContextText").text_content()) or "")
            self.assertIn("recorded zero successful requests", (await page.locator("#infoBanner").text_content()) or "")
            self.assertIn("throughput", (await page.locator("#resultsTableWrap").text_content()) or "")
            self.assertIn("All benchmark requests failed", (await page.locator("#stderrBox").text_content()) or "")

            await browser.close()

    async def test_mobile_primary_action_is_near_top(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(viewport={"width": 390, "height": 844})
            await page.goto(self.base_url)

            box = await page.locator("#startButton").bounding_box()
            self.assertIsNotNone(box)
            assert box is not None
            self.assertLess(box["y"], 700)
            self.assertEqual("not-allowed", await page.locator("#stopButton").evaluate("el => getComputedStyle(el).cursor"))

            await browser.close()

    async def test_mobile_validation_error_is_scrolled_into_view(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(viewport={"width": 390, "height": 844})
            await page.goto(self.base_url)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.locator("summary", has_text="Runtime And Connection Controls").click()
            await page.locator('input[name="bench_path_override"]').fill('"bad')
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")

            banner_box = await page.locator("#errorBanner").bounding_box()
            self.assertIsNotNone(banner_box)
            assert banner_box is not None
            self.assertGreater(banner_box["y"] + banner_box["height"], 0)
            self.assertLess(banner_box["y"], 844)

            await browser.close()

    async def test_disabled_export_controls_are_buttons_and_not_focusable_links(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            csv_state = await page.locator("#csvExport").evaluate(
                "el => ({tag: el.tagName, disabled: el.disabled, matchesDisabled: el.matches(':disabled'), ariaDisabled: el.getAttribute('aria-disabled'), tabIndex: el.getAttribute('tabindex')})"
            )
            jsonl_state = await page.locator("#jsonlExport").evaluate(
                "el => ({tag: el.tagName, disabled: el.disabled, matchesDisabled: el.matches(':disabled'), ariaDisabled: el.getAttribute('aria-disabled'), tabIndex: el.getAttribute('tabindex')})"
            )
            self.assertEqual("BUTTON", csv_state["tag"])
            self.assertTrue(csv_state["disabled"])
            self.assertTrue(csv_state["matchesDisabled"])
            self.assertEqual("true", csv_state["ariaDisabled"])
            self.assertEqual("-1", csv_state["tabIndex"])
            self.assertEqual("BUTTON", jsonl_state["tag"])
            self.assertTrue(jsonl_state["disabled"])
            self.assertTrue(jsonl_state["matchesDisabled"])
            self.assertEqual("true", jsonl_state["ariaDisabled"])
            self.assertEqual("-1", jsonl_state["tabIndex"])

            focused_ids: list[str] = []
            for _ in range(24):
                await page.keyboard.press("Tab")
                focused_ids.append(
                    await page.evaluate("document.activeElement && document.activeElement.id ? document.activeElement.id : ''")
                )
            self.assertNotIn("csvExport", focused_ids)
            self.assertNotIn("jsonlExport", focused_ids)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            self.assertTrue(await page.locator("#csvExport").is_enabled())
            self.assertTrue(await page.locator("#jsonlExport").is_enabled())
            csv_enabled_state = await page.locator("#csvExport").evaluate(
                "el => ({ariaDisabled: el.getAttribute('aria-disabled'), tabIndex: el.getAttribute('tabindex')})"
            )
            jsonl_enabled_state = await page.locator("#jsonlExport").evaluate(
                "el => ({ariaDisabled: el.getAttribute('aria-disabled'), tabIndex: el.getAttribute('tabindex')})"
            )
            self.assertEqual("false", csv_enabled_state["ariaDisabled"])
            self.assertIsNone(csv_enabled_state["tabIndex"])
            self.assertEqual("false", jsonl_enabled_state["ariaDisabled"])
            self.assertIsNone(jsonl_enabled_state["tabIndex"])

            await browser.close()

    async def test_export_download_filenames_include_job_id_and_change_per_run(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("model-a")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_a = (await page.locator("#jobId").text_content()) or ""
            async with page.expect_download() as download_a_info:
                await page.locator("#csvExport").click()
            download_a = await download_a_info.value
            self.assertEqual(f"llmbench-{job_a}.csv", download_a.suggested_filename)

            await page.locator('input[name="model"]').fill("model-b")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function(
                """
                (previousJobId) => document.getElementById('jobId').textContent !== previousJobId
                """,
                arg=job_a,
            )
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_b = (await page.locator("#jobId").text_content()) or ""
            async with page.expect_download() as download_b_info:
                await page.locator("#csvExport").click()
            download_b = await download_b_info.value
            self.assertEqual(f"llmbench-{job_b}.csv", download_b.suggested_filename)
            self.assertNotEqual(job_a, job_b)
            self.assertNotEqual(download_a.suggested_filename, download_b.suggested_filename)

            await context.close()
            await browser.close()

    async def test_export_during_new_submit_stays_bound_to_displayed_job(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("first-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_a = (await page.locator("#jobId").text_content()) or ""

            delayed = {"pending": 1}

            async def delay_start(route):
                request = route.request
                if request.method == "POST" and delayed["pending"] > 0:
                    delayed["pending"] -= 1
                    await asyncio.sleep(0.3)
                await route.continue_()

            await page.route("**/api/jobs", delay_start)
            await page.locator('input[name="model"]').fill("second-model")
            await page.get_by_role("button", name="Start Benchmark").click()

            async with page.expect_download() as download_info:
                await page.locator("#csvExport").click()
            export_from_displayed = await download_info.value
            self.assertEqual(f"llmbench-{job_a}.csv", export_from_displayed.suggested_filename)
            self.assertEqual(job_a, (await page.locator("#jobId").text_content()) or "")

            await page.unroute("**/api/jobs", delay_start)
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_b = (await page.locator("#jobId").text_content()) or ""
            self.assertNotEqual(job_a, job_b)

            await context.close()
            await browser.close()

    async def test_completed_results_table_and_exports_redact_host_local_record_paths(self) -> None:
        local_path = "/home/leo/Qwen3.5-0.8B"
        local_base_url = "https://internal.example.local:8443/v1"
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_id = (await page.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

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

            await page.reload()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            await page.wait_for_function(
                """
                () => {
                  const text = document.getElementById("resultsTableWrap").textContent || "";
                  return text.includes("browser-configured-tokenizer-")
                    && text.includes("browser-configured-model-")
                    && text.includes("browser-configured-base-url-");
                }
                """
            )

            table_text = (await page.locator("#resultsTableWrap").text_content()) or ""
            self.assertIn("browser-configured-tokenizer-", table_text)
            self.assertIn("browser-configured-model-", table_text)
            self.assertIn("browser-configured-base-url-", table_text)
            self.assertIn("Qwen3.5-0.8B", table_text)
            self.assertNotIn(local_path, table_text)
            self.assertNotIn("~/Qwen3.5-0.8B", table_text)
            self.assertNotIn(local_base_url, table_text)

            async with page.expect_download() as csv_download_info:
                await page.locator("#csvExport").click()
            csv_download = await csv_download_info.value
            csv_text = Path(await csv_download.path()).read_text()
            self.assertIn("browser-configured-tokenizer-", csv_text)
            self.assertIn("browser-configured-model-", csv_text)
            self.assertIn("browser-configured-base-url-", csv_text)
            self.assertIn("Qwen3.5-0.8B", csv_text)
            self.assertNotIn(local_path, csv_text)
            self.assertNotIn("~/Qwen3.5-0.8B", csv_text)
            self.assertNotIn(local_base_url, csv_text)

            async with page.expect_download() as jsonl_download_info:
                await page.locator("#jsonlExport").click()
            jsonl_download = await jsonl_download_info.value
            jsonl_text = Path(await jsonl_download.path()).read_text()
            self.assertIn("browser-configured-tokenizer-", jsonl_text)
            self.assertIn("browser-configured-model-", jsonl_text)
            self.assertIn("browser-configured-base-url-", jsonl_text)
            self.assertIn("Qwen3.5-0.8B", jsonl_text)
            self.assertNotIn(local_path, jsonl_text)
            self.assertNotIn("~/Qwen3.5-0.8B", jsonl_text)
            self.assertNotIn(local_base_url, jsonl_text)

            await context.close()
            await browser.close()

    async def test_success_without_structured_rows_explains_table_state(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("sweep")
            await page.locator("summary", has_text="Runtime And Connection Controls").click()
            await page.locator('input[name="bench_path_override"]').fill("sweep plot")
            await page.locator('textarea[name="extra_args"]').fill("--dry-run")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")

            info = await page.locator("#infoBanner").text_content()
            self.assertIn("did not emit structured rows", info or "")
            self.assertIn("No structured result rows", await page.locator("#resultsTableWrap").text_content())

            await browser.close()

    async def test_completed_results_stay_distinct_from_next_draft_command(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            job_id = await page.locator("#jobId").text_content()

            await page.locator('input[name="model"]').fill("next-model")
            self.assertIn("Draft launch command", (await page.locator("#commandPreviewLabel").text_content()) or "")
            self.assertIn("next-model", (await page.locator("#commandPreview").text_content()) or "")
            self.assertIn(f"job {job_id}", (await page.locator("#resultContextText").text_content()) or "")
            self.assertIn("demo-model", (await page.locator("#resultCommand").text_content()) or "")

            await browser.close()

    async def test_reload_completed_job_restores_form_context_for_displayed_results(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("origin-model")
            await page.locator('textarea[name="extra_args"]').fill("--request-rate 9 --max-concurrency 5")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            completed_job_id = await page.locator("#jobId").text_content()
            self.assertIsNotNone(completed_job_id)

            await page.locator('input[name="model"]').fill("draft-model")
            self.assertIn("draft-model", (await page.locator("#commandPreview").text_content()) or "")

            await page.reload()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            self.assertEqual(completed_job_id, await page.locator("#jobId").text_content())
            self.assertEqual("origin-model", await page.locator('input[name="model"]').input_value())

            preview = (await page.locator("#commandPreview").text_content()) or ""
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("origin-model", preview)
            self.assertIn("origin-model", result_command)
            self.assertNotIn("draft-model", preview)
            self.assertNotIn("draft-model", result_command)
            self.assertIn(f"job {completed_job_id}", (await page.locator("#resultContextText").text_content()) or "")

            await browser.close()

    async def test_reload_preserves_explicit_output_json_dash_flag_in_form_context(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("throughput")
            await page.locator('input[name="model"]').fill("tp-model")
            await page.locator('textarea[name="extra_args"]').fill("--output-json - --fake-live-output")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            self.assertEqual("--output-json - --fake-live-output", await page.locator('textarea[name="extra_args"]').input_value())
            self.assertIn("--output-json - --fake-live-output", (await page.locator("#commandPreview").text_content()) or "")

            await page.reload()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            self.assertEqual("--output-json - --fake-live-output", await page.locator('textarea[name="extra_args"]').input_value())
            self.assertIn("--output-json - --fake-live-output", (await page.locator("#commandPreview").text_content()) or "")

            await browser.close()

    async def test_throughput_random_preview_uses_random_length_flags(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("throughput")
            await page.locator("summary", has_text="Secondary Scenario Controls").click()
            await page.locator('select[name="dataset_name"]').select_option("random")
            await page.locator('input[name="input_len"]').fill("8")
            await page.locator('input[name="output_len"]').fill("8")

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--random-input-len 8", preview)
            self.assertIn("--random-output-len 8", preview)
            self.assertNotIn("--input-len 8", preview)
            self.assertNotIn("--output-len 8", preview)

            await browser.close()

    async def test_running_job_surfaces_live_stdout_and_stderr(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-live-output --fake-sleep 5")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            await page.wait_for_function("document.getElementById('stdoutBox').textContent.includes('fake live stdout chunk 1')")
            await page.wait_for_function("document.getElementById('stderrBox').textContent.includes('fake live stderr chunk 1')")

            self.assertIn("Current Benchmark", (await page.locator("#resultContextTitle").text_content()) or "")
            self.assertIn("is running", (await page.locator("#resultContextText").text_content()) or "")
            self.assertEqual("running", await page.locator("#jobStatus").text_content())
            self.assertNotEqual("-", await page.locator("#jobElapsed").text_content())
            self.assertIn("fake live stdout chunk 1", (await page.locator("#stdoutBox").text_content()) or "")
            self.assertIn("fake live stderr chunk 1", (await page.locator("#stderrBox").text_content()) or "")

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await browser.close()

    async def test_running_job_surfaces_carriage_return_live_output_and_redacts_command_values(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("/home/leo/Qwen3.5-0.8B")
            await page.locator('input[name="tokenizer"]').fill("/home/leo/Qwen3.5-0.8B")
            await page.locator('input[name="base_url"]').fill("http://127.0.0.1:18011")
            await page.locator('textarea[name="extra_args"]').fill("--fake-live-cr-output --fake-sleep 5")
            await page.wait_for_function(
                """
                () => {
                  const text = document.getElementById("commandPreview").textContent || "";
                  return text.includes("--model browser-configured-model-")
                    && text.includes("--tokenizer browser-configured-tokenizer-")
                    && text.includes("--base-url browser-configured-base-url-");
                }
                """
            )

            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--model browser-configured-model-", preview)
            self.assertIn("--tokenizer browser-configured-tokenizer-", preview)
            self.assertIn("--base-url browser-configured-base-url-", preview)
            self.assertNotIn("/home/leo/Qwen3.5-0.8B", preview)
            self.assertNotIn("http://127.0.0.1:18011", preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            await page.wait_for_function("document.getElementById('stdoutBox').textContent.includes('fake live stdout carriage 1')")
            await page.wait_for_function("document.getElementById('stderrBox').textContent.includes('fake live stderr carriage 1')")

            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("--model browser-configured-model-", result_command)
            self.assertIn("--tokenizer browser-configured-tokenizer-", result_command)
            self.assertIn("--base-url browser-configured-base-url-", result_command)
            self.assertNotIn("/home/leo/Qwen3.5-0.8B", result_command)
            self.assertNotIn("http://127.0.0.1:18011", result_command)

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await browser.close()

    async def test_browser_visible_identifiers_stay_stable_across_preview_result_and_export(self) -> None:
        local_model = "/home/leo/Qwen3.5-0.8B"
        local_tokenizer = "~/Qwen3.5-0.8B"
        local_base_url = "http://127.0.0.1:18011"

        def extract_flag_value(command: str, flag: str) -> str:
            match = re.search(rf"{re.escape(flag)}\s+([A-Za-z0-9._-]+)", command)
            self.assertIsNotNone(match, msg=f"missing {flag} in command: {command}")
            assert match is not None
            return match.group(1)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill(local_model)
            await page.locator('input[name="tokenizer"]').fill(local_tokenizer)
            await page.locator('input[name="base_url"]').fill(local_base_url)
            await page.wait_for_function(
                """
                () => {
                  const text = document.getElementById("commandPreview").textContent || "";
                  return text.includes("--model browser-configured-model-")
                    && text.includes("--tokenizer browser-configured-tokenizer-")
                    && text.includes("--base-url browser-configured-base-url-");
                }
                """
            )
            preview_command = (await page.locator("#commandPreview").text_content()) or ""
            preview_model = extract_flag_value(preview_command, "--model")
            preview_tokenizer = extract_flag_value(preview_command, "--tokenizer")
            preview_base_url = extract_flag_value(preview_command, "--base-url")

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")

            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertEqual(preview_model, extract_flag_value(result_command, "--model"))
            self.assertEqual(preview_tokenizer, extract_flag_value(result_command, "--tokenizer"))
            self.assertEqual(preview_base_url, extract_flag_value(result_command, "--base-url"))

            async with page.expect_download() as jsonl_download_info:
                await page.locator("#jsonlExport").click()
            jsonl_download = await jsonl_download_info.value
            jsonl_text = Path(await jsonl_download.path()).read_text()
            self.assertIn(preview_model, jsonl_text)
            self.assertIn(preview_base_url, jsonl_text)
            self.assertNotIn(local_model, jsonl_text)
            self.assertNotIn(local_base_url, jsonl_text)

            await context.close()
            await browser.close()

    async def test_completed_job_clears_browser_visible_placeholders_from_editable_form(self) -> None:
        local_model = "/home/leo/Qwen3.5-0.8B"
        local_tokenizer = "~/Qwen3.5-0.8B"
        local_base_url = "http://127.0.0.1:18011"

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill(local_model)
            await page.locator('input[name="tokenizer"]').fill(local_tokenizer)
            await page.locator('input[name="base_url"]').fill(local_base_url)
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")

            self.assertEqual("", await page.locator('input[name="model"]').input_value())
            self.assertEqual("", await page.locator('input[name="tokenizer"]').input_value())
            self.assertEqual("", await page.locator('input[name="base_url"]').input_value())

            preview = (await page.locator("#commandPreview").text_content()) or ""
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertNotIn("browser-configured-model-", preview)
            self.assertNotIn("browser-configured-tokenizer-", preview)
            self.assertNotIn("browser-configured-base-url-", preview)
            self.assertIn("browser-configured-model-", result_command)
            self.assertIn("browser-configured-tokenizer-", result_command)
            self.assertIn("browser-configured-base-url-", result_command)
            self.assertNotIn(local_model, result_command)
            self.assertNotIn(local_base_url, result_command)

            await browser.close()

    async def test_placeholder_configuration_cannot_be_submitted_as_new_benchmark(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("guard-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            completed_job_id = (await page.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", completed_job_id)

            post_count = {"value": 0}

            async def count_job_posts(route):
                if route.request.method == "POST":
                    post_count["value"] += 1
                await route.continue_()

            await page.route("**/api/jobs", count_job_posts)
            await page.locator('input[name="model"]').fill("browser-configured-model-guard-12345678")
            await page.locator('input[name="tokenizer"]').fill("browser-configured-tokenizer-guard-12345678")
            await page.locator('input[name="base_url"]').fill("browser-configured-base-url-http-private-12345678")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")

            error_text = (await page.locator("#errorBanner").text_content()) or ""
            self.assertIn("browser-visible placeholders", error_text)
            self.assertIn("Model, Tokenizer, and Base URL", error_text)
            self.assertEqual(0, post_count["value"])
            self.assertEqual(completed_job_id, (await page.locator("#jobId").text_content()) or "")
            self.assertEqual("completed", (await page.locator("#jobStatus").text_content()) or "")

            await page.unroute("**/api/jobs", count_job_posts)
            await browser.close()

    async def test_stopped_job_does_not_surface_failure_banner(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("stop-model")
            await page.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")

            self.assertTrue(await page.locator("#errorBanner").is_hidden())
            self.assertIn("Benchmark Stopped", (await page.locator("#resultContextTitle").text_content()) or "")
            self.assertIn("was stopped before structured result rows were captured", (await page.locator("#resultContextText").text_content()) or "")
            self.assertNotIn("Benchmark failed", (await page.locator("#stderrBox").text_content()) or "")

            await browser.close()

    async def test_multi_tab_reload_recovers_each_tab_own_job(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page_a = await context.new_page()
            page_b = await context.new_page()
            await page_a.goto(self.base_url)
            await page_b.goto(self.base_url)

            await page_a.locator('input[name="model"]').fill("model-a")
            await page_a.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page_a.get_by_role("button", name="Start Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_a = await page_a.locator("#jobId").text_content()

            await page_b.locator('input[name="model"]').fill("model-b")
            await page_b.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page_b.get_by_role("button", name="Start Benchmark").click()
            await page_b.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_b = await page_b.locator("#jobId").text_content()

            self.assertNotEqual(job_a, job_b)

            await page_a.reload()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertEqual(job_a, await page_a.locator("#jobId").text_content())
            self.assertEqual("model-a", await page_a.locator('input[name="model"]').input_value())

            await page_a.get_by_role("button", name="Stop Benchmark").click()
            await page_b.get_by_role("button", name="Stop Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await page_b.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context.close()
            await browser.close()

    async def test_cloned_session_storage_rotates_tab_identity_and_blocks_foreign_job_control(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page_a = await context.new_page()
            await page_a.goto(self.base_url)

            await page_a.locator('input[name="model"]').fill("cloned-tab-model")
            await page_a.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page_a.get_by_role("button", name="Start Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            job_id = (await page_a.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", job_id)

            cloned_storage = await page_a.evaluate(
                """
                () => {
                  const entries = {};
                  for (let index = 0; index < window.sessionStorage.length; index += 1) {
                    const key = window.sessionStorage.key(index);
                    if (key) {
                      entries[key] = window.sessionStorage.getItem(key);
                    }
                  }
                  return entries;
                }
                """
            )
            original_tab_id = cloned_storage.get("llmbench.tabId", "")
            self.assertNotEqual("", original_tab_id)

            page_b = await context.new_page()
            await page_b.add_init_script(
                f"""
                (() => {{
                  const entries = {json.dumps(cloned_storage)};
                  for (const [key, value] of Object.entries(entries || {{}})) {{
                    window.sessionStorage.setItem(key, value);
                  }}
                }})();
                """,
            )
            await page_b.goto(self.base_url)
            await page_b.wait_for_function("window.sessionStorage.getItem('llmbench.currentJobId') === null")
            await page_b.wait_for_function("document.getElementById('jobStatus').textContent === 'idle'")

            cloned_tab_id = await page_b.evaluate("window.sessionStorage.getItem('llmbench.tabId') || ''")
            self.assertNotEqual(original_tab_id, cloned_tab_id)
            self.assertEqual("-", await page_b.locator("#jobId").text_content())
            self.assertTrue(await page_b.locator("#stopButton").is_disabled())
            self.assertIn(
                "still active in another tab",
                ((await page_b.locator("#infoBanner").text_content()) or "").lower(),
            )
            self.assertNotIn("Unknown job id", (await page_b.locator("#errorBanner").text_content()) or "")

            foreign_status = await page_b.evaluate(
                """
                async (targetJobId) => {
                  const tabId = window.sessionStorage.getItem('llmbench.tabId') || '';
                  const response = await fetch(`/api/jobs/${targetJobId}`, {
                    headers: {"X-llmbench-tab-id": tabId},
                  });
                  return response.status;
                }
                """,
                job_id,
            )
            self.assertEqual(409, foreign_status)

            await page_a.get_by_role("button", name="Stop Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context.close()
            await browser.close()

    async def test_results_table_stays_inside_scroll_container(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")

            page_scroll = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            wrap_overflow = await page.locator("#resultsTableWrap").evaluate(
                "el => ({overflowX: getComputedStyle(el).overflowX, clientWidth: el.clientWidth, scrollWidth: el.scrollWidth})"
            )

            self.assertLessEqual(page_scroll, 8)
            self.assertEqual("auto", wrap_overflow["overflowX"])
            self.assertLessEqual(wrap_overflow["clientWidth"], 1440)

            await browser.close()

    async def test_results_table_supports_keyboard_horizontal_scroll(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(viewport={"width": 1024, "height": 768})
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo-model")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            await page.locator("#resultsTableWrap").focus()
            await page.keyboard.press("End")
            scroll_left = await page.locator("#resultsTableWrap").evaluate("el => el.scrollLeft")
            self.assertGreater(scroll_left, 0)

            await browser.close()

    async def test_error_banner_is_announced_accessibly(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('textarea[name="extra_args"]').fill('--fake-sleep "2')
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_selector("#errorBanner:not([hidden])")

            banner_state = await page.locator("#errorBanner").evaluate(
                "el => ({id: el.id, role: el.getAttribute('role'), live: el.getAttribute('aria-live'), activeId: document.activeElement && document.activeElement.id})"
            )
            self.assertEqual("alert", banner_state["role"])
            self.assertEqual("assertive", banner_state["live"])
            self.assertEqual("errorBanner", banner_state["activeId"])

            await browser.close()

    async def test_same_session_running_job_is_not_listed_as_recoverable(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page_a = await context.new_page()
            page_b = await context.new_page()
            await page_a.goto(self.base_url)
            await page_b.goto(self.base_url)

            await page_a.locator('input[name="model"]').fill("same-session-model")
            await page_a.locator('textarea[name="extra_args"]').fill("--fake-sleep 8")
            await page_a.get_by_role("button", name="Start Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            running_job_id = (await page_a.locator("#jobId").text_content()) or ""
            self.assertNotEqual("", running_job_id)

            await page_b.get_by_role("button", name="Refresh Jobs").click()
            await page_b.wait_for_timeout(250)
            recoverable_values = await page_b.evaluate(
                """
                () => Array.from(document.getElementById('recoverJobSelect').options).map((option) => option.value)
                """
            )
            self.assertEqual([""], recoverable_values)

            await page_a.get_by_role("button", name="Stop Benchmark").click()
            await page_a.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context.close()
            await browser.close()
