from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession, web
from playwright.async_api import async_playwright

from llmbench.webapp import create_app


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

            await page.locator('input[name="model"]').fill("recovery-model")
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
            self.assertEqual("recovery-model", await page.locator('input[name="model"]').input_value())

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await browser.close()

    async def test_fresh_session_requires_explicit_adopt_for_running_job(self) -> None:
        async with ClientSession() as session:
            response = await session.post(
                f"{self.base_url}/api/jobs",
                json={
                    "subcommand": "serve",
                    "model": "fresh-session-model",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, response.status)
            payload = await response.json()
            job_id = payload["job_id"]

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.base_url)

            self.assertEqual("idle", await page.locator("#jobStatus").text_content())
            self.assertEqual("-", await page.locator("#jobId").text_content())
            await page.wait_for_function(
                f"""
                () => {{
                  const select = document.getElementById('recoverJobSelect');
                  if (!select) return false;
                  return Array.from(select.options).some((option) => option.value === {json.dumps(job_id)});
                }}
                """
            )
            self.assertEqual(None, await page.evaluate("window.sessionStorage.getItem('llmbench.currentJobId')"))

            await page.select_option("#recoverJobSelect", job_id)
            await page.get_by_role("button", name="Adopt Job").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertEqual(job_id, await page.locator("#jobId").text_content())
            self.assertEqual("fresh-session-model", await page.locator('input[name="model"]').input_value())
            self.assertEqual(job_id, await page.evaluate("window.sessionStorage.getItem('llmbench.currentJobId')"))

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context.close()
            await browser.close()

    async def test_recovery_selector_lists_multiple_jobs_and_can_adopt_older_one(self) -> None:
        async with ClientSession() as session:
            first = await session.post(
                f"{self.base_url}/api/jobs",
                json={
                    "subcommand": "serve",
                    "model": "older-model",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, first.status)
            first_payload = await first.json()
            older_job_id = first_payload["job_id"]

            second = await session.post(
                f"{self.base_url}/api/jobs",
                json={
                    "subcommand": "serve",
                    "model": "newer-model",
                    "extra_args": "--fake-sleep 8",
                },
            )
            self.assertEqual(200, second.status)
            second_payload = await second.json()
            newer_job_id = second_payload["job_id"]

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.base_url)
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

            await page.select_option("#recoverJobSelect", older_job_id)
            await page.get_by_role("button", name="Adopt Job").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'running'")
            self.assertEqual(older_job_id, await page.locator("#jobId").text_content())
            self.assertEqual("older-model", await page.locator('input[name="model"]').input_value())

            await page.get_by_role("button", name="Stop Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'stopped'")
            await context.close()
            await browser.close()

        async with ClientSession() as cleanup:
            await cleanup.post(f"{self.base_url}/api/jobs/{newer_job_id}/stop")

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
            self.assertIn("python3", preview)
            self.assertIn("fake_vllm.py", preview)
            self.assertIn("bench serve", preview)

            await browser.close()

    async def test_command_trust_cues_quote_arguments_with_spaces(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('input[name="model"]').fill("demo model")
            await page.locator('input[name="dataset_path"]').fill("/tmp/data dir/input.json")
            await page.locator('textarea[name="extra_args"]').fill("--request-id-prefix 'bench run'")
            preview = (await page.locator("#commandPreview").text_content()) or ""
            self.assertIn("--model 'demo model'", preview)
            self.assertIn("--dataset-path '/tmp/data dir/input.json'", preview)
            self.assertIn("--request-id-prefix 'bench run'", preview)

            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'completed'")
            result_command = (await page.locator("#resultCommand").text_content()) or ""
            self.assertIn("--model 'demo model'", result_command)
            self.assertIn("--dataset-path '/tmp/data dir/input.json'", result_command)
            self.assertIn("--request-id-prefix 'bench run'", result_command)

            await browser.close()

    async def test_result_command_fallback_quotes_space_delimited_subcommand(self) -> None:
        bad_output_target = Path(self.tempdir.name) / "not a directory"
        bad_output_target.write_text("x", encoding="utf-8")
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("sweep")
            await page.locator('select[name="sweep_subcommand"]').select_option("serve_sla")
            await page.locator('textarea[name="extra_args"]').fill(f"--output-dir '{bad_output_target}'")
            await page.get_by_role("button", name="Start Benchmark").click()
            await page.wait_for_function("document.getElementById('jobStatus').textContent === 'failed'")
            self.assertEqual("'sweep serve_sla'", (await page.locator("#resultCommand").text_content()) or "")

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
                    await page.wait_for_timeout(300)
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

    async def test_success_without_structured_rows_explains_table_state(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(self.base_url)

            await page.locator('select[name="subcommand"]').select_option("sweep")
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
