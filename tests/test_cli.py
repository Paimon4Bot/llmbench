from __future__ import annotations

import io
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llmbench.cli import main
from llmbench.vllm_runner import (
    CaptureConfig,
    _all_benchmark_requests_failed,
    build_vllm_command,
    load_result_records,
    run_benchmark_sync,
)


class CliModeTest(unittest.TestCase):
    def test_cli_supports_csv_export_file(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.csv"
            rc = main(
                [
                    "cli",
                    "serve",
                    "--app-vllm-binary",
                    fake_binary,
                    "--app-artifact-root",
                    tmp,
                    "--app-output",
                    "csv",
                    "--app-output-path",
                    str(output_path),
                    "--model",
                    "demo-model",
                    "--backend",
                    "openai",
                    "--request-rate",
                    "4",
                ]
            )
            self.assertEqual(0, rc)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("metrics.ttft_ms", text)
            self.assertIn("demo-model", text)

    def test_cli_supports_stdout_mode(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "latency",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "stdout",
                        "--model",
                        "latency-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn("status   : completed", output)
            self.assertIn("metrics.ttft_ms", output)
            self.assertEqual(1, output.count("fake vllm bench latency completed"))
            self.assertNotIn("fake vllm bench latency completed", stderr.getvalue())

    def test_cli_serve_treats_all_failed_requests_as_wrapper_failure(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "stdout",
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                        "--tokenizer",
                        "demo-model",
                        "--fake-all-requests-fail",
                    ]
                )
            self.assertEqual(1, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("code     : 1", stdout.getvalue())
            self.assertIn("Successful requests", stdout.getvalue())
            self.assertRegex(stderr.getvalue().lower(), r"all (benchmark )?requests failed")

    def test_cli_serve_treats_stdout_only_failed_requests_as_wrapper_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "stdout_only_failed_requests.py"
            script.write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import json",
                        "import sys",
                        "",
                        "args = sys.argv[1:]",
                        "result_dir = Path(args[args.index('--result-dir') + 1])",
                        "result_dir.mkdir(parents=True, exist_ok=True)",
                        "result_path = result_dir / args[args.index('--result-filename') + 1] if '--result-filename' in args else result_dir / 'result.json'",
                        "result_path.write_text(json.dumps({'subcommand': 'serve', 'config': {'model': 'demo-model'}}), encoding='utf-8')",
                        "print('Successful requests: 0')",
                        "print('Failed requests: 1')",
                        "print('All requests failed. This is likely due to a misconfiguration on the benchmark arguments.', file=sys.stderr)",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        f"{sys.executable} {script}",
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "stdout",
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                        "--tokenizer",
                        "demo-model",
                    ]
                )
            self.assertEqual(1, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("code     : 1", stdout.getvalue())
            self.assertIn("Successful requests: 0", stdout.getvalue())
            self.assertIn("Failed requests: 1", stdout.getvalue())
            self.assertRegex(stderr.getvalue().lower(), r"all (benchmark )?requests failed")

    def test_all_failed_request_detection_accepts_real_vllm_summary_keys(self) -> None:
        self.assertTrue(_all_benchmark_requests_failed([{"completed": 0, "failed": 1}]))
        self.assertFalse(_all_benchmark_requests_failed([{"completed": 1, "failed": 0}]))

    def test_cli_accepts_nested_bench_paths(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "stdout",
                        "--model",
                        "sweep-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertGreaterEqual(output.count("sweep serve"), 2)
            self.assertEqual(1, output.count("fake vllm bench sweep serve completed"))
            self.assertNotIn("fake vllm bench sweep serve completed", stderr.getvalue())

    def test_cli_forwards_help_to_vllm(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--help",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn("fake vllm help for sweep serve", stdout.getvalue())

    def test_cli_long_help_shows_wrapper_help_without_vllm(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(["cli", "--help"])
        self.assertEqual(0, rc)
        self.assertIn("Wrapper options:", stdout.getvalue())
        self.assertIn("Top-level vLLM bench commands:", stdout.getvalue())
        self.assertIn("llmbench cli sweep serve --help", stdout.getvalue())
        self.assertNotIn("serve_sla", stdout.getvalue())
        self.assertEqual("", stderr.getvalue())

    def test_cli_short_help_shows_wrapper_help_without_vllm(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(["cli", "-h"])
        self.assertEqual(0, rc)
        self.assertIn("Wrapper options:", stdout.getvalue())
        self.assertIn("llmbench cli serve --help", stdout.getvalue())
        self.assertIn("llmbench cli sweep serve --help", stdout.getvalue())
        self.assertNotIn("serve_sla", stdout.getvalue())
        self.assertEqual("", stderr.getvalue())

    def test_real_vllm_sweep_help_matches_latest_surface_when_available(self) -> None:
        real_vllm = Path(__file__).resolve().parents[1] / ".venv-vllm" / "bin" / "vllm"
        if not real_vllm.exists():
            self.skipTest("real vLLM bench binary is not available")
        completed = subprocess.run(
            [str(real_vllm), "bench", "sweep", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(0, completed.returncode, msg=completed.stderr)
        self.assertIn("{serve,serve_workload,startup,plot,plot_pareto}", completed.stdout)
        self.assertNotIn("serve_sla", completed.stdout)

    def test_cli_accepts_sweep_plot_pareto_help_path(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "plot_pareto",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--help",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn("fake vllm help for sweep plot_pareto", stdout.getvalue())

    def test_cli_handles_missing_binary_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        "definitely-not-a-command",
                        "--app-artifact-root",
                        tmp,
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertNotEqual(0, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("Failed to launch vLLM binary", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_handles_empty_binary_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        "",
                        "--app-artifact-root",
                        tmp,
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertNotEqual(0, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("Failed to prepare vLLM benchmark invocation", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_handles_invalid_app_raw_args_gracefully(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(
                [
                    "cli",
                    "serve",
                    "--app-vllm-binary",
                    fake_binary,
                    "--app-raw-args",
                    '--fake-sleep "2',
                ]
            )
        self.assertEqual(2, rc)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("invalid --app-raw-args", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_output_path_that_is_a_directory(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            output_dir = Path(tmp) / "result-dir"
            output_dir.mkdir()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_dir),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(1, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("failed to prepare output file", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_handles_invalid_result_dir_without_traceback(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "notadir"
            bad_path.write_text("x", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--result-dir",
                        str(bad_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertNotEqual(0, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("Failed to prepare vLLM benchmark invocation", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_handles_unreadable_output_json_target_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_vllm = Path(tmp) / "bad_vllm.py"
            bad_vllm.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "",
                        "argv = sys.argv",
                        "target = Path(argv[argv.index('--output-json') + 1])",
                        "target.parent.mkdir(parents=True, exist_ok=True)",
                        "target.write_text('not json', encoding='utf-8')",
                        "print('bad benchmark output written')",
                    ]
                ),
                encoding="utf-8",
            )
            fake_binary = f"{sys.executable} {bad_vllm}"
            bad_output_path = Path(tmp) / "out.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--output-json",
                        str(bad_output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(1, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("llmbench failed to load benchmark results", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_fails_cleanly_when_output_json_target_is_directory(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_target = Path(tmp) / "result.json"
            output_target.mkdir(parents=True, exist_ok=True)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--output-json",
                        str(output_target),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("Failed to prepare vLLM benchmark invocation", stderr.getvalue())
            self.assertIn("is a directory", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertNotIn("fake vllm bench throughput completed", stderr.getvalue())

    def test_cli_throughput_random_rewrites_length_flags_for_current_vllm_surface(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "stdout",
                        "--model",
                        "demo-model",
                        "--dataset-name",
                        "random",
                        "--input-len",
                        "8",
                        "--output-len",
                        "9",
                        "--num-prompts",
                        "1",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn("--random-input-len 8", output)
            self.assertIn("--random-output-len 9", output)
            self.assertNotIn("--input-len 8", output)
            self.assertNotIn("--output-len 9", output)

    def test_cli_forwards_equals_style_help_to_vllm(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--help=listgroup",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn("fake vllm help for serve", stdout.getvalue())

    def test_load_result_records_does_not_fallback_to_stdout_when_capture_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = CaptureConfig(
                mode="output_json",
                target_path=Path(tmp) / "missing.json",
                command_args=[],
                baseline_signatures={},
            )
            records = load_result_records(capture, '{"subcommand":"serve","metrics":{"ttft_ms":1}}', allow_stdout_fallback=False)
            self.assertEqual([], records)

    def test_cli_writing_csv_file_keeps_stdout_machine_safe(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.csv"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("fake vllm bench serve completed", stderr.getvalue())
            written = output_path.read_text(encoding="utf-8")
            self.assertIn("metrics.ttft_ms", written)
            self.assertNotIn("status   :", written)

    def test_cli_writing_jsonl_file_keeps_stdout_machine_safe(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.jsonl"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "jsonl",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("fake vllm bench serve completed", stderr.getvalue())
            written = output_path.read_text(encoding="utf-8")
            self.assertIn('"subcommand": "serve"', written)
            self.assertNotIn("status   :", written)

    def test_cli_rejects_output_path_when_stdout_mode_selected(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.txt"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-output",
                        "stdout",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("--app-output-path requires --app-output csv or jsonl", stderr.getvalue())
            self.assertFalse(output_path.exists())

    def test_cli_streaming_does_not_duplicate_child_output_after_completion(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--output-json",
                        "-",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            output_stream = stdout.getvalue()
            error_stream = stderr.getvalue()
            self.assertIn("status   : completed", output_stream)
            self.assertEqual(1, output_stream.count("fake vllm bench throughput completed"))
            self.assertNotIn("status   : completed", error_stream)
            self.assertNotIn('"subcommand": "throughput"', output_stream)
            self.assertNotIn('"subcommand": "throughput"', error_stream)

    def test_cli_preserves_existing_output_file_when_result_parsing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_vllm = Path(tmp) / "bad_vllm.py"
            bad_vllm.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "target = Path(sys.argv[sys.argv.index('--output-json') + 1])",
                        "target.parent.mkdir(parents=True, exist_ok=True)",
                        "target.write_text('not json', encoding='utf-8')",
                        "print('done')",
                    ]
                ),
                encoding="utf-8",
            )
            fake_binary = f"{sys.executable} {bad_vllm}"
            output_path = Path(tmp) / "result.csv"
            output_path.write_text("KEEP ME", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--output-json",
                        str(Path(tmp) / "broken.json"),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(1, rc)
            self.assertEqual("KEEP ME", output_path.read_text(encoding="utf-8"))
            self.assertIn("preserving existing output file", stderr.getvalue())

    def test_cli_does_not_overwrite_existing_output_file_on_failed_run_without_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            failing_binary = (
                f"{sys.executable} -c \"import sys; print('simulated failure', file=sys.stderr); raise SystemExit(2)\""
            )
            output_path = Path(tmp) / "result.csv"
            output_path.write_text("KEEP THIS", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        failing_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            self.assertEqual("KEEP THIS", output_path.read_text(encoding="utf-8"))
            self.assertIn("preserving existing output file", stderr.getvalue())

    def test_cli_does_not_create_new_output_file_on_failed_run_without_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            failing_binary = (
                f"{sys.executable} -c \"import sys; print('simulated failure', file=sys.stderr); raise SystemExit(2)\""
            )
            output_path = Path(tmp) / "new-result.csv"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        failing_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            self.assertFalse(output_path.exists())
            self.assertIn("skipping output file", stderr.getvalue())

    def test_cli_refreshes_existing_output_file_on_success_without_records(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.csv"
            output_path.write_text("KEEP THIS", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "plot",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--dry-run",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual("", output_path.read_text(encoding="utf-8"))
            self.assertIn("wrote empty output file", stderr.getvalue())

    def test_cli_creates_empty_output_file_on_success_without_records(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "new-result.csv"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "plot",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--dry-run",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertTrue(output_path.exists())
            self.assertEqual("", output_path.read_text(encoding="utf-8"))
            self.assertIn("wrote empty output file", stderr.getvalue())

    def test_cli_rejects_malformed_structured_output_and_preserves_existing_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            malformed = Path(tmp) / "bad_shape_vllm.py"
            malformed.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "target = Path(sys.argv[sys.argv.index('--output-json') + 1])",
                        "target.parent.mkdir(parents=True, exist_ok=True)",
                        "target.write_text('[\"not-a-record\"]', encoding='utf-8')",
                        "print('wrote malformed output')",
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp) / "result.csv"
            output_path.write_text("KEEP THIS\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        f"{sys.executable} {malformed}",
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertNotEqual(0, rc)
            self.assertEqual("KEEP THIS\n", output_path.read_text(encoding="utf-8"))
            self.assertIn("preserving existing output file", stderr.getvalue())
            self.assertIn("llmbench failed to load benchmark results", stderr.getvalue())
            self.assertNotIn("wrote empty output file", stderr.getvalue())

    def test_cli_rejects_partially_malformed_jsonl_and_preserves_existing_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            malformed = Path(tmp) / "partial_malformed_vllm.py"
            malformed.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "target = Path(sys.argv[sys.argv.index('--output-json') + 1])",
                        "target.parent.mkdir(parents=True, exist_ok=True)",
                        "target.write_text('{\"metrics\": {\"ttft_ms\": 1}, \"subcommand\": \"throughput\"}\\nnot-json\\n', encoding='utf-8')",
                        "print('wrote partial output')",
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp) / "result.csv"
            output_path.write_text("KEEP THIS\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        f"{sys.executable} {malformed}",
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertNotEqual(0, rc)
            self.assertEqual("KEEP THIS\n", output_path.read_text(encoding="utf-8"))
            self.assertIn("preserving existing output file", stderr.getvalue())
            self.assertIn("llmbench failed to load benchmark results", stderr.getvalue())

    def test_cli_output_json_dash_ignores_stdout_when_wrapper_manages_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            malformed = Path(tmp) / "partial_stdout_vllm.py"
            malformed.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "print(json.dumps({'subcommand': 'throughput', 'metrics': {'ttft_ms': 1}}, ensure_ascii=True), flush=True)",
                        "print('{\"broken\":', flush=True)",
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp) / "result.csv"
            output_path.write_text("KEEP THIS\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        f"{sys.executable} {malformed}",
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--app-output",
                        "csv",
                        "--app-output-path",
                        str(output_path),
                        "--output-json",
                        "-",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn('{"subcommand": "throughput"', stderr.getvalue())
            self.assertEqual("", output_path.read_text(encoding="utf-8"))
            self.assertIn("wrote empty output file", stderr.getvalue())
            self.assertNotIn("llmbench failed to load benchmark results", stderr.getvalue())

    def test_cli_ctrl_c_returns_130_without_traceback(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llmbench.cli",
                    "cli",
                    "serve",
                    "--app-vllm-binary",
                    fake_binary,
                    "--app-artifact-root",
                    tmp,
                    "--model",
                    "demo-model",
                    "--fake-sleep",
                    "10",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                time.sleep(1.0)
                process.send_signal(signal.SIGINT)
                stdout_text, stderr_text = process.communicate(timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            self.assertEqual(130, process.returncode)
            self.assertNotIn("Traceback", stdout_text)
            self.assertNotIn("Traceback", stderr_text)

    def test_cli_ctrl_c_terminates_child_process_group_and_returns(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            runner = Path(tmp) / "spawn_vllm.py"
            runner.write_text(
                "\n".join(
                    [
                        "import signal",
                        "import subprocess",
                        "import sys",
                        "import time",
                        "from pathlib import Path",
                        f"pid_file = Path({str(pid_file)!r})",
                        "def on_term(signum, frame):",
                        "    while True:",
                        "        time.sleep(1)",
                        "signal.signal(signal.SIGTERM, on_term)",
                        "if len(sys.argv) >= 3 and sys.argv[1] == 'bench' and sys.argv[2] == 'serve':",
                        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                        "    pid_file.write_text(str(child.pid), encoding='utf-8')",
                        "    while True:",
                        "        time.sleep(1)",
                        "raise SystemExit(2)",
                    ]
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llmbench.cli",
                    "cli",
                    "serve",
                    "--app-vllm-binary",
                    f"{sys.executable} {runner}",
                    "--app-artifact-root",
                    tmp,
                    "--model",
                    "demo-model",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            child_pid = None
            try:
                deadline = time.time() + 10
                while time.time() < deadline and not pid_file.exists():
                    time.sleep(0.1)
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text(encoding="utf-8"))
                os.killpg(process.pid, signal.SIGINT)
                stdout_text, stderr_text = process.communicate(timeout=20)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            self.assertEqual(130, process.returncode)
            self.assertNotIn("Traceback", stdout_text)
            self.assertNotIn("Traceback", stderr_text)
            if child_pid is not None:
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)

    def test_cli_ctrl_c_terminates_fd_holding_descendants_and_returns(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            child_pid_file = Path(tmp) / "descendant.pid"
            child = Path(tmp) / "fd_holder_child.py"
            child.write_text(
                "\n".join(
                    [
                        "import signal",
                        "import time",
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                        "signal.signal(signal.SIGINT, signal.SIG_IGN)",
                        "while True:",
                        "    time.sleep(1)",
                    ]
                ),
                encoding="utf-8",
            )
            runner = Path(tmp) / "fd_holder_vllm.py"
            runner.write_text(
                "\n".join(
                    [
                        "import signal",
                        "import subprocess",
                        "import sys",
                        "import time",
                        "from pathlib import Path",
                        f"pid_file = Path({str(child_pid_file)!r})",
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                        "signal.signal(signal.SIGINT, signal.SIG_IGN)",
                        "if len(sys.argv) >= 3 and sys.argv[1] == 'bench' and sys.argv[2] == 'serve':",
                        f"    child = subprocess.Popen([sys.executable, {str(child)!r}])",
                        "    pid_file.write_text(str(child.pid), encoding='utf-8')",
                        "    print('runner ready', flush=True)",
                        "    while True:",
                        "        time.sleep(1)",
                        "raise SystemExit(2)",
                    ]
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llmbench.cli",
                    "cli",
                    "serve",
                    "--app-vllm-binary",
                    f"{sys.executable} {runner}",
                    "--app-artifact-root",
                    tmp,
                    "--model",
                    "demo-model",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            child_pid = None
            try:
                deadline = time.time() + 10
                while time.time() < deadline and not child_pid_file.exists():
                    time.sleep(0.1)
                self.assertTrue(child_pid_file.exists())
                child_pid = int(child_pid_file.read_text(encoding="utf-8"))
                os.killpg(process.pid, signal.SIGINT)
                stdout_text, stderr_text = process.communicate(timeout=20)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            self.assertEqual(130, process.returncode)
            self.assertNotIn("Traceback", stdout_text)
            self.assertNotIn("Traceback", stderr_text)
            if child_pid is not None:
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)

    def test_cli_csv_stdout_export_without_records_writes_zero_bytes(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "plot",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--dry-run",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual("", stdout.getvalue())

    def test_cli_jsonl_stdout_export_without_records_writes_zero_bytes(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "plot",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "jsonl",
                        "--dry-run",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual("", stdout.getvalue())

    def test_cli_does_not_load_stale_explicit_output_json_from_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale_output = Path(tmp) / "stale.json"
            stale_output.write_text('{"subcommand":"stale-run","metrics":{"ttft_ms":999}}', encoding="utf-8")
            failing_binary = (
                f"{sys.executable} -c \"import sys; print('simulated failure', file=sys.stderr); raise SystemExit(2)\""
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        failing_binary,
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--output-json",
                        str(stale_output),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            report = stdout.getvalue()
            self.assertIn("No benchmark records were produced.", report)
            self.assertNotIn("stale-run", report)
            self.assertIn("results  : -", report)

    def test_cli_does_not_load_stale_output_dir_json_from_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale_dir = Path(tmp) / "stale-dir" / "serve"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "stale.json").write_text('{"subcommand":"stale-run"}', encoding="utf-8")
            failing_binary = (
                f"{sys.executable} -c \"import sys; print('simulated failure', file=sys.stderr); raise SystemExit(2)\""
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "serve",
                        "--app-vllm-binary",
                        failing_binary,
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--output-dir",
                        str(Path(tmp) / "stale-dir"),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            report = stdout.getvalue()
            self.assertIn("No benchmark records were produced.", report)
            self.assertNotIn("stale-run", report)

    def test_cli_output_json_dash_with_csv_keeps_stdout_csv_only(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--output-json",
                        "-",
                        "--fake-live-output",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            out_text = stdout.getvalue()
            err_text = stderr.getvalue()
            self.assertIn("metrics.ttft_ms", out_text)
            self.assertNotIn("fake live stdout chunk 1", out_text)
            self.assertIn("fake live stdout chunk 1", err_text)
            self.assertIn("fake vllm bench throughput completed", err_text)
            self.assertNotIn("metrics.ttft_ms", err_text)

    def test_cli_output_json_dash_with_jsonl_keeps_stdout_jsonl_only(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "jsonl",
                        "--output-json",
                        "-",
                        "--fake-live-output",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            out_text = stdout.getvalue()
            err_text = stderr.getvalue()
            self.assertEqual(1, out_text.count('"subcommand": "throughput"'))
            self.assertNotIn("fake live stdout chunk 1", out_text)
            self.assertIn("fake live stdout chunk 1", err_text)
            self.assertIn("fake vllm bench throughput completed", err_text)

    def test_cli_csv_stdout_export_keeps_stdout_machine_safe(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "csv",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            out_text = stdout.getvalue()
            err_text = stderr.getvalue()
            self.assertIn("metrics.ttft_ms", out_text)
            self.assertNotIn("fake vllm bench serve completed", out_text)
            self.assertIn("fake vllm bench serve completed", err_text)

    def test_cli_jsonl_stdout_export_keeps_stdout_machine_safe(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "jsonl",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            out_text = stdout.getvalue()
            err_text = stderr.getvalue()
            self.assertIn('"subcommand": "serve"', out_text)
            self.assertNotIn("fake vllm bench serve completed", out_text)
            self.assertIn("fake vllm bench serve completed", err_text)

    def test_cli_accepts_tilde_in_composite_vllm_binary(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as home_tmp:
            home_tmp_path = Path(home_tmp)
            binary_path = home_tmp_path / "python-review"
            binary_path.symlink_to(Path(sys.executable))
            tilde_binary = f"~/{home_tmp_path.name}/python-review {Path(__file__).with_name('fake_vllm.py')}"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        tilde_binary,
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn("subcommand                      : serve", stdout.getvalue())

    def test_cli_structured_mode_redacts_bearer_header_in_mirrored_stderr(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        token = "topsecret-token"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-output",
                        "jsonl",
                        "--header",
                        f"Authorization=Bearer {token}",
                        "--fake-header-echo",
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn('"subcommand": "serve"', stdout.getvalue())
            self.assertNotIn(token, stderr.getvalue())
            self.assertIn("header=['Authorization=<redacted-auth-value>']", stderr.getvalue())

    def test_cli_stdout_report_redacts_bearer_header_in_command_summary(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        token = "topsecret-token"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--header",
                        f"Authorization: Bearer {token}",
                        "--fake-header-echo",
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertNotIn(token, stdout.getvalue())
            self.assertNotIn(token, stderr.getvalue())
            self.assertIn("Authorization: <redacted-auth-value>", stdout.getvalue())
            self.assertIn("header=['Authorization: <redacted-auth-value>']", stdout.getvalue())

    def test_cli_structured_mode_uses_cache_artifacts_outside_cwd_by_default(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cache_root = Path(tmp) / "cache-home"
            cwd.mkdir()
            stdout = io.StringIO()
            previous_cwd = Path.cwd()
            try:
                os.chdir(cwd)
                with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(cache_root)}, clear=False), redirect_stdout(stdout):
                    rc = main(
                        [
                            "cli",
                            "serve",
                            "--app-vllm-binary",
                            fake_binary,
                            "--app-output",
                            "jsonl",
                            "--model",
                            "demo-model",
                            "--backend",
                            "openai",
                        ]
                    )
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(0, rc)
            self.assertIn('"subcommand": "serve"', stdout.getvalue())
            self.assertFalse((cwd / ".llmbench_runs").exists())
            self.assertTrue((cache_root / "llmbench" / "runs").exists())

    def test_cli_output_file_mode_uses_cache_artifacts_outside_cwd_by_default(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cache_root = Path(tmp) / "cache-home"
            output_path = cwd / "result.csv"
            cwd.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(cwd)
                with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(cache_root)}, clear=False):
                    rc = main(
                        [
                            "cli",
                            "serve",
                            "--app-vllm-binary",
                            fake_binary,
                            "--app-output",
                            "csv",
                            "--app-output-path",
                            str(output_path),
                            "--model",
                            "demo-model",
                            "--backend",
                            "openai",
                        ]
                    )
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(0, rc)
            self.assertTrue(output_path.exists())
            self.assertFalse((cwd / ".llmbench_runs").exists())
            self.assertTrue((cache_root / "llmbench" / "runs").exists())

    def test_cli_expands_tilde_for_app_artifact_root(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory(dir=Path.home()) as home_tmp:
            home_tmp_path = Path(home_tmp)
            artifact_root = Path("~") / home_tmp_path.name / "llmbench-artifacts"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        str(artifact_root),
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                    ]
                )
            self.assertEqual(0, rc)
            expanded_root = artifact_root.expanduser().resolve()
            self.assertIn(f"run_dir  : {expanded_root}", stdout.getvalue())
            self.assertTrue(expanded_root.exists())
            self.assertFalse((Path.cwd() / "~" / home_tmp_path.name / "llmbench-artifacts").exists())

    def test_web_mode_expands_tilde_for_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as home_tmp:
            home_tmp_path = Path(home_tmp)
            artifact_root = Path("~") / home_tmp_path.name / "web-artifacts"
            fake_app = object()
            with mock.patch("llmbench.webapp.create_app", return_value=fake_app) as create_app, mock.patch(
                "aiohttp.web.run_app"
            ) as run_app:
                rc = main(
                    [
                        "web",
                        "--artifact-root",
                        str(artifact_root),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "18099",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual(artifact_root.expanduser().resolve(), create_app.call_args.kwargs["artifact_root"])
            self.assertFalse((Path.cwd() / "~" / home_tmp_path.name / "web-artifacts").exists())
            run_app.assert_called_once_with(fake_app, host="127.0.0.1", port=18099)

    def test_build_vllm_command_resolves_default_binary_next_to_active_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_bin = Path(tmp) / "bin"
            env_bin.mkdir(parents=True, exist_ok=True)
            fake_python = env_bin / "python"
            fake_python.symlink_to(Path(sys.executable))
            fake_vllm = env_bin / "vllm"
            fake_vllm.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        f'exec "{sys.executable}" "{Path(__file__).with_name("fake_vllm.py")}" "$@"',
                    ]
                ),
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)

            with mock.patch.object(sys, "executable", str(fake_python)), mock.patch.object(
                sys,
                "argv",
                [str(env_bin / "llmbench")],
            ), mock.patch("llmbench.vllm_runner.shutil.which", return_value=None):
                command, _ = build_vllm_command(
                    vllm_binary="vllm",
                    bench_path=["serve"],
                    raw_args=["--model", "demo-model"],
                    artifact_dir=Path(tmp) / "artifacts",
                )
            self.assertEqual(str(fake_vllm), command[0])

    def test_build_vllm_command_resolves_default_binary_when_active_python_is_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_bin = Path(tmp) / "venv" / "bin"
            env_bin.mkdir(parents=True, exist_ok=True)
            uv_cache_bin = Path(tmp) / "uv-cache" / "cpython" / "bin"
            uv_cache_bin.mkdir(parents=True, exist_ok=True)
            cache_python = uv_cache_bin / "python"
            cache_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            cache_python.chmod(0o755)
            active_python = env_bin / "python"
            active_python.symlink_to(cache_python)
            fake_vllm = env_bin / "vllm"
            fake_vllm.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        f'exec "{sys.executable}" "{Path(__file__).with_name("fake_vllm.py")}" "$@"',
                    ]
                ),
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)

            with mock.patch.object(sys, "executable", str(cache_python)), mock.patch.object(
                sys,
                "argv",
                [str(active_python)],
            ), mock.patch("llmbench.vllm_runner.shutil.which", return_value=None):
                command, _ = build_vllm_command(
                    vllm_binary="vllm",
                    bench_path=["serve"],
                    raw_args=["--model", "demo-model"],
                    artifact_dir=Path(tmp) / "artifacts",
                )
            self.assertEqual(str(fake_vllm), command[0])

    def test_cli_expands_tilde_for_output_json_destination(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory(dir=Path.home()) as home_tmp:
            home_tmp_path = Path(home_tmp)
            output_path = Path("~") / home_tmp_path.name / "capture.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--output-json",
                        str(output_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            expanded_output = output_path.expanduser()
            self.assertTrue(expanded_output.exists())
            self.assertIn(f"--output-json {expanded_output}", stdout.getvalue())

    def test_cli_stdout_report_shows_effective_result_location(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            external_result = Path(tmp) / "external" / "result.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--output-json",
                        str(external_result),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn(f"results  : {external_result}", output)
            self.assertIn(f"run_dir  : {Path(tmp) / 'artifacts'}", output)

    def test_cli_rejects_missing_bench_path_without_swallowing_vllm_option_value(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(2, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("missing vLLM bench command path", stderr.getvalue())
            self.assertNotIn("bench demo-model --model", stderr.getvalue())

    def test_cli_requires_explicit_bench_path_when_forwarded_option_precedes_it(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "cli",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--model",
                        "demo-model",
                        "serve",
                    ]
                )
            self.assertEqual(2, rc)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("ambiguous vLLM bench invocation", stderr.getvalue())
            self.assertIn("--app-bench-path", stderr.getvalue())

    def test_cli_supports_explicit_bench_path_with_forwarded_options_before_it(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-bench-path",
                        "serve",
                        "--model",
                        "demo-model",
                        "--backend",
                        "openai",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn("subcommand                      : serve", output)
            self.assertIn("config.model                    : demo-model", output)

    def test_cli_consumes_wrapper_separator_before_forwarding_args(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--",
                        "serve",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn("subcommand                      : serve", output)
            self.assertIn("command  : ", output)
            self.assertNotIn(" bench -- serve", output)
            self.assertIn(" bench serve --model demo-model", output)

    def test_cli_strips_wrapper_separator_with_explicit_bench_path(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--app-bench-path",
                        "serve",
                        "--",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn("subcommand                      : serve", output)
            self.assertNotIn(" bench serve -- --model demo-model", output)
            self.assertIn(" bench serve --model demo-model", output)

    def test_throughput_uses_output_json_capture(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--model",
                        "throughput-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn("--output-json", output)
            self.assertNotIn("--save-result", output)
            self.assertIn("subcommand                      : throughput", output)

    def test_cli_respects_user_supplied_result_dir(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            user_result_dir = Path(tmp) / "user-dir"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--result-dir",
                        str(user_result_dir),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertIn(f"--result-dir {user_result_dir}", output)
            self.assertNotIn("--result-filename", output)
            self.assertTrue((user_result_dir / "result.json").exists())

    def test_cli_respects_equals_style_result_destinations(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            user_result_dir = Path(tmp) / "user-dir"
            user_output_json = Path(tmp) / "user.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        f"--result-dir={user_result_dir}",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn(f"--result-dir={user_result_dir}", stdout.getvalue())
            self.assertNotIn("--result-filename", stdout.getvalue())
            self.assertTrue((user_result_dir / "result.json").exists())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "throughput",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts-2"),
                        f"--output-json={user_output_json}",
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn(f"--output-json={user_output_json}", stdout.getvalue())
            self.assertTrue(user_output_json.exists())

    def test_cli_rejects_empty_result_dir_equals_value(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cwd"
            run_dir.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()
            previous_cwd = Path.cwd()
            try:
                os.chdir(run_dir)
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main(
                        [
                            "cli",
                            "serve",
                            "--app-vllm-binary",
                            fake_binary,
                            "--app-artifact-root",
                            str(Path(tmp) / "artifacts"),
                            "--result-dir=",
                            "--model",
                            "demo-model",
                            "--backend",
                            "openai",
                        ]
                    )
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(2, rc)
            self.assertIn("status   : failed", stdout.getvalue())
            self.assertIn("--result-dir requires a non-empty value", stderr.getvalue())
            self.assertFalse((run_dir / "result.json").exists())

    def test_build_vllm_command_rejects_empty_equals_style_capture_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            with self.assertRaisesRegex(ValueError, "--output-json requires a non-empty value"):
                build_vllm_command(
                    vllm_binary="vllm",
                    bench_path=["throughput"],
                    raw_args=["--output-json=", "--model", "demo-model"],
                    artifact_dir=artifact_dir,
                )
            with self.assertRaisesRegex(ValueError, "--output-dir requires a non-empty value"):
                build_vllm_command(
                    vllm_binary="vllm",
                    bench_path=["sweep", "serve"],
                    raw_args=["--output-dir=", "--model", "demo-model"],
                    artifact_dir=artifact_dir,
                )

    def test_cli_respects_short_output_dir_alias_for_sweep(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            user_output_dir = Path(tmp) / "sweep-user-output"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "-o",
                        str(user_output_dir),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn(f"-o {user_output_dir}", stdout.getvalue())
            self.assertTrue((user_output_dir / "serve" / "summary.json").exists())

    def test_cli_accepts_vllm_binary_path_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper_dir = Path(tmp) / "dir with spaces"
            wrapper_dir.mkdir(parents=True, exist_ok=True)
            wrapper_path = wrapper_dir / "fake-vllm"
            wrapper_path.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        f'exec "{sys.executable}" "{Path(__file__).with_name("fake_vllm.py")}" "$@"',
                    ]
                ),
                encoding="utf-8",
            )
            wrapper_path.chmod(0o755)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        str(wrapper_path),
                        "--app-artifact-root",
                        str(Path(tmp) / "artifacts"),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn(shlex.quote(str(wrapper_path)), stdout.getvalue())

    def test_cli_stdout_report_uses_shell_safe_command_rendering(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dir with spaces" / "data.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("{}", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--dataset-path",
                        str(dataset_path),
                        "--model",
                        "demo-model",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn(shlex.quote(str(dataset_path)), stdout.getvalue())

    def test_sweep_plot_does_not_inject_structured_result_capture_flags(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "plot",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--dry-run",
                    ]
                )
            self.assertEqual(0, rc)
            output = stdout.getvalue()
            self.assertNotIn("--save-result", output)
            self.assertNotIn("--output-json", output)
            self.assertNotIn("--output-dir", output)
            self.assertIn("No benchmark records were produced.", output)

    def test_run_benchmark_sync_does_not_parse_stdout_json_for_capture_mode_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_stdout_json = Path(tmp) / "fake_stdout_json.py"
            fake_stdout_json.write_text(
                "\n".join(
                    [
                        "import json",
                        "print(json.dumps({'subcommand': 'fabricated-from-stdout', 'metrics': {'ttft_ms': 1}}))",
                    ]
                ),
                encoding="utf-8",
            )
            execution = run_benchmark_sync(
                vllm_binary=f"{sys.executable} {fake_stdout_json}",
                bench_path=["sweep", "plot"],
                raw_args=["--dry-run"],
                artifact_root=Path(tmp),
            )
        self.assertEqual("completed", execution.status)
        self.assertIn("fabricated-from-stdout", execution.stdout)
        self.assertEqual([], execution.records)
        self.assertIsNone(execution.result_path)

    def test_load_result_records_output_dir_ignores_unrelated_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_dir = output_dir / "serve"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run-0.json").write_text('{"subcommand":"run-0"}', encoding="utf-8")
            (run_dir / "summary.json").write_text('{"subcommand":"summary"}', encoding="utf-8")
            (run_dir / "random.json").write_text('{"subcommand":"unrelated-direct"}', encoding="utf-8")
            (run_dir / "nested").mkdir(parents=True, exist_ok=True)
            (run_dir / "nested" / "run-999.json").write_text('{"subcommand":"unrelated-nested"}', encoding="utf-8")
            (output_dir / "noise.json").write_text('{"subcommand":"unrelated-root"}', encoding="utf-8")
            (output_dir / "custom" / "run-1.json").parent.mkdir(parents=True, exist_ok=True)
            (output_dir / "custom" / "run-1.json").write_text('{"subcommand":"unrelated-custom"}', encoding="utf-8")
            records = load_result_records(
                CaptureConfig(mode="output_dir", target_path=output_dir, command_args=[], baseline_signatures={})
            )
        self.assertEqual(
            [{"subcommand": "run-0"}, {"subcommand": "summary"}],
            records,
        )

    def test_build_vllm_command_uses_output_dir_for_sweep_serve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["sweep", "serve"],
                raw_args=["--model", "demo-model"],
                artifact_dir=Path(tmp),
            )
            self.assertIn("--output-dir", command)
            self.assertNotIn("--save-result", command)
            self.assertEqual("output_dir", capture.mode)

    def test_build_vllm_command_respects_user_supplied_serve_result_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp) / "user-output"
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["serve"],
                raw_args=["--result-dir", str(user_dir), "--model", "demo-model"],
                artifact_dir=Path(tmp) / "artifacts",
            )
        self.assertEqual("serve_result_dir", capture.mode)
        self.assertEqual(user_dir, capture.target_path)
        self.assertEqual(1, command.count("--result-dir"))
        self.assertIn("--save-result", command)
        self.assertNotIn("--result-filename", command)
        self.assertNotIn(str(Path(tmp) / "artifacts"), command)

    def test_build_vllm_command_respects_equals_style_capture_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp) / "user-output"
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["serve"],
                raw_args=[f"--result-dir={user_dir}", "--model", "demo-model", "--result-filename=custom.json"],
                artifact_dir=Path(tmp) / "artifacts",
            )
            self.assertEqual(user_dir / "custom.json", capture.target_path)
            self.assertEqual(0, command.count("--result-dir"))
            self.assertEqual(0, command.count("--result-filename"))
            self.assertIn(f"--result-dir={user_dir}", command)
            self.assertIn("--result-filename=custom.json", command)

            output_json = Path(tmp) / "user.json"
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["throughput"],
                raw_args=[f"--output-json={output_json}", "--model", "demo-model"],
                artifact_dir=Path(tmp) / "artifacts-2",
            )
            self.assertEqual(output_json, capture.target_path)
            self.assertNotIn("--output-json", command)
            self.assertIn(f"--output-json={output_json}", command)

    def test_build_vllm_command_rewrites_output_json_dash_to_wrapper_capture_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["throughput"],
                raw_args=["--output-json", "-", "--model", "demo-model"],
                artifact_dir=Path(tmp) / "artifacts",
            )
            self.assertEqual(Path(tmp) / "artifacts" / "vllm-result.json", capture.target_path)
            self.assertEqual(1, command.count("--output-json"))
            self.assertIn(str(Path(tmp) / "artifacts" / "vllm-result.json"), command)
            self.assertNotIn("-", command)

    def test_run_benchmark_sync_rewrites_output_json_dash_to_wrapper_capture_file(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            execution = run_benchmark_sync(
                vllm_binary=fake_binary,
                bench_path=["throughput"],
                raw_args=["--output-json", "-", "--model", "demo-model"],
                artifact_root=Path(tmp),
            )
        self.assertEqual("completed", execution.status)
        self.assertEqual(str(Path(tmp) / execution.job_id / "vllm-result.json"), execution.result_path)
        self.assertEqual(1, len(execution.records))
        self.assertEqual("throughput", execution.records[0]["subcommand"])
        self.assertEqual("demo-model", execution.records[0]["config"]["model"])

    def test_build_vllm_command_respects_short_output_dir_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "user-output"
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["sweep", "serve"],
                raw_args=["-o", str(output_dir), "--model", "demo-model"],
                artifact_dir=Path(tmp) / "artifacts",
            )
        self.assertEqual(output_dir, capture.target_path)
        self.assertEqual(0, command.count("--output-dir"))
        self.assertIn("-o", command)

    def test_stdout_report_quotes_shell_arguments_with_spaces(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = str(Path(tmp) / "dir with spaces" / "data.json")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "serve",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--dataset-path",
                        dataset_path,
                        "--model",
                        "demo-model",
                    ]
                )
        self.assertEqual(0, rc)
        self.assertIn(f"--dataset-path '{dataset_path}'", stdout.getvalue())

    def test_load_result_records_includes_summary_and_run_json_for_sweep_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_dir = output_dir / "serve"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run-0.json").write_text('{"subcommand": "run-0"}', encoding="utf-8")
            (run_dir / "summary.json").write_text('{"subcommand": "summary"}', encoding="utf-8")
            records = load_result_records(
                CaptureConfig(mode="output_dir", target_path=output_dir, command_args=[], baseline_signatures={})
            )
        self.assertEqual(
            [{"subcommand": "run-0"}, {"subcommand": "summary"}],
            records,
        )

    def test_load_result_records_includes_nested_experiment_summary_and_run_json_for_sweep_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_dir = output_dir / "repro" / "STARTUP-demo"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run=0.json").write_text('{"subcommand": "nested-run"}', encoding="utf-8")
            (run_dir / "summary.json").write_text('{"subcommand": "nested-summary"}', encoding="utf-8")
            records = load_result_records(
                CaptureConfig(mode="output_dir", target_path=output_dir, command_args=[], baseline_signatures={})
            )
        self.assertEqual(
            [{"subcommand": "nested-run"}, {"subcommand": "nested-summary"}],
            records,
        )

    def test_cli_sweep_startup_exports_nested_experiment_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "sweep_nested.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "from pathlib import Path",
                        "",
                        "args = sys.argv[1:]",
                        'output_dir = Path(args[args.index("--output-dir") + 1])',
                        'experiment = args[args.index("-e") + 1] if "-e" in args else "default"',
                        'run_dir = output_dir / experiment / "STARTUP-demo"',
                        "run_dir.mkdir(parents=True, exist_ok=True)",
                        '(run_dir / "run=0.json").write_text(json.dumps({"subcommand": "nested-run"}), encoding="utf-8")',
                        '(run_dir / "summary.json").write_text(json.dumps({"subcommand": "nested-summary"}), encoding="utf-8")',
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp) / "out.csv"
            rc = main(
                [
                    "cli",
                    "sweep",
                    "startup",
                    "--app-vllm-binary",
                    f"{sys.executable} {script_path}",
                    "--app-artifact-root",
                    str(Path(tmp) / "artifacts"),
                    "--app-output",
                    "csv",
                    "--app-output-path",
                    str(output_path),
                    "-e",
                    "repro",
                ]
            )
            self.assertEqual(0, rc)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("nested-run", text)
            self.assertIn("nested-summary", text)

    def test_run_benchmark_sync_propagates_explicit_binary_dir_to_nested_sweep_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            script_path = bin_dir / "vllm"
            script_path.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json",
                        "import os",
                        "import subprocess",
                        "import sys",
                        "from pathlib import Path",
                        "",
                        "args = sys.argv[1:]",
                        'if args[:3] == ["bench", "sweep", "startup"]:',
                        '    output_dir = Path(args[args.index("--output-dir") + 1])',
                        '    experiment = args[args.index("-e") + 1] if "-e" in args else "default"',
                        '    os.environ["LLMBENCH_SWEEP_OUTPUT_DIR"] = str(output_dir)',
                        '    os.environ["LLMBENCH_SWEEP_EXPERIMENT"] = experiment',
                        '    raise SystemExit(subprocess.run(["vllm", "bench", "startup"], check=False).returncode)',
                        'if args[:2] == ["bench", "startup"]:',
                        '    output_dir = Path(os.environ["LLMBENCH_SWEEP_OUTPUT_DIR"])',
                        '    experiment = os.environ["LLMBENCH_SWEEP_EXPERIMENT"]',
                        '    run_dir = output_dir / experiment / "STARTUP-demo"',
                        "    run_dir.mkdir(parents=True, exist_ok=True)",
                        '    (run_dir / "run=0.json").write_text(json.dumps({"subcommand": "nested-startup"}), encoding="utf-8")',
                        '    (run_dir / "summary.json").write_text(json.dumps({"subcommand": "nested-summary"}), encoding="utf-8")',
                        "    raise SystemExit(0)",
                        "raise SystemExit(2)",
                    ]
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=False):
                execution = run_benchmark_sync(
                    vllm_binary=str(script_path),
                    bench_path=["sweep", "startup"],
                    raw_args=["-e", "repro"],
                    artifact_root=Path(tmp) / "artifacts",
                )
            self.assertEqual("completed", execution.status)
            self.assertEqual(2, len(execution.records))
            self.assertEqual({"nested-startup", "nested-summary"}, {record["subcommand"] for record in execution.records})

    def test_run_benchmark_sync_streams_carriage_return_progress_live(self) -> None:
        class _TimedSink:
            def __init__(self) -> None:
                self.events: list[tuple[float, str]] = []

            def write(self, text: str) -> int:
                self.events.append((time.monotonic(), text))
                return len(text)

            def flush(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "cr_progress.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import sys",
                        "import time",
                        "",
                        "args = sys.argv[1:]",
                        'if args[:2] != ["bench", "serve"]:',
                        "    raise SystemExit(2)",
                        "sys.stdout.write('progress stdout 1\\r')",
                        "sys.stdout.flush()",
                        "sys.stderr.write('progress stderr 1\\r')",
                        "sys.stderr.flush()",
                        "time.sleep(0.45)",
                        "sys.stdout.write('progress stdout 2\\r')",
                        "sys.stdout.flush()",
                        "sys.stderr.write('progress stderr 2\\r')",
                        "sys.stderr.flush()",
                        "time.sleep(0.45)",
                        "print('stream done', file=sys.stderr, flush=True)",
                    ]
                ),
                encoding="utf-8",
            )
            sink = _TimedSink()
            started = time.monotonic()
            execution = run_benchmark_sync(
                vllm_binary=f"{sys.executable} {script_path}",
                bench_path=["serve"],
                raw_args=[],
                artifact_root=Path(tmp) / "artifacts",
                capture_results=False,
                stream_output=True,
                live_stdout=sink,
                live_stderr=sink,
            )
            finished = time.monotonic()

        self.assertEqual("completed", execution.status)
        self.assertEqual(0, execution.returncode)
        self.assertGreater(finished - started, 0.75)
        self.assertTrue(sink.events)
        first_progress_event = min(
            timestamp
            for timestamp, text in sink.events
            if "progress stdout 1\r" in text or "progress stderr 1\r" in text
        )
        self.assertLess(first_progress_event - started, 0.7)
        self.assertIn("progress stdout 1\r", execution.stdout)
        self.assertIn("progress stderr 1\r", execution.stderr)
        self.assertIn("stream done", execution.stderr)

    def test_help_invocation_does_not_create_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
            rc = main(
                [
                    "cli",
                    "serve",
                    "--app-vllm-binary",
                    fake_binary,
                    "--app-artifact-root",
                    tmp,
                    "--help",
                ]
            )
            self.assertEqual(0, rc)
            self.assertEqual([], list(Path(tmp).iterdir()))
