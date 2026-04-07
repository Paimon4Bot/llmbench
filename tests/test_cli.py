from __future__ import annotations

import io
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llmbench.cli import main
from llmbench.vllm_runner import CaptureConfig, build_vllm_command, load_result_records, run_benchmark_sync


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
            self.assertNotIn("fake vllm bench latency completed", output)
            self.assertEqual(1, stderr.getvalue().count("fake vllm bench latency completed"))

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
                        "serve_sla",
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
            self.assertGreaterEqual(output.count("sweep serve_sla"), 2)
            self.assertNotIn("fake vllm bench sweep serve_sla completed", output)
            self.assertEqual(1, stderr.getvalue().count("fake vllm bench sweep serve_sla completed"))

    def test_cli_forwards_help_to_vllm(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "cli",
                        "sweep",
                        "serve_sla",
                        "--app-vllm-binary",
                        fake_binary,
                        "--app-artifact-root",
                        tmp,
                        "--help",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertIn("fake vllm help for sweep serve_sla", stdout.getvalue())

    def test_cli_long_help_shows_wrapper_help_without_vllm(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(["cli", "--help"])
        self.assertEqual(0, rc)
        self.assertIn("Wrapper options:", stdout.getvalue())
        self.assertIn("Top-level vLLM bench commands:", stdout.getvalue())
        self.assertEqual("", stderr.getvalue())

    def test_cli_short_help_shows_wrapper_help_without_vllm(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(["cli", "-h"])
        self.assertEqual(0, rc)
        self.assertIn("Wrapper options:", stdout.getvalue())
        self.assertIn("llmbench cli serve --help", stdout.getvalue())
        self.assertEqual("", stderr.getvalue())

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

    def test_cli_streams_child_output_to_stderr_while_writing_csv_file(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.csv"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
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
            self.assertIn("fake vllm bench serve completed", stderr.getvalue())

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
            self.assertEqual(1, output_stream.count('"subcommand": "throughput"'))
            self.assertNotIn("status   : completed", output_stream)
            self.assertIn("status   : completed", error_stream)
            self.assertEqual(1, error_stream.count("fake vllm bench throughput completed"))
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

    def test_cli_does_not_overwrite_existing_output_file_on_success_without_records(self) -> None:
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
            self.assertEqual("KEEP THIS", output_path.read_text(encoding="utf-8"))
            self.assertIn("preserving existing output file", stderr.getvalue())

    def test_cli_does_not_create_new_output_file_on_success_without_records(self) -> None:
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
            self.assertFalse(output_path.exists())
            self.assertIn("skipping output file", stderr.getvalue())

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
            stale_dir = Path(tmp) / "stale-dir" / "serve_sla"
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
                        "serve_sla",
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
            self.assertNotIn('"subcommand": "throughput"', out_text)
            self.assertNotIn("fake live stdout chunk 1", out_text)
            self.assertIn('"subcommand": "throughput"', err_text)
            self.assertIn("fake live stdout chunk 1", err_text)
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

    def test_cli_preserves_wrapper_separator_with_explicit_bench_path(self) -> None:
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
            self.assertIn(" bench serve -- --model demo-model", output)

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
            self.assertTrue((user_result_dir / "vllm-result.json").exists())

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
            self.assertTrue((user_result_dir / "vllm-result.json").exists())

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
                        "serve_sla",
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
            self.assertTrue((user_output_dir / "serve_sla" / "summary.json").exists())

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
            run_dir = output_dir / "serve_sla"
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

    def test_build_vllm_command_uses_output_dir_for_sweep_serve_sla(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["sweep", "serve_sla"],
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
        self.assertEqual(user_dir / "vllm-result.json", capture.target_path)
        self.assertEqual(1, command.count("--result-dir"))
        self.assertIn("--save-result", command)
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

    def test_build_vllm_command_preserves_output_json_dash_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["throughput"],
                raw_args=["--output-json", "-", "--model", "demo-model"],
                artifact_dir=Path(tmp) / "artifacts",
            )
            self.assertEqual(Path("-"), capture.target_path)
            self.assertEqual(1, command.count("--output-json"))
            self.assertIn("-", command)

    def test_run_benchmark_sync_reads_output_json_dash_from_stdout(self) -> None:
        fake_binary = f"{sys.executable} {Path(__file__).with_name('fake_vllm.py')}"
        with tempfile.TemporaryDirectory() as tmp:
            execution = run_benchmark_sync(
                vllm_binary=fake_binary,
                bench_path=["throughput"],
                raw_args=["--output-json", "-", "--model", "demo-model"],
                artifact_root=Path(tmp),
            )
        self.assertEqual("completed", execution.status)
        self.assertEqual("-", execution.result_path)
        self.assertEqual(1, len(execution.records))
        self.assertEqual("throughput", execution.records[0]["subcommand"])
        self.assertEqual("demo-model", execution.records[0]["config"]["model"])

    def test_build_vllm_command_respects_short_output_dir_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "user-output"
            command, capture = build_vllm_command(
                vllm_binary="vllm",
                bench_path=["sweep", "serve_sla"],
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
            run_dir = output_dir / "serve_sla"
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
