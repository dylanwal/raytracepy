#!/usr/bin/env python3
"""Tests for ``scripts/check_environment.py``.

Standard library only, matching the script's own constraint: the HarmonyOS PC
may have no third-party packages at all, so these tests must run there with a
bare interpreter and no pytest.

Run them with either::

    python scripts/tests/test_check_environment.py -v
    python -m unittest discover -s scripts/tests -t scripts/tests

They are deliberately not placed in ``tests/``: ``pyproject.toml`` points
``testpaths`` at ``tests/`` for the package's own suite (whose conftest imports
raytracepy, numpy and numba), while these tests describe a script that is
claimed to run on a bare interpreter with nothing installed.  Keeping them
apart means this file can be run on the HarmonyOS PC itself.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_environment as ce  # noqa: E402  (path set up above)

REFERENCE_JSON = (
    REPO_ROOT / "artifacts" / "reference" / "windows-python310" / "environment.json"
)


def completed(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a ``CompletedProcess`` stand-in."""

    return subprocess.CompletedProcess(
        args=["stub"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def present_probe(
    import_name: str,
    severity: str = ce.SEVERITY_REQUIRED,
    **extra: object,
) -> dict:
    """A probe result that reported a successful import."""

    payload = {"import_name": import_name, "status": "present", "severity": severity}
    payload.update(extra)
    return payload


def missing_probe(
    import_name: str,
    severity: str = ce.SEVERITY_REQUIRED,
    error_type: str = "ModuleNotFoundError",
) -> dict:
    """A probe result that reported a failure."""

    return {
        "import_name": import_name,
        "status": "error",
        "severity": severity,
        "error_type": error_type,
        "message": f"No module named {import_name!r}",
    }


def healthy_inputs() -> dict:
    """Argument dicts for ``summarize`` describing a machine that can host it."""

    system = {"machine": "AMD64", "system": "Windows"}
    python_info = {"python_version": "3.10.21", "satisfies_python_requires": True}
    toolchain = {
        "pip_available": True,
        "venv_importable": True,
        "setuptools_version": "69.0.0",
        "setuptools_meets_minimum": True,
        "wheel_version": "0.42.0",
        "compilers": {"gcc": "gcc :: 13.2.0"},
    }
    probes = [present_probe("numpy"), present_probe("numba")]
    return {
        "system": system,
        "python_info": python_info,
        "toolchain": toolchain,
        "probes": probes,
    }


# ---------------------------------------------------------------------------
# --label validation
# ---------------------------------------------------------------------------


class TestLabelValidation(unittest.TestCase):
    def test_accepts_ordinary_labels(self):
        for label in ("windows-python310", "harmonyos_aarch64", "v1.2", "x", "0"):
            with self.subTest(label=label):
                self.assertEqual(ce.valid_label(label), label)

    def test_rejects_path_traversal_and_separators(self):
        for label in (
            "../../etc",
            "..",
            ".",
            "/absolute",
            "a/b",
            "a\\b",
            "-leading-dash",
            "_leading_underscore",
            "",
        ):
            with self.subTest(label=label):
                with self.assertRaises(argparse.ArgumentTypeError):
                    ce.valid_label(label)

    def test_rejects_embedded_newline(self):
        # re.match with \Z must not be fooled by a trailing newline the way an
        # unanchored "$" would be.
        with self.assertRaises(argparse.ArgumentTypeError):
            ce.valid_label("windows\n")

    def test_derived_default_label_is_itself_path_safe(self):
        # The default bypasses the --label type function, so it has to satisfy
        # the same whitelist on its own.
        self.assertEqual(ce.valid_label(ce.default_label()), ce.default_label())

    def test_default_label_falls_back_when_platform_is_odd(self):
        with mock.patch.object(ce.platform, "system", return_value="Weird System/../x"):
            self.assertEqual(ce.default_label(), "unknown-python")


# ---------------------------------------------------------------------------
# --probe-timeout validation
# ---------------------------------------------------------------------------


class TestProbeTimeout(unittest.TestCase):
    def test_accepts_positive_finite_values(self):
        self.assertEqual(ce.positive_finite_seconds("180"), 180.0)
        self.assertEqual(ce.positive_finite_seconds("0.001"), 0.001)

    def test_rejects_nan_infinity_and_zero(self):
        for text in ("nan", "NaN", "inf", "-inf", "Infinity", "0", "-1", "-0.5", "abc", ""):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    ce.positive_finite_seconds(text)

    def test_argparse_reports_non_finite_timeout_as_a_usage_error(self):
        # Previously "nan" reached subprocess and raised ValueError from inside
        # the standard library, killing the run with a traceback and no report.
        for value in ("nan", "inf", "0"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as caught:
                    ce.parse_arguments(["--no-write", "--probe-timeout", value])
                self.assertEqual(caught.exception.code, 2)


# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------


class TestCapture(unittest.TestCase):
    def test_returns_first_line_of_stdout(self):
        self.assertEqual(ce.capture((sys.executable, "-c", "print('hello')")), "hello")

    def test_returns_none_when_the_command_fails(self):
        # The pip case: a failing command still writes a message to stderr, and
        # treating that text as a version reported pip as available.
        command = (
            sys.executable,
            "-c",
            "import sys; sys.stderr.write(\"No module named pip\"); sys.exit(1)",
        )
        self.assertIsNone(ce.capture(command))

    def test_check_false_returns_output_despite_failure(self):
        command = (
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('musl libc Version 1.2.5'); sys.exit(1)",
        )
        self.assertEqual(ce.capture(command, check=False), "musl libc Version 1.2.5")

    def test_falls_back_to_stderr_when_stdout_is_empty(self):
        command = (sys.executable, "-c", "import sys; sys.stderr.write('from stderr')")
        self.assertEqual(ce.capture(command), "from stderr")

    def test_survives_output_that_is_not_utf8(self):
        command = (
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'\\xff\\xfeok')",
        )
        self.assertIsInstance(ce.capture(command), str)

    def test_missing_executable_returns_none(self):
        self.assertIsNone(ce.capture(("definitely-not-a-real-program-xyz",)))

    def test_capture_from_path_returns_none_for_missing_program(self):
        self.assertIsNone(ce.capture_from_path("definitely-not-a-real-program-xyz"))


# ---------------------------------------------------------------------------
# run_probe()
# ---------------------------------------------------------------------------


class TestRunProbe(unittest.TestCase):
    def test_success_payload_is_parsed(self):
        outcome = ce.run_probe(
            "import json; print('__PROBE__' + json.dumps({'status': 'present'}))",
            timeout=30.0,
            import_name="demo",
        )
        self.assertEqual(outcome["status"], "present")
        self.assertEqual(outcome["import_name"], "demo")

    def test_payload_from_the_last_marker_wins(self):
        # An imported module may print to stdout; the probe marker must still be
        # found reliably.
        snippet = (
            "import json\n"
            "print('__PROBE__' + json.dumps({'status': 'error'}))\n"
            "print('__PROBE__' + json.dumps({'status': 'present'}))\n"
        )
        outcome = ce.run_probe(snippet, timeout=30.0, import_name="demo")
        self.assertEqual(outcome["status"], "present")

    def test_timeout_path_still_carries_import_name(self):
        # The original bug: summarize() read probe["import_name"], which the
        # timeout path did not set, so the whole report was lost.
        outcome = ce.run_probe(
            "import time; time.sleep(30)",
            timeout=0.001,
            import_name="slow",
        )
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_type"], "Timeout")
        self.assertEqual(outcome["import_name"], "slow")
        self.assertIn("0.001", outcome["message"])

    def test_silent_death_path_carries_import_name_and_returncode(self):
        outcome = ce.run_probe(
            "import os; os._exit(3)",
            timeout=30.0,
            import_name="dies",
        )
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_type"], "NoProbeOutput")
        self.assertEqual(outcome["import_name"], "dies")
        self.assertEqual(outcome["returncode"], 3)

    def test_negative_returncode_is_reported_as_a_crash_with_a_signal(self):
        # Not reproducible portably, so drive the branch directly.
        with mock.patch.object(ce.subprocess, "run", return_value=completed(-11)):
            outcome = ce.run_probe("pass", timeout=30.0, import_name="numba.jit")
        self.assertEqual(outcome["error_type"], "Crashed")
        self.assertEqual(outcome["import_name"], "numba.jit")
        self.assertTrue(outcome["signal_name"])
        self.assertIn("returncode -11", outcome["message"])

    def test_unusable_timeout_is_reported_instead_of_raised(self):
        # Defense in depth: argparse rejects NaN, but a caller of run_probe must
        # not be able to take the report down with it either.
        with mock.patch.object(
            ce.subprocess, "run", side_effect=ValueError("Invalid value NaN")
        ):
            outcome = ce.run_probe("pass", timeout=30.0, import_name="demo")
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_type"], "ValueError")
        self.assertEqual(outcome["import_name"], "demo")

    def test_every_error_path_sets_the_keys_the_summary_reads(self):
        snippets = (
            "import time; time.sleep(30)",
            "import os; os._exit(3)",
            "raise SystemExit(1)",
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                timeout = 0.001 if "sleep" in snippet else 30.0
                outcome = ce.run_probe(snippet, timeout=timeout, import_name="demo")
                # These are exactly the keys summarize() and render_report()
                # subscript without a default.
                self.assertIn("import_name", outcome)
                self.assertEqual(outcome["status"], "error")
                self.assertIsInstance(outcome["message"], str)

    def test_jit_snippet_verifies_the_result(self):
        # Compiling and calling is the point: an import-only probe cannot see
        # the HarmonyOS case where the bindings load and the call segfaults.
        self.assertIn("jit_result", ce.JIT_SNIPPET)
        self.assertIn("njit", ce.JIT_SNIPPET)

    def test_jit_probe_is_skipped_without_numba(self):
        # A missing numba is already a required-dependency blocker; probing the
        # JIT on top of it would only add a confusing second failure.
        self.assertIsNone(ce.collect_jit_probe(5.0, numba_present=False))

    def test_jit_probe_is_marked_required(self):
        with mock.patch.object(ce, "run_probe", return_value=present_probe("numba.jit")):
            outcome = ce.collect_jit_probe(5.0, numba_present=True)
        self.assertEqual(outcome["severity"], ce.SEVERITY_REQUIRED)


# ---------------------------------------------------------------------------
# parse_version()
# ---------------------------------------------------------------------------


class TestParseVersion(unittest.TestCase):
    def test_converts_dotted_versions(self):
        cases = {
            "1.26.4": (1, 26, 4),
            "0.53.1": (0, 53, 1),
            "3.10.15": (3, 10, 15),
            "42": (42,),
            "0.0.1": (0, 0, 1),
            "1.26.4+mkl": (1, 26, 4),
            "1.0rc1": (1, 0),
            "2.22.3.post1": (2, 22, 3),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ce.parse_version(text), expected)

    def test_handles_none_and_empty(self):
        self.assertEqual(ce.parse_version(None), ())
        self.assertEqual(ce.parse_version(""), ())

    def test_ignores_a_leading_dot(self):
        self.assertEqual(ce.parse_version(".5"), ())

    def test_comparison_behaviour_used_by_the_setuptools_gate(self):
        self.assertFalse(ce.parse_version("41.0.1") >= ce.MIN_SETUPTOOLS)
        self.assertTrue(ce.parse_version("42") >= ce.MIN_SETUPTOOLS)
        self.assertTrue(ce.parse_version("69.5.0") >= ce.MIN_SETUPTOOLS)
        self.assertFalse(ce.parse_version("garbage") >= ce.MIN_SETUPTOOLS)


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


class TestSummarize(unittest.TestCase):
    def summarize(self, **overrides) -> dict:
        inputs = healthy_inputs()
        inputs.update(overrides)
        return ce.summarize(
            inputs["system"],
            inputs["python_info"],
            inputs["toolchain"],
            inputs["probes"],
            inputs.get("jit"),
        )

    def test_healthy_machine_is_ok(self):
        summary = self.summarize()
        self.assertEqual(summary["verdict"], "OK")
        self.assertFalse(summary["blocked"])
        self.assertEqual(summary["blockers"], [])

    def test_python_310_boundary(self):
        self.assertTrue(
            self.summarize(
                python_info={"python_version": "3.10.0", "satisfies_python_requires": True}
            )["verdict"]
            == "OK"
        )
        summary = self.summarize(
            python_info={"python_version": "3.9.18", "satisfies_python_requires": False}
        )
        self.assertTrue(summary["blocked"])
        self.assertTrue(
            any("does not satisfy >=3.10" in item for item in summary["blockers"])
        )

    def test_missing_required_dependency_blocks(self):
        summary = self.summarize(
            probes=[present_probe("numpy"), missing_probe("numba")]
        )
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["missing_required"], ["numba"])

    def test_missing_important_dependency_only_warns(self):
        summary = self.summarize(
            probes=[
                present_probe("numpy"),
                missing_probe("scipy", severity=ce.SEVERITY_IMPORTANT),
            ]
        )
        self.assertFalse(summary["blocked"])
        self.assertEqual(summary["verdict"], "OK_WITH_WARNINGS")
        self.assertEqual(summary["missing_important"], ["scipy"])

    def test_missing_optional_dependency_is_only_listed(self):
        summary = self.summarize(
            probes=[
                present_probe("numpy"),
                missing_probe("datashader", severity=ce.SEVERITY_OPTIONAL),
            ]
        )
        self.assertEqual(summary["verdict"], "OK")
        self.assertEqual(summary["missing_optional"], ["datashader"])
        self.assertEqual(summary["warnings"], [])

    def test_probe_without_a_severity_fails_closed_instead_of_raising(self):
        probe = {"import_name": "numpy", "status": "error", "message": "boom"}
        summary = self.summarize(probes=[probe])
        self.assertTrue(summary["blocked"])
        self.assertIn("numpy", summary["missing_required"])

    def test_probe_with_an_unknown_severity_fails_closed(self):
        # A typo in "severity" used to create a fourth bucket that nothing read,
        # so the dependency disappeared from the verdict entirely.
        probe = missing_probe("numpy", severity="catastrophic")
        summary = self.summarize(probes=[probe])
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["missing_required"], ["numpy"])

    def test_pip_missing_blocks(self):
        toolchain = dict(healthy_inputs()["toolchain"], pip_available=False)
        summary = self.summarize(toolchain=toolchain)
        self.assertTrue(summary["blocked"])
        self.assertTrue(any("pip is not available" in item for item in summary["blockers"]))

    def test_missing_setuptools_is_reported_not_silent(self):
        toolchain = dict(
            healthy_inputs()["toolchain"],
            setuptools_version=None,
            setuptools_meets_minimum=None,
        )
        summary = self.summarize(toolchain=toolchain)
        self.assertTrue(
            any("setuptools is not installed" in item for item in summary["warnings"])
        )

    def test_old_setuptools_is_reported(self):
        toolchain = dict(
            healthy_inputs()["toolchain"],
            setuptools_version="41.0.1",
            setuptools_meets_minimum=False,
        )
        summary = self.summarize(toolchain=toolchain)
        self.assertTrue(any("older than 42" in item for item in summary["warnings"]))

    def test_missing_compiler_warns(self):
        toolchain = dict(healthy_inputs()["toolchain"], compilers={})
        summary = self.summarize(toolchain=toolchain)
        self.assertTrue(any("no C compiler" in item for item in summary["warnings"]))

    def test_jit_failure_blocks(self):
        # Import success is not execution success: this is the HarmonyOS case.
        jit = {
            "import_name": "numba.jit",
            "status": "error",
            "error_type": "NoProbeOutput",
            "message": "probe process died from SIGSEGV (returncode -11)",
        }
        summary = self.summarize(jit=jit)
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["jit_status"], "error")
        self.assertTrue(any("numba JIT cannot execute" in item for item in summary["blockers"]))

    def test_jit_timeout_blocks_and_tells_you_what_to_do(self):
        jit = {
            "import_name": "numba.jit",
            "status": "error",
            "error_type": "Timeout",
            "message": "probe exceeded the 5s timeout",
        }
        summary = self.summarize(jit=jit)
        self.assertTrue(summary["blocked"])
        self.assertTrue(
            any("--probe-timeout" in item for item in summary["blockers"])
        )

    def test_successful_jit_does_not_block(self):
        jit = {"import_name": "numba.jit", "status": "present", "jit_result": 45.0}
        summary = self.summarize(jit=jit)
        self.assertFalse(summary["blocked"])
        self.assertEqual(summary["jit_status"], "present")

    def test_jit_not_probed_is_reported_but_not_a_blocker(self):
        summary = self.summarize()
        self.assertEqual(summary["jit_status"], "not probed")
        self.assertFalse(summary["blocked"])


# ---------------------------------------------------------------------------
# URL redaction
# ---------------------------------------------------------------------------


class TestRedactUrl(unittest.TestCase):
    def test_strips_userinfo(self):
        self.assertEqual(
            ce.redact_url("https://buildbot:faketoken@nexus.example/simple"),
            "https://<redacted>@nexus.example/simple",
        )

    def test_strips_user_only_credentials(self):
        self.assertEqual(
            ce.redact_url("https://token@nexus.example/simple"),
            "https://<redacted>@nexus.example/simple",
        )

    def test_keeps_credentials_out_of_a_bare_authority(self):
        redacted = ce.redact_url("buildbot:secret@nexus.example/simple")
        self.assertNotIn("secret", redacted)

    def test_leaves_plain_urls_alone(self):
        for value in (
            "https://pypi.org/simple/",
            "https://nexus.example:8443/simple",
            "not-a-url",
        ):
            with self.subTest(value=value):
                self.assertEqual(ce.redact_url(value), value)

    def test_passes_through_empty_values(self):
        self.assertIsNone(ce.redact_url(None))
        self.assertEqual(ce.redact_url(""), "")

    def test_toolchain_never_records_the_secret(self):
        secret = "https://buildbot:SUPERSECRETTOKEN@nexus.example/simple"
        with mock.patch.dict(os.environ, {"PIP_INDEX_URL": secret}):
            toolchain = ce.collect_toolchain()
        self.assertNotIn("SUPERSECRETTOKEN", json.dumps(toolchain))
        self.assertIn("<redacted>", toolchain["pip_index_url"])


# ---------------------------------------------------------------------------
# report shape
# ---------------------------------------------------------------------------


def fake_report() -> dict:
    """Build a report from stubbed collectors, without spawning anything."""

    probes = [present_probe("numpy"), present_probe("numba")]
    with mock.patch.object(ce, "collect_system", return_value={
        "platform": "Windows-10-10.0.26200-SP0",
        "system": "Windows",
        "release": "10",
        "version": "10.0.26200",
        "machine": "AMD64",
        "processor": "Intel64 Family 6",
        "is_64bit": True,
        "cpu_count": 8,
        "hostname": "host",
        "uname": None,
        "libc": None,
    }), mock.patch.object(ce, "collect_platform_files", return_value={}), \
        mock.patch.object(ce, "collect_python", return_value={
            "python_implementation": "CPython",
            "python_version": "3.10.21",
            "python_full_version": "3.10.21",
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_base_prefix": sys.base_prefix,
            "python_cache_tag": "cpython-310",
            "pyspec_platform": "win-amd64",
            "python_debug_build": False,
            "gil_disabled": False,
            "host_gnu_type": None,
            "in_virtualenv": False,
            "satisfies_python_requires": True,
        }), mock.patch.object(ce, "collect_toolchain", return_value={
            "pip_available": True,
            "pip_version": "24.0",
            "pip_path": None,
            "pip_index_url": None,
            "setuptools_version": "69.0.0",
            "setuptools_meets_minimum": True,
            "wheel_version": "0.42.0",
            "venv_importable": True,
            "ensurepip_importable": True,
            "ssl_available": True,
            "openssl_version": "OpenSSL 3.0.13",
            "compilers": {"gcc": "gcc :: 13.2.0"},
            "make_available": False,
        }), mock.patch.object(ce, "collect_probes", return_value=probes), \
        mock.patch.object(ce, "collect_raytracepy", return_value={
            "distribution_version": "0.0.1",
            "from_environment": present_probe("raytracepy", file="C:/x/__init__.py"),
            "from_source_tree": present_probe("raytracepy", file="C:/src/__init__.py"),
        }), mock.patch.object(ce, "collect_jit_probe", return_value=None), \
        mock.patch.object(ce, "collect_git_commit", return_value="deadbeef"), \
        mock.patch.object(ce, "collect_windows_only_artifacts", return_value=[]):
        return ce.build_report(ce.parse_arguments(["--no-write"]))


class TestReportShape(unittest.TestCase):
    def test_every_reference_key_is_present(self):
        # The two files are meant to be diffed key by key.
        reference = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
        report = fake_report()
        self.assertEqual(sorted(set(reference) - set(report)), [])

    def test_packages_match_the_reference_keys_exactly(self):
        reference = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
        report = fake_report()
        self.assertEqual(sorted(report["packages"]), sorted(reference["packages"]))
        self.assertEqual(sorted(report["packages"]), sorted(ce.COMPARABLE_PACKAGES))

    def test_build_tooling_is_not_in_packages(self):
        report = fake_report()
        for name in ("pip", "setuptools", "wheel"):
            self.assertNotIn(name, report["packages"])

    def test_flat_section_drops_cache_tag_but_keeps_it_under_python(self):
        report = fake_report()
        self.assertNotIn("python_cache_tag", report)
        self.assertEqual(report["python"]["python_cache_tag"], "cpython-310")

    def test_import_path_prefers_the_source_tree(self):
        report = fake_report()
        self.assertEqual(report["raytracepy_import_path"], "C:/src/__init__.py")

    def test_report_is_json_serialisable(self):
        # Guards against a tuple or set reaching json.dump.
        json.dumps(fake_report())

    def test_jit_probe_is_recorded_under_jit(self):
        self.assertIn("jit", fake_report())

    def test_summary_is_consistent_with_the_probe_list(self):
        report = fake_report()
        self.assertEqual(report["summary"]["jit_status"], "not probed")
        self.assertFalse(report["summary"]["blocked"])


# ---------------------------------------------------------------------------
# render_report()
# ---------------------------------------------------------------------------


class TestRenderReport(unittest.TestCase):
    def render(self, report: dict) -> str:
        return ce.render_report(report)

    def test_successful_report_mentions_the_verdict(self):
        text = self.render(fake_report())
        self.assertIn("RESULT       : OK\n", text)
        self.assertIn("RayTracePy", text)

    def test_jit_section_is_rendered_when_the_probe_ran(self):
        report = fake_report()
        report["jit"] = {
            "import_name": "numba.jit",
            "status": "present",
            "jit_result": 45.0,
            "numba_version": "0.61.2",
            "jit_enabled": True,
        }
        report["summary"]["jit_status"] = "present"
        text = self.render(report)
        self.assertIn("Numba JIT execution", text)
        self.assertIn("returned 45.0", text)

    def test_jit_failure_is_rendered_with_the_returncode(self):
        report = fake_report()
        report["jit"] = {
            "import_name": "numba.jit",
            "status": "error",
            "error_type": "NoProbeOutput",
            "message": "probe process died from SIGSEGV",
            "returncode": -11,
        }
        text = self.render(report)
        self.assertIn("njit compile+call: FAILED", text)
        self.assertIn("-11", text)
        self.assertIn("unusable", text)

    def test_jit_section_is_absent_when_there_was_no_probe(self):
        report = fake_report()
        report["jit"] = None
        self.assertNotIn("Numba JIT execution", self.render(report))

    def test_missing_probe_details_are_rendered(self):
        report = fake_report()
        report["probes"] = [missing_probe("numpy")]
        text = self.render(report)
        self.assertIn("MISSING", text)
        self.assertIn("numpy", text)

    def test_windows_only_artifacts_are_flagged(self):
        report = fake_report()
        report["windows_only_artifacts"] = ["math_custom.cp310-win_amd64.pyd"]
        self.assertIn("not usable on a non-Windows", self.render(report))

    def test_empty_blockers_and_warnings_render_as_none(self):
        text = self.render(fake_report())
        self.assertIn("blockers     : none", text)
        self.assertIn("warnings     : none", text)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


class TestCommandLine(unittest.TestCase):
    """End-to-end runs of the real script.

    Every probe is given an unusably small timeout so the run stays fast; the
    assertions are about the plumbing, not about which packages are installed.
    """

    FAST = "--probe-timeout=0.001"

    def run_cli(self, *arguments: str, env: dict | None = None):
        return subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "check_environment.py"), *arguments],
            capture_output=True,
            text=True,
            errors="replace",
            cwd=str(REPO_ROOT),
            env={**os.environ, **(env or {})},
            check=False,
        )

    def test_runs_and_reports_without_writing(self):
        result = self.run_cli("--no-write", self.FAST)
        self.assertIn("RESULT       : BLOCKED", result.stdout)
        self.assertIn("RayTracePy native Python environment check", result.stdout)
        # Every probe timed out at 1 ms, so required dependencies are missing.
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)

    def test_rejects_a_traversing_label_without_a_traceback(self):
        result = self.run_cli("--no-write", "--label=../../escape", self.FAST)
        self.assertEqual(result.returncode, 2)
        self.assertIn("argument --label", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_rejects_a_non_finite_timeout_without_a_traceback(self):
        # Before the fix this exited with ValueError/OverflowError and no report.
        for value in ("nan", "inf", "0"):
            with self.subTest(value=value):
                result = self.run_cli("--no-write", f"--probe-timeout={value}")
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)

    def test_rejects_no_write_together_with_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "environment.json"
            result = self.run_cli("--no-write", f"--output={target}", self.FAST)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(target.exists())

    def test_writes_a_valid_json_document_on_a_legacy_console(self):
        # PYTHONIOENCODING=ascii is the cheapest way to reproduce a console that
        # cannot encode the U+FFFD that errors="replace" puts into decoded probe
        # output.  Before the fix this raised UnicodeEncodeError and exited
        # before the JSON was written.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "environment.json"
            result = self.run_cli(
                f"--output={target}", self.FAST, env={"PYTHONIOENCODING": "ascii"}
            )
            self.assertNotIn("Traceback", result.stderr)
            self.assertTrue(target.exists(), result.stderr)
            document = json.loads(target.read_text(encoding="utf-8"))
            for key in ("summary", "packages", "probes", "python", "toolchain", "jit"):
                self.assertIn(key, document)
            self.assertEqual(document["summary"]["verdict"], "BLOCKED")
            self.assertEqual(sorted(document["packages"]), sorted(ce.COMPARABLE_PACKAGES))

    def test_unwritable_output_is_a_failure_not_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("x", encoding="utf-8")
            result = self.run_cli(f"--output={blocker / 'environment.json'}", self.FAST)
            self.assertEqual(result.returncode, 1)
            self.assertIn("ERROR: could not write", result.stdout)

    def test_unencodable_report_text_is_replaced_not_fatal(self):
        # Drive main() with a stdout whose encoding cannot represent U+FFFD.
        buffer = io.TextIOWrapper(io.BytesIO(), encoding="ascii", newline="")
        report = {"label": "unit", "summary": {"blocked": False}}
        with mock.patch.object(ce, "build_report", return_value=report), \
                mock.patch.object(
                    ce, "render_report", return_value="bad \ufffd char"
                ), \
                mock.patch.object(sys, "stdout", buffer):
            with self.assertRaises(UnicodeEncodeError):
                # Sanity check: this stream is strict by default.
                sys.stdout.write("\ufffd")
            exit_code = ce.main(["--no-write"])
        self.assertEqual(exit_code, 0)
        buffer.flush()
        written = buffer.detach().getvalue().decode("ascii")
        self.assertEqual(written, "bad ? char\n")

    def test_stdout_without_reconfigure_is_tolerated(self):
        buffer = io.StringIO()
        report = {"label": "unit", "summary": {"blocked": False}}
        with mock.patch.object(ce, "build_report", return_value=report), \
                mock.patch.object(ce, "render_report", return_value="plain text"), \
                mock.patch.object(sys, "stdout", buffer):
            exit_code = ce.main(["--no-write"])
        self.assertEqual(exit_code, 0)
        self.assertIn("plain text", buffer.getvalue())

    def test_printing_failure_does_not_lose_the_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "environment.json"
            report = {"label": "unit", "summary": {"blocked": False}}
            broken = mock.MagicMock()
            broken.write.side_effect = OSError("pipe closed")
            with mock.patch.object(ce, "build_report", return_value=report), \
                    mock.patch.object(ce, "render_report", return_value="text"), \
                    mock.patch.object(sys, "stdout", broken):
                exit_code = ce.main([f"--output={target}"])
            self.assertEqual(exit_code, 0)
            self.assertTrue(target.exists())

    def test_help_documents_the_exit_status(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("exit status", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
