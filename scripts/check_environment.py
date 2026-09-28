#!/usr/bin/env python3
"""Check whether a native Python environment can host RayTracePy.

This script is the deliverable of the HarmonyOS PC Python-environment task.
Run it unchanged on the Windows reference machine and on the HarmonyOS PC, then
compare the two generated ``environment.json`` files.

Design constraints -- do not break these:

* Standard library only.  The target machine may not have NumPy, Numba or any
  other third-party package yet, so nothing outside the standard library may be
  imported at module level.
* A missing dependency is a *result*, not an error.  Every probe is guarded and
  reported, and the script never aborts because a single check failed.
* Import probes run in a subprocess.  Importing Numba pulls in LLVM and can be
  slow or hard-crash on an unsupported architecture; the checker must survive
  that and still write its report.
* Emit both a human-readable report and a machine-readable JSON document whose
  keys line up with ``artifacts/reference/windows-python310/environment.json``
  so the reference and HarmonyOS results can be diffed directly.
* Nothing may abort the run before the JSON is written.  A probe that crashes,
  hangs or returns nonsense is a result; so is an unusable ``--probe-timeout``
  or a character the console cannot encode.  The report is only useful if it
  survives the situation it was written for.

Exit status: ``0`` for OK or OK_WITH_WARNINGS, ``1`` for BLOCKED (which
includes a report that could not be written), ``2`` for bad arguments.

Usage::

    python scripts/check_environment.py --label windows-python310
    python scripts/check_environment.py --no-write
    python scripts/check_environment.py --check-network
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "environment"

SEVERITY_REQUIRED = "required"
SEVERITY_IMPORTANT = "important"
SEVERITY_OPTIONAL = "optional"

#: Failures ``subprocess`` can raise that must be turned into a reported
#: result.  ``ValueError`` and ``OverflowError`` are included because an
#: unusable timeout (NaN, infinity) surfaces from deep inside ``subprocess`` as
#: one of those, and an escaping traceback would destroy the whole report.
SUBPROCESS_FAILURES = (OSError, ValueError, OverflowError, subprocess.SubprocessError)

#: Minimum setuptools version required by ``pyproject.toml``.
MIN_SETUPTOOLS = (42,)

#: Distributions placed in the flattened ``packages`` map.  These are exactly
#: the keys used by ``artifacts/reference/windows-python310/environment.json``,
#: so the reference and the target file can be compared key by key.  The build
#: tooling (pip, setuptools, wheel) is deliberately excluded because the
#: reference file does not record it; it is reported under ``toolchain``.
COMPARABLE_PACKAGES = (
    "raytracepy",
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "pandas",
    "plotly",
    "datashader",
    "pytest",
    "pytest-cov",
)

#: Characters allowed in ``--label``.  The first character must be
#: alphanumeric so that ``.``, ``..`` and absolute paths are rejected and the
#: report cannot be written outside ``artifacts/environment``.
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

#: C compilers that would be needed to build a dependency from source.
COMPILER_NAMES = ("cc", "gcc", "clang", "g++", "clang++", "cl")

#: Platform files copied into the report as raw evidence.
PLATFORM_FILE_CANDIDATES = (
    "/etc/os-release",
    "/proc/version",
    "/proc/meminfo",
    "/proc/cpuinfo",
)


@dataclass(frozen=True)
class Probe:
    """A single third-party import that the port depends on."""

    distribution: str
    import_name: str
    severity: str
    purpose: str


PROBES: tuple[Probe, ...] = (
    Probe(
        "numpy",
        "numpy",
        SEVERITY_REQUIRED,
        "core array maths; imported by every RayTracePy module",
    ),
    Probe(
        "numba",
        "numba",
        SEVERITY_REQUIRED,
        "imported at `import raytracepy` time; JIT backend",
    ),
    Probe(
        "plotly",
        "plotly",
        SEVERITY_REQUIRED,
        "imported by light, plane, raytrace and light_layouts",
    ),
    Probe(
        "pandas",
        "pandas",
        SEVERITY_REQUIRED,
        "imported by raytracepy.plane",
    ),
    Probe(
        "scipy",
        "scipy",
        SEVERITY_IMPORTANT,
        "scipy.integrate quadrature in raytracepy.ref_data",
    ),
    Probe(
        "llvmlite",
        "llvmlite",
        SEVERITY_IMPORTANT,
        "LLVM bindings required by numba",
    ),
    Probe(
        "datashader",
        "datashader",
        SEVERITY_OPTIONAL,
        "heatmap figures in examples only; NOT needed to import the package",
    ),
    Probe(
        "pytest",
        "pytest",
        SEVERITY_OPTIONAL,
        "automated test runner",
    ),
)

#: Snippet executed in a fresh interpreter for each import probe.  A hard crash
#: (segfault, ``SystemExit``, dead LLVM) only kills the child process.
PROBE_SNIPPET = """
import importlib
import json
import sys

name = sys.argv[1]
payload = {"import_name": name}
try:
    module = importlib.import_module(name)
except BaseException as exc:  # noqa: BLE001 - anything is a reportable result
    payload["status"] = "error"
    payload["error_type"] = type(exc).__name__
    payload["message"] = str(exc)[:500]
else:
    payload["status"] = "present"
    payload["file"] = getattr(module, "__file__", None)
    payload["module_version"] = getattr(module, "__version__", None)
print("__PROBE__" + json.dumps(payload))
"""

#: Snippet that probes ``raytracepy`` with ``src`` prepended to ``sys.path``.
RAYTRACEPY_SNIPPET = """
import json
import sys

sys.path.insert(0, sys.argv[1])
payload = {"import_name": "raytracepy"}
try:
    import raytracepy
except BaseException as exc:  # noqa: BLE001
    payload["status"] = "error"
    payload["error_type"] = type(exc).__name__
    payload["message"] = str(exc)[:500]
else:
    payload["status"] = "present"
    payload["file"] = getattr(raytracepy, "__file__", None)
print("__PROBE__" + json.dumps(payload))
"""

#: Snippet that compiles *and calls* an ``@njit`` function.  Importing numba
#: only proves the bindings load; the JIT emits machine code for the host CPU
#: on first call, and that is where an unsupported or emulated platform fails.
#: A failure here is reported as a crash because the child process dies.
JIT_SNIPPET = """
import json
import sys

payload = {"import_name": "numba.jit", "status": "present"}
try:
    import numba
    import numpy as np

    @numba.njit
    def _sum(values):
        total = 0.0
        for value in values:
            total += value
        return total

    got = float(_sum(np.arange(10.0)))
except BaseException as exc:  # noqa: BLE001 - anything is a reportable result
    payload["status"] = "error"
    payload["error_type"] = type(exc).__name__
    payload["message"] = str(exc)[:500]
else:
    expected = 45.0
    if abs(got - expected) > 1e-9:
        payload["status"] = "error"
        payload["error_type"] = "WrongResult"
        payload["message"] = "njit returned %r, expected %r" % (got, expected)
    else:
        payload["jit_result"] = got
        payload["numba_version"] = numba.__version__
        payload["jit_enabled"] = not numba.config.DISABLE_JIT
print("__PROBE__" + json.dumps(payload))
"""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def capture(
    command: tuple[str, ...],
    timeout: float = 30.0,
    check: bool = True,
) -> str | None:
    """Run ``command`` and return its first line of output, or ``None``.

    A non-zero exit status yields ``None`` unless ``check`` is false.  Without
    that check an interpreter lacking pip is reported as having it: the failing
    ``python -m pip --version`` writes "No module named pip" to stderr, and
    returning that text as the version marks pip as available, so the
    "pip is not available" blocker can never fire.
    """

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except SUBPROCESS_FAILURES:
        return None

    if check and completed.returncode != 0:
        return None

    for stream in (completed.stdout, completed.stderr):
        text = (stream or "").strip()
        if text:
            return text.splitlines()[0].strip()
    return None


def parse_version(text: str | None) -> tuple[int, ...]:
    """Extract a leading dotted-numeric version from ``text``."""

    if not text:
        return ()
    digits: list[str] = []
    parts: list[str] = []
    for char in text:
        if char.isdigit():
            digits.append(char)
        elif char == "." and digits:
            parts.append("".join(digits))
            digits = []
        else:
            break
    if digits:
        parts.append("".join(digits))
    return tuple(int(part) for part in parts if part)


def distribution_version(name: str) -> str | None:
    """Return an installed distribution version without importing it."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - malformed metadata must not be fatal
        return None


def default_label() -> str:
    """Derive a report label such as ``windows-python310``.

    The label becomes a path component, so the derived default is held to the
    same whitelist as ``--label``: an unexpected value from ``platform.system()``
    must not be able to point the output somewhere else.
    """

    system = (platform.system() or "unknown").lower()
    info = sys.version_info
    candidate = f"{system}-python{info.major}{info.minor}"
    return candidate if LABEL_PATTERN.match(candidate) else "unknown-python"


def redact_url(value: str | None) -> str | None:
    """Strip credentials embedded in a URL.

    ``PIP_INDEX_URL`` commonly carries a token for a private index, and the
    report is written into the repository and pushed, so the secret must never
    reach the file or the terminal.
    """

    if not value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        parts = None
    if parts is not None and parts.netloc and "@" in parts.netloc:
        host = parts.netloc.rsplit("@", 1)[1]
        return urlunsplit(
            (
                parts.scheme,
                f"<redacted>@{host}",
                parts.path,
                parts.query,
                parts.fragment,
            )
        )
    # No parseable authority.  A bare ``user:secret@host`` would be split as
    # scheme ``user`` and netloc ````, so fall back to dropping everything in
    # front of the last "@".
    if "@" in value:
        return "<redacted>@" + value.rsplit("@", 1)[1]
    return value


# ---------------------------------------------------------------------------
# system information
# ---------------------------------------------------------------------------


def collect_system() -> dict[str, Any]:
    """Collect operating-system and CPU information."""

    uname = platform.uname()
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "is_64bit": sys.maxsize > 2**32,
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "uname": capture_from_path("uname", "-a"),
        # ``ldd --version`` exits non-zero on musl while still printing the libc
        # version, and libc identity is exactly what this report needs, so the
        # exit status is deliberately not treated as failure here.
        "libc": capture_from_path("ldd", "--version", check=False),
    }


def capture_from_path(
    program: str,
    *arguments: str,
    check: bool = True,
) -> str | None:
    """Look ``program`` up on ``PATH`` and run it."""

    executable = shutil.which(program)
    if executable is None:
        return None
    return capture((executable, *arguments), check=check)


def summarize_platform_file(path: Path) -> str | None:
    """Read a platform file, reducing the very large ones to useful lines."""

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if path.name == "meminfo":
        keys = ("MemTotal", "MemAvailable", "SwapTotal")
        kept = [line.strip() for line in raw.splitlines() if line.startswith(keys)]
        return "\n".join(kept) or None

    if path.name == "cpuinfo":
        markers = (
            "model name",
            "Hardware",
            "Model",
            "CPU implementer",
            "CPU part",
            "CPU architecture",
            "Features",
        )
        kept = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and any(marker in line for marker in markers)
        ]
        return "\n".join(kept[:40]) or None

    return raw.strip()[:2000] or None


def collect_platform_files() -> dict[str, str]:
    """Return raw ``/etc`` and ``/proc`` evidence for the platform."""

    candidates = [Path(item) for item in PLATFORM_FILE_CANDIDATES]

    etc = Path("/etc")
    if etc.is_dir():
        try:
            candidates.extend(sorted(etc.glob("*release*"))[:10])
        except OSError:
            pass

    collected: dict[str, str] = {}
    for candidate in candidates:
        if str(candidate) in collected:
            continue
        summary = summarize_platform_file(candidate)
        if summary:
            collected[str(candidate)] = summary
    return collected


# ---------------------------------------------------------------------------
# python runtime and toolchain
# ---------------------------------------------------------------------------


def collect_python() -> dict[str, Any]:
    """Collect interpreter identity, ABI and capability information."""

    implementation = platform.python_implementation()
    return {
        "python_implementation": implementation,
        "python_version": platform.python_version(),
        "python_full_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "python_cache_tag": sys.implementation.cache_tag,
        "pyspec_platform": sysconfig.get_platform(),
        "python_debug_build": bool(sysconfig.get_config_var("Py_DEBUG")),
        "gil_disabled": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "host_gnu_type": sysconfig.get_config_var("HOST_GNU_TYPE"),
        "in_virtualenv": sys.prefix != sys.base_prefix,
        "satisfies_python_requires": (
            sys.version_info.major == 3 and sys.version_info.minor >= 10
        ),
    }


def collect_toolchain() -> dict[str, Any]:
    """Collect pip/setuptools/wheel and build-capability information."""

    pip_version = None
    pip_path = None
    pip_output = capture((sys.executable, "-m", "pip", "--version"))
    if pip_output:
        head, separator, tail = pip_output.partition(" from ")
        pip_version = (head.split(" ", 1)[1].strip() if " " in head else head.strip()) or None
        if separator:
            pip_path = tail.split(" (python")[0].strip() or None

    setuptools_version = distribution_version("setuptools")
    wheel_version = distribution_version("wheel")

    setuptools_ok: bool | None
    if setuptools_version is None:
        setuptools_ok = None
    else:
        parsed = parse_version(setuptools_version)
        setuptools_ok = bool(parsed) and parsed >= MIN_SETUPTOOLS

    compilers: dict[str, str] = {}
    for name in COMPILER_NAMES:
        executable = shutil.which(name)
        if executable is None:
            continue
        version = capture((executable, "--version")) or capture((executable,))
        compilers[name] = f"{executable} :: {version or 'version unknown'}"

    return {
        "pip_available": pip_version is not None,
        "pip_version": pip_version,
        "pip_path": pip_path,
        "pip_index_url": redact_url(os.environ.get("PIP_INDEX_URL")),
        "setuptools_version": setuptools_version,
        "setuptools_meets_minimum": setuptools_ok,
        "wheel_version": wheel_version,
        "venv_importable": module_importable("venv"),
        "ensurepip_importable": module_importable("ensurepip"),
        "ssl_available": module_importable("ssl"),
        "openssl_version": ssl_version(),
        "compilers": compilers,
        "make_available": shutil.which("make") is not None,
    }


def module_importable(name: str) -> bool:
    """Check that a standard-library module can be imported."""

    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ssl_version() -> str | None:
    """Return the linked OpenSSL version, if any."""

    try:
        import ssl
    except ImportError:
        return None
    return getattr(ssl, "OPENSSL_VERSION", None)


# ---------------------------------------------------------------------------
# import probes
# ---------------------------------------------------------------------------


def run_probe(
    snippet: str,
    *arguments: str,
    timeout: float,
    import_name: str,
) -> dict[str, Any]:
    """Execute ``snippet`` in a fresh interpreter and parse its result.

    ``import_name`` is copied into every returned document, so a failure
    carries the same key as a success.  Callers depend on that: the key used to
    be absent from the timeout and crash paths, and ``summarize()`` raised
    ``KeyError`` while assembling the verdict.  That destroyed the entire
    report in precisely the case the checker exists for -- a probe that hangs
    or takes the interpreter down -- leaving no text report and no JSON.
    """

    command = (sys.executable, "-c", snippet, *arguments)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "import_name": import_name,
            "status": "error",
            "error_type": "Timeout",
            "message": f"probe exceeded the {timeout:g}s timeout",
        }
    except SUBPROCESS_FAILURES as exc:
        return {
            "import_name": import_name,
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith("__PROBE__"):
            try:
                payload = json.loads(line[len("__PROBE__") :])
            except json.JSONDecodeError:
                break
            payload.setdefault("import_name", import_name)
            if completed.returncode != 0:
                payload["returncode"] = completed.returncode
            return payload

    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    failure: dict[str, Any] = {
        "import_name": import_name,
        "status": "error",
        "error_type": "NoProbeOutput",
        "message": detail[-1] if detail else "the probe process produced no result",
        "returncode": completed.returncode,
    }
    if completed.returncode < 0:
        failure["error_type"] = "Crashed"
        failure["signal_name"] = signal_name(-completed.returncode)
        failure["message"] = (
            f"probe process died from {failure['signal_name']} "
            f"(returncode {completed.returncode})"
        )
    return failure


def signal_name(number: int) -> str:
    """Return the name of a signal number, or a readable fallback."""

    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def collect_probes(timeout: float) -> list[dict[str, Any]]:
    """Probe every dependency, plus ``raytracepy`` itself."""

    results: list[dict[str, Any]] = []
    for probe in PROBES:
        outcome = run_probe(
            PROBE_SNIPPET,
            probe.import_name,
            timeout=timeout,
            import_name=probe.import_name,
        )
        outcome.update(
            {
                "distribution": probe.distribution,
                "severity": probe.severity,
                "purpose": probe.purpose,
                "distribution_version": distribution_version(probe.distribution),
            }
        )
        results.append(outcome)
    return results


def collect_jit_probe(timeout: float, numba_present: bool) -> dict[str, Any] | None:
    """Compile and call an ``@njit`` function to see whether the JIT executes.

    Import success does not imply the backend works.  On HarmonyOS PC the
    bindings load and compilation succeeds, but the call segfaults, so a
    package whose hot paths are all ``@njit`` cannot run.  Only meaningful when
    numba imported in the first place; otherwise the missing dependency is
    already reported and this returns ``None``.
    """

    if not numba_present:
        return None

    outcome = run_probe(JIT_SNIPPET, timeout=timeout, import_name="numba.jit")
    outcome["severity"] = SEVERITY_REQUIRED
    outcome["purpose"] = (
        "compiles and calls an @njit kernel; RayTracePy's hot paths are all @njit"
    )
    return outcome


def collect_raytracepy(timeout: float) -> dict[str, Any]:
    """Try ``import raytracepy`` from the environment and from ``src/``."""

    from_environment = run_probe(
        PROBE_SNIPPET,
        "raytracepy",
        timeout=timeout,
        import_name="raytracepy",
    )
    from_environment["source"] = "installed / ambient sys.path"

    from_source = run_probe(
        RAYTRACEPY_SNIPPET,
        str(PROJECT_ROOT / "src"),
        timeout=timeout,
        import_name="raytracepy",
    )
    from_source["source"] = "src/ prepended to sys.path"

    return {
        "distribution_version": distribution_version("raytracepy"),
        "from_environment": from_environment,
        "from_source_tree": from_source,
    }


def collect_git_commit() -> str | None:
    """Return the HEAD commit of the repository containing this script.

    Recorded so a report can be tied to the exact source revision, matching the
    ``git_commit`` key of the reference ``environment.json``.
    """

    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            (executable, "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            errors="replace",
            cwd=str(PROJECT_ROOT),
            timeout=15.0,
            check=False,
        )
    except SUBPROCESS_FAILURES:
        return None
    if completed.returncode != 0:
        return None
    return (completed.stdout or "").strip() or None


def collect_windows_only_artifacts() -> list[str]:
    """List binary artefacts that are known to be platform specific."""

    compile_dir = PROJECT_ROOT / "src" / "raytracepy" / "compile"
    if not compile_dir.is_dir():
        return []
    try:
        return sorted(path.name for path in compile_dir.glob("*.pyd"))
    except OSError:
        return []


def check_network(timeout: float) -> dict[str, Any]:
    """Optional, opt-in reachability check for the package index."""

    import urllib.error
    import urllib.request

    url = os.environ.get("PIP_INDEX_URL", "https://pypi.org/simple/")
    # The request needs the credentials, the report does not.
    reported = redact_url(url)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"url": reported, "reachable": True, "status": response.status}
    except urllib.error.HTTPError as exc:
        return {"url": reported, "reachable": True, "status": exc.code}
    except Exception as exc:  # noqa: BLE001 - offline is a normal result
        return {
            "url": reported,
            "reachable": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def summarize(
    system: dict[str, Any],
    python_info: dict[str, Any],
    toolchain: dict[str, Any],
    probes: list[dict[str, Any]],
    jit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce everything to a verdict for the day-5 decision gate."""

    missing: dict[str, list[str]] = {
        SEVERITY_REQUIRED: [],
        SEVERITY_IMPORTANT: [],
        SEVERITY_OPTIONAL: [],
    }
    for probe in probes:
        if probe.get("status") == "present":
            continue
        # An unlabelled probe must not vanish from the verdict, and an unknown
        # severity must not create a fourth bucket that nobody reads: both fail
        # closed as required, because the answer this feeds is "can we ship".
        severity = probe.get("severity", SEVERITY_REQUIRED)
        if severity not in missing:
            severity = SEVERITY_REQUIRED
        missing[severity].append(probe.get("import_name", "<unnamed probe>"))

    blockers: list[str] = []
    if not python_info["satisfies_python_requires"]:
        blockers.append(
            f"python {python_info['python_version']} does not satisfy >=3.10"
        )
    if not toolchain["pip_available"]:
        blockers.append("pip is not available for this interpreter")
    if not toolchain["venv_importable"]:
        blockers.append("the venv module is not importable")
    blockers.extend(f"required dependency missing: {name}" for name in missing[SEVERITY_REQUIRED])
    if jit is not None and jit.get("status") != "present":
        detail = f"{jit.get('error_type', 'error')}: {jit.get('message', '')}".strip()
        detail = detail.strip(":").strip() or "no detail reported"
        if jit.get("error_type") == "Timeout":
            detail += (
                "; the first @njit call compiles LLVM code and needs longer than "
                "an import, so re-run with a larger --probe-timeout"
            )
        blockers.append(
            f"numba JIT cannot execute on this host ({detail}); "
            "RayTracePy's core loops are all @njit"
        )

    warnings: list[str] = []
    if toolchain["setuptools_version"] is None:
        warnings.append("setuptools is not installed; building from source may fail")
    elif toolchain["setuptools_meets_minimum"] is False:
        warnings.append(
            f"setuptools {toolchain['setuptools_version']} is older than 42"
        )
    if toolchain["wheel_version"] is None:
        warnings.append("wheel is not installed")
    if toolchain["compilers"] == {}:
        warnings.append(
            "no C compiler found; any dependency without a prebuilt wheel "
            "cannot be built from source"
        )
    warnings.extend(
        f"important dependency missing: {name}" for name in missing[SEVERITY_IMPORTANT]
    )

    return {
        "architecture": system["machine"],
        "system": system["system"],
        "python_version": python_info["python_version"],
        "blocked": bool(blockers),
        "blockers": blockers,
        "warnings": warnings,
        "missing_required": missing[SEVERITY_REQUIRED],
        "missing_important": missing[SEVERITY_IMPORTANT],
        "missing_optional": missing[SEVERITY_OPTIONAL],
        "jit_status": jit.get("status") if jit is not None else "not probed",
        "verdict": "BLOCKED" if blockers else ("OK_WITH_WARNINGS" if warnings else "OK"),
    }


def render_report(report: dict[str, Any]) -> str:
    """Render the human-readable text report."""

    lines: list[str] = []
    add = lines.append
    rule = "=" * 78

    system = report["system_info"]
    python_info = report["python"]
    toolchain = report["toolchain"]
    summary = report["summary"]

    add(rule)
    add("RayTracePy native Python environment check")
    add(rule)
    add(f"label        : {report['label']}")
    add(f"captured     : {report['captured_at_utc']}")
    add(f"hostname     : {system['hostname']}")
    add(f"project root : {PROJECT_ROOT}")
    add(f"script       : {Path(__file__).resolve()}")
    add("")

    add("-- System " + "-" * 68)
    add(f"system       : {system['system']}")
    add(f"release      : {system['release']}")
    add(f"machine      : {system['machine']}   <- CPU architecture")
    add(f"processor    : {system['processor']}")
    add(f"platform     : {system['platform']}")
    add(f"64-bit       : {system['is_64bit']}")
    add(f"cpu count    : {system['cpu_count']}")
    if system.get("uname"):
        add(f"uname        : {system['uname']}")
    if system.get("libc"):
        add(f"libc         : {system['libc']}")
    add("")

    if report["platform_files"]:
        add("-- Platform evidence (/etc, /proc) " + "-" * 44)
        for name, text in report["platform_files"].items():
            add(f"{name}:")
            for line in text.splitlines():
                add(f"    {line}")
        add("")

    add("-- Python " + "-" * 68)
    add(f"implementation: {python_info['python_implementation']}")
    add(f"version      : {python_info['python_version']}")
    add(f"cache tag    : {python_info['python_cache_tag']}   <- Python 3.10 reports cpython-310")
    add(f"debug build  : {python_info['python_debug_build']}")
    add(f"executable   : {python_info['python_executable']}")
    add(f"prefix       : {python_info['python_prefix']}")
    add(f"virtualenv   : {python_info['in_virtualenv']}")
    add(f"sysconfig    : {python_info['pyspec_platform']}")
    if python_info.get("host_gnu_type"):
        add(f"host gnu type: {python_info['host_gnu_type']}   <- target triple for C extensions")
    add(
        "requires>=3.10: "
        f"{python_info['satisfies_python_requires']}"
    )
    add("")

    add("-- Toolchain " + "-" * 65)
    if toolchain["pip_available"]:
        add(f"pip          : present {toolchain['pip_version']}")
        if toolchain["pip_path"]:
            add(f"pip path     : {toolchain['pip_path']}")
    else:
        add("pip          : MISSING")
    add(f"setuptools   : {toolchain['setuptools_version']} (>=42: {toolchain['setuptools_meets_minimum']})")
    add(f"wheel        : {toolchain['wheel_version']}")
    add(f"venv         : {toolchain['venv_importable']}")
    add(f"ensurepip    : {toolchain['ensurepip_importable']}")
    add(f"ssl          : {toolchain['ssl_available']} ({toolchain['openssl_version']})")
    add(f"make         : {toolchain['make_available']}")
    if toolchain["pip_index_url"]:
        add(f"PIP_INDEX_URL: {toolchain['pip_index_url']}")
    if toolchain["compilers"]:
        for name, detail in toolchain["compilers"].items():
            add(f"compiler     : {name} -> {detail}")
    else:
        add("compiler     : none found")
    if report.get("network"):
        add(f"network      : {report['network']}")
    add("")

    add("-- Dependency probes " + "-" * 57)
    for probe in report["probes"]:
        severity = probe["severity"]
        name = probe["import_name"]
        status = probe["status"]
        if status == "present":
            version = probe.get("distribution_version") or probe.get("module_version")
            detail = f"present {version or 'unknown version'}"
            if probe.get("file"):
                detail += f"  {probe['file']}"
        else:
            detail = (
                f"MISSING {probe.get('error_type', '')} "
                f"{probe.get('message', '')}"
            ).strip()
        add(f"[{severity:<8}] {name:<12} {detail}")
    add("")

    raytracepy = report["raytracepy"]
    add("-- RayTracePy " + "-" * 64)
    add(f"distribution : {raytracepy['distribution_version']}")
    for key in ("from_environment", "from_source_tree"):
        outcome = raytracepy[key]
        detail = outcome.get("status")
        if outcome.get("status") != "present":
            detail = f"{detail} {outcome.get('error_type', '')} {outcome.get('message', '')}"
        add(f"{key:<13}: {detail.strip()}")
    if report["windows_only_artifacts"]:
        add(
            "windows-only : "
            + ", ".join(report["windows_only_artifacts"])
            + "   <- not usable on a non-Windows/AMD64 host"
        )
    add("")

    jit = report.get("jit")
    if jit is not None:
        add("-- Numba JIT execution " + "-" * 55)
        if jit.get("status") == "present":
            add(
                "njit compile+call: OK   "
                f"returned {jit.get('jit_result')}   numba {jit.get('numba_version')}"
            )
            add(f"DISABLE_JIT  : {not jit.get('jit_enabled', True)}")
        else:
            add(
                "njit compile+call: FAILED "
                f"{jit.get('error_type', '')} {jit.get('message', '')}"
            )
            if jit.get("returncode") is not None:
                add(f"returncode   : {jit['returncode']}")
            add(
                "note         : import succeeds but the compiled kernel cannot "
                "be called, so @njit code paths are unusable"
            )
        add("")

    add("-- Summary " + "-" * 67)
    add(f"architecture : {summary['architecture']}")
    add(f"python       : {summary['python_version']}")
    add(f"jit          : {summary['jit_status']}")
    for label, key in (
        ("blockers", "blockers"),
        ("warnings", "warnings"),
    ):
        items = summary[key]
        if items:
            for item in items:
                add(f"{label:<12} : {item}")
        else:
            add(f"{label:<12} : none")
    add("")
    add(f"RESULT       : {summary['verdict']}")
    add(rule)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def valid_label(text: str) -> str:
    """Reject labels that could escape ``artifacts/environment``.

    The label becomes a path component, so ``../../x`` would otherwise write the
    report anywhere the process can reach.
    """

    if not LABEL_PATTERN.match(text):
        raise argparse.ArgumentTypeError(
            "label must start with a letter or digit and contain only "
            "letters, digits, dot, dash and underscore"
        )
    return text


def positive_finite_seconds(text: str) -> float:
    """Parse a strictly positive, finite number of seconds.

    ``subprocess`` decides whether a deadline has passed with a ``>``
    comparison.  A NaN or infinite timeout makes that comparison useless, so a
    probe that hangs -- exactly the case this checker exists to report -- would
    hang the checker instead, and the traceback would take the report with it.
    Zero is rejected too: it says nothing useful and only races the process
    start.
    """

    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a positive, finite number of seconds"
        )
    return value


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report whether this machine can host RayTracePy, and write a "
            "comparable environment.json next to the reference baseline."
        ),
        epilog=(
            "exit status: 0 = OK or OK_WITH_WARNINGS, 1 = BLOCKED (including a "
            "report that could not be written), 2 = bad arguments"
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        type=valid_label,
        help=(
            "name of the environment being checked, used for the output "
            "directory (default: derived from the OS and Python version)"
        ),
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        default=None,
        help="explicit path for environment.json",
    )
    destination.add_argument(
        "--no-write",
        action="store_true",
        help="print the report only; do not write any file",
    )
    parser.add_argument(
        "--check-network",
        action="store_true",
        help="also probe the package index over the network (off by default)",
    )
    parser.add_argument(
        "--probe-timeout",
        type=positive_finite_seconds,
        default=180.0,
        help=(
            "seconds allowed per import probe (default: 180).  Applied "
            "separately to each probe, so the worst case is about 11 times "
            "this value."
        ),
    )
    parser.add_argument(
        "--boot-banner",
        action="store_true",
        help="prefix the report with the interpreter banner for the log",
    )
    return parser.parse_args(argv)


def build_report(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run every collector and assemble the report document."""

    label = arguments.label or default_label()

    system = collect_system()
    python_info = collect_python()
    toolchain = collect_toolchain()
    probes = collect_probes(arguments.probe_timeout)
    raytracepy = collect_raytracepy(arguments.probe_timeout)

    numba_present = any(
        probe["import_name"] == "numba" and probe.get("status") == "present"
        for probe in probes
    )
    jit = collect_jit_probe(arguments.probe_timeout, numba_present)

    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "script": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "system_info": system,
        "platform_files": collect_platform_files(),
        "python": python_info,
        "toolchain": toolchain,
        "probes": probes,
        "jit": jit,
        "raytracepy": raytracepy,
        "windows_only_artifacts": collect_windows_only_artifacts(),
    }

    if arguments.check_network:
        report["network"] = check_network(timeout=15.0)

    # Flatten the fields the reference environment.json already uses so the two
    # files can be compared key by key.
    report["packages"] = {
        name: distribution_version(name) for name in COMPARABLE_PACKAGES
    }
    report.update(
        {
            "git_commit": collect_git_commit(),
            "platform": system["platform"],
            "system": system["system"],
            "release": system["release"],
            "machine": system["machine"],
            "processor": system["processor"],
            "python_implementation": python_info["python_implementation"],
            "python_version": python_info["python_version"],
            "python_executable": python_info["python_executable"],
            "raytracepy_import_path": (
                raytracepy["from_source_tree"].get("file")
                or raytracepy["from_environment"].get("file")
            ),
        }
    )
    report["summary"] = summarize(system, python_info, toolchain, probes, jit)
    return report


def emit(text: str) -> bool:
    """Write ``text`` to stdout, tolerating a console that cannot show it.

    Probe output is decoded with ``errors="replace"``, so the report itself can
    contain U+FFFD; a pipe can also be closed at any moment.  Neither is a good
    reason to lose the JSON document and the exit status, which are the parts
    other people consume.
    """

    try:
        print(text)
    except (OSError, UnicodeError) as exc:
        try:
            print(
                f"WARNING: could not print to stdout: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        except (OSError, UnicodeError):
            pass
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    # The report contains text decoded from the platform with errors="replace",
    # so it can hold characters the console encoding cannot represent (U+FFFD on
    # a legacy code page, for instance).  Printing must never be able to destroy
    # the run before the JSON is written.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

    if arguments.boot_banner:
        emit(f"# interpreter: {sys.executable}")
        emit(f"# banner: {sys.version.splitlines()[0]}")
        emit("")

    report = build_report(arguments)

    exit_code = 1 if report["summary"]["blocked"] else 0

    emit(render_report(report))

    if not arguments.no_write:
        output = arguments.output or (
            DEFAULT_OUTPUT_ROOT / report["label"] / "environment.json"
        )
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError as exc:
            # The report claims to have produced a JSON document; failing to do
            # so must not look like success to a caller.
            emit(f"ERROR: could not write {output}: {exc}")
            exit_code = 1
        else:
            emit(f"JSON written to {output}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
