#!/usr/bin/env python3
"""Install the HarmonyOS PC dependency set for RayTracePy.

Verified end-to-end on a clean venv on the target device (2026-09-28); see
docs/harmonyos-pc/DEPENDENCY_MATRIX.md and docs/harmonyos-pc.md.

The order below is not optional:

  1. numpy              PyPI ships a musllinux aarch64 wheel
  2. llvmlite + numba   PyPI has no musllinux build; install the device-specific
                        aarch64 wheels first, otherwise pip builds from source
  3. remaining pins     requirements-harmonyos.txt fixes the whole transitive
                        set (a current dask would pull pyarrow and break the
                        datashader import)
  4. fixall.py          ELF metadata + musl compat + code signing; without a
                        .codesign section OpenHarmony refuses to dlopen the
                        extension ("Permission denied")
  5. add_stlshim.py     links libstlshim into numba's extensions (libstdc++ does
                        not export std::bad_array_new_length's constructor)
  6. raytracepy         editable install of this repository

The pip on the device is configured with an OpenHarmony mirror that does not
carry numpy, so the official index is used unless --index-url says otherwise.

    python scripts/install_dependencies_harmonyos.py
    python scripts/install_dependencies_harmonyos.py --dry-run
    python scripts/install_dependencies_harmonyos.py --skip-raytracepy
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements-harmonyos.txt"
HNP_BIN = Path("/data/service/hnp/bin")
DEFAULT_TOOLS = Path.home() / ".local" / "ohos-python-tools"
DEFAULT_INDEX = "https://pypi.org/simple"


def run(cmd: list, *, env: dict, dry_run: bool) -> None:
    printable = [str(part) for part in cmd]
    print("+", " ".join(printable), flush=True)
    if dry_run:
        return
    subprocess.run(printable, check=True, text=True, cwd=ROOT, env=env)


def pinned(requirements: Path, name: str) -> str:
    """Return the ``name==version`` pin declared in a requirements file."""

    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}=="):
            return line
    raise SystemExit(f"{requirements} has no pin for {name}")


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def main() -> int:
    p = argparse.ArgumentParser(description="Install HarmonyOS dependency pins.")
    p.add_argument("--requirements", type=Path, default=REQ)
    p.add_argument(
        "--tools-dir",
        type=Path,
        default=DEFAULT_TOOLS,
        help="holds fixall.py / add_stlshim.py and wheels/ (default: ~/.local/ohos-python-tools)",
    )
    p.add_argument("--index-url", default=DEFAULT_INDEX)
    p.add_argument("--skip-postprocess", action="store_true")
    p.add_argument("--skip-raytracepy", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if sys.version_info < (3, 10):
        raise SystemExit(f"need Python >= 3.10, got {sys.version}")
    if not args.requirements.is_file():
        raise SystemExit(f"missing {args.requirements}")

    env = dict(os.environ)
    env["PIP_INDEX_URL"] = args.index_url
    if HNP_BIN.is_dir():
        hnp = str(HNP_BIN)
        if hnp not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = hnp + os.pathsep + env.get("PATH", "")

    wheel_dir = args.tools_dir / "wheels"
    wheels = sorted(wheel_dir.glob("llvmlite-*.whl")) + sorted(
        wheel_dir.glob("numba-*.whl")
    )
    if len(wheels) < 2:
        raise SystemExit(
            f"missing the device-specific llvmlite/numba wheels under {wheel_dir}"
        )

    pip = [sys.executable, "-m", "pip", "install", "--index-url", args.index_url]
    if args.dry_run:
        pip.append("--dry-run")

    print("== 1/5 numpy ==", flush=True)
    run(pip + [pinned(args.requirements, "numpy")], env=env, dry_run=args.dry_run)

    print("== 2/5 llvmlite + numba (device wheels) ==", flush=True)
    run(pip + [str(wheel) for wheel in wheels], env=env, dry_run=args.dry_run)

    print("== 3/5 remaining pins ==", flush=True)
    run(pip + ["-r", str(args.requirements)], env=env, dry_run=args.dry_run)

    if not args.skip_postprocess:
        print("== 4/5 fixall (ELF metadata + musl compat + signing) ==", flush=True)
        run(
            [sys.executable, args.tools_dir / "fixall.py", site_packages()],
            env=env,
            dry_run=args.dry_run,
        )
        print("== 5/5 add_stlshim (numba extensions) ==", flush=True)
        run(
            [sys.executable, args.tools_dir / "add_stlshim.py", site_packages()],
            env=env,
            dry_run=args.dry_run,
        )

    if not args.skip_raytracepy:
        print("== raytracepy ==", flush=True)
        run(
            [sys.executable, "-m", "pip", "install", "-e", str(ROOT)],
            env=env,
            dry_run=args.dry_run,
        )

    print(flush=True)
    print("DONE. Verify with:", flush=True)
    print(f"  {sys.executable} scripts/check_environment.py", flush=True)
    print(
        f"  NUMBA_DISABLE_JIT=1 {sys.executable} -m pytest -q tests/",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
