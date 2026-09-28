"""Generate a reproducible RayTracePy reference baseline.

The configuration mirrors ``examples/single/single_light.py``.  NumPy and
Numba use separate random-number states, so both are seeded before each run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numba
import numpy as np
import raytracepy as rpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "fixtures" / "single_light_reference_input.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reference" / "windows-python310"
PACKAGE_NAMES = (
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


@numba.njit(cache=False)
def _seed_numba(seed: int) -> None:
    """Seed Numba's RNG, which is independent from NumPy's Python RNG."""

    np.random.seed(seed)


def seed_random_generators(seed: int) -> None:
    """Reset every random-number generator used by RayTracePy."""

    np.random.seed(seed)
    _seed_numba(seed)


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a baseline input file."""

    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    required = {"schema_version", "seed", "plane", "light", "simulation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing baseline configuration keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError(f"Unsupported schema version: {config['schema_version']}")
    return config


def run_simulation(config: dict[str, Any]) -> tuple[rpy.RayTrace, rpy.Plane]:
    """Run the configured single-light simulation and return it with its plane."""

    seed_random_generators(int(config["seed"]))
    plane_config = config["plane"]
    light_config = config["light"]
    simulation_config = config["simulation"]

    ground = rpy.Plane(
        name=plane_config["name"],
        position=np.asarray(plane_config["position"], dtype="float64"),
        normal=np.asarray(plane_config["normal"], dtype="float64"),
        length=float(plane_config["length"]),
        width=float(plane_config["width"]),
        bins=tuple(plane_config["bins"]),
    )
    light = rpy.Light(
        name=light_config["name"],
        position=np.asarray(light_config["position"], dtype="float64"),
        direction=np.asarray(light_config["direction"], dtype="float64"),
        num_traces=int(light_config["num_traces"]),
        theta_func=int(light_config["theta_func"]),
    )
    simulation = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=int(simulation_config["total_num_rays"]),
        bounce_max=int(simulation_config["bounce_max"]),
    )
    simulation.run()
    return simulation, ground


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def calculate_metrics(
    config: dict[str, Any], simulation: rpy.RayTrace, ground: rpy.Plane
) -> dict[str, Any]:
    """Create deterministic numeric metrics suitable for cross-platform comparison."""

    hits = ground.hits
    histogram = ground.histogram.values
    canonical_histogram = np.asarray(histogram, dtype="<f8", order="C")
    total_rays = int(simulation.total_num_rays)
    hit_count = int(hits.shape[0])

    return {
        "schema_version": 1,
        "case": config["case"],
        "seed": int(config["seed"]),
        "result": {
            "total_rays": total_rays,
            "hit_count": hit_count,
            "miss_count": total_rays - hit_count,
            "hit_rate": hit_count / total_rays,
            "hit_coordinate_mean": _float_list(np.mean(hits, axis=0)),
            "hit_coordinate_std": _float_list(np.std(hits, axis=0)),
            "hit_coordinate_min": _float_list(np.min(hits, axis=0)),
            "hit_coordinate_max": _float_list(np.max(hits, axis=0)),
            "histogram_shape": list(histogram.shape),
            "histogram_sum": int(np.sum(histogram)),
            "histogram_mean": float(np.mean(histogram)),
            "histogram_std": float(np.std(histogram)),
            "histogram_min": float(np.min(histogram)),
            "histogram_max": float(np.max(histogram)),
            "histogram_percentiles": {
                str(percentile): float(np.percentile(histogram, percentile))
                for percentile in (1, 5, 10, 50, 90, 95, 99)
            },
            "histogram_sha256": hashlib.sha256(canonical_histogram.tobytes()).hexdigest(),
        },
        "comparison_policy": {
            "reference_environment": "exact deterministic comparison",
            "harmonyos_provisional_rtol": 1e-6,
            "harmonyos_provisional_atol": 1e-8,
            "note": "Recalibrate on HarmonyOS during week two using repeated-run evidence.",
        },
    }


def collect_environment() -> dict[str, Any]:
    """Capture the software and source versions that define this baseline."""

    package_versions = {
        name: importlib.metadata.version(name)
        for name in PACKAGE_NAMES
    }
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "git_commit": git_commit,
        "raytracepy_import_path": inspect.getfile(rpy),
        "packages": package_versions,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_baseline(
    input_path: Path,
    output_dir: Path,
    expected_output: Path | None,
    create_plot: bool,
    verify_repeat: bool,
) -> dict[str, Any]:
    """Run one baseline case and persist all reproducibility artifacts."""

    config = load_config(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_dir / "single_light_input.json")

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        simulation, ground = run_simulation(config)
    duration_seconds = time.perf_counter() - started
    metrics = calculate_metrics(config, simulation, ground)

    repeat_identical: bool | None = None
    if verify_repeat:
        repeated_simulation, repeated_ground = run_simulation(config)
        repeated_metrics = calculate_metrics(config, repeated_simulation, repeated_ground)
        repeat_identical = repeated_metrics["result"] == metrics["result"]
        if not repeat_identical:
            raise RuntimeError("Repeated baseline run produced different numeric results")

    write_json(output_dir / "single_light_metrics.json", metrics)
    write_json(output_dir / "environment.json", collect_environment())
    installed_packages = sorted({
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    })
    (output_dir / "installed-packages.txt").write_text(
        "\n".join(installed_packages) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "single_light_histogram.npz",
        values=ground.histogram.values,
        xedges=ground.histogram.xedges,
        yedges=ground.histogram.yedges,
    )

    if create_plot:
        simulation.plot_report(
            str(output_dir / "single_light_report.html"),
            auto_open=False,
            plot_rdf=True,
        )

    log_lines = [
        "RayTracePy reference baseline",
        f"input={input_path}",
        f"git_commit={subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()}",
        f"python={platform.python_version()}",
        f"seed={config['seed']}",
        f"total_rays={metrics['result']['total_rays']}",
        f"hit_count={metrics['result']['hit_count']}",
        f"hit_rate={metrics['result']['hit_rate']:.12f}",
        f"histogram_sha256={metrics['result']['histogram_sha256']}",
        f"duration_seconds={duration_seconds:.6f}",
        f"repeat_identical={repeat_identical}",
    ]
    for captured_warning in captured_warnings:
        warning_text = " ".join(str(captured_warning.message).splitlines())
        log_lines.append(
            f"warning={captured_warning.category.__name__}: {warning_text}"
        )
    (output_dir / "single_light_run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    if expected_output is not None:
        write_json(expected_output, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-output", type=Path)
    parser.add_argument("--plot", action="store_true", help="Generate the HTML visual report")
    parser.add_argument(
        "--verify-repeat",
        action="store_true",
        help="Run the case twice and require bit-for-bit identical metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = generate_baseline(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_output=args.expected_output.resolve() if args.expected_output else None,
        create_plot=args.plot,
        verify_repeat=args.verify_repeat,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
