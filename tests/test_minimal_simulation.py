import json
from pathlib import Path

import numpy as np


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_minimal_simulation_completes_with_finite_hits(smoke_run):
    simulation, ground, metrics = smoke_run

    assert simulation._run is True
    assert simulation.total_num_rays == 10_000
    assert ground.hits.ndim == 2
    assert ground.hits.shape[1] == 3
    assert ground.hits.shape[0] > 0
    assert np.isfinite(ground.hits).all()
    assert metrics["result"]["histogram_sum"] == ground.hits.shape[0]


def test_fixed_input_matches_reference_metrics(smoke_run):
    _, _, actual = smoke_run
    expected = json.loads(
        (FIXTURES_DIR / "single_light_smoke_expected.json").read_text(encoding="utf-8")
    )

    assert actual["schema_version"] == expected["schema_version"]
    assert actual["case"] == expected["case"]
    assert actual["seed"] == expected["seed"]
    assert actual["result"]["total_rays"] == expected["result"]["total_rays"]
    assert actual["result"]["hit_count"] == expected["result"]["hit_count"]
    assert actual["result"]["histogram_sum"] == expected["result"]["histogram_sum"]
    assert actual["result"]["histogram_shape"] == expected["result"]["histogram_shape"]
    assert actual["result"]["histogram_sha256"] == expected["result"]["histogram_sha256"]
    np.testing.assert_allclose(
        actual["result"]["hit_coordinate_mean"],
        expected["result"]["hit_coordinate_mean"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual["result"]["hit_coordinate_std"],
        expected["result"]["hit_coordinate_std"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_seeded_runs_are_repeatable(smoke_config):
    from scripts.generate_reference_baseline import calculate_metrics, run_simulation

    first_simulation, first_ground = run_simulation(smoke_config)
    second_simulation, second_ground = run_simulation(smoke_config)
    first = calculate_metrics(smoke_config, first_simulation, first_ground)
    second = calculate_metrics(smoke_config, second_simulation, second_ground)

    assert first["result"] == second["result"]
