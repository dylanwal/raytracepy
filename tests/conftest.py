from pathlib import Path

import pytest

from scripts.generate_reference_baseline import calculate_metrics, load_config, run_simulation


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def smoke_config():
    return load_config(FIXTURES_DIR / "single_light_smoke_input.json")


@pytest.fixture(scope="session")
def smoke_run(smoke_config):
    simulation, ground = run_simulation(smoke_config)
    metrics = calculate_metrics(smoke_config, simulation, ground)
    return simulation, ground, metrics
