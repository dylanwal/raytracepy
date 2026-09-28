import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "reference" / "windows-python310"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_committed_reference_artifacts_match_expected_metrics():
    actual = json.loads(
        (ARTIFACTS_DIR / "single_light_metrics.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (FIXTURES_DIR / "single_light_reference_expected.json").read_text(
            encoding="utf-8"
        )
    )

    assert actual == expected


def test_committed_histogram_is_complete_and_has_expected_digest():
    expected = json.loads(
        (FIXTURES_DIR / "single_light_reference_expected.json").read_text(
            encoding="utf-8"
        )
    )["result"]
    with np.load(ARTIFACTS_DIR / "single_light_histogram.npz") as histogram_data:
        values = histogram_data["values"]

    assert list(values.shape) == expected["histogram_shape"]
    assert int(values.sum()) == expected["histogram_sum"]
    assert np.isfinite(values).all()
    digest = hashlib.sha256(
        np.asarray(values, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    assert digest == expected["histogram_sha256"]
