from importlib.metadata import version


def test_raytracepy_import_and_version():
    import raytracepy

    assert version("raytracepy") == "0.0.1"
    assert raytracepy.__all__ == [
        "CirclePattern",
        "GridPattern",
        "OffsetGridPattern",
        "SpiralPattern",
        "Light",
        "Plane",
        "RayTrace",
    ]


def test_public_api_is_constructible():
    import raytracepy

    assert callable(raytracepy.Light)
    assert callable(raytracepy.Plane)
    assert callable(raytracepy.RayTrace)
