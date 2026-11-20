"""
Import and call rules when optional dependencies are missing.

CI installs everything, so each scenario blocks a few imports in a
subprocess (see import_check) and checks two rules: every module
either imports or fails with its curated message, and every guarded
function fails with "pip install sertit[<extra>]".

Simulating per-dependency gaps here is cheaper than a venv matrix,
and a real base-only install can't isolate one dependency anyway.
The end-to-end install itself is covered by ci/minimal in its own
CI job.

Only the scenarios are listed below: the curated failure messages
live in expected_imports, and modules and guarded functions are
discovered at runtime.
"""

from functools import cache

import pytest

from ci import expected_imports, import_check

# scenario -> imports to block (one scenario per optional dependency)
SCENARIOS = {
    "no_click": ("click",),
    "no_colorlog": ("colorlog",),
    "no_dask": ("dask",),
    "no_dask_geopandas": ("dask_geopandas",),
    "no_geopandas": ("geopandas",),
    "no_geoutils": ("geoutils",),
    "no_odc_geo": ("odc_geo",),
    "no_pythonjsonlogger": ("pythonjsonlogger",),
    "no_rasterio": ("rasterio",),
    "no_rioxarray": ("rioxarray",),
    "no_s3fs": ("s3fs",),
    "no_sklearn": ("sklearn",),
    "no_xdem": ("xdem",),
}

# scenario -> modules expected to fail to import, with the curated
# message to check for (see expected_imports). Every other module
# must import cleanly.
IMPORT_RULES = {
    "no_rasterio": expected_imports.NO_RASTERIO,
    "no_rioxarray": expected_imports.NO_RIOXARRAY,
    "no_geopandas": expected_imports.NO_GEOPANDAS,
    "no_geoutils": expected_imports.NO_GEOUTILS,
    "no_click": expected_imports.NO_CLICK,
}


@cache
def _results(blocked: tuple[str, ...]):
    """Run the import check once per scenario, and cache its output."""
    return import_check.check(blocked)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_modules_import_rules(scenario):
    """Every module imports cleanly, or fails with its curated message."""
    imports, _ = _results(SCENARIOS[scenario])
    expected = set(import_check.modules())
    assert set(imports) == expected, (expected - set(imports), set(imports) - expected)

    expected_failures = IMPORT_RULES.get(scenario, {})
    for module, message in expected_failures.items():
        line = imports[module]
        assert line.startswith("ModuleNotFoundError"), (module, line)
        assert message in line, (module, line)

    for module, line in imports.items():
        if module not in expected_failures:
            assert line == "OK", (module, line)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_guarded_call_rules(scenario):
    """Every guarded function with a missing dependency fails friendly."""
    _, calls = _results(SCENARIOS[scenario])
    for dotted, (expected, result) in calls.items():
        if expected == "OK":
            continue
        module, extra = expected.split("|")
        assert result.startswith("ImportError"), (dotted, result)
        assert f"{module} is required" in result, (dotted, result)
        assert f"pip install sertit[{extra}]" in result, (dotted, result)
