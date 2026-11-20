"""
Import rules for a base-only install, checked against the real environment.

Runs in the pytest_minimal job (bare python image, no extras).
Complements test_optional_imports.py, which simulates missing
dependencies but can't verify that a plain `pip install -e .`
actually yields a working package.

The file skips itself when any optional dependency is installed, so
the regular jobs (which collect all of ci/) and local full
environments ignore it. The job runs with --noconftest because
ci/conftest.py imports rasterio at module level.

Modules and guarded functions are discovered at runtime with
import_check, as in test_optional_imports.py; the expected import
failures come from expected_imports.MINIMAL.
"""

import importlib
import importlib.util

import pytest

from ci import expected_imports, import_check

# Optional deps that must be absent for the rules below to hold.
_OPTIONAL = ("rasterio", "geopandas", "geoutils", "click")

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is not None for name in _OPTIONAL),
    reason="requires a base-only install (no optional extras)",
)


def test_module_import_rules():
    """Every module imports cleanly, or fails with its curated message."""
    for module in import_check.modules():
        if module in expected_imports.MINIMAL:
            with pytest.raises(
                ModuleNotFoundError, match=expected_imports.MINIMAL[module]
            ):
                importlib.import_module(f"sertit.{module}")
        else:
            importlib.import_module(f"sertit.{module}")


def test_guarded_function_rules():
    """Every guarded function fails with the friendly pip install message."""
    for module in import_check.modules():
        if module in expected_imports.MINIMAL:
            continue
        try:
            guarded = list(import_check.guarded_calls(module))
        except ModuleNotFoundError:
            continue
        for _dotted, deps, fn in guarded:
            missing = next(
                (dep for dep, _ in deps if importlib.util.find_spec(dep) is None), None
            )
            if missing is None:
                continue
            extra = dict(deps)[missing]
            with pytest.raises(
                ImportError,
                match=f"{missing} is required.*pip install sertit\\[{extra}\\]",
            ):
                fn()
