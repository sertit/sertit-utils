# Sertit-Utils Technical Guide

Technical reference for the **Sertit-Utils** codebase: architecture, build & test, code style, CI/CD, pitfalls.
It is the "read this" target for the task-based navigation in [AGENTS.md](https://github.com/sertit/sertit-utils/blob/main/AGENTS.md).
Human contribution guidelines: [CONTRIBUTING.md](https://github.com/sertit/sertit-utils/blob/main/CONTRIBUTING.md).
AI-assisted contribution policy: [AI_POLICY.md](https://github.com/sertit/sertit-utils/blob/main/AI_POLICY.md).

Where this file and the code disagree, the code wins and this file should be fixed.

---

## Project Overview

**Sertit-Utils** is the shared Python ground under SERTIT's Earth observation projects.
It gathers the generic tools (paths, rasters, vectors, archives, XML, logging, dask, SNAP, S3)
that the higher-level libraries such as [EOReader](https://github.com/sertit/eoreader) and [EOSets](https://github.com/sertit/eosets) build on.

- **Language**: Python 3.11–3.14
- **Package name**: `sertit`
- **License**: Apache 2.0
- **Repository**: https://github.com/sertit/sertit-utils, default branch `main`
- **Author**: ICube-SERTIT (https://sertit.unistra.fr/)
---

## Key Files to Read First (by task)

| Task | Read Order |
|---|---|
| Understanding the optional-dependency contract | `sertit/utils.py` → `ci/expected_imports.py` → `ci/import_check.py` |
| Adding a new function | the module it belongs to → `ci/test_<module>.py` |
| Adding a raster function | `sertit/rasters.py` → `sertit/rasters_rio.py` → `ci/test_rasters.py` |
| Path / archive handling | `sertit/path.py` → `sertit/files.py` |
| Dask | `sertit/dask.py` |
| S3 / infrastructure | `sertit/s3.py` → `sertit/unistra.py` |
| SNAP | `sertit/snap.py` |
| Testing | `ci/conftest.py` → `ci/script_utils.py` → `ci/test_<module>.py` |
| CI/CD | `.gitlab-ci.yml` → `.github/workflows/` |

---

## Architecture

### Core Concepts

1. **Flat package**: one module per domain (paths, rasters, vectors, files, XML...), no class hierarchy. The `sertit/` directory is the source of truth for the module list.
2. **Optional-dependency contract**: with a base-only install (`pip install sertit`), every module imports, and every guarded function fails fast with a friendly `ImportError` naming the pip extra to install. Guards are declared with `@requires(("module", "extra"))` (`sertit/utils.py`); `ci/import_check.py` discovers them through the `_sertit_requires` marker, so a new guard is tested automatically.
3. **Lazy by default**: with dask enabled, raster functions keep the arrays chunked until the user computes. Tests enforce this with `CI_SERTIT_TEST_LAZY` and `assert_chunked_computed` (`ci/script_utils.py`); not-yet-lazy functions are marked with `is_not_lazy_yet`.
4. **Environment variables as configuration**: no config file. Behaviour-changing variables are declared with docstrings in the module that uses them; the docstrings are the source of truth and are rendered in the API docs.
5. **Cloud-ready paths**: anything that may point to S3 goes through `AnyPath` (re-exported from `sertit/__init__.py`). Never use raw `os.path` on such paths.
6. **Errors**: user-facing errors derive from `SertitException` (`sertit/exception.py`).

---

## Build & Test Setup

### Dev Environment

The installed set is described by `pyproject.toml` (base dependencies plus the extras `colorlog`, `geo`, `nearest_neighbors`, `dask`, `arcpy`, `full`) and `requirements.txt` (development):

```bash
# Conda
conda env create -f environment.yml
conda activate sertit
pip install -e .

# or Pip
pip install -r requirements.txt
pip install -e .
```

The `dev` extra installs the test and lint tools on top of the `full` extra: `pip install -e ".[dev]"`.

### Linting

`pre-commit` runs `ruff` (check + format) plus JSON/YAML/XML and large-file checks; the exact same hooks run in CI:

```bash
pre-commit install
pre-commit run --all-files
```

The ruff ruleset lives in `pyproject.toml` (`[tool.ruff.lint]`).

### Running the Tests

Tests live in `ci/`, one `test_*.py` file per module:

```bash
python -m pytest ci/test_rasters.py             # a single module
python -m pytest ci/test_rasters.py::test_read  # a single test
python -m pytest ci/minimal --noconftest        # base-only imports, anywhere
```

> [!IMPORTANT]
> The test suite is **not self-contained**: it reads test data from SERTIT infrastructure
> (S3 bucket `s3://sertit-ci/sertit_utils` or the local DB3 database, not redistributed).
> Most contributors cannot run the full suite locally; CI validates for you.
> Set `CI_SERTIT_USE_S3=0` to read the data from disk (DB3 layout `CI/sertit_utils/DATA/`).

---

## Testing

- `ci/`: one `test_*.py` per module, run on every push. `ci/minimal/`: base-only imports, run on a bare Python image without the optional extras.
- `ci/conftest.py`: shared fixtures. `ci/script_utils.py`: shared helpers (`dask_env`, `get_output`, `assert_chunked_computed`, `is_not_lazy_yet`) and the `CI_SERTIT_*` variables.
- `ci/expected_imports.py`: curated import-failure rules, asserted verbatim by `ci/test_optional_imports.py` and `ci/minimal/test_minimal_imports.py`.

Assertion style:

- Compare properties (shape, dtype, CRS, nodata, metadata) rather than full arrays when possible, reusing the `sertit.ci.assert_*` helpers.
- Raster tests run in both the numpy and the dask paths, multithreaded and with a local cluster (`dask_env`); outputs are written per backend with `get_output`.
- Laziness is asserted when `CI_SERTIT_TEST_LAZY=1`; not-yet-lazy functions are marked with `is_not_lazy_yet`.

---

## Code Style

### Copyright header

Every Python file starts with the Apache 2.0 header (year updated at each release):

```python
# Copyright 202x, SERTIT-ICube - France, https://sertit.unistra.fr/
# This file is part of sertit-utils project
#     https://github.com/sertit/sertit-utils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

New files include it. Do not touch the year in unrelated files.

### Docstrings & typing

- Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Example:`); public functions are rendered in the Sphinx API docs. `Example:` code must be runnable as a doctest.
- Type hints on every new function and method, including tests. `from __future__ import annotations` is used in the core modules; follow the local pattern.
- Public enumerations are `ListEnum` (`sertit.misc`); private helpers are prefixed with `_`.

### Imports

```python
from __future__ import annotations

import os  # stdlib
import numpy as np  # third-party
from cloudpathlib import AnyPath  # cloudpathlib

from sertit import path  # sertit submodules
```

---

## Key Patterns for Contributors

### Adding a New Function

1. Add it to the module that owns the domain; guard it with `@requires(("module", "extra"))` if it needs an optional dependency.
2. Google-style docstring, type hints, `SertitException` subtypes for user-facing errors.
3. Tests in `ci/test_<module>.py`, reusing the `ci/conftest.py` fixtures.
4. `CHANGES.md` entry with the right prefix.

### Adding or Changing an Optional Dependency

1. Declare the extra in `pyproject.toml` and the package in `requirements.txt`.
2. Guard the functions with `@requires` (discovered automatically by `ci/import_check.py`).
3. Update `ci/expected_imports.py` if an import-failure message changes (asserted verbatim).
4. Run `python -m pytest ci/test_optional_imports.py` and `python -m pytest ci/minimal --noconftest`.

### Adding an Environment Variable

Declare it with a docstring (name, default, behaviour; part of the API docs) in the module that uses it, then read it from `os.environ` with a default.

---

## CI/CD

GitHub is the primary repo; pushes are mirrored to SERTIT's GitLab, which runs the tests
(stages `lint` → `test` → `weekly_tests`, SERTIT `geo` containers):
the main suite with data on disk, the main suite with data read from S3,
the base-only imports on a bare Python image, and the newest-Python check on schedule.
The exact jobs are maintained in `.gitlab-ci.yml` (source of truth).

GitHub Actions: `test.yml` mirrors every push to GitLab and triggers CI;
`publish.yml` builds with uv and publishes to PyPI (trusted publishing) on tag `v*`.

---

## Release Procedure

Releases are made by maintainers (see [CONTRIBUTING.md § Release Sertit-Utils](https://github.com/sertit/sertit-utils/blob/main/CONTRIBUTING.md#release-sertit-utils));
conda-forge packaging follows automatically.

> [!CAUTION]
> **A coding agent must NEVER create a new release** (no tags, no version bumps, no publish workflow dispatches).
> See the hard limits in [AGENTS.md](https://github.com/sertit/sertit-utils/blob/main/AGENTS.md).

---

## Common Pitfalls & Gotchas

1. **Optional-dependency cascade**: geoutils imports rasterio and rioxarray at module level, so a missing one fails the geoutils import too. `ci/expected_imports.py` encodes these cascades; update it when a message changes.
2. **Base install**: `pytest_minimal` runs on a bare Python image. A module-level hard import of an optional dependency breaks it; the import suites catch it, the regular tests do not.
3. **Dask laziness**: never force materialisation in a function that must stay lazy; use `is_not_lazy_yet` for the exceptions, and keep them few.
4. **Cloud paths**: always `AnyPath`; S3 objects are byte-ranged, so reading a whole file from S3 when a window suffices is a bug.
5. **Test data**: do NOT commit test rasters, products, stacks, or derived outputs. `check-added-large-files` (1600 kB) is a backstop, not a license.
6. **SNAP**: changes touching `snap.py` need the `gpt` executable; without it, state so in the PR description and let CI validate.
7. **Copyright year**: updated project-wide at release time (`DOC:` commit), never as a side effect.

---

## Documentation

- **Source**: `docs/` (Sphinx + MyST, notebooks rendered by `myst_nb`). Build: `sphinx-build docs docs/_build/html -E` (needs `requirements-doc.txt`).
- **API docs**: autosummary from module docstrings. Online: https://sertit-utils.readthedocs.io/latest/
- `docs/history.md` includes `CHANGES.md`, which stays the source of truth.
