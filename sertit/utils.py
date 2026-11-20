# Copyright 2026, SERTIT-ICube - France, https://sertit.unistra.fr/
# This file is part of sertit-utils project
#     https://github.com/sertit/sertit-utils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic utilities
"""

import functools
import importlib


def requires(*deps: tuple[str, str]):
    """
    Decorator that guards a function with one or more optional dependencies.

    The check is deferred: each dependency is only verified (by attempting to import it)
    when the decorated function is actually called. If any of them is missing, an
    :code:`ImportError` is raised with a hint on which pip extra to install; otherwise
    the function runs as usual.

    This lets a module be importable even when an optional dependency is not installed,
    while still failing fast with a clear message when a function that needs it is used.

    Args:
        *deps (tuple[str, str]): One or more :code:`(module, extra)` pairs. :code:`module`
            is the import name to check (e.g. :code:`"geoutils"`) and :code:`extra` is the
            pip extra to install if the import fails (e.g. :code:`"rasters"`).

    The wrapped function records the required dependencies, in check order, as the
    :code:`_sertit_requires` attribute. The CI import check (``ci/import_check.py``)
    discovers guarded functions through it, so new guards are tested automatically.

    Example:
        >>> @requires(("geoutils", "rasters"))
        ... def process(raster):
        ...     import geoutils as gu
        ...     return gu.open_raster(raster)
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for module, extra in deps:
                try:
                    importlib.import_module(module)
                except ImportError as exc:
                    raise ImportError(
                        f"{module} is required for {fn.__qualname__}(). "
                        f"Install it with: pip install sertit[{extra}]"
                    ) from exc
            return fn(*args, **kwargs)

        # used by the CI import check to discover guarded functions
        wrapper._sertit_requires = deps + getattr(fn, "_sertit_requires", ())

        return wrapper

    return deco
