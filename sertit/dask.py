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
"""Dask-related utilities (chunking, guarding, computation)."""

import contextlib
import logging
import os
from contextlib import contextmanager
from functools import wraps

import psutil
import xarray as xr

from sertit import logs, misc
from sertit.types import AnyXrDataStructure

LOGGER = logging.getLogger(logs.SU_NAME)

DEFAULT_CHUNKS = "auto"
""" Default chunks used in Sertit library (if dask is installed) """

SERTIT_DEFAULT_CHUNKS = "SERTIT_DEFAULT_CHUNKS"
"""
Environment variable to override default chunks.

Available keywords (case agnostic): 
- Give :code:`NONE` to set :code:`None`
- Give :code:`TRUE` to set :code:`True`
- Give :code:`AUTO` to set :code:`"auto"`
"""


def is_dask_installed() -> bool:
    """Is dask (with dask.distributed) installed?"""
    try:
        from dask import optimize  # noqa: F401
        from dask.distributed import get_client  # noqa: F401

        return True
    except ModuleNotFoundError:  # pragma: no cover
        return False


def get_client():
    """
    Get the default Dask client, if any.

    Returns:
        dask.distributed.Client | None: The default client, or None if dask is not installed
            or no client exists
    """
    client = None
    if is_dask_installed():
        from dask.distributed import get_client

        with contextlib.suppress(ValueError):
            # Return default client
            client = get_client()
    else:  # pragma: no cover
        LOGGER.warning(
            "Can't import 'dask'. If you experience out-of-memory issues, consider installing 'dask'."
        )

    return client


@contextmanager
def get_or_create_dask_client(
    processes=False,
    env_vars: dict | None = None,
    *,
    # processes parameters
    available_ram_pct: float = 0.9,
    worker_mem: float = 16,
    threads_per_worker: int = 4,
    **kwargs,
):
    """
    Return the default Dask client, or create a local cluster and a linked client if none exists.

    Args:
        processes (bool): If True, create a multi-process cluster. Otherwise, create a threaded cluster.
        env_vars (dict | None): Environment variables to set on the workers.
        available_ram_pct (float): Percentage of the available RAM to use when sizing the cluster.
        worker_mem (float): Memory limit per worker, in GB.
        threads_per_worker (int): Number of threads per worker.
        **kwargs: Other arguments passed to :code:`dask.distributed.Client`.

    Returns:
        dask.distributed.Client: The client (to be used in a :code:`with` block)
    """
    client = None
    try:
        if is_dask_installed():
            from dask.distributed import Client, get_client

            try:
                # Return default client
                client = get_client(
                    **misc.select_dict(
                        kwargs, ["address", "timeout", "resolve_address"]
                    )
                )
            except ValueError:
                if processes:
                    # Gather information to create a client adapted to the computer
                    ram_info = psutil.virtual_memory()
                    available_ram = ram_info.available / 1024 / 1024 / 1024
                    available_ram = available_ram_pct * available_ram

                    n_workers = 1
                    memory_limit = f"{available_ram}Gb"
                    if available_ram >= worker_mem:
                        n_workers = available_ram // worker_mem
                        memory_limit = f"{worker_mem}Gb"

                    # Create a local cluster and return client
                    LOGGER.warning(
                        f"Init local cluster with {n_workers} workers and {memory_limit} per worker"
                    )
                    client = Client(
                        n_workers=int(n_workers),
                        threads_per_worker=threads_per_worker,
                        memory_limit=memory_limit,
                        **kwargs,
                    )
                else:
                    # Create a local cluster (threaded)
                    LOGGER.warning("Init local cluster (threaded)")
                    client = Client(
                        processes=processes,
                        **kwargs,
                    )
            if env_vars:
                client.run(
                    lambda: os.environ.update({k: v for k, v in env_vars.items()})
                )

            yield client

        else:  # pragma: no cover
            LOGGER.warning(
                "Can't import 'dask'. If you experience out-of-memory issues, consider installing 'dask'."
            )
    finally:
        try:
            if client is not None:
                client.close()
        except Exception as ex:  # pragma: no cover
            LOGGER.warning(ex)


def get_dask_lock(name):
    """
    Get a Dask lock with the given name.
    This lock is used in rioxarray.to_raster (see https://vylor.github.io/rioxarray/stable/examples/dask_read_write.html)

    If a Dask client exists, returns a :code:`distributed.Lock()`, else :code:`None`.

    Args:
        name (str): The name of the lock

    Returns:
        dask.distributed.Lock | None: The lock, or None if dask is not installed or no client exists
    """
    lock = None
    if is_dask_installed():
        from dask.distributed import Lock

        current_client = get_client()
        if current_client:
            lock = Lock(name)
    else:  # pragma: no cover
        LOGGER.warning(
            "Can't import 'dask'. If you experience out-of-memory issues, consider installing 'dask'."
        )
    return lock


def is_chunked(array: AnyXrDataStructure) -> bool:
    """
    Returns true if the array is still chunked.
    (i.e. its data has not been computed and is not loaded into memory as a numpy array)

    Args:
        array (AnyXrDataStructure): Array to check

    Returns: True if array is still chunked
    """
    try:
        if isinstance(array, xr.DataArray):
            is_chunked = array.chunks is not None
        else:
            is_chunked = len(array.chunks) > 0
    except AttributeError:
        is_chunked = False
    return is_chunked


def is_computed(array: AnyXrDataStructure) -> bool:
    """
    Returns true if the array has been computed.
    (i.e. its data is loaded into memory as a numpy array, and not chunked)

    Args:
        array (AnyXrDataStructure): Array to check

    Returns: True if the array is computed
    """
    return not is_chunked(array)


def get_default_chunks():
    """
    Get the default chunks:

    - check if dask is available
    - check :code:`SERTIT_DEFAULT_CHUNKS` env variable
    - defaults on DEFAULT_CHUNKS
    """
    chunks = None
    if is_dask_installed():
        chunks = os.getenv(SERTIT_DEFAULT_CHUNKS, DEFAULT_CHUNKS)
        if chunks.lower() == "none":
            chunks = None
        elif chunks.lower() == "auto":
            chunks = "auto"
        elif chunks.lower() == "true":
            chunks = True

    return chunks


# From xarray: https://github.com/pydata/xarray/blob/30743945538ca2d276fc28eb221afa7bcb03978a/xarray/tests/__init__.py#L236-L259
class _CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    Reference: https://stackoverflow.com/questions/53289286/"""

    def __init__(
        self,
        max_computes=0,
        nof_computes=None,
        dont_raise=False,
        force_synchronous=False,
        name: str = "",
    ):
        self.total_computes = 0
        self.nof_computes = nof_computes

        if max_computes is None or (
            nof_computes is not None and max_computes < nof_computes
        ):
            max_computes = nof_computes

        self.max_computes = max_computes
        self.dont_raise = dont_raise
        self.debug = force_synchronous
        self.name = name

        # In case of debug, use the dask basic scheduler
        if self.debug:
            import dask

            self.get = dask.get
        else:
            self.get = None

    def __call__(self, dsk, keys, **kwargs):
        self.total_computes += 1

        # Log where the compute happens
        import traceback

        stack = traceback.extract_stack()
        for tb in stack[::-1]:
            # Change this condition if we are using dask elsewhere than rasters.py and we want to display the tb
            if (
                tb.filename.lower().split(os.sep)[:2]
                == __file__.lower().split(os.sep)[:2]
                and "rasters.py" in tb.filename
            ):  # pragma: no cover
                LOGGER.debug(
                    f"{self.name}: Computation number {self.total_computes}: {tb.line} | {tb.name} in {tb.filename} at line {tb.lineno}"
                )
                break

        # Raise or warn if too many computes have occurred
        if self.total_computes > self.max_computes:
            text = f"{self.name}: Too many computes. Total: {self.total_computes} > max: {self.max_computes}."
            if self.dont_raise:
                LOGGER.warning(text)
            else:
                raise RuntimeError(text)

        # Use the wanted get
        if self.get is None:
            client = get_client()
            if client:
                self.get = client.get

        if self.get is not None:
            return self.get(dsk, keys, **kwargs)

    def check_total_nof_computes(self):
        if self.nof_computes is not None and self.total_computes != self.nof_computes:
            text = f"{self.name}: Unexpected number of computes. Total: {self.total_computes} != Expected: {self.nof_computes}."

            if self.dont_raise:  # pragma: no cover
                LOGGER.warning(text)
            else:
                raise RuntimeError(text)
        return True


@contextmanager
def raise_if_dask_computes(
    max_computes=0,
    nof_computes=None,
    dont_raise=False,
    force_synchronous=False,
    name: str = "",
):
    """
    Context manager that counts the dask computes happening inside it
    and raises (or warns) if there are too many.

    If dask is not installed, yields a dummy context manager,
    so that this can also be used for non-dask objects.

    Args:
        max_computes (int): Maximum number of computes allowed before an error is raised
        nof_computes (int): Exact number of computes expected, checked once the context ends
        dont_raise (bool): If True, warn instead of raising
        force_synchronous (bool): If True, force the dask synchronous scheduler (useful for debugging)
        name (str): Name of the context, used in the logs and errors

    Returns:
        dask.config.Context: The dask config context with the counting scheduler set
    """
    # return a dummy context manager so that this can be used for non-dask objects
    if not is_dask_installed():  # pragma: no cover
        yield contextlib.nullcontext()

    import dask

    scheduler = _CountingScheduler(
        max_computes=max_computes,
        nof_computes=nof_computes,
        dont_raise=dont_raise,
        force_synchronous=force_synchronous,
        name=name if name else "Dask Context manager",
    )
    try:
        yield dask.config.set(scheduler=scheduler)
    finally:
        scheduler.check_total_nof_computes()
        # Make sure the counting scheduler is removed
        dask.config.set(scheduler=None)


def guard_compute(
    max_computes=0,
    nof_computes=0,
):
    """
    Decorator that guards a function against unexpected (and usually unintended) dask computes.

    When the :code:`CI_SERTIT_USE_DASK` environment variable is set to :code:`"1"` (the default),
    the wrapped function is run inside a :func:`raise_if_dask_computes` context so that every
    compute it triggers is tracked and bounded. Otherwise, it runs unchanged.

    Args:
        max_computes (int): Maximum number of computes allowed before an error is raised.
        nof_computes (int): Exact number of computes expected, checked once the function returns.
    """

    def guard_compute_decorator(function):

        @wraps(function)
        def guard_compute_wrapper(*_args, **_kwargs):
            """Guard compute wrapper"""
            raise_on_compute = os.getenv("CI_SERTIT_USE_DASK", "1") == "1"

            if raise_on_compute:
                LOGGER.info(
                    f"Guarding '{function.__name__}' against dask computes. Maximum number of computes allowed is 0."
                )
                with raise_if_dask_computes(
                    max_computes=max_computes,
                    nof_computes=nof_computes,
                    dont_raise=not raise_on_compute,
                    name=function.__name__,
                ):
                    return function(*_args, **_kwargs)
            else:
                return function(*_args, **_kwargs)

        return guard_compute_wrapper

    return guard_compute_decorator
