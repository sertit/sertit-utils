import contextlib
import logging
import os
from contextlib import contextmanager

import psutil
import xarray as xr

from sertit import logs
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


def is_dask_installed():
    try:
        from dask import optimize  # noqa: F401
        from dask.distributed import get_client  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def get_client():
    client = None
    if is_dask_installed():
        from dask.distributed import get_client

        with contextlib.suppress(ValueError):
            # Return default client
            client = get_client()
    else:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )

    return client


@contextmanager
def get_or_create_dask_client(processes=False):
    """
    Return default Dask client or create a local cluster and linked  client if not existing
    Returns:
    """
    client = None
    if is_dask_installed():
        from dask.distributed import Client, get_client

        try:
            # Return default client
            client = get_client()
        except ValueError:
            if processes:
                # Gather information to create a client adapted to the computer
                ram_info = psutil.virtual_memory()
                available_ram = ram_info.available / 1024 / 1024 / 1024
                available_ram = 0.9 * available_ram

                n_workers = 1
                memory_limit = f"{available_ram}Gb"
                if available_ram >= 16:
                    n_workers = available_ram // 16
                    memory_limit = f"{16}Gb"

                # Create a local cluster and return client
                LOGGER.warning(
                    f"Init local cluster with {n_workers} workers and {memory_limit} per worker"
                )
                client = Client(
                    n_workers=int(n_workers),
                    threads_per_worker=4,
                    memory_limit=memory_limit,
                )
            else:
                # Create a local cluster (threaded)
                LOGGER.warning("Init local cluster (threaded)")
                client = Client(
                    processes=processes,
                )

        yield client

    else:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )

    try:
        if client is not None:
            client.close()
    except Exception as ex:
        LOGGER.warning(ex)


def get_dask_lock(name):
    """
    Get a dask lock with given name. This lock uses the default client if existing;
    or create a local cluster (:py:func:`get_or_create_dask_client`) otherwise.
    Args:
        name: The name of the lock
    Returns:
    """
    lock = None
    if is_dask_installed():
        from dask.distributed import Lock

        if get_client():
            lock = Lock(name)
    else:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )
    return lock


def is_chunked(array: AnyXrDataStructure) -> bool:
    """
    Returns true if the array is still chunked.
    (i.e. its data is not computed, bnot loaded into memory as a numpy array)

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
    Returns true if the array is still chunked.
    (i.e. its data is not computed, bnot loaded into memory as a numpy array)

    Args:
        array (AnyXrDataStructure): Array to check

    Returns: True if array is still chunked
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
