import contextlib
import logging
from contextlib import contextmanager

import psutil

from sertit import logs

LOGGER = logging.getLogger(logs.SU_NAME)


def get_client():
    client = None
    try:
        from dask.distributed import get_client

        with contextlib.suppress(ValueError):
            # Return default client
            client = get_client()
    except ModuleNotFoundError:
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
    try:
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

    except ModuleNotFoundError:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )
    finally:
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
    try:
        from dask.distributed import Lock

        if get_client():
            lock = Lock(name)
    except ModuleNotFoundError:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )
    return lock
