import logging
from contextlib import contextmanager

import psutil

from sertit import logs

LOGGER = logging.getLogger(logs.SU_NAME)


@contextmanager
def get_or_create_dask_client(processes=False):
    """
    Return default Dask client or create a local cluster and linked  client if not existing
    Returns:
    """

    try:
        from dask.distributed import Client, get_client

        ram_info = psutil.virtual_memory()
        available_ram = ram_info.available / 1024 / 1024 / 1024
        available_ram = 0.9 * available_ram

        n_workers = 1
        memory_limit = f"{available_ram}Gb"
        if available_ram >= 16:
            n_workers = available_ram // 16
            memory_limit = f"{16}Gb"
        try:
            # Return default client
            yield get_client()
        except ValueError:
            if processes:
                # Create a local cluster and return client
                LOGGER.warning(
                    f"Init local cluster with {n_workers} workers and {memory_limit} per worker"
                )
                yield Client(
                    n_workers=int(n_workers),
                    threads_per_worker=4,
                    memory_limit=memory_limit,
                )
            else:
                # Create a local cluster (threaded)
                LOGGER.warning("Init local cluster (threaded)")
                yield Client(
                    processes=processes,
                )

    except ModuleNotFoundError:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )

    return None


def get_dask_lock(name):
    """
    Get a dask lock with given name. This lock uses the default client if existing;
    or create a local cluster (get_or_create_dask_client) otherwise.
    Args:
        name: The name of the lock
    Returns:
    """

    try:
        from dask.distributed import Lock

        return Lock(name)
    except ModuleNotFoundError:
        LOGGER.warning(
            "Can't import 'dask'. If you experiment out of memory issue, consider installing 'dask'."
        )
        return None
