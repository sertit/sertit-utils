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
def get_or_create_dask_client(processes=False, env_vars=None):
    """
    Return default Dask client or create a local cluster and linked  client if not existing
    Returns:
    """
    client = None
    try:
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
            if env_vars:
                client.run(
                    lambda: os.environ.update({k: v for k, v in env_vars.items()})
                )

            yield client

        else:
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
    Get a dask lock with given name.
    This lock is used in rioxarray.to_raster (see https://corteva.github.io/rioxarray/stable/examples/dask_read_write.html)

    If a Multiple worker client exists: returns a distributed.Lock()
    Elif a Multithreaded client exists: returns a threading.Lock()
    else: returns None


    Args:
        name: The name of the lock
    Returns:
    """
    lock = None
    if is_dask_installed():
        from dask.distributed import Lock

        current_client = get_client()
        if current_client:
            lock = Lock(name, client=current_client)
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

        for tb in traceback.extract_stack()[::-1]:
            # Change this condition if we are using dask elsewhere than rasters.py and we want to display the tb
            if (
                tb.filename.lower().startswith("/home/data")
                and "/rasters.py" in tb.filename
            ):
                LOGGER.debug(
                    f"Computation number {self.total_computes}: {tb.line} | {tb.name} in {tb.filename} at line {tb.lineno}"
                )
                break

        # Raise or warn if too many computes have occurred
        if self.total_computes > self.max_computes:
            text = f"Too many computes. Total: {self.total_computes} > max: {self.max_computes}."
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
            text = f"Unexpected number of computes. Total: {self.total_computes} != {self.nof_computes}."

            if self.dont_raise:
                LOGGER.warning(text)
            else:
                raise RuntimeError(text)
        return True


@contextmanager
def raise_if_dask_computes(
    max_computes=0, nof_computes=None, dont_raise=False, force_synchronous=False
):
    # return a dummy context manager so that this can be used for non-dask objects
    if not is_dask_installed():
        yield contextlib.nullcontext()

    import dask

    scheduler = _CountingScheduler(
        max_computes=max_computes,
        nof_computes=nof_computes,
        dont_raise=dont_raise,
        force_synchronous=force_synchronous,
    )
    try:
        yield dask.config.set(scheduler=scheduler)
    finally:
        scheduler.check_total_nof_computes()
        # Make sure the counting scheduler is removed
        dask.config.set(scheduler=None)
