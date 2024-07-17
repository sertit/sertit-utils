# -*- coding: utf-8 -*-
# Copyright 2024, SERTIT-ICube - France, https://sertit.unistra.fr/
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
Unistra tools
"""
import logging
import os
from contextlib import contextmanager

from sertit import AnyPath, s3
from sertit.logs import SU_NAME
from sertit.s3 import temp_s3
from sertit.types import AnyPathType

LOGGER = logging.getLogger(SU_NAME)

UNISTRA_S3_ENPOINT = "s3.unistra.fr"
"""
Unistra S3 compatible storage endpoint: s3.unistra.fr
"""


def s3_env(*args, **kwargs):
    """
    Create Unistra's S3 compatible storage environment.
    You must export the variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environement. Y
    ou can use ready-to-use environements provided by the Sertit or asks for s3 credentials.

    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function

    Example:
        >>> from sertit.unistra import s3_env
        >>> from sertit import AnyPath
        >>> @s3_env
        >>> def file_exists(path: str):
        >>>     pth = AnyPath(path)
        >>>     print(pth.exists())
        >>> file_exists("s3://sertit-geodatastore/GLOBAL/COPDEM_30m/COPDEM_30m.vrt")
        True
    """
    return s3.s3_env(endpoint=UNISTRA_S3_ENPOINT)(*args, **kwargs)


@contextmanager
def unistra_s3() -> None:
    """
    Initialize a temporary S3 environment as a context manager, with Unistra endpoint

    Args:
        default_endpoint (str):Default Endpoint to look for

    Example:
        >>> from sertit.unistra import unistra_s3
        >>> from sertit import AnyPath
        >>> def file_exists(path: str):
        >>>     with unistra_s3():
        >>>         pth = AnyPath(path)
        >>>         print(pth.exists())
        >>> file_exists("s3://sertit-geodatastore/GLOBAL/COPDEM_30m/COPDEM_30m.vrt")
        True
    """
    try:
        with temp_s3(endpoint=UNISTRA_S3_ENPOINT):
            yield
    finally:
        pass


def define_s3_client():
    """
    Define Unistra's S3 client
    """
    return s3.define_s3_client(endpoint=UNISTRA_S3_ENPOINT)


def get_geodatastore() -> AnyPathType:
    """
    Get database directory.

    If the environment variable USE_S3_STORAGE=1, this function returns `AnyPath("s3://sertit-geodatastore")`.

    If USE_S3_STORAGE=0, it returns path to the DS2: `AnyPath("//ds2/database02/BASES_DE_DONNEES")` or `AnyPath(/home/ds2_db2/BASE_DE_DONNESS`)`

    Returns:
        AnyPath: Database directory

    Example:
        Don't set manually USE_S3_STORAGE with os.environ !

        >>> from sertit.unistra import get_geodatastore
        >>> import os
        >>> os.environ["USE_S3_STORAGE"] = "1"
        >>> print(get_geodatastore())
        s3://sertit-geodatastore

        >>> from sertit.unistra import get_geodatastore
        >>> import os
        >>> os.environ["USE_S3_STORAGE"] = "0"
        >>> print(get_geodatastore())
        //ds2/database02/BASES_DE_DONNEES/GLOBAL
    """
    if int(os.getenv(s3.USE_S3_STORAGE, 0)):
        # Define S3 client for S3 paths
        s3.define_s3_client()
        return AnyPath("s3://sertit-geodatastore")
    else:
        # on the DS2
        db_dir = AnyPath(r"//ds2/database02/BASES_DE_DONNEES")

        if not db_dir.is_dir():
            try:
                db_dir = AnyPath(get_db2_path(), "BASES_DE_DONNEES")
            except NotADirectoryError:
                db_dir = AnyPath("/home", "ds2_db2", "BASES_DE_DONNEES")

        if not db_dir.is_dir():
            raise NotADirectoryError("Impossible to open database directory !")

    return AnyPath(db_dir)


def get_mnt_path() -> str:
    """
    Return mounting directory :code:`/mnt`.

    .. WARNING::
        This won't work on Windows !

    Returns:
        str: Mounting directory

    Example:
        >>> get_mnt_path()
        '/mnt'
    """
    return r"/mnt"


def _get_db_path(db_nb=2) -> str:
    """
    Returns DSx database0x path

    - :code:`/mnt/ds2_dbx` when mounted (docker...)
    - :code:`//ds2/database0x` on windows
    """
    db_path = f"{get_mnt_path()}/ds2_db{db_nb}"

    if not os.path.isdir(db_path):
        db_path = rf"\\DS2\database0{db_nb}"

    if not os.path.isdir(db_path):
        raise NotADirectoryError(f"Impossible to open ds2/database0{db_nb}!")
    return db_path


def get_db2_path() -> str:
    """
    Returns DS2 database02 path

    - :code:`/mnt/ds2_db2` when mounted (docker...)
    - :code:`//ds2/database02` on windows

    Returns:
        str: Mounted directory

    Example:
        >>> get_db2_path()
        '/mnt/ds2_db2'
    """
    return _get_db_path(2)


def get_db3_path() -> str:
    """
    Returns DS2 database03 path

    - :code:`/mnt/ds2_db3` when mounted (docker...)
    - :code:`//ds2/database03` on windows

    Returns:
        str: Mounted directory

    Example:
        >>> get_db3_path()
        '/mnt/ds2_db3'
    """
    return _get_db_path(3)


def get_db4_path() -> str:
    """
    Returns DS2 database04 path

    - :code:`/mnt/ds2_db4` when mounted (docker...)
    - :code:`//ds2/database04` on windows

    Returns:
        str: Mounted directory

    Example:
        >>> get_db4_path()
        '/mnt/ds2_db4'
    """
    return _get_db_path(4)
