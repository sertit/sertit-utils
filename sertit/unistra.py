# -*- coding: utf-8 -*-
# Copyright 2023, SERTIT-ICube - France, https://sertit.unistra.fr/
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

from cloudpathlib import AnyPath

from sertit import s3
from sertit.logs import SU_NAME
from sertit.types import AnyPathType

LOGGER = logging.getLogger(SU_NAME)

UNISTRA_S3_ENPOINT = "s3.unistra.fr"
"""
Unistra S3 compatible storage endpoint: s3.unistra.fr
"""


def s3_env(*args, **kwargs):
    """
    Create Unistra's S3 compatible storage environment

    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
    """
    return s3.s3_env(default_endpoint=UNISTRA_S3_ENPOINT)(*args, **kwargs)


def define_s3_client():
    """
    Define Unistra's S3 client
    """
    return s3.define_s3_client(default_endpoint=UNISTRA_S3_ENPOINT)


def get_geodatastore() -> AnyPathType:
    """
    Get database directory in the DS2

    Returns:
        AnyPathType: Database directory
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

    .. code-block:: python

        >>> get_mnt_path()
        '/mnt'

    Returns:
        str: Mounting directory
    """
    return r"/mnt"


def _get_db_path(db_nb=2) -> str:
    """
    Returns DSx database0x path

    - :code:`/mnt/ds2_dbx` when mounted (docker...)
    - :code:`\\ds2\database0x` on windows
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
    - :code:`\\ds2\database02` on windows

    .. code-block:: python

        >>> get_db2_path()
        '/mnt/ds2_db2'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(2)


def get_db3_path() -> str:
    """
    Returns DS2 database03 path

    - :code:`/mnt/ds2_db3` when mounted (docker...)
    - :code:`\\ds2\database03` on windows

    .. code-block:: python

        >>> get_db3_path()
        '/mnt/ds2_db3'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(3)


def get_db4_path() -> str:
    """
    Returns DS2 database04 path

    - :code:`/mnt/ds2_db4` when mounted (docker...)
    - :code:`\\ds2\database04` on windows

    Returns:
        str: Mounted directory
    """
    return _get_db_path(4)
