# Copyright 2025, SERTIT-ICube - France, https://sertit.unistra.fr/
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

import configparser
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from sertit import AnyPath, s3
from sertit.logs import SU_NAME
from sertit.s3 import USE_S3_STORAGE, temp_s3
from sertit.types import AnyPathType

LOGGER = logging.getLogger(SU_NAME)

UNISTRA_S3_ENDPOINT = "s3.unistra.fr"
"""
Unistra S3 compatible storage endpoint: s3.unistra.fr
"""

UNISTRA_S3_ENPOINT = UNISTRA_S3_ENDPOINT
# Legacy, to be removed in v2.0


def s3_env(*args, **kwargs):
    """
    Create Unistra's S3 compatible storage environment.

    This function searches for S3 configuration in many places.
    It does apply configuration variables precedence, and you might have a use for it.
    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    #. AWS profile
    #. AWS environment variable

    Profile unistra is first read from X:/SI/Secrets/config and X:/SI/Secrets/credentials.
    If this file does not exist, it fallbacks to local file $USER/.aws/config and $USER/.aws/credentials.

    You can use ready-to-use environements provided by the Sertit or asks for s3 credentials.

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
    _set_aws_file_path()
    use_s3 = kwargs.pop("use_s3_env_var", USE_S3_STORAGE)
    extra_args = {"profile_name": "unistra"} if does_unistra_profile_exist() else {}
    extra_args["endpoint"] = UNISTRA_S3_ENDPOINT
    return s3.s3_env(use_s3_env_var=use_s3, **extra_args)(*args, **kwargs)


@contextmanager
def unistra_s3() -> None:
    """
    Initialize a temporary S3 environment as a context manager, with Unistra endpoint

    This function searches for S3 configuration in many places.
    It does apply configuration variables precedence, and you might have a use for it.
    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    #. AWS profile "unistra"
    #. AWS environment variable

    Profile unistra is first read from X:/SI/Secrets/config and X:/SI/Secrets/credentials.
    If this file does not exist, it fallbacks to local file $USER/.aws/config and $USER/.aws/credentials.

    You can use ready-to-use environements provided by the Sertit or asks for s3 credentials.

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
    _set_aws_file_path()
    try:
        extra_args = {"profile_name": "unistra"} if does_unistra_profile_exist() else {}
        extra_args["endpoint"] = UNISTRA_S3_ENDPOINT
        with temp_s3(**extra_args):
            yield
    finally:
        pass


def define_s3_client():
    """
    Define Unistra's S3 client

    This function searches for S3 configuration in many places.
    It does apply configuration variables precedence, and you might have a use for it.
    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    #. AWS profile
    #. AWS environment variable

    Profile unistra is first read from X:/SI/Secrets/config and X:/SI/Secrets/credentials.
    If this file does not exist, it fallbacks to local file $USER/.aws/config and $USER/.aws/credentials.

    You can use ready-to-use environements provided by the Sertit or asks for s3 credentials.

    """
    _set_aws_file_path()
    profile_arg = {"profile_name": "unistra"} if does_unistra_profile_exist() else {}
    return s3.define_s3_client(endpoint=UNISTRA_S3_ENDPOINT, **profile_arg)


def get_geodatastore() -> AnyPathType:
    """
    Get database directory.

    If :any:`USE_S3_STORAGE` is set to ``1``, this function returns ``AnyPath("s3://sertit-geodatastore")``.

    If :any:`USE_S3_STORAGE` is set to ``0`` it returns:

    * ``AnyPath("//ds2/database02/BASES_DE_DONNEES")`` if code is running on Windows
    * ``AnyPath(/home/ds2_db2/BASE_DE_DONNESS`)`` if code is running in a Docker containers on Windows

    Returns:
        AnyPath: Database directory

    Example:
        Don't set manually ``USE_S3_STORAGE`` with ``os.environ`` !

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
        define_s3_client()
        return AnyPath("s3://sertit-geodatastore")
    else:
        try:
            db_dir = AnyPath(get_db2_path(), "BASES_DE_DONNEES")
        except NotADirectoryError:
            db_dir = AnyPath("/home", "ds2_db2", "BASES_DE_DONNEES")

        if not db_dir.is_dir():
            raise NotADirectoryError("Impossible to open database directory !")

    return AnyPath(db_dir)


def does_unistra_profile_exist() -> bool:
    """
    This function checks if "unistra" AWS profile exists.

    Returns: True if "unistra" AWS profile is present on the machine, False otherwise

    """
    config = configparser.ConfigParser()
    default_credentials_path = Path.home() / ".aws" / "credentials"
    credentials_path = os.getenv(
        "AWS_SHARED_CREDENTIALS_FILE", str(default_credentials_path)
    )
    credentials_path = AnyPath(credentials_path)
    if not credentials_path.exists():
        return False
    config.read(credentials_path)
    return "unistra" in config.sections()


def _set_aws_file_path() -> None:
    """
    This function sets AWS filepath config and credentials inside X network drive "X:/SI/Secrets/AWS"
    Returns:

    """
    config_file = AnyPath("X:") / "SI" / "Secrets" / "AWS" / "config"
    credentials_file = AnyPath("X:") / "SI" / "Secrets" / "AWS" / "credentials"
    if config_file.exists():
        os.environ["AWS_CONFIG_FILE"] = str(config_file)
    if credentials_file.exists():
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(credentials_file)


def get_mnt_path() -> str:
    """
    Return mounting directory :code:`/mnt`.

    Warnings:
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
    """
    db_path = f"{get_mnt_path()}/ds2_db{db_nb}"

    if not os.path.isdir(db_path):
        raise NotADirectoryError(f"Impossible to open {db_path}!")
    return db_path


def get_db2_path() -> str:
    """
    Returns DS2 database02 path

    - :code:`/mnt/ds2_db2` when mounted (docker...)

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

    Returns:
        str: Mounted directory

    Example:
        >>> get_db4_path()
        '/mnt/ds2_db4'
    """
    return _get_db_path(4)
