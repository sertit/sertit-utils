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
S3 tools
"""

import logging
import os
from contextlib import contextmanager
from functools import wraps

from cloudpathlib import S3Client

from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)

AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
"""
Environment variable linked to AWS Access Key ID.
"""

AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
"""
Environment variable linked to AWS Secret Access Key.
"""

AWS_S3_ENDPOINT = "AWS_S3_ENDPOINT"
"""
Environment variable linked to AWS endpoint.
"""

USE_S3_STORAGE = "USE_S3_STORAGE"
"""
Environment variable created to use Unistra's S3 bucket.
"""


def s3_env(*args, **kwargs):
    """
    Create S3 compatible storage environment
    You need to set endpoint url if you use s3 compatible storage
    since GDAL/Rasterio does not read endpoint url from config file.

    This function searches for S3 configuration in many places.
    It does apply configuration variables precedence, and you might have a use for it.
    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    1. AWS profile
    2. Given endpoint_url as function argument
    3. AWS environment variable

    Returns:
        Callable: decorated function

    Example:
        >>> from sertit.s3 import s3_env
        >>> from sertit import AnyPath
        >>> @s3_env(endpoint="s3.unistra.fr")
        >>> def file_exists(path: str):
        >>>     pth = AnyPath(path)
        >>>     print(pth.exists())
        >>> file_exists("s3://sertit-geodatastore/GLOBAL/COPDEM_30m/COPDEM_30m.vrt")
        True
    """
    import rasterio

    use_s3 = kwargs.get("use_s3_env_var", USE_S3_STORAGE)
    requester_pays = kwargs.get("requester_pays")
    no_sign_request = kwargs.get("no_sign_request")
    endpoint = os.getenv(AWS_S3_ENDPOINT, kwargs.get("endpoint"))
    profile_name = kwargs.get("profile_name")

    def decorator(function):
        @wraps(function)
        def s3_env_wrapper(*_args, **_kwargs):
            """S3 environment wrapper"""
            if int(os.getenv(use_s3, 1)):
                args_rasterio = {
                    "profile_name": profile_name,
                    "CPL_CURL_VERBOSE": False,
                    "GDAL_DISABLE_READDIR_ON_OPEN": False,
                    "AWS_NO_SIGN_REQUEST": "YES" if no_sign_request else "NO",
                    "AWS_REQUEST_PAYER": "requester" if requester_pays else None,
                }
                args_s3_client = {
                    "profile_name": profile_name,
                    "requester_pays": requester_pays,
                    "no_sign_request": no_sign_request,
                }
                args_s3_client.update(kwargs)

                if endpoint is not None:
                    args_rasterio["AWS_S3_ENDPOINT"] = endpoint
                    args_s3_client["endpoint_url"] = (
                        f"https://{endpoint}"  # cloudpathlib can read endpoint from config file
                    )

                # Define S3 client for S3 paths
                define_s3_client(**args_s3_client)
                os.environ[use_s3] = "1"
                LOGGER.info("Using S3 files")
                with rasterio.Env(**args_rasterio):
                    return function(*_args, **_kwargs)

            else:
                os.environ[use_s3] = "0"
                LOGGER.info("Using on disk files")
                return function(*_args, **_kwargs)

        return s3_env_wrapper

    return decorator


@contextmanager
def temp_s3(
    endpoint: str = None,
    profile_name: str = None,
    requester_pays: bool = False,
    no_sign_request: bool = False,
    **kwargs,
) -> None:
    """
    Initialize a temporary S3 environment as a context manager
    You need to set endpoint url if you use s3 compatible storage
    since GDAL/Rasterio does not read endpoint url from config file.

    This function searches for S3 configuration in many places.
    It does apply configuration variables precedence, and you might have a use for it.
    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    1. AWS profile
    2. Given endpoint_url as function argument
    3. AWS environment variable

    Args:
        endpoint: Endpoint to s3 path in the form s3.yourdomain.com
        profile_name: The name of your AWS profile
        requester_pays (bool): True if the endpoint says 'requester pays'
        no_sign_request (bool): True if the endpoint is open access

    Example:
        >>> from sertit.s3 import temp_s3
        >>> from sertit import AnyPath
        >>> def file_exists(path: str):
        >>>     with temp_s3(endpoint="s3.unistra.fr"):
        >>>         pth = AnyPath(path)
        >>>         print(pth.exists())
        >>> file_exists("s3://sertit-geodatastore/GLOBAL/COPDEM_30m/COPDEM_30m.vrt")
        True
    """
    import rasterio

    # Define S3 client for S3 paths
    try:
        args_rasterio = {
            "profile_name": profile_name,
            "CPL_CURL_VERBOSE": False,
            "GDAL_DISABLE_READDIR_ON_OPEN": False,
            "AWS_NO_SIGN_REQUEST": "YES" if no_sign_request else "NO",
            "AWS_REQUEST_PAYER": "requester" if requester_pays else None,
        }
        args_s3_client = {
            "profile_name": profile_name,
            "requester_pays": requester_pays,
            "no_sign_request": no_sign_request,
        }
        args_s3_client.update(kwargs)

        endpoint = os.getenv(
            AWS_S3_ENDPOINT, endpoint
        )  # Give the precedence to AWS_S3_ENDPOINT
        if endpoint is not None and endpoint != "":
            args_rasterio["AWS_S3_ENDPOINT"] = endpoint
            args_s3_client["endpoint_url"] = (
                f"https://{endpoint}"  # cloudpathlib can read endpoint from config file
            )

        with rasterio.Env(**args_rasterio):
            yield define_s3_client(**args_s3_client)
    finally:
        # Clean env
        S3Client().set_as_default_client()


def define_s3_client(
    endpoint_url=None,
    profile_name=None,
    requester_pays: bool = False,
    no_sign_request: bool = False,
    **kwargs,
):
    """
    Define S3 client
    This function searches for S3 configuration in many places.
    It does apply configuration variables precedence, and you might have a use for it.
    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    1. AWS profile
    2. Given endpoint_url as function argument
    3. AWS environment variable

    Args:
        endpoint_url: The endpoint url in the form https://s3.yourdomain.com
        profile_name: The name of the aws profile. Default to default profile in AWS configuration file.
        requester_pays (bool): True if the endpoint says 'requester pays'
        no_sign_request (bool): True if the endpoint is open access
    """

    endpoint_url = os.environ.get(AWS_S3_ENDPOINT)
    if endpoint_url is not None and endpoint_url != "":
        endpoint_url = kwargs.pop(
            "endpoint_url", f"https://{os.environ.get(AWS_S3_ENDPOINT)}"
        )

    aws_access_key_id = kwargs.pop("aws_access_key_id", os.getenv(AWS_ACCESS_KEY_ID))
    aws_secret_access_key = kwargs.pop(
        "aws_secret_access_key", os.getenv(AWS_SECRET_ACCESS_KEY)
    )
    if not no_sign_request:
        no_sign_request = kwargs.pop("no_sign_request", False)

    s3_client_args = [
        "aws_session_token",
        "botocore_session",
        "profile_name",
        "boto3_session",
        "file_cache_mode",
        "local_cache_dir",
        "boto3_transfer_config",
        "content_type_method",
        "extra_args",
    ]
    s3_client_kwargs = {key: kwargs.get(key) for key in s3_client_args if key in kwargs}

    if requester_pays:
        if "extra_args" in s3_client_kwargs:
            s3_client_kwargs["extra_args"].update({"RequestPayer": "requester"})
        else:
            s3_client_kwargs["extra_args"] = {"RequestPayer": "requester"}

    # ON S3
    args_s3_client = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "profile_name": profile_name,
        "no_sign_request": no_sign_request,
    }
    args_s3_client.update(s3_client_kwargs)

    if endpoint_url is not None and endpoint_url != "":
        args_s3_client["endpoint_url"] = endpoint_url

    client = S3Client(**args_s3_client)

    client.set_as_default_client()
