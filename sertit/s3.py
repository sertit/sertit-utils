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


def in_house_s3_configs():
    """
    This function declares in-house S3 configurations for rasterio, pyogrio and cloudpathlib.

    You can control how rasterio, pyogrio and cloudpathlib connects to S3 server by changing the input configurations bellow.
    Then, configuration files are translated for each library into their own configuration system.

    To get more information read the docstring of the function args_dict.

    Returns:

    """
    import rasterio

    args_rasterio_env = rasterio.env.getenv() if rasterio.env.hasenv() else {}
    args_rasterio_l = [
        {"kwargs": "profile_name"},
        {"name": "AWS_S3_ENDPOINT", "kwargs": "endpoint", "env": "AWS_S3_ENDPOINT"},
        {"name": "CPL_AWS_CREDENTIALS_FILE", "env": "AWS_SHARED_CREDENTIALS_FILE"},
        {"name": "AWS_CONFIG_FILE", "env": "AWS_CONFIG_FILE"},
        {
            "name": "CPL_CURL_VERBOSE",
            "default": args_rasterio_env.get("CPL_CURL_VERBOSE", False),
        },
        {
            "name": "GDAL_DISABLE_READDIR_ON_OPEN",
            "default": args_rasterio_env.get("GDAL_DISABLE_READDIR_ON_OPEN", False),
        },
        {"name": "AWS_NO_SIGN_REQUEST", "kwargs": "no_sign_request"},
        {"name": "AWS_REQUEST_PAYER", "kwargs": "requester_pays"},
    ]

    args_pyogrio_l = [
        {"name": "AWS_PROFILE", "kwargs": "profile_name", "env": "AWS_PROFILE"},
        {"name": "AWS_S3_ENDPOINT", "kwargs": "endpoint", "env": "AWS_S3_ENDPOINT"},
        {"name": "CPL_AWS_CREDENTIALS_FILE", "env": "AWS_SHARED_CREDENTIALS_FILE"},
        {"name": "AWS_CONFIG_FILE", "env": "AWS_CONFIG_FILE"},
        {"name": "AWS_NO_SIGN_REQUEST", "kwargs": "no_sign_request"},
        {"name": "AWS_REQUEST_PAYER", "kwargs": "requester_pays"},
    ]

    s3_client_args_l = [
        {
            "name": "endpoint_url",
            "kwargs": "endpoint",
            "env": "AWS_S3_ENDPOINT",
            "prefix": "https://",
        },
        {"kwargs": "aws_access_key_id", "env": "AWS_ACCESS_KEY_ID"},
        {"kwargs": "aws_secret_access_key", "env": "AWS_SECRET_ACCESS_KEY"},
        {"kwargs": "requester_pays"},
        {"kwargs": "no_sign_request", "default": False},
        {"kwargs": "profile_name", "env": "AWS_S3_PROFILE"},
        {"kwargs": "aws_session_token"},
        {"kwargs": "botocore_session"},
        {"kwargs": "boto3_session"},
        {"kwargs": "file_cache_mode"},
        {"kwargs": "local_cache_dir"},
        {"kwargs": "boto3_transfer_config"},
        {"kwargs": "content_type_method"},
        {"kwargs": "extra_args"},
    ]
    return {
        "cloudpathlib": s3_client_args_l,
        "pyogrio": args_pyogrio_l,
        "rasterio": args_rasterio_l,
    }


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
    2. Given arguments
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
    from pyogrio import set_gdal_config_options

    use_s3 = kwargs.get("use_s3_env_var", USE_S3_STORAGE)
    args = s3_args(*args, **kwargs)
    args_rasterio = args["rasterio"]
    args_pyogrio = args["pyogrio"]

    def decorator(function):
        @wraps(function)
        def s3_env_wrapper(*_args, **_kwargs):
            """S3 environment wrapper"""
            if int(os.getenv(use_s3, 1)):
                # Define S3 client for S3 paths
                define_s3_client(**kwargs)
                os.environ[use_s3] = "1"
                LOGGER.info("Using S3 files")
                set_gdal_config_options(args_pyogrio)

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
    2. Given arguments
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
    from pyogrio import set_gdal_config_options

    kwargs_cp = kwargs.copy()
    kwargs_cp["endpoint"] = endpoint
    kwargs_cp["profile_name"] = profile_name
    kwargs_cp["requester_pays"] = requester_pays
    kwargs_cp["no_sign_request"] = no_sign_request
    args = s3_args(**kwargs_cp)
    args_rasterio = args["rasterio"]
    args_pyogrio = args["pyogrio"]

    # Define S3 client for S3 paths
    try:
        set_gdal_config_options(args_pyogrio)
        with rasterio.Env(**args_rasterio):
            yield define_s3_client(
                endpoint=endpoint,
                profile_name=profile_name,
                requester_pays=requester_pays,
                no_sign_request=no_sign_request,
                **kwargs,
            )
    finally:
        # Clean env
        S3Client().set_as_default_client()


def define_s3_client(
    endpoint=None,
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
    2. Given arguments
    3. AWS environment variable

    Args:
        endpoint: The s3 endpoint (s3.yourdomain.com)
        profile_name: The name of the aws profile. Default to default profile in AWS configuration file.
        requester_pays (bool): True if the endpoint says 'requester pays'
        no_sign_request (bool): True if the endpoint is open access
    """
    kwargs_cp = kwargs.copy()
    kwargs_cp["endpoint"] = endpoint
    kwargs_cp["profile_name"] = profile_name
    kwargs_cp["requester_pays"] = requester_pays
    kwargs_cp["no_sign_request"] = no_sign_request

    args = s3_args(**kwargs_cp)
    args_s3_client = args["cloudpathlib"]
    client = S3Client(**args_s3_client)

    client.set_as_default_client()


def s3_args(*args, **kwargs) -> dict:
    """
    This function returns ready to use configurations for rasterio, pyogrio and cloudpathlib.
    For each in-house input configurations, it applies the function args_dict.

    Args:
        *args:
        **kwargs:

    Returns:

    """
    import rasterio

    in_house_configs = in_house_s3_configs()

    s3_client_args_l = in_house_configs["cloudpathlib"]
    args_pyogrio_l = in_house_configs["pyogrio"]
    args_rasterio_l = in_house_configs["rasterio"]

    s3_client_args = args_dict(s3_client_args_l, kwargs)
    args_pyogrio = args_dict(args_pyogrio_l, kwargs)

    args_rasterio = args_dict(args_rasterio_l, kwargs)
    args_rasterio_env = rasterio.env.getenv() if rasterio.env.hasenv() else {}
    args_rasterio_env.update(args_rasterio)

    return {
        "cloudpathlib": s3_client_args,
        "pyogrio": args_pyogrio,
        "rasterio": args_rasterio,
    }


def args_dict(args_l: list[dict], kwargs) -> dict:
    """
    This function converts a single in-house S3 configuration to a dictionary containing ready to use key/value parameters.

    The input is a list of dict to process. Each dictionary can contain the following key:
    - name: The name of the key in the output dict. If not given, the name of the key is the value of kwargs.
    - kwargs: The name of the kwargs argument whose value is taken to set the value in output dict.
    - env: The name of the envrionment variable whose value is taken to set the value in output dict.
    - prefix: A prefix prefixed to the value in the ouput dict.

    For example, if the environment variable "AWS_PROFILE" is unset, the following input:
    args_l = [{"name": "AWS_PROFILE", "kwargs": "profile_name", "env": "AWS_PROFILE"}], kwargs={"profile_name": "unistra"}
    will output:
    {"AWS_PROFILE": "unistra"}

    Here is the order of precedence from least to greatest
    (the last listed configuration variables override all other variables):

    1. Value from kwargs.
    2. Default value from key "default".
    3. Value from environment variable.

    If no value is found, the output dict will not contain the wanted parameter.

    Args:
        args_l: A list of parameters to extract. Each element will give a single key/value parameter.
        kwargs: Considered kwargs input to set parameters.

    Returns:

    """
    ret = {}
    for arg in args_l:
        arg_name = arg["name"] if "name" in arg else arg["kwargs"]

        # First, set with default value
        arg_value = arg.get("default")
        if arg_value is not None:
            ret[arg_name] = arg_value
        # Override with kwargs
        arg_value = kwargs.get(arg["kwargs"]) if arg.get("kwargs") is not None else None
        if arg_value is not None:
            ret[arg_name] = arg_value
        # Override with environment variable
        arg_value = os.getenv(arg.get("env")) if arg.get("env") is not None else None
        if arg_value is not None and arg_value != "":
            ret[arg_name] = arg_value

        if arg.get("prefix") is not None and ret.get(arg_name) is not None:
            ret[arg_name] = arg.get("prefix") + ret[arg_name]

    # Some exceptions
    if ret.get("AWS_NO_SIGN_REQUEST") is not None:
        ret["AWS_NO_SIGN_REQUEST"] = "YES" if ret["AWS_NO_SIGN_REQUEST"] else "NO"
    if ret.get("AWS_REQUEST_PAYER") is not None:
        ret["AWS_REQUEST_PAYER"] = "requester" if ret["AWS_REQUEST_PAYER"] else None

    if ret.get("requester_pays") is not None:
        if ret.get("extra_args") is not None:
            ret["extra_args"].update({"RequestPayer": "requester"})
        else:
            ret["extra_args"] = {"RequestPayer": "requester"}
        ret.pop("requester_pays")
    return ret
