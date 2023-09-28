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
S3 tools
"""
import logging
import os
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
    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
    """
    import rasterio

    use_s3 = kwargs.get("use_s3_env_var", USE_S3_STORAGE)
    default_endpoint = kwargs.get("default_endpoint")

    def decorator(function):
        @wraps(function)
        def s3_env_wrapper(*_args, **_kwargs):
            """S3 environment wrapper"""
            if int(os.getenv(use_s3, 1)) and os.getenv(AWS_SECRET_ACCESS_KEY):
                # Define S3 client for S3 paths
                define_s3_client(default_endpoint)
                os.environ[use_s3] = "1"
                LOGGER.info("Using S3 files")
                with rasterio.Env(
                    CPL_CURL_VERBOSE=False,
                    AWS_VIRTUAL_HOSTING=False,
                    AWS_S3_ENDPOINT=os.getenv(AWS_S3_ENDPOINT, default_endpoint),
                    GDAL_DISABLE_READDIR_ON_OPEN=False,
                ):
                    function(*_args, **_kwargs)

            else:
                os.environ[use_s3] = "0"
                LOGGER.info("Using on disk files")
                function(*_args, **_kwargs)

        return s3_env_wrapper

    return decorator


def define_s3_client(default_endpoint=None):
    """
    Define S3 client
    """
    # ON S3
    client = S3Client(
        endpoint_url=f"https://{os.getenv(AWS_S3_ENDPOINT, default_endpoint)}",
        aws_access_key_id=os.getenv(AWS_ACCESS_KEY_ID),
        aws_secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY),
    )
    client.set_as_default_client()
