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
"""Performance module"""

import os

MAX_CORES = os.cpu_count() - 2
"""
Default maximum number of cores to use (total number of CPU minus 2).
"""

SU_MAX_CORE = "SERTIT_UTILS_MAX_CORES"
"""
Maximum core to use, default is total number of CPU minus 2.
You can update it with the environment variable :code:`SERTIT_UTILS_MAX_CORES`.
"""


def get_max_cores() -> int:
    """
    Retrieves the maximum number of CPU cores to be used by any function.

    This function determines the number of maximum allowable cores by first attempting
    to fetch the value from the environment variable `SERTIT_UTILS_MAX_CORES`. If the environment
    variable is not set, it defaults to the total number of available CPU cores minus 2.

    Returns:
        int: The maximum number of processor cores to be used.
    """
    return int(os.getenv(SU_MAX_CORE, MAX_CORES))
