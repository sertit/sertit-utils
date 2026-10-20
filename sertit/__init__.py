# -*- coding: utf-8 -*-
# Copyright 2026, SERTIT-ICube - France, https://sertit.unistra.fr/
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
**Source Code**: https://github.com/sertit/sertit-utils
"""

U_PATHLIB = "universal_pathlib"
CLD_PATHLIB = "cloudpathlib"

# Try first with Universal Pathlib
try:
    from upath import UPath

    AnyPath = UPath
    CLD_LB = U_PATHLIB

except ImportError:
    # If not found, try with cloudpathlib
    try:
        from cloudpathlib import AnyPath

        AnyPath = AnyPath
    except ImportError:
        CLD_LB = None


def use_universal_pathlib():
    return CLD_LB == U_PATHLIB


def use_cloudpathlib():
    return CLD_LB == CLD_PATHLIB


# flake8: noqa
from .__meta__ import (
    __version__,
)
