# -*- coding: utf-8 -*-
# Copyright 2021, SERTIT-ICube - France, https://sertit.unistra.fr/
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
import os
from enum import unique

from sertit.misc import ListEnum


@unique
class Polarization(ListEnum):
    """SAR Polarizations"""

    hh = "HH"
    vv = "VV"
    vh = "VH"
    hv = "HV"


def get_proj_path():
    """Get project path"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_ci_data_path():
    """Get CI DATA path"""
    return os.path.join(get_proj_path(), "CI", "DATA")


RASTER_DATA = os.path.join(get_ci_data_path(), "rasters")
GEO_DATA = os.path.join(get_ci_data_path(), "vectors")
FILE_DATA = os.path.join(get_ci_data_path(), "files")
DISPLAY_DATA = os.path.join(get_ci_data_path(), "display")
