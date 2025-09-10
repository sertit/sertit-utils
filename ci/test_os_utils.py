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
"""Script testing the os_utils module"""

from sertit import ci, os_utils

ci.reduce_verbosity()


def test_str_containing_version():
    """Test str_containing_version function"""
    ci.assert_val(
        os_utils._compare_str_containing_version("qgis 3.34.5", "qgis 3.35.6"),
        -1,
        "Version <=",
    )
    ci.assert_val(
        os_utils._compare_str_containing_version("qgis 3.34.5", "qgis 3.34.5"),
        0,
        "Version ==",
    )
    ci.assert_val(
        os_utils._compare_str_containing_version("qgis 3.37.5", "qgis 3.35.6"),
        1,
        "Version >=",
    )


def test_qgis_bin():
    """Test qgis_bin function"""
    # We are on linux, this fct returns nothing
    ci.assert_val(os_utils.qgis_bin(), None, "Empty qgis_bin")


def test_gdalbuildvrt():
    """Test gdalbuildvrt function"""
    # We are on linux, this fct returns the name of the exe only
    ci.assert_val(os_utils.gdalbuildvrt_exe(), "gdalbuildvrt", "gdalbuildvrt on Linux")
