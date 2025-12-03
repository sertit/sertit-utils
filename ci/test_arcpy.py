# -*- coding: utf-8 -*-
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
"""Script testing the arcpy module"""

import pickle
import sys

from sertit import arcpy, ci

ci.reduce_verbosity()


# flake8: noqa
def test_arcpy():
    """Test CI functions"""
    arcpy.init_conda_arcpy_env()
    import geopandas as gpd
    import rasterio
    from lxml import etree  # Should work


def test_create_conda_env_cli(tmp_path):
    """Test create_conda_env_cli function"""

    # Create a dumb fct
    def my_tool_core(a):
        return a + 2, "test"

    # Pickle paths
    in_pkl_path = tmp_path / "input.pkl"
    out_pkl_path = tmp_path / "output.pkl"

    # Add input for the fct in a pickle
    with open(in_pkl_path, "wb") as fp_in:
        pickle.dump({"a": 1}, fp_in)

    # Create CLI and run fct
    sys.argv = [__file__, "-i", str(in_pkl_path), "-o", str(out_pkl_path)]
    arcpy.create_conda_env_cli(standalone_mode=False)(my_tool_core)

    # Load output
    with open(out_pkl_path, "rb") as fp_out:
        out = pickle.load(fp_out)

    # Check output
    ci.assert_val(out, (3, "test"), "Test create_backend_cli")
