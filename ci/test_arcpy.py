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
"""Script testing the arcpy module"""

import pickle
import sys

import pytest

from ci.script_utils import arcpy_path, s3_env
from sertit import arcpy, ci, misc, rasters, vectors

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


def test_gdb_raster(tmp_path):
    gdb_path = arcpy_path() / "Default.gdb"
    rio_path = arcpy.from_gdb_raster_to_rio_path(
        str(gdb_path / "Segmented_202604151030385081819")
    )

    assert rio_path.endswith("Default.gdb:Segmented_202604151030385081819")
    assert rio_path.startswith("OpenFileGDB:")

    # This fails in docker sadly, for an unknown reason (both locally and on s3) -> see local fallback if needed
    # raster = rasters.read(rio_path)
    # ci.assert_val(raster.rio.width, 224, "Raster width")
    # ci.assert_val(raster.rio.height, 179, "Raster height")
    # ci.assert_val(raster.rio.count, 3, "Raster count")


# Fallback on disk test
@pytest.mark.skipif(misc.in_docker(), reason="Only works outside docker")
def test_gdb_raster_local(tmp_path):
    from cloudpathlib import AnyPath

    gdb_path = AnyPath(r"D:\_ARCGIS\DATA\vectorise\vectorise_test_2\Default.gdb")
    assert gdb_path.exists()
    raster = rasters.read(
        arcpy.from_gdb_raster_to_rio_path(
            str(gdb_path / "Segmented_202604151030385081819")
        )
    )
    ci.assert_val(raster.rio.width, 224, "Raster width")
    ci.assert_val(raster.rio.height, 179, "Raster height")
    ci.assert_val(raster.rio.count, 3, "Raster count")


@s3_env
def test_gdb_vector(tmp_path):
    gdb_path = arcpy_path() / "Default.gdb"
    assert gdb_path.exists()
    vect = vectors.read(gdb_path / "A1_area_of_interest_a")
    assert not vect.empty
