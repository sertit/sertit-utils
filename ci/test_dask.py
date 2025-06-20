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
"""Script testing dask functions"""

from ci.script_utils import rasters_path, s3_env
from sertit import ci, dask, rasters

ci.reduce_verbosity()


@s3_env
def test_computed_lazy():
    """Test read function"""
    raster_path = rasters_path().joinpath("raster.tif")

    # Read already computed
    np_arr = rasters.read(raster_path, chunks=None)
    assert np_arr is not None
    assert not dask.is_chunked(np_arr)
    ci.assert_computed(np_arr)

    # Read lazy
    da_arr = rasters.read(raster_path)
    assert da_arr is not None
    assert dask.is_chunked(da_arr)
    ci.assert_chunked(da_arr)

    # Assert computed after compute
    ci.assert_computed(da_arr.compute())
