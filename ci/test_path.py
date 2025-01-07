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
"""Script testing the files"""

import os
import shutil
import tempfile

import pytest

from ci.script_utils import files_path, get_s3_ci_path, s3_env
from sertit import AnyPath, ci, misc, path, vectors

ci.reduce_verbosity()


def test_paths():
    """Test path functions"""
    curr_file = AnyPath(__file__).resolve()
    curr_dir = curr_file.parent
    with misc.chdir(curr_dir):
        # Relative path
        curr_rel_path = path.real_rel_path(curr_file, curr_dir)
        assert curr_rel_path == AnyPath(os.path.join(".", os.path.basename(__file__)))

        # Abspath
        abs_file = path.to_abspath(curr_rel_path)
        assert abs_file == curr_file

        with pytest.raises(FileNotFoundError):
            path.to_abspath("haha.txt", raise_file_not_found=True)

        # with not pytest.raises(FileNotFoundError):
        path.to_abspath("haha.txt", raise_file_not_found=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = path.to_abspath(os.path.join(tmp_dir, "haha"))
            assert os.path.isdir(tmp)

        # Listdir abspath
        list_abs = path.listdir_abspath(curr_dir)
        assert curr_file in list_abs

        # Root path
        assert str(abs_file).startswith(str(path.get_root_path()))

        # Writeable
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert path.is_writable(tmp_dir)  # Writeable

        assert not path.is_writable(get_s3_ci_path())  # Not writable
        assert not path.is_writable("cvfgbherth")  # Non-existing


@s3_env
def test_archived_paths():
    landsat_name = "LM05_L1TP_200030_20121230_20200820_02_T2_CI"
    ok_folder = files_path().joinpath(landsat_name)
    zip_file = files_path().joinpath(f"{landsat_name}.zip")
    tar_file = files_path().joinpath(f"{landsat_name}.tar")
    targz_file = files_path().joinpath(f"{landsat_name}.tar.gz")
    sz_file = files_path().joinpath(f"{landsat_name}.7z")

    # Archive file
    tif_name = "LM05_L1TP_200030_20121230_20200820_02_T2_QA_RADSAT.TIF"
    tif_ok = f"{ok_folder.name}/{tif_name}"
    tif_regex = f".*{tif_name}"
    assert tif_ok == path.get_archived_path(zip_file, tif_regex)
    assert tif_ok == path.get_archived_path(zip_file, tif_regex, as_list=True)[0]
    assert tif_ok == path.get_archived_path(tar_file, ".*RADSAT")

    # RASTERIO
    tif_zip = path.get_archived_rio_path(zip_file, tif_regex)
    tif_list = path.get_archived_rio_path(zip_file, tif_regex, as_list=True)
    tif_tar = path.get_archived_rio_path(tar_file, ".*RADSAT")
    tif_ok = ok_folder.joinpath(tif_name)
    ci.assert_raster_equal(tif_ok, tif_zip)
    ci.assert_raster_equal(tif_ok, tif_list[0])
    ci.assert_raster_equal(tif_ok, tif_tar)

    file_list = path.get_archived_file_list(zip_file)
    ci.assert_raster_equal(
        tif_ok, path.get_archived_rio_path(zip_file, tif_regex, file_list=file_list)
    )

    # VECTORS
    vect_name = "map-overlay.kml"
    vec_ok_path = ok_folder.joinpath(vect_name)
    if shutil.which("ogr2ogr"):  # Only works if ogr2ogr can be found.
        vect_regex = f".*{vect_name}"
        vect_zip = vectors.read(zip_file, archive_regex=vect_regex)
        vect_tar = vectors.read(tar_file, archive_regex=r".*overlay\.kml")
        vect_ok = vectors.read(vec_ok_path)
        assert not vect_ok.empty
        ci.assert_geom_equal(vect_ok, vect_zip)
        ci.assert_geom_equal(vect_ok, vect_tar)

    # ERRORS
    with pytest.raises(TypeError):
        path.get_archived_rio_path(targz_file, tif_regex)
    with pytest.raises(TypeError):
        path.get_archived_rio_path(sz_file, tif_regex)
    with pytest.raises(FileNotFoundError):
        path.get_archived_rio_path(zip_file, "cdzeferf")


def test_get_file_name():
    """Test get_file_name"""
    file_name = path.get_filename(__file__)
    assert file_name == "test_path"
    file_name = path.get_filename(__file__ + "\\")
    assert file_name == "test_path"
    file_name = path.get_filename(__file__ + "/")
    assert file_name == "test_path"

    file = "/fkjzeh-r_éfertg.tar.gz"
    assert file[1:] == path.get_filename(file) + "." + path.get_ext(file)

    # Multi point files
    fn = r"/test/HLS.L30.T42RVR.2022240T055634.v2.0.B01.tif"
    file_name = path.get_filename(fn)
    assert file_name == "HLS.L30.T42RVR.2022240T055634.v2.0.B01"

    fn = (
        r"/test/S3B_SL_1_RBT____20200909T104016_0179_043_165_2340_LN2_O_NT_004.SEN3.zip"
    )
    file_name = path.get_filename(fn)
    assert file_name == "S3B_SL_1_RBT____20200909T104016_0179_043_165_2340_LN2_O_NT_004"

    fn = r"/test/S2A_MSIL1C_20200824T110631_N0209_R137_T30TTK_20200824T150432.SAFE.zip"
    file_name = path.get_filename(fn)
    assert file_name == "S2A_MSIL1C_20200824T110631_N0209_R137_T30TTK_20200824T150432"

    fn = r"/test/S2A_MSIL1C_20200824T110631_N0209_R137_T30TTK_20200824T150432.SAFE.tar.gz.zip"
    file_name = path.get_filename(fn, other_exts=".gz.zip")
    assert file_name == "S2A_MSIL1C_20200824T110631_N0209_R137_T30TTK_20200824T150432"


def test_find_files():
    """Test find_files"""
    names = os.path.basename(__file__)
    root_paths = AnyPath(__file__).parent
    max_nof_files = 1
    get_as_str = True

    # Test
    found_path = path.find_files(names, root_paths, max_nof_files, get_as_str)

    assert found_path == AnyPath(__file__)


def test_get_file_in_dir():
    """Test get_file_in_dir"""
    # Get parent dir
    folder = os.path.dirname(os.path.realpath(__file__))

    # Test
    file = path.get_file_in_dir(
        folder, "path", ".py", filename_only=False, get_list=True, exact_name=False
    )
    filename = path.get_file_in_dir(
        folder,
        path.get_filename(__file__),
        "py",
        filename_only=True,
        get_list=False,
        exact_name=True,
    )

    assert file[0] == AnyPath(__file__)
    assert filename == os.path.basename(__file__)
