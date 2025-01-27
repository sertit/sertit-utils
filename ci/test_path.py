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
import tempfile

import pytest

from ci.script_utils import get_s3_ci_path
from sertit import AnyPath, ci, misc, path

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
