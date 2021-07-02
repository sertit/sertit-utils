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
""" Script testing the files """
import os
import shutil
import tempfile
from datetime import date, datetime

import numpy as np
import pytest
from cloudpathlib import AnyPath, CloudPath
from lxml import etree

from CI.SCRIPTS.script_utils import Polarization, files_path, s3_env
from sertit import ci, files, misc, vectors


def test_paths():
    """Test path functions"""
    curr_file = AnyPath(__file__).resolve()
    curr_dir = curr_file.parent
    with misc.chdir(curr_dir):
        # Relative path
        curr_rel_path = files.real_rel_path(curr_file, curr_dir)
        assert curr_rel_path == AnyPath(os.path.join(".", os.path.basename(__file__)))

        # Abspath
        abs_file = files.to_abspath(curr_rel_path)
        assert abs_file == curr_file

        with pytest.raises(FileNotFoundError):
            files.to_abspath("haha.txt")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = files.to_abspath(os.path.join(tmp_dir, "haha"))
            assert os.path.isdir(tmp)

        # Listdir abspath
        list_abs = files.listdir_abspath(curr_dir)
        assert curr_file in list_abs

        # Root path
        assert str(abs_file).startswith(str(files.get_root_path()))


def test_archive():
    """Test extracting functions"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Archives
        zip_file = files_path().joinpath("test_zip.zip")
        zip2_file = files_path().joinpath("test_zip.zip")  # For overwrite
        tar_file = files_path().joinpath("test_tar.tar")
        tar_gz_file = files_path().joinpath("test_targz.tar.gz")

        # Core dir
        core_dir = files_path().joinpath("core")
        folder = core_dir
        archives = [zip_file, tar_file, tar_gz_file, folder, zip2_file]

        # Extract
        extracted_dirs = files.extract_files(archives, tmp_dir, overwrite=True)
        files.extract_files([zip2_file], tmp_dir, overwrite=False)  # Already existing

        # Test
        for ex_dir in extracted_dirs:
            ci.assert_dir_equal(core_dir, ex_dir)

        # Archive
        archive_base = os.path.join(tmp_dir, "archive")
        for fmt in ["zip", "tar", "gztar"]:
            archive_fn = files.archive(
                folder_path=core_dir, archive_path=archive_base, fmt=fmt
            )
            out = files.extract_file(archive_fn, tmp_dir)
            if fmt == "zip":
                ci.assert_dir_equal(core_dir, out)
            else:
                # For tar and tar.gz, an additional folder is created because these formats dont have any file tree
                out_dir = files.listdir_abspath(out)[0]
                ci.assert_dir_equal(core_dir, out_dir)

            # Remove out directory in order to avoid any interferences
            files.remove(out)

        # Add to zip
        zip_out = archive_base + ".zip"
        core_copy = files.copy(core_dir, os.path.join(tmp_dir, "core2"))
        files.add_to_zip(zip_out, core_copy)

        # Extract
        unzip_out = os.path.join(tmp_dir, "out")
        files.extract_file(zip_out, unzip_out)

        # Test
        unzip_dirs = files.listdir_abspath(unzip_out)

        assert len(unzip_dirs) == 2
        ci.assert_dir_equal(unzip_dirs[0], unzip_dirs[1])


@s3_env
def test_archived_files():
    landsat_name = "LM05_L1TP_200030_20121230_20200820_02_T2_CI"
    ok_folder = files_path().joinpath(landsat_name)
    zip_file = files_path().joinpath(f"{landsat_name}.zip")
    tar_file = files_path().joinpath(f"{landsat_name}.tar")
    targz_file = files_path().joinpath(f"{landsat_name}.tar.gz")
    sz_file = files_path().joinpath(f"{landsat_name}.7z")

    # RASTERIO
    tif_name = "LM05_L1TP_200030_20121230_20200820_02_T2_QA_RADSAT.TIF"
    tif_regex = f".*{tif_name}"
    tif_zip = files.get_archived_rio_path(zip_file, tif_regex)
    tif_list = files.get_archived_rio_path(zip_file, tif_regex, as_list=True)
    tif_tar = files.get_archived_rio_path(tar_file, ".*RADSAT")
    tif_ok = ok_folder.joinpath(tif_name)
    ci.assert_raster_equal(tif_ok, tif_zip)
    ci.assert_raster_equal(tif_ok, tif_list[0])
    ci.assert_raster_equal(tif_ok, tif_tar)

    xml_name = "LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml"
    xml_ok_path = ok_folder.joinpath(xml_name)

    vect_name = "map-overlay.kml"
    vec_ok_path = ok_folder.joinpath(vect_name)

    # VECTORS
    if shutil.which("ogr2ogr"):  # Only works if ogr2ogr can be found.
        vect_regex = f".*{vect_name}"
        vect_zip = vectors.read(zip_file, archive_regex=vect_regex)
        vect_tar = vectors.read(tar_file, archive_regex=r".*overlay\.kml")
        vect_ok = vectors.read(vec_ok_path)
        assert not vect_ok.empty
        ci.assert_geom_equal(vect_ok, vect_zip)
        ci.assert_geom_equal(vect_ok, vect_tar)

    # XML
    if isinstance(files_path(), CloudPath):
        xml_ok_path = xml_ok_path.fspath
    else:
        xml_ok_path = str(xml_ok_path)

    xml_regex = f".*{xml_name}"
    xml_zip = files.read_archived_xml(zip_file, xml_regex)
    xml_tar = files.read_archived_xml(tar_file, r".*_MTL\.xml")
    xml_ok = etree.parse(xml_ok_path).getroot()
    ci.assert_xml_equal(xml_ok, xml_zip)
    ci.assert_xml_equal(xml_ok, xml_tar)

    # ERRORS
    with pytest.raises(TypeError):
        files.get_archived_rio_path(targz_file, tif_regex)
    with pytest.raises(TypeError):
        files.get_archived_rio_path(sz_file, tif_regex)
    with pytest.raises(FileNotFoundError):
        files.get_archived_rio_path(zip_file, "cdzeferf")
    with pytest.raises(TypeError):
        files.read_archived_xml(targz_file, xml_regex)
    with pytest.raises(TypeError):
        files.read_archived_xml(sz_file, xml_regex)
    with pytest.raises(FileNotFoundError):
        files.read_archived_xml(zip_file, "cdzeferf")


def test_get_file_name():
    """Test get_file_name"""
    file_name = files.get_filename(__file__)
    assert file_name == "test_files"
    file_name = files.get_filename(__file__ + "\\")
    assert file_name == "test_files"
    file_name = files.get_filename(__file__ + "/")
    assert file_name == "test_files"


def test_cp_rm():
    """Test CP/RM functions"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_tmp = os.listdir(tmp_dir)

        # Copy file
        curr_path = os.path.realpath(__file__)
        file_1 = files.copy(curr_path, tmp_dir)
        file_2 = files.copy(curr_path, os.path.join(tmp_dir, "test_pattern.py"))

        # Copy dir
        dir_path = os.path.dirname(curr_path)
        test_dir = files.copy(
            dir_path, os.path.join(tmp_dir, os.path.basename(dir_path))
        )

        # Test copy
        assert os.path.isfile(file_1)
        assert os.path.isfile(file_2)
        assert os.path.isdir(test_dir)

        # Remove file
        files.remove(file_1)
        files.remove("non_existing_file.txt")
        files.remove_by_pattern(tmp_dir, name_with_wildcard="*pattern*", extension="py")

        # Remove dir
        files.remove(test_dir)

        # Assert tempfile is empty
        assert os.listdir(tmp_dir) == empty_tmp


def test_find_files():
    """Test find_files"""
    names = os.path.basename(__file__)
    root_paths = AnyPath(__file__).parent
    max_nof_files = 1
    get_as_str = True

    # Test
    path = files.find_files(names, root_paths, max_nof_files, get_as_str)

    assert path == AnyPath(__file__)


def test_json():
    """Test json functions"""

    test_dict = {
        "A": 3,
        "C": "m2",  # Can be parsed as a date, we do not want that !
        "D": datetime.today(),
        "Dbis": date.today(),
        "E": np.int64(15),
        "F": Polarization.vv,
        "G": True,
        "H": AnyPath("/home/data"),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = os.path.join(tmp_dir, "test.json")

        # Save pickle
        files.save_json(json_file, test_dict)

        # Load pickle
        obj = files.read_json(json_file)

        assert (
            obj.pop("F") == test_dict.pop("F").value
        )  # Enum are stored following their value

        assert obj.pop("H") == str(
            test_dict.pop("H")
        )  # Enum are stored following their value
        assert obj == test_dict


def test_pickle():
    """Test pickle functions"""

    test_dict = {
        "A": 3,
        "B": np.zeros((3, 3)),
        "C": "str",
        "D": datetime.today(),
        "E": np.int64(15),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_file = os.path.join(tmp_dir, "test.pkl")

        # Save pickle
        files.save_obj(test_dict, pkl_file)

        # Load pickle
        obj = files.load_obj(pkl_file)

        # Test (couldn't compare the dicts as they contain numpy arrays)
        np.testing.assert_equal(obj, test_dict)


def test_get_file_in_dir():
    """Test get_file_in_dir"""
    # Get parent dir
    folder = os.path.dirname(os.path.realpath(__file__))

    # Test
    file = files.get_file_in_dir(
        folder, "file", ".py", filename_only=False, get_list=True, exact_name=False
    )
    filename = files.get_file_in_dir(
        folder,
        files.get_filename(__file__),
        "py",
        filename_only=True,
        get_list=False,
        exact_name=True,
    )

    assert file[0] == AnyPath(__file__)
    assert filename == os.path.basename(__file__)


def test_hash_file_content():
    """Test hash_file_content"""
    file_content = "This is a test."

    # Test
    hashed = files.hash_file_content(file_content)

    # Test
    assert hashed == "16c5bf1fc5"
