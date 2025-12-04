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
from datetime import date, datetime

import numpy as np
import pytest
from lxml import etree, html

from ci.script_utils import Polarization, files_path, s3_env
from sertit import AnyPath, ci, files, path, vectors

ci.reduce_verbosity()


@s3_env
def test_archive(tmp_path):
    """Test extracting functions"""
    # Archives
    zip_file = files_path().joinpath("test_zip.zip")
    zip2_file = files_path().joinpath("test_zip.zip")  # For overwrite
    zip_without_directory = files_path().joinpath("test_zip_without_directory.zip")
    tar_file = files_path().joinpath("test_tar.tar")
    tar_gz_file = files_path().joinpath("test_targz.tar.gz")

    # Core dir
    core_dir = files_path().joinpath("core")
    folder = core_dir
    archives = [
        zip_file,
        tar_file,
        tar_gz_file,
        folder,
        zip2_file,
        zip_without_directory,
    ]

    # Extract
    extracted_dirs = files.extract_files(archives, tmp_path, overwrite=True)
    files.extract_files([zip2_file], tmp_path, overwrite=False)  # Already existing

    # Test
    for ex_dir in extracted_dirs:
        ci.assert_dir_equal(core_dir, ex_dir)

    # Archive
    archive_base = os.path.join(tmp_path, "archive")
    for fmt in ["zip", "tar", "gztar"]:
        archive_fn = files.archive(
            folder_path=core_dir, archive_path=archive_base, fmt=fmt
        )
        out = files.extract_file(archive_fn, tmp_path)
        # an additional folder is created
        out_dir = path.listdir_abspath(out)[0]
        ci.assert_dir_equal(core_dir, out_dir)

        # Remove out directory in order to avoid any interferences
        files.remove(out)

    # Add to zip
    zip_out = zip2_file if path.is_cloud_path(zip2_file) else archive_base + ".zip"
    core_copy = files.copy(core_dir, os.path.join(tmp_path, "core2"))
    zip_out = files.add_to_zip(zip_out, core_copy)

    # Non existing
    with pytest.raises(FileNotFoundError):
        files.add_to_zip("not_existing.zip", core_copy)

    # Extract
    unzip_out = os.path.join(tmp_path, "out")
    unzip_out = files.extract_file(zip_out, unzip_out)

    # Test
    unzip_dirs = path.listdir_abspath(unzip_out)

    assert len(unzip_dirs) == 2
    ci.assert_dir_equal(unzip_dirs[0], unzip_dirs[1])


@s3_env
def test_archive_cloud(tmp_path):
    landsat_name = "LM05_L1TP_200030_20121230_20200820_02_T2_CI"
    folder = files_path().joinpath(landsat_name)
    arch_path = files.archive(folder, tmp_path / landsat_name)
    ci.assert_val(arch_path.name, landsat_name + ".zip", "archive name")


def test_archived_err(tmp_path):
    txt_path = tmp_path / "hello.txt"
    txt_path.write_text("test")

    with pytest.raises(TypeError):
        files.extract_file(txt_path, tmp_path)


@s3_env
def test_archived_files(tmp_path):
    landsat_name = "LM05_L1TP_200030_20121230_20200820_02_T2_CI"
    ok_folder = files_path().joinpath(landsat_name)
    zip_file = files_path().joinpath(f"{landsat_name}.zip")
    tar_file = files_path().joinpath(f"{landsat_name}.tar")
    targz_file = files_path().joinpath(f"{landsat_name}.tar.gz")
    sz_file = files_path().joinpath(f"{landsat_name}.7z")

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

    # XML
    xml_name = "LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml"
    xml_ok_path = ok_folder.joinpath(xml_name)
    if path.is_cloud_path(files_path()):
        xml_ok_path = str(xml_ok_path.download_to(tmp_path))
    else:
        xml_ok_path = str(xml_ok_path)

    xml_regex = f".*{xml_name}"
    xml_zip = files.read_archived_xml(zip_file, xml_regex)
    xml_tar = files.read_archived_xml(tar_file, r".*_MTL\.xml")
    xml_ok = etree.parse(xml_ok_path).getroot()
    ci.assert_xml_equal(xml_ok, xml_zip)
    ci.assert_xml_equal(xml_ok, xml_tar)

    # FILE + HTML
    html_zip_file = files_path().joinpath("productPreview.zip")
    html_tar_file = files_path().joinpath("productPreview.tar")
    html_name = "productPreview.html"
    html_ok_path = files_path().joinpath(html_name)
    if path.is_cloud_path(files_path()):
        html_ok_path = str(html_ok_path.download_to(tmp_path))
    else:
        html_ok_path = str(html_ok_path)

    html_regex = f".*{html_name}"

    # FILE
    file_zip = files.read_archived_file(html_zip_file, html_regex)
    file_tar = files.read_archived_file(html_tar_file, html_regex)
    html_ok = html.parse(html_ok_path).getroot()
    ci.assert_html_equal(html_ok, html.fromstring(file_zip))
    ci.assert_html_equal(html_ok, html.fromstring(file_tar))

    file_list = path.get_archived_file_list(html_zip_file)
    ci.assert_html_equal(
        html_ok,
        html.fromstring(
            files.read_archived_file(html_zip_file, html_regex, file_list=file_list)
        ),
    )

    # HTML
    html_zip = files.read_archived_html(html_zip_file, html_regex)
    html_tar = files.read_archived_html(html_tar_file, html_regex)
    ci.assert_html_equal(html_ok, html_zip)
    ci.assert_html_equal(html_ok, html_tar)
    ci.assert_html_equal(
        html_ok,
        files.read_archived_html(
            html_tar_file,
            html_regex,
            file_list=path.get_archived_file_list(html_tar_file),
        ),
    )

    # ERRORS
    with pytest.raises(TypeError):
        files.read_archived_file(targz_file, xml_regex)
    with pytest.raises(TypeError):
        files.read_archived_file(sz_file, xml_regex)
    with pytest.raises(FileNotFoundError):
        files.read_archived_file(zip_file, "cdzeferf")


def test_cp_rm(tmp_path):
    """Test CP/RM functions"""
    empty_tmp = os.listdir(tmp_path)

    # Copy file
    curr_path = os.path.realpath(__file__)
    file_1 = files.copy(curr_path, tmp_path)

    # Copy a folder onto a file: OSError: Copy error: File exists
    with pytest.raises(OSError):
        files.copy(tmp_path, file_1)

    file_2 = files.copy(curr_path, os.path.join(tmp_path, "test_pattern.py"))

    # Copy dir
    dir_path = os.path.dirname(curr_path)
    test_dir = files.copy(dir_path, os.path.join(tmp_path, os.path.basename(dir_path)))

    # Test copy
    assert os.path.isfile(file_1)
    assert os.path.isfile(file_2)
    assert os.path.isdir(test_dir)

    # Remove file
    files.remove(file_1)
    files.remove("non_existing_file.txt")
    files.remove_by_pattern(tmp_path, name_with_wildcard="*pattern*", extension="py")

    # Remove dir
    files.remove(test_dir)

    # Assert tempfile is empty
    assert os.listdir(tmp_path) == empty_tmp


@s3_env
def test_copy_cloud(tmp_path):
    zip_file = files_path().joinpath("test_zip.zip")
    zip_copy = files.copy(zip_file, tmp_path)
    assert zip_copy.is_file()
    assert zip_copy.name == "test_zip.zip"


def test_json(tmp_path):
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
        "U": 3.1415,
    }

    json_file = os.path.join(tmp_path, "test.json")

    # Save JSON
    files.save_json(test_dict, json_file)

    # Load JSON
    obj = files.read_json(json_file)

    assert (
        obj.pop("F") == test_dict.pop("F").value
    )  # Enum are stored following their value

    assert obj.pop("H") == str(
        test_dict.pop("H")
    )  # Enum are stored following their value
    assert obj == test_dict


def test_pickle(tmp_path):
    """Test pickle functions"""

    test_dict = {
        "A": 3,
        "B": np.zeros((3, 3)),
        "C": "str",
        "D": datetime.today(),
        "E": np.int64(15),
    }

    pkl_file = os.path.join(tmp_path, "test.pkl")

    # Save pickle
    files.save_obj(test_dict, pkl_file)

    # Load pickle
    obj = files.load_obj(pkl_file)

    # Test (couldn't compare the dicts as they contain numpy arrays)
    np.testing.assert_equal(obj, test_dict)


def test_hash_file_content():
    """Test hash_file_content"""
    file_content = "This is a test."

    # Test
    hashed = files.hash_file_content(file_content)

    # Test
    assert hashed == "16c5bf1fc5"
