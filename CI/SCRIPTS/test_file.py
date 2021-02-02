""" Script testing the file_utils """
import logging
import os
import tempfile

import numpy as np
from datetime import datetime, date
from sertit_utils.core import file_utils, sys_utils
from CI.SCRIPTS import script_utils

from sertit_utils.core.log_utils import SU_NAME

LOGGER = logging.getLogger(SU_NAME)
FILE_DATA = os.path.join(script_utils.get_ci_data_path(), "file_utils")


def test_paths():
    """ Test path functions """
    curr_file = os.path.realpath(__file__)
    curr_dir = os.path.dirname(curr_file)
    with sys_utils.chdir(curr_dir):
        # Relative path
        curr_rel_path = file_utils.real_rel_path(curr_file, curr_dir)
        assert curr_rel_path == os.path.join(".", os.path.basename(__file__))

        # Abspath
        abs_file = file_utils.to_abspath(curr_rel_path)
        assert abs_file == curr_file

        # Listdir abspath
        list_abs = file_utils.listdir_abspath(curr_dir)
        assert curr_file in list_abs

        # Root path
        assert abs_file.startswith(file_utils.get_root_path())


def test_archive():
    """ Test extracting functions """
    tmp_dir = tempfile.TemporaryDirectory()

    # Archives
    zip_file = os.path.join(FILE_DATA, "test_zip.zip")
    tar_file = os.path.join(FILE_DATA, "test_tar.tar")
    tar_gz_file = os.path.join(FILE_DATA, "test_targz.tar.gz")
    archives = [zip_file, tar_file, tar_gz_file]

    # Core dir
    core_dir = os.path.join(FILE_DATA, "core")

    # Extract
    extracted_dirs = file_utils.extract_files(archives, tmp_dir.name)

    # Test
    for ex_dir in extracted_dirs:
        script_utils.assert_dir_equal(core_dir, ex_dir)

    # Archive
    archive_base = os.path.join(tmp_dir.name, "archive")
    for fmt in ["zip", "tar", "gztar"]:
        archive_fn = file_utils.archive(folder_path=core_dir,
                                        archive_path=archive_base,
                                        fmt=fmt)
        out = file_utils.extract_file(archive_fn, tmp_dir.name)
        if fmt == "zip":
            script_utils.assert_dir_equal(core_dir, out)
        else:
            # For tar and tar.gz, an additional folder is created because these formats dont have any given file tree
            out_dir = file_utils.listdir_abspath(out)[0]
            script_utils.assert_dir_equal(core_dir, out_dir)

        # Remove out directory in order to avoid any interferences
        file_utils.remove(out)

    # Add to zip
    zip_out = archive_base + ".zip"
    core_copy = file_utils.copy(core_dir, os.path.join(tmp_dir.name, "core2"))
    file_utils.add_to_zip(zip_out, core_copy)

    # Extract
    unzip_out = os.path.join(tmp_dir.name, "out")
    file_utils.extract_file(zip_out, unzip_out)

    # Test
    unzip_dirs = file_utils.listdir_abspath(unzip_out)
    LOGGER.info(unzip_dirs)
    assert len(unzip_dirs) == 2
    script_utils.assert_dir_equal(unzip_dirs[0], unzip_dirs[1])

    # Cleanup
    tmp_dir.cleanup()


def test_get_file_name():
    """ Test get_file_name """
    file_name = file_utils.get_file_name(__file__)
    assert file_name == "test_file"
    file_name = file_utils.get_file_name(__file__ + "\\")
    assert file_name == "test_file"
    file_name = file_utils.get_file_name(__file__ + "/")
    assert file_name == "test_file"


def test_cp_rm():
    """ Test CP/RM functions """
    tmp_dir = tempfile.TemporaryDirectory()
    empty_tmp = os.listdir(tmp_dir.name)

    # Copy file
    curr_path = os.path.realpath(__file__)
    file_1 = file_utils.copy(curr_path, tmp_dir.name)
    file_2 = file_utils.copy(curr_path, os.path.join(tmp_dir.name, "test_pattern.py"))

    # Copy dir
    dir_path = os.path.dirname(curr_path)
    test_dir = file_utils.copy(dir_path, os.path.join(tmp_dir.name, os.path.basename(dir_path)))

    # Test copy
    assert os.path.isfile(file_1)
    assert os.path.isfile(file_2)
    assert os.path.isdir(test_dir)

    # Remove file
    file_utils.remove(file_1)
    file_utils.remove("non_existing_file.txt")
    file_utils.remove_by_pattern(tmp_dir.name, name_with_wildcard="*pattern*", extension="py")

    # Remove dir
    file_utils.remove(test_dir)

    # Assert tempfile is empty
    assert os.listdir(tmp_dir.name) == empty_tmp

    # Cleanup
    tmp_dir.cleanup()


def test_find_files():
    """ Test find_files """
    names = os.path.basename(__file__)
    root_paths = script_utils.get_proj_path()
    max_nof_files = 1
    get_as_str = True

    # Test
    path = file_utils.find_files(names, root_paths, max_nof_files, get_as_str)

    assert path == os.path.realpath(__file__)


def test_json():
    """ Test json functions """
    test_dict = {"A": 3,
                 "C": "m2",  # Can be parsed as a date, we do not want that !
                 "D": datetime.today(),
                 "Dbis": date.today(),
                 "E": np.int64(15)}

    tmp_dir = tempfile.TemporaryDirectory()
    json_file = os.path.join(tmp_dir.name, "test.json")

    # Save pickle
    file_utils.save_json(json_file, test_dict)

    # Load pickle
    obj = file_utils.read_json(json_file)

    # Clean-up
    tmp_dir.cleanup()

    assert obj == test_dict


def test_pickle():
    """ Test pickle functions """

    test_dict = {"A": 3,
                 "B": np.zeros((3, 3)),
                 "C": "str",
                 "D": datetime.today(),
                 "E": np.int64(15)}

    tmp_dir = tempfile.TemporaryDirectory()
    pkl_file = os.path.join(tmp_dir.name, "test.pkl")

    # Save pickle
    file_utils.save_obj(test_dict, pkl_file)

    # Load pickle
    obj = file_utils.load_obj(pkl_file)

    # Clean-up
    tmp_dir.cleanup()

    # Test (couldn't compare the dicts as they contain numpy arrays)
    np.testing.assert_equal(obj, test_dict)


def test_get_file_in_dir():
    """ Test get_file_in_dir """
    # Get parent dir
    folder = os.path.dirname(os.path.realpath(__file__))

    # Test
    file = file_utils.get_file_in_dir(folder, "file", ".py", filename_only=False, get_list=True, exact_name=False)
    filename = file_utils.get_file_in_dir(folder, file_utils.get_file_name(__file__), "py",
                                          filename_only=True, get_list=False, exact_name=True)

    assert file[0] == __file__
    assert filename == os.path.basename(__file__)


def test_hash_file_content():
    """ Test hash_file_content """
    file_content = "This is a test."

    # Test
    hashed = file_utils.hash_file_content(file_content)

    # Test
    assert hashed == '16c5bf1fc5'
