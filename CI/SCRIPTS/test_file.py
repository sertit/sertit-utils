""" Script testing the file_utils """
import os
import tempfile
import filecmp
import numpy as np
from datetime import datetime
from sertit_utils.core import file_utils
from CI.SCRIPTS import script_utils


def test_paths():
    """ Test path functions """
    # Relative path
    curr_file = os.path.realpath(__file__)
    curr_dir = os.path.dirname(curr_file)
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


def test_extract():
    """ Test extracting functions """
    tmp_dir = tempfile.TemporaryDirectory()

    # Archives
    zip_file = os.path.join(script_utils.get_ci_data_path(), "test_zip.zip")
    tar_file = os.path.join(script_utils.get_ci_data_path(), "test_tar.tar")
    tar_gz_file = os.path.join(script_utils.get_ci_data_path(), "test_targz.tar.gz")
    archives = [zip_file, tar_file, tar_gz_file]

    # Core dir
    core_dir = os.path.join(script_utils.get_ci_data_path(), "core")

    # Extract
    extracted_dirs = file_utils.extract_files(archives, tmp_dir.name)

    # Test
    for ex_dir in extracted_dirs:
        dcmp = filecmp.dircmp(core_dir, ex_dir)
        assert os.path.isdir(ex_dir)
        assert dcmp.left_only == []
        assert dcmp.right_only == []

    # Cleanup
    tmp_dir.cleanup()


def test_get_file_name():
    """ Test get_file_name """
    file_name = file_utils.get_file_name(__file__)
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
                 "C": "str",
                 "D": datetime.today(),
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
    pattern = "file"
    extension = ".py"
    filename_only = False
    get_list = False
    exact_name = False

    # Test
    file = file_utils.get_file_in_dir(folder, pattern, extension, filename_only, get_list, exact_name)

    assert file == __file__


def test_hash_file_content():
    """ Test hash_file_content """
    file_content = "This is a test."

    # Test
    hashed = file_utils.hash_file_content(file_content)

    # Test
    assert hashed == '16c5bf1fc5'
