# Copyright 2024, SERTIT-ICube - France, https://sertit.unistra.fr/
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
from datetime import date, datetime

import numpy as np
from CI.SCRIPTS.script_utils import Polarization

from sertit import AnyPath, ci, files

ci.reduce_verbosity()


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


def test_hash_file_content():
    """Test hash_file_content"""
    file_content = "This is a test."

    # Test
    hashed = files.hash_file_content(file_content)

    # Test
    assert hashed == "16c5bf1fc5"
