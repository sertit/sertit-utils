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
""" Script testing the miscellaneous functions """
import os

import pytest

from CI.SCRIPTS.script_utils import Polarization
from sertit import misc


def test_run_command():
    """Test run_command"""
    cmd = "cd .."
    misc.run_cli(
        cmd, in_background=False, cwd="/"
    )  # Just ensure no exception is thrown
    cmd = ["cd", ".."]
    misc.run_cli(cmd, in_background=True, cwd="/")  # Just ensure no exception is thrown


def test_get_function_name():
    """Test get_function_name"""
    assert misc.get_function_name() == "test_get_function_name"


def test_in_docker():
    """Test in_docker"""
    # Hack: in docker if ni linux
    misc.in_docker()  # Just hope it doesn't crash


def test_chdir():
    """Testing chdir functions"""
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    old_pwd = os.getcwd()
    with misc.chdir(curr_dir):
        pwd = os.getcwd()
        assert pwd == curr_dir

    assert os.getcwd() == old_pwd


def test_list_dict():
    """Test dict functions"""
    test_list = ["A", "T", "R", "", 3, None]
    test_dict = {"A": "T", "R": 3}
    test_list = misc.remove_empty_values(test_list)
    assert test_list == ["A", "T", "R", 3]

    # List to dict
    assert misc.list_to_dict(test_list) == test_dict

    # Nested set
    res_dict = {"A": "T", "R": 3, "B": {"C": {"D": "value"}}}
    misc.nested_set(test_dict, ["B", "C", "D"], "value")
    assert test_dict == res_dict

    # Mandatory keys
    misc.check_mandatory_keys(test_dict, ["A", "B"])  # True
    with pytest.raises(ValueError):
        misc.check_mandatory_keys(test_dict, ["C"])  # False

    # Find by key
    assert misc.find_by_key(test_dict, "D") == "value"


def test_enum():
    """Test ListEnum"""
    assert Polarization.list_values() == ["HH", "VV", "VH", "HV"]
    assert Polarization.list_names() == ["hh", "vv", "vh", "hv"]
    assert Polarization.from_value("HH") == Polarization.hh
    assert Polarization.from_value(Polarization.hh) == Polarization.hh

    with pytest.raises(ValueError):
        Polarization.from_value("ZZ")

    assert Polarization.convert_from(["HH", "vv", Polarization.hv]) == [
        Polarization.hh,
        Polarization.vv,
        Polarization.hv,
    ]
