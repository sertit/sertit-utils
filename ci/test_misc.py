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
"""Script testing the miscellaneous functions"""

import os

import pytest

from ci.script_utils import Polarization
from sertit import __version__, ci, misc
from sertit.misc import compare, compare_version

ci.reduce_verbosity()


def test_run_command():
    """Test run_command"""
    cmd = "cd .."
    misc.run_cli(
        cmd, in_background=False, cwd="/"
    )  # Just ensure no exception is thrown
    cmd = ["cd", ".."]
    misc.run_cli(cmd, in_background=True, cwd="/")  # Just ensure no exception is thrown

    with pytest.raises(TypeError):
        misc.run_cli(
            1, in_background=True, cwd="/"
        )  # Just ensure no exception is thrown


def test_get_function_name():
    """Test get_function_name"""
    ci.assert_val(
        misc.get_function_name(), "test_get_function_name", "get_function_name"
    )


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

    ci.assert_val(os.getcwd(), old_pwd, "test_chdir")


def test_select_prune_dict():
    keys = ["a", "b", "c"]
    d = {"a": 1, "b": 2, "c": 3, "d": 4}

    # Select dict
    ci.assert_val(misc.select_dict(d, keys), {"a": 1, "b": 2, "c": 3}, "Select dict")

    # Prune dict
    ci.assert_val(misc.prune_dict(d, keys), {"d": 4}, "Prune dict")


def test_strip_kwargs():
    """Test the strip_kwargs decorator"""
    calls = []

    # Engine rejecting a keyword argument with a TypeError (e.g. Python ufunc)
    @misc.strip_kwargs
    def engine_type_error(a, **kwargs):
        calls.append(kwargs.copy())
        if "bad" in kwargs:
            raise TypeError(
                "engine_type_error() got an unexpected keyword argument 'bad'"
            )
        return "ok"

    ci.assert_val(engine_type_error(1, bad=0, good=1), "ok", "Strip kwargs (TypeError)")
    ci.assert_val(
        calls, [{"bad": 0, "good": 1}, {"good": 1}], "Retry without bad kwarg"
    )

    # Engine rejecting an option with a ValueError (e.g. pyogrio)
    calls.clear()

    @misc.strip_kwargs
    def engine_value_error(path, **kwargs):
        calls.append(kwargs.copy())
        if "badopt" in kwargs:
            # The engine reports the option name in uppercase
            raise ValueError("unrecognized option 'BADOPT' for driver 'ESRI Shapefile'")
        return "ok"

    ci.assert_val(
        engine_value_error("path", badopt=0, good=1),
        "ok",
        "Strip kwargs (ValueError, case-insensitive)",
    )
    ci.assert_val(
        calls, [{"badopt": 0, "good": 1}, {"good": 1}], "Retry without bad option"
    )

    # Any other error is propagated as is
    @misc.strip_kwargs
    def engine_div_zero(**kwargs):
        return 1 / 0

    with pytest.raises(ZeroDivisionError):
        engine_div_zero(bad=0)

    @misc.strip_kwargs
    def engine_other_error(**kwargs):
        raise ValueError("some other error")

    with pytest.raises(ValueError, match="some other error"):
        engine_other_error(bad=0)

    # Function name is kept by the decorator
    ci.assert_val(
        engine_type_error.__name__, "engine_type_error", "Decorator keeps name"
    )


def test_list_dict():
    """Test dict functions"""
    test_list = ["A", "T", "R", "", 3, None, "0"]
    test_dict = {"A": "T", "R": 3, "B": None, "C": "", "D": "0"}

    # Remove empty values (list)
    test_list = misc.remove_empty_values(test_list, other_empty_values=["0"])
    ci.assert_val(test_list, ["A", "T", "R", 3], "Remove empty values from list")

    # Remove empty values (dict)
    test_dict = misc.remove_empty_values(test_dict, other_empty_values=["", "0"])
    ci.assert_val(test_dict, {"A": "T", "R": 3}, "Remove empty values from dict")

    # List to dict
    ci.assert_val(misc.list_to_dict(test_list), test_dict, "List to dict")

    # Nested set
    res_dict = {"A": "T", "R": 3, "B": {"C": {"D": "value"}}}
    misc.nested_set(test_dict, ["B", "C", "D"], "value")
    ci.assert_val(test_dict, res_dict, "Nested set")

    # Mandatory keys
    misc.check_mandatory_keys(test_dict, ["A", "B"])  # True
    with pytest.raises(ValueError):
        misc.check_mandatory_keys(test_dict, ["C"])  # False

    # Find by key
    ci.assert_val(misc.find_by_key(test_dict, "D"), "value", "Find by key")


def test_enum():
    """Test ListEnum"""

    ci.assert_val(Polarization.list_values(), ["HH", "VV", "VH", "HV"], "Values")
    ci.assert_val(Polarization.list_names(), ["hh", "vv", "vh", "hv"], "Names")
    ci.assert_val(Polarization.from_value("HH"), Polarization.hh, "From string value")
    ci.assert_val(
        Polarization.from_value(Polarization.hh), Polarization.hh, "From enum value"
    )

    with pytest.raises(ValueError):
        Polarization.from_value("ZZ")

    ci.assert_val(
        Polarization.convert_from(["HH", "vv", Polarization.hv]),
        [
            Polarization.hh,
            Polarization.vv,
            Polarization.hv,
        ],
        "convert_from",
    )

    ci.assert_val(
        Polarization.convert_from("HH"),
        [
            Polarization.hh,
        ],
        "convert_from no list",
    )

    with pytest.raises(TypeError):
        Polarization.convert_from(["ZZ"])


def test_unique():
    """Test unique function"""
    non_unique = [1, 2, 20, 6, 210, 2, 1]
    unique = [1, 2, 20, 6, 210]
    ci.assert_val(unique, misc.unique(non_unique), "Unique")


def test_comparisons():
    """Test comparisons"""
    # -- True --
    # Equalities
    assert compare(1, 1, "==")
    assert compare(1, 1, ">=")
    assert compare(1, 1, "<=")

    # Inequalities
    assert compare(1, 2, "<")
    assert compare(1, 0.5, ">")

    # -- False --
    # Equalities
    assert not compare(1, 2, "==")
    assert not compare(1, 5, ">=")
    assert not compare(1, 0.5, "<=")

    # Inequalities
    assert not compare(1, 0.5, "<")
    assert not compare(1, 2, ">")


def test_compare_versions():
    """"""
    assert compare_version("geopandas", "0.10.0", ">=")
    assert compare_version("geopandas", "5.0.0", "<")
    assert compare_version(__version__, "5.0.0", "<")
    assert compare_version(__version__, "1.0.0", ">")
    from packaging.version import InvalidVersion

    with pytest.raises((TypeError, InvalidVersion)):
        assert compare_version(__version__, 1, ">")
