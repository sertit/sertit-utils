from pathlib import Path
from typing import Union

import numpy as np
from cloudpathlib import CloudPath

from sertit import AnyPath
from sertit.types import AnyPathType, is_iterable, make_iterable


def test_types():
    """Test some type aliases"""
    assert AnyPathType == Union[Path, CloudPath]


def test_is_iterable():
    """Test is_iterable"""
    assert is_iterable((1, 2, 3))
    assert is_iterable([1, 2, 3])
    assert is_iterable({1, 2, 3})
    assert is_iterable(np.array([1, 2, 3]))
    assert is_iterable("1, 2, 3")
    assert not is_iterable(1)
    assert not is_iterable(AnyPath("1, 2, 3"))


def test_make_iterable():
    """Test make_iterable"""

    def assert_mi(val, is_it=True):
        if not is_it:
            val = [val]

        comp = val == make_iterable(val)
        try:
            assert comp
        except ValueError:
            assert all(comp)

    assert_mi((1, 2, 3))
    assert_mi([1, 2, 3])
    assert_mi({1, 2, 3})
    assert_mi(np.array([1, 2, 3]))
    assert_mi("1, 2, 3")
    assert_mi(1, is_it=False)
    assert_mi(AnyPath("1, 2, 3"), is_it=False)
