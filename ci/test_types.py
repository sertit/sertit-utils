from pathlib import Path

import numpy as np
from cloudpathlib import CloudPath

from sertit import AnyPath, ci
from sertit.types import AnyPathType, is_iterable, make_iterable


def test_types():
    """Test some type aliases"""
    assert AnyPathType == Path | CloudPath


def test_is_iterable():
    """Test is_iterable"""
    assert is_iterable((1, 2, 3))
    assert is_iterable([1, 2, 3])
    assert is_iterable({1, 2, 3})
    assert is_iterable(np.array([1, 2, 3]))
    assert not is_iterable("1, 2, 3")
    assert is_iterable("1, 2, 3", str_allowed=True)
    assert not is_iterable(1)
    assert not is_iterable(AnyPath("1, 2, 3"))


def test_make_iterable():
    """Test make_iterable"""

    def assert_mi(val, is_it=True, **kwargs):
        true_val = [val] if not is_it else val
        out_val = make_iterable(val, **kwargs)

        try:
            ci.assert_val(out_val, true_val, "")
        except AssertionError:
            if not kwargs.get("convert_none", False) and val is None:
                ci.assert_val(out_val, None, "")

    assert_mi((1, 2, 3))
    assert_mi([1, 2, 3])
    assert_mi({1, 2, 3})
    assert_mi(np.array([1, 2, 3]))
    assert_mi("1, 2, 3", str_allowed=True, is_it=False)
    assert_mi("1, 2, 3", str_allowed=True)
    assert_mi("1, 2, 3", str_allowed=False, is_it=False)
    assert_mi(1, is_it=False)
    assert_mi(None, convert_none=True, is_it=False)
    assert_mi(None, convert_none=True)
    assert_mi(None, convert_none=False, is_it=False)
