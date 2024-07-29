from pathlib import Path
from typing import Union

import numpy as np
from cloudpathlib import CloudPath

from sertit import AnyPath
from sertit.types import AnyPathType, is_iterable


def test_types():
    assert AnyPathType == Union[Path, CloudPath]


def test_iterable():
    assert is_iterable((1, 2, 3))
    assert is_iterable([1, 2, 3])
    assert is_iterable({1, 2, 3})
    assert is_iterable(np.array([1, 2, 3]))
    assert is_iterable("1, 2, 3")
    assert not is_iterable(1)
    assert not is_iterable(AnyPath("1, 2, 3"))
