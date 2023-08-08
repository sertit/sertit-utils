from pathlib import Path
from typing import Union

from cloudpathlib import CloudPath

from sertit.types import AnyPathType


def test_types():
    assert AnyPathType == Union[Path, CloudPath]
