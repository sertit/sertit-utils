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
"""Operating system related utilities."""

import logging
import re
from functools import cmp_to_key
from pathlib import Path

from sertit import strings
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def _compare_str_containing_version(str1: str, str2: str) -> int:
    """
    Compare two strings containing a software version, for example "qgis 3.34.5" and "qgis 3.35.6".

    Args:
        str1 (str): First version string
        str2 (str): Second version string

    Returns:
        int: Let's say str1 and str2 contain respectively the versions v1 and v2.
            This function returns:
            -1 if v1 < v2
            0 if v1 == v2
            1 if v1 > v2

    """
    from semver import Version

    match_pattern = r"\d+\.\d+\.?\d+?"
    v1 = Version.parse(
        re.search(match_pattern, str1).group(), optional_minor_and_patch=True
    )
    v2 = Version.parse(
        re.search(match_pattern, str2).group(), optional_minor_and_patch=True
    )
    if v1 == v2:
        return 0
    elif v1 < v2:
        return -1
    else:
        return 1


def qgis_bin() -> Path | None:
    """
    Look for the QGIS bin directory in the Windows filesystem and return it.

    Returns:
        Path | None: None if the QGIS bin directory is not found,
            or the path to the QGIS bin directory otherwise.
    """
    # Find QGis in ProgramFiles
    qgis_dirs = []
    parent_path_qgis = Path("C:/") / "Program Files"
    if parent_path_qgis.exists():  # pragma: no cover
        qgis_dirs = [
            str(pt)
            for pt in parent_path_qgis.iterdir()
            if pt.stem.lower().startswith("qgis")
        ]

    if len(qgis_dirs) > 0:  # pragma: no cover
        # Take the latest one available
        qgis_dirs = sorted(qgis_dirs, key=cmp_to_key(_compare_str_containing_version))
        qgis_valid_dirs = []
        for path in qgis_dirs:
            bin_path = Path(path) / "bin"
            if bin_path.exists():
                qgis_valid_dirs.append(bin_path)

        qgis_dir = None if len(qgis_valid_dirs) == 0 else qgis_valid_dirs[-1]

    # Find QGis in osgeo4w
    else:
        osgeo_path = Path("C:/") / "osgeo4w" / "bin"
        qgis_dir = osgeo_path if osgeo_path.exists() else None

    return qgis_dir


def gdalbuildvrt_exe() -> str:
    """
    Look for the gdalbuildvrt executable from the PATH or inside the QGIS bin directory.

    Returns:
        str: The gdalbuildvrt executable

    """
    qgis = qgis_bin()

    if qgis is not None:  # pragma: no cover
        gdal_build_vrt_exe = strings.to_cmd_string(str(qgis / "gdalbuildvrt.exe"))
    else:
        gdal_build_vrt_exe = "gdalbuildvrt"

    return gdal_build_vrt_exe
