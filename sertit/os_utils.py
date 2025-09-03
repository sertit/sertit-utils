from functools import cmp_to_key
from pathlib import Path
from typing import Union
import logging
import os
import pprint
import subprocess
import sys
from typing import Union
from sertit.logs import SU_NAME
from semver import Version
import re

LOGGER = logging.getLogger(SU_NAME)

def compare_str_containing_version(str1: str, str2: str):
    match_pattern = r"\d+\.\d+\.?\d+?"
    LOGGER.debug(f"Compare string {str1} with string {str2}")
    v1 = Version.parse(
        re.search(match_pattern, str1).group(), optional_minor_and_patch=True
    )
    v2 = Version.parse(
        re.search(match_pattern, str2).group(), optional_minor_and_patch=True
    )
    if v1 <= v2:
        return -1
    if v1 == v2:
        return 0
    else:
        return 1


def qgis_bin():
    # Find QGis in ProgramFiles
    qgis_dir = ""
    qgis_dirs = []
    parent_path_qgis = Path("C:/") / "Program Files"
    if parent_path_qgis.exists():
        qgis_dirs = [
            str(pt)
            for pt in parent_path_qgis.iterdir()
            if pt.stem.lower().startswith("qgis")
        ]

    if len(qgis_dirs) > 0:
        # Take the latest one available
        qgis_dirs = sorted(qgis_dir, key=cmp_to_key(compare_str_containing_version))
        qgis_valid_dirs = []
        for path in qgis_dirs:
            bin_path = Path(path) / "bin"
            if bin_path.exists():
                qgis_valid_dirs.append(bin_path)

        qgis_dir = None if len(qgis_valid_dirs) == 0 else qgis_valid_dirs[-1]

    # Find QGis in osgeo4w
    else:
        osgeo_path =  Path("C:/") / "osgeo4w" / "bin"
        qgis_dir = osgeo_path if osgeo_path.exists() else None

    return qgis_dir
