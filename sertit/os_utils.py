import logging
import re
from functools import cmp_to_key
from pathlib import Path

from sertit import strings
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def _compare_str_containing_version(str1: str, str2: str):
    """
    Compare two strings containing a software version, for example "qgis 3.34.5" and "qgis 3.35.6".

    Args:
        str1:
        str2:

    Returns:
        Let's say str1 and str2 contains respectively the versions v1 and v2. So, this function returns:
        -1 if v1 <= v2
        0 if v1 == v2
        1 if v1 >= v2

    """
    from semver import Version

    match_pattern = r"\d+\.\d+\.?\d+?"
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


def qgis_bin() -> Path:
    """
    Looking for qgis bin directory in Windows filesystem and return it.

    Returns: None if qgis bin directory is not found or a path to the qgis bin directory otherwise.
    """
    # Find QGis in ProgramFiles
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
    Looking for gdalbuildvrt exe from the path or inside qgis bin directory.

    Returns:
        str: gdalbuildvrt exe

    """
    qgis = qgis_bin()

    if qgis is not None:
        gdal_build_vrt_exe = strings.to_cmd_string(str(qgis / "gdalbuildvrt.exe"))
    else:
        gdal_build_vrt_exe = "gdalbuildvrt"

    return gdal_build_vrt_exe
